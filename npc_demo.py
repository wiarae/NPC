import torch
# from diffusers import FluxPipeline
from models.flux_stepwise import FluxPipelineStepwise, FluxPipelineStepwiseDNG, FluxPipeline_new, FluxPipelineStep
import time 
import argparse
import os, io, re, json, base64
from PIL import Image
from typing import List, Dict, Any
from openai import OpenAI
from pathlib import Path
import shutil
from torch.nn import functional as F
from transformers import AutoTokenizer, T5EncoderModel

DEFAULT_FALLBACKS = [
    "cgi", "3d render", "cartoon", "illustration", "digital painting",
    "text", "logo", "watermark", "jpeg artifacts",
    "blurry", "low quality", "overexposed", "underexposed", "harsh lighting"
]

# ========= Proxy-SNR helpers (FLUX T5 text-only) =========

_SNR_CACHE = {}

_NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
    "a":"1","an":"1"
}
_STOP = {
    "a","an","the","and","or","with","of","in","on","at","to","for","from",
    "photo","picture","image","scene","view","close-up","closeup","shot"
}

def _dtype_map(name: str):
    return dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16).get(name, torch.float32)

def _snr_load_encoder(repo: str, dtype: str, device: str):
    key = (repo, dtype, device)
    if key in _SNR_CACHE:
        return _SNR_CACHE[key]
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer_2")
    te2 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2",
                                         torch_dtype=_dtype_map(dtype))
    te2.to(device); te2.eval()
    _SNR_CACHE[key] = (tok, te2)
    return tok, te2

def _mean_pool_t5(encoder_out: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.float().unsqueeze(-1)      # [B,L,1]
    summed = (encoder_out * mask).sum(dim=1)    # [B,D]
    denom  = mask.sum(dim=1).clamp(min=1e-6)    # [B,1]
    return summed / denom                       # [B,D]

def _encode_sentence(tok, te2, text: str, device: str) -> torch.Tensor:
    toks = tok(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = te2(**toks)
    last = out.last_hidden_state                 # [1,L,D]
    sent = _mean_pool_t5(last, toks["attention_mask"]).squeeze(0)  # [D]
    return sent

def _phrase_vec_hidden(tok, te2, phrase: str, device: str) -> torch.Tensor:
    toks = tok(phrase, padding=False, truncation=True, return_tensors="pt", add_special_tokens=False)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = te2(**toks)
    last = out.last_hidden_state
    mask = toks.get("attention_mask", torch.ones(last.shape[:2], device=last.device))
    vec = _mean_pool_t5(last, mask).squeeze(0)
    return vec

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

def _singularize(w: str) -> str:
    if re.search(r'ies$', w): return re.sub(r'ies$', 'y', w)
    if re.search(r'(xes|ches|shes|ses|zes)$', w): return re.sub(r'es$', '', w)
    if re.search(r's$', w) and not re.search(r'ss$', w): return w[:-1]
    return w

def _pluralize(w: str) -> str:
    if re.search(r'([sxz]|[cs]h)$', w): return w + 'es'
    if re.search(r'[^aeiou]y$', w): return re.sub(r'y$', 'ies', w)
    return w + 's'

def auto_extract_qty_objs(prompt: str):
    text = re.sub(r'[^0-9a-zA-Z\- ]+', ' ', prompt.lower())
    qty_tokens = []
    obj_candidates = []

    pat = re.compile(r'\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b\s+(?:of\s+)?\b([a-z][a-z\-]*)')
    for m in pat.finditer(text):
        q_raw = m.group(1)
        q_norm = _NUM_WORDS.get(q_raw, q_raw)  
    
        if q_raw in _NUM_WORDS:
            qty_tokens.append(q_raw)       # Word form
            qty_tokens.append(_NUM_WORDS[q_raw])  # Numeric form
        else:
            qty_tokens.append(q_raw)       # Numeric form
        noun = m.group(2)
        if noun not in _STOP and len(noun) > 1:
            obj_candidates.append(noun)

    # Fallback: Extract nouns after prepositions if no quantity found
    if not obj_candidates:
        for w in text.split():
            if w.isalpha() and w not in _STOP and len(w) > 2:
                obj_candidates.append(w)

    # Refine: Include both singular and plural forms
    objs_set = set()
    for w in obj_candidates:
        s = _singularize(w)
        p = _pluralize(s)
        objs_set.update([s, p])

    # Default values for quantity
    if not qty_tokens:
        qty_tokens = ["one","two","three","1","2","3"]

    # Default values for objects
    if not objs_set:
        last_word = (text.split() or ["object"])[-1]
        s = _singularize(last_word)
        objs_set.update([s, _pluralize(s)])

    # Unique while preserving order
    def _unique(seq): 
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return _unique(qty_tokens), _unique(list(objs_set))

def snr_rank_negatives(
    cond: str,
    cands: list[str],
    qty: list[str],
    objs: list[str],
    repo: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cpu",
    dtype: str = "float32"
):
    """
    Returns:
      sorted_list: Candidate list sorted by mean-normalized proxy score S = s_qty + s_obj - s_cand
      rows_sorted: Detailed score list (sorted by score descending)
    """
    if not cands:
        return [], []

    device_eff = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    tok, te2 = _snr_load_encoder(repo, dtype, device_eff)

    with torch.no_grad():
        e_pos = _encode_sentence(tok, te2, cond, device_eff)
        qty_vecs = [_phrase_vec_hidden(tok, te2, w, device_eff) for w in qty]
        obj_vecs = [_phrase_vec_hidden(tok, te2, w, device_eff) for w in objs]

    n_q = len(qty_vecs)
    n_o = len(obj_vecs)

    rows = []
    for cand in cands:
        with torch.no_grad():
            e_neg = _encode_sentence(tok, te2, cand, device_eff)
            delta = e_pos - e_neg

            v_cand = _phrase_vec_hidden(tok, te2, cand, device_eff)
            s_cand = _cos(delta, v_cand)

            # Normalization: Scale scores to [-1, 1] range
            if n_q > 0:
                s_qty = sum((_cos(delta, v) for v in qty_vecs), 0.0) / n_q
            else:
                s_qty = 0.0

            if n_o > 0:
                s_obj = sum((_cos(delta, v) for v in obj_vecs), 0.0) / n_o
            else:
                s_obj = 0.0

            contrast_qty = s_qty - s_cand
            retention_obj = s_obj

            # Final Score calculation
            score = s_qty + s_obj - s_cand

            rows.append({
                "candidate": cand,
                "s_qty": float(s_qty),
                "s_obj": float(s_obj),
                "s_cand": float(s_cand),
                "contrast_qty": float(contrast_qty),
                "retention_obj": float(retention_obj),
                "score": float(score),
            })

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    sorted_list = [r["candidate"] for r in rows_sorted]
    return sorted_list, rows_sorted

def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())

def load_correct_prompts(json_path: str) -> set[str]:
    """
    Returns a set of prompts where correct == 1 from the evaluation JSON.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[warn] eval json not found: {json_path}")
        return set()
    except Exception as e:
        print(f"[warn] failed to parse {json_path}: {e}")
        return set()

    items = data.get("overall_results", [])
    out = set()
    for obj in items:
        if not isinstance(obj, dict):
            continue
        res = obj.get("result", {})
        try:
            correct = int(res.get("correct", 0))
        except Exception:
            correct = 0
        if correct == 1:
            p = obj.get("prompt", "")
            if p:
                out.add(_norm(p))
    return out

def should_skip_prompt(prompt: str, correct_set: set[str]) -> bool:
    return _norm(prompt) in correct_set

def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def gpt_negatives_candidates(
    positive_prompt: str,
    caption: str,
    k: int,
    system_txt: str,
    user_tpl: str,
    model: str,
    failure_json: str = "",
    temperature: float = 0.12
) -> List[str]:
    client = OpenAI()

    user_txt = (user_tpl
                .replace("$positive_prompt", positive_prompt)
                .replace("$caption", caption)
                .replace("$k", str(k))
                .replace("$failure_json", failure_json))

    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user_txt},
        ],
    )
    raw = (r.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(raw)
    cands = data.get("candidates", [])
    
    cleaned = []
    seen = set()
    for s in cands:
        if not isinstance(s, str):
            continue
        s2 = " ".join(s.strip().lower().split())
        if not s2:
            continue
        # Length check: 1-6 words
        if not (1 <= len(s2.split()) <= 6):
            continue
        # Negative word filter
        if any(tok in s2.split() for tok in ("no", "without", "not")):
            continue
        # Remove trailing punctuation
        while s2 and s2[-1] in ",.;:!?":
            s2 = s2[:-1].rstrip()
        if not s2:
            continue
        # Duplicate filter
        if s2 in seen:
            continue
        seen.add(s2)
        cleaned.append(s2)
    return cleaned[:k]

def read_prompts(txt_path: Path):
    prompts = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def gpt_caption(image: Image.Image, system_txt: str, user_txt: str,
                model: str, temperature: float = 0.2) -> str:
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_txt},
            {"role": "user", "content": [
                {"type": "text", "text": user_txt},
                {"type": "image_url", "image_url": {"url": to_data_url(image)}},
            ]}
        ],
    )
    return (r.choices[0].message.content or "").strip()

def slugify(text: str, max_len: int = 60) -> str:
    # Basic filename sanitization
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-_.]", "", text, flags=re.UNICODE)
    return (text[:max_len] or "image").strip("._")

def metadata_to_explanation(metadata: dict) -> str:
    parts = []

    def fmt(item: dict) -> str:
        cls = item["class"]
        count = item.get("count", 1)
        color = item.get("color")
        region = item.get("region")
        size = item.get("size")
        noun = f"{count} {cls + 's' if count > 1 else cls}"
        desc = []
        if color: desc.append(f"{color}-colored")
        if size:  desc.append(size)
        if desc:
            noun = f"{' '.join(desc)} {noun}"
        if region:
            return f"{noun} located in the {region} part of the image"
        return f"{noun} present in the image"

    for it in metadata.get("include", []):
        parts.append(f"- {fmt(it)}.")
    for it in metadata.get("exclude", []):
        cls = it["class"]
        count = it.get("count", 1)
        noun = f"{cls + 's' if count > 1 else cls}"
        parts.append(f"- No more than {count - 1} {noun} should appear.")
    return "This image should contain:\n" + "\n".join(parts)

def call_verifier_with_files(
    image: Image.Image,
    instruction: str,
    checklist: str,
    system_txt: str,
    user_tpl: str,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_retries: int = 3
) -> Dict[str, Any]:
    client = OpenAI()
    user_filled = (user_tpl
                   .replace("$instruction", instruction)
                   .replace("$checklist", checklist or ""))

    msg = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": [
            {"type": "text", "text": user_filled},
            {"type": "image_url", "image_url": {"url": to_data_url(image, fmt="JPEG")}},
        ]}
    ]

    last_err = None
    for _ in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=msg,
                max_tokens=200
            )
            data_raw = r.choices[0].message.content or "{}"
            data = json.loads(data_raw)

            # Safe parsing
            corr = 1 if int(data.get("correct", 0)) == 1 else 0
            # score expected 0-1, default to 0.0 if missing
            try:
                score = float(data.get("score", 0.0))
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))

            reason = str(data.get("reason", ""))
            return {"correct": corr, "score": score, "reason": reason}
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return {"correct": -1, "score": 0.0, "reason": f"verifier error: {last_err}"}

def _clean_candidates(cands: List[str], k: int) -> List[str]:
    cleaned, seen = [], set()
    for s in cands or []:
        if not isinstance(s, str):
            continue
        s2 = " ".join(s.strip().lower().split())
        if not s2:
            continue
        if not (1 <= len(s2.split()) <= 6):
            continue
        toks = s2.split()
        if any(tok in toks for tok in ("no", "without", "not")):
            continue
        while s2 and s2[-1] in ",.;:!?":
            s2 = s2[:-1].rstrip()
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        cleaned.append(s2)
        if len(cleaned) >= k:
            break
    return cleaned

def _render_user(user_tpl: str, variables: Dict[str, Any], k: int) -> str:
    txt = user_tpl
    for key, val in (variables or {}).items():
        repl = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
        txt = txt.replace(f"${key}", repl)
    txt = txt.replace("$k", str(k))
    return txt

def _parse_float(x, default=None):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _clean_phrase(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s2 = " ".join(s.strip().lower().split())
    if not s2:
        return None
    if not (1 <= len(s2.split()) <= 6):
        return None
    toks = s2.split()
    if any(tok in toks for tok in ("no", "without", "not")):
        return None
    while s2 and s2[-1] in ",.;:!?":
        s2 = s2[:-1].rstrip()
    return s2 or None

def _clean_and_sort_scored(items: List[dict], k: int) -> List[dict]:
    out, seen = [], set()
    for it in items or []:
        txt = _clean_phrase(it.get("text", it.get("candidate", "")))
        if not txt: 
            continue
        if txt in seen:
            continue
        seen.add(txt)

        vqa = _parse_float(it.get("vqa"), default=None)
        llm = _parse_float(it.get("llm"), default=None)
        imp = _parse_float(it.get("importance"), default=None)

        # Defaults and clamping
        if vqa is None: vqa = 0.0
        if llm is None: llm = 0.0
        vqa = max(0.0, min(1.0, vqa))
        llm = max(0.0, min(1.0, llm))
        if imp is None:
            imp = vqa - llm
        else:
            imp = max(-1.0, min(1.0, imp))

        out.append({"text": txt, "vqa": vqa, "llm": llm, "importance": imp})

    # Sort by Importance -> VQA -> Length (Shorter first on ties)
    out.sort(key=lambda d: (d["importance"], d["vqa"], -len(d["text"])), reverse=True)
    return out[:k]

def _llm_scored_candidates_from_templates(system_txt: str, user_tpl: str,
                                          variables: Dict[str, Any],
                                          model: str, k: int,
                                          temperature: float = 0.12) -> Dict[str, Any]:
    client = OpenAI()
    user_txt = _render_user(user_tpl, variables, k)
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user_txt},
        ],
    )
    raw = (r.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)

    scored_raw = data.get("scored", [])
    scored = _clean_and_sort_scored(scored_raw, k)

    cands = data.get("candidates", [])
    cleaned = _clean_candidates(cands, k)
    if not cleaned:
        cleaned = [d["text"] for d in scored]

    return {"candidates": cleaned[:k], "scored": scored}

def _llm_candidates_from_templates(system_txt: str, user_tpl: str,
                                   variables: Dict[str, Any],
                                   model: str, k: int,
                                   temperature: float = 0.12) -> List[str]:
    client = OpenAI()
    user_txt = _render_user(user_tpl, variables, k)
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user_txt},
        ],
    )
    raw = (r.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    cands = data.get("candidates", [])
    return _clean_candidates(cands, k)

def _combine_pairs(a: List[str], b: List[str], k: int, mode: str = "zip") -> List[str]:
    pairs: List[str] = []
    if mode == "cartesian":
        for x in a:
            for y in b:
                pairs.append(f"{x}, {y}")
    else:
        for x, y in zip(a, b):
            pairs.append(f"{x}, {y}")

    out, seen = [], set()
    for s in pairs:
        if s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= k:
                break
    return out

def generate_negative_pairs(
    positive_prompt: str,
    caption: str,
    failure_json: Any,
    k: int,
    prompt_dir: str = "gpt_prompts",
    caption_model: str = "gpt-4o-mini",
    failure_model: str = "gpt-4o-mini",
    combine_mode: str = "zip"
) -> Dict[str, List[str]]:

    # 0) Normalize failure_reason
    if isinstance(failure_json, dict):
        failure_reason = failure_json.get("result", {}).get("reason", "") or failure_json.get("reason", "")
    else:
        failure_reason = str(failure_json or "")

    # 1) Load templates
    sys_path = os.path.join(prompt_dir, "caption_candidates.system.txt")
    usr_path = os.path.join(prompt_dir, "caption_candidates.user.txt")

    neg_sys = read_txt(sys_path)
    neg_usr = read_txt(usr_path)
    
    # 2) Variable binding
    variables = {
        "positive_prompt": positive_prompt,
        "caption": caption,
        "failure_reason": failure_reason,
        "fallbacks": DEFAULT_FALLBACKS,
    }

    # 3) LLM call
    cands = _llm_candidates_from_templates(
        system_txt=neg_sys,
        user_tpl=neg_usr,
        variables=variables,
        model=caption_model,
        k=k,
        temperature=0.12
    )

    print(cands)
    return {
        "combined": cands
    }

def load_failure_index(json_path: str):
    """
    Returns:
    - by_prompt: normalized prompt -> record
    - by_id: index -> record
    - correct_set: set of normalized prompts where correct == 1
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[warn] eval json not found: {json_path}")
        return {}, {}, set()
    except Exception as e:
        print(f"[warn] failed to parse {json_path}: {e}")
        return {}, {}, set()

    items = data.get("overall_results", [])
    by_prompt, by_id, correct_set = {}, {}, set()
    for obj in items:
        if not isinstance(obj, dict):
            continue

        p = obj.get("prompt", "")
        if p:
            by_prompt[_norm(p)] = obj

        try:
            _id = int(obj.get("id", -1))
            if _id >= 0:
                by_id[_id] = obj
                by_id[_id - 1] = obj
        except Exception:
            pass

        try:
            res = obj.get("result", {})
            if int(res.get("correct", 0)) == 1 and p:
                correct_set.add(_norm(p))
        except Exception:
            pass

    return by_prompt, by_id, correct_set

def main(args):
    pipe = FluxPipelineStep.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    PROMPT = args.prompt

    if not PROMPT.strip():
        raise ValueError("--prompt must be a non-empty string")

    # === Templates ===
    cap_sys = read_txt(os.path.join(args.prompt_dir, args.caption_system_file))
    cap_usr = read_txt(os.path.join(args.prompt_dir, args.caption_user_file))
    ver_sys = read_txt(os.path.join(args.prompt_dir, args.verifier_system_file))
    ver_usr = read_txt(os.path.join(args.prompt_dir, args.verifier_user_file))

    images_dir = Path(f"{args.save_dir}/images")
    meta_dir = Path(f"{args.save_dir}/meta")
    corrected_dir = Path(f"{args.save_dir}/corrected")
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    # 1. ORIGINAL GENERATION
    orig_img = pipe(
        PROMPT,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]

    orig_path = images_dir / "original.png"
    orig_img.save(orig_path)

    # 2. ORIGINAL VERIFICATION
    ver0 = call_verifier_with_files(
        image=orig_img,
        instruction=PROMPT,
        checklist="",
        system_txt=ver_sys,
        user_tpl=ver_usr,
        model=args.verifier_model,
        temperature=0.0
    )

    v0_correct = int(ver0.get("correct", 0))
    failure_reason_used = ver0.get("reason", "")

    status = "original_correct"
    # Skip if original is already correct
    if v0_correct == 1:
        shutil.copy2(orig_path, corrected_dir / orig_path.name)

        meta = {
            "prompt": PROMPT,
            "status": status,
            "original_verifier": ver0,
            "failure_reason_used": "",
            "caption": None,
            "negatives": [],
            "snr": None,
            "best": None,
            "attempts": []
        }

        with (meta_dir / "run.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print("✅ ORIGINAL IMAGE CORRECT — META SAVED")
        return

    # 3. CAPTION
    caption = gpt_caption(
        image=orig_img,
        system_txt=cap_sys,
        user_txt=cap_usr,
        model=args.caption_model
    )

    # 4. NEGATIVE PROPOSAL (Using failure reason)
    negs = generate_negative_pairs(
        positive_prompt=PROMPT,
        caption=caption,
        failure_json=failure_reason_used,
        k=5,
        prompt_dir=args.prompt_dir,
        caption_model=args.caption_model,
        failure_model=args.caption_model,
    )["combined"]

    # 5. PROXY-SNR RANKING
    qty_tokens, obj_tokens = auto_extract_qty_objs(PROMPT)
    neg_sorted, snr_rows = snr_rank_negatives(
        cond=PROMPT,
        cands=negs,
        qty=qty_tokens,
        objs=obj_tokens,
        repo=args.snr_repo,
        device=args.snr_device,
        dtype=args.snr_dtype,
    )

    # 6. RE-GENERATE + VERIFY
    attempts = []
    best = {"score": -1}

    for i, neg in enumerate(neg_sorted):
        img = pipe(
            PROMPT,
            negative_prompt=neg,
            height=1024,
            width=1024,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=3.5,
            num_inference_steps=50,
            neg_first_n_steps=args.n_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed)
        ).images[0]

        out_path = images_dir / f"attempt_{i}_{slugify(neg)}.png"
        img.save(out_path)

        ver = call_verifier_with_files(
            image=img,
            instruction=PROMPT,
            checklist="",
            system_txt=ver_sys,
            user_tpl=ver_usr,
            model=args.verifier_model,
            temperature=0.0
        )

        attempts.append({
            "negative_prompt": neg,
            "image_path": str(out_path),
            "verifier": ver
        })

        if ver["score"] > best["score"]:
            best = {
                "negative_prompt": neg,
                "image_path": str(out_path),
                "score": ver["score"]
            }

        if ver["correct"] == 1:
            shutil.copy2(out_path, corrected_dir / out_path.name)
            break

    # 7. META SAVE
    meta = {
        "prompt": PROMPT,
        "original_verifier": ver0,
        "failure_reason_used": failure_reason_used,
        "caption": caption,
        "negatives": neg_sorted,
        "snr": snr_rows,
        "best": best,
        "attempts": attempts
    }

    with (meta_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--true-cfg-scale", type=float, default=1.8)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--caption_model", default="gpt-4o-mini", type=str)
    parser.add_argument("--prompt_dir", default="gpt_prompts", type=str)
    parser.add_argument("--caption_system_file", default="caption_long.system.txt", type=str)
    parser.add_argument("--caption_user_file", default="caption_long.user.txt", type=str)
    parser.add_argument("--propose_system_file", default="prompts_from_failure.system.txt", type=str)
    parser.add_argument("--propose_user_file", default="prompts_from_failure.user.txt", type=str)
    parser.add_argument("--verifier_system_file", default="verifier.system.txt", type=str)
    parser.add_argument("--verifier_user_file", default="verifier.user.txt", type=str)
    parser.add_argument("--propose_model", default="gpt-4o-mini", type=str)
    parser.add_argument("--verifier_model", default="gpt-4.1", type=str)
    parser.add_argument("--eval_json", default="Output_seed0.json", type=str)
    parser.add_argument("--ablation", type=str, default='combined')
    parser.add_argument("--save-dir", type=str, default="npc_demo")
    parser.add_argument("--snr_repo", default="black-forest-labs/FLUX.1-dev", type=str)
    parser.add_argument("--snr_device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--snr_dtype", choices=["float32","float16","bfloat16"], default="float32")
    parser.add_argument("--precheck_original", action="store_true", default=True)
    parser.add_argument("--use_eval_json_skip", action="store_true", default=False)
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the target image"
    )

    args = parser.parse_args()
    main(args)