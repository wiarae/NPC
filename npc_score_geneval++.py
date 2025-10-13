import torch
# from diffusers import FluxPipeline
from models.flux_stepwise import FluxPipelineStepwise, FluxPipelineStepwiseDNG, FluxPipeline_new, FluxPipelineStep
import time 
import argparse
import os, io, re, json, base64, argparse
from PIL import Image
import os 
from typing import List
import json
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Any
import shutil
import re
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
    """
    프롬프트에서 (대략적으로) 수량 토큰과 객체 토큰을 뽑아냅니다.
    - qty: 'three', 'two', '3', '2' 처럼 단어/숫자 혼합
    - objs: 수량 바로 뒤에 오는 명사 후보 + 간단한 단/복수 변형
    """
    text = re.sub(r'[^0-9a-zA-Z\- ]+', ' ', prompt.lower())
    qty_tokens = []
    obj_candidates = []

    # 패턴: (수량) [of] (명사)
    pat = re.compile(r'\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b\s+(?:of\s+)?\b([a-z][a-z\-]*)')
    for m in pat.finditer(text):
        q_raw = m.group(1)
        q_norm = _NUM_WORDS.get(q_raw, q_raw)   # 단어→숫자 문자열
        # qty에 단어형/숫자형 모두 넣기
        if q_raw in _NUM_WORDS:    # 단어형 입력이었으면
            qty_tokens.append(q_raw)       # 단어형
            qty_tokens.append(_NUM_WORDS[q_raw])  # 숫자형
        else:
            qty_tokens.append(q_raw)       # 숫자형
        noun = m.group(2)
        if noun not in _STOP and len(noun) > 1:
            obj_candidates.append(noun)

    # 여분: 전치사 of/with 뒤 명사들도 후보로 (수량 못 찾았을 때 대비)
    if not obj_candidates:
        for w in text.split():
            if w.isalpha() and w not in _STOP and len(w) > 2:
                obj_candidates.append(w)

    # 정제: 객체 토큰은 단/복수 모두 포함
    objs_set = set()
    for w in obj_candidates:
        s = _singularize(w)
        p = _pluralize(s)
        objs_set.update([s, p])

    # qty 기본값 (비었으면)
    if not qty_tokens:
        qty_tokens = ["one","two","three","1","2","3"]

    # objs 기본값 (비었으면)
    if not objs_set:
        # 아주 보수적으로 프롬프트의 마지막 단어를 넣어둠
        last_word = (text.split() or ["object"])[-1]
        s = _singularize(last_word)
        objs_set.update([s, _pluralize(s)])

    # 순서 보존 + 중복 제거
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
    return:
      sorted_list: mean-normalized proxy score S = s_qty + s_obj - s_cand 기준 정렬된 후보 리스트
      rows_sorted: 상세 점수(dict) 리스트 (score 내림차순)
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

            # candidate phrase vector
            v_cand = _phrase_vec_hidden(tok, te2, cand, device_eff)
            s_cand = _cos(delta, v_cand)

            # 평균 정규화: [-1,1] 범위로 스케일 맞춤
            if n_q > 0:
                s_qty = sum((_cos(delta, v) for v in qty_vecs), 0.0) / n_q
            else:
                s_qty = 0.0

            if n_o > 0:
                s_obj = sum((_cos(delta, v) for v in obj_vecs), 0.0) / n_o
            else:
                s_obj = 0.0

            # 유지: 해석 편의를 위한 파생 값들
            contrast_qty = s_qty - s_cand        # (정규화 버전)
            retention_obj = s_obj

            # 최종 단일 스코어 (직관적)
            score = s_qty + s_obj - s_cand

            rows.append({
                "candidate": cand,
                "s_qty": float(s_qty),
                "s_obj": float(s_obj),
                "s_cand": float(s_cand),
                "contrast_qty": float(contrast_qty),  # backward-compat
                "retention_obj": float(retention_obj),# backward-compat
                "score": float(score),
            })

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    sorted_list = [r["candidate"] for r in rows_sorted]
    return sorted_list, rows_sorted

# =========================================================


def _norm(s: str) -> str:
    # 공백/대소문자 차이로 인한 미스매치 방지
    return " ".join((s or "").lower().split())

def load_correct_prompts(json_path: str) -> set[str]:
    """
    {"overall_results": [ { "prompt": "...", "result": {"correct": 0/1, ...} }, ... ]}
    구조의 파일에서 correct==1인 prompt만 세트로 반환
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
    """
    LLM이 {"candidates": [...]} 를 반환한다고 가정하고
    깨끗한 리스트[str]로 돌려줍니다.
    """
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
    # 정리: 문자열만, 소문자/길이/금지어/중복 필터
    cleaned = []
    seen = set()
    for s in cands:
        if not isinstance(s, str):
            continue
        s2 = " ".join(s.strip().lower().split())
        if not s2:
            continue
        # 1–6 단어
        if not (1 <= len(s2.split()) <= 6):
            continue
        # 금지 토큰
        if any(tok in s2.split() for tok in ("no", "without", "not")):
            continue
        # 끝문장부호 제거
        while s2 and s2[-1] in ",.;:!?":
            s2 = s2[:-1].rstrip()
        if not s2:
            continue
        # 중복 제거
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
    # 간단한 파일명 안전화
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

            # 안전 파싱
            corr = 1 if int(data.get("correct", 0)) == 1 else 0
            # score는 0~1 실수 기대. 없으면 보수적 기본값.
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
        # 1–6 단어
        if not (1 <= len(s2.split()) <= 6):
            continue
        # 금지 토큰
        toks = s2.split()
        if any(tok in toks for tok in ("no", "without", "not")):
            continue
        # 끝문장부호 제거
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
    # 길이 1–6 단어
    if not (1 <= len(s2.split()) <= 6):
        return None
    # 금지 토큰
    toks = s2.split()
    if any(tok in toks for tok in ("no", "without", "not")):
        return None
    # 끝문장부호 제거
    while s2 and s2[-1] in ",.;:!?":
        s2 = s2[:-1].rstrip()
    return s2 or None

def _clean_and_sort_scored(items: List[dict], k: int) -> List[dict]:
    """LLM이 반환한 scored 아이템 목록을 정제/정렬."""
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

        # 기본값 및 보정
        if vqa is None: vqa = 0.0
        if llm is None: llm = 0.0
        # [0,1]로 클램프
        vqa = max(0.0, min(1.0, vqa))
        llm = max(0.0, min(1.0, llm))
        if imp is None:
            imp = vqa - llm
        else:
            imp = max(-1.0, min(1.0, imp))

        out.append({"text": txt, "vqa": vqa, "llm": llm, "importance": imp})

    # 중요도 → VQA → 강한 모순(추정 불가하니 importance 동률 시 길이 짧은 것) 순
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

    # candidates(문자열) + scored(객체) 모두 허용
    scored_raw = data.get("scored", [])
    scored = _clean_and_sort_scored(scored_raw, k)

    # candidates가 없다면 scored에서 추출
    cands = data.get("candidates", [])
    cleaned = _clean_candidates(cands, k)  # 기존 유틸 재사용
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

# ---- 조합 로직 ----
def _combine_pairs(a: List[str], b: List[str], k: int, mode: str = "zip") -> List[str]:
    pairs: List[str] = []
    if mode == "cartesian":
        for x in a:
            for y in b:
                pairs.append(f"{x}, {y}")
    else:  # zip (기본)
        for x, y in zip(a, b):
            pairs.append(f"{x}, {y}")

    # 중복 제거 + 상한 K
    out, seen = [], set()
    for s in pairs:
        if s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= k:
                break
    return out

# ---- 메인 헬퍼 ----
def generate_negative_pairs(
    positive_prompt: str,
    caption: str,
    failure_json: Any,                # dict 또는 str 모두 OK
    k: int,
    prompt_dir: str = "gpt_prompts",
    caption_model: str = "gpt-4o-mini",   # 그대로 두되, 아래 LLM 호출에 사용
    failure_model: str = "gpt-4o-mini",   # (둘 다 같은 모델이면 상관 없음)
    combine_mode: str = "zip"             # 더 이상 사용하지 않지만 시그니처는 유지
) -> Dict[str, List[str]]:

    # 0) failure_reason 정규화
    if isinstance(failure_json, dict):
        failure_reason = failure_json.get("result", {}).get("reason", "") or failure_json.get("reason", "")
    else:
        failure_reason = str(failure_json or "")

    # 1) 템플릿 로드 (없으면 폴백)
    sys_path = os.path.join(prompt_dir, "caption_candidates.system.txt")
    usr_path = os.path.join(prompt_dir, "caption_candidates.user.txt")
    # sys_path = os.path.join(prompt_dir, "propose_from_caption_candidates.system.txt")
    # usr_path = os.path.join(prompt_dir, "propose_from_caption_candidates.user.txt")

    neg_sys = read_txt(sys_path)
    neg_usr = read_txt(usr_path)
    

    # 2) 변수 바인딩
    variables = {
        "positive_prompt": positive_prompt,
        "caption": caption,
        "failure_reason": failure_reason,
        "fallbacks": DEFAULT_FALLBACKS,
    }

    # 3) LLM 호출 (한 번에 후보 생성)
    #    - 모델은 caption_model 파라미터 사용 (원하면 failure_model로 바꿔도 됨)
   
    # 한 번의 LLM 호출로 스코어 포함 결과 수집
    cands = _llm_candidates_from_templates(
        system_txt=neg_sys,
        user_tpl=neg_usr,
        variables=variables,
        model=caption_model,
        k=k,
        temperature=0.12
    )
    # 4) 반환 (호환성 위해 combined 키도 동일 리스트로 제공)
    print(cands)
    return {
        "combined": cands
    }

def load_failure_index(json_path: str):
    """
    Output_seed0.json에서 overall_results를 읽어
    - by_prompt: 정규화된 prompt -> 레코드
    - by_id: 인덱스(0/1-based 모두) -> 레코드
    - correct_set: correct==1인 prompt들의 정규화 세트
    를 반환
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

        # prompt 인덱스
        p = obj.get("prompt", "")
        if p:
            by_prompt[_norm(p)] = obj

        # id 인덱스 (0/1 기반 모두 대응)
        try:
            _id = int(obj.get("id", -1))
            if _id >= 0:
                by_id[_id] = obj        # 1-based일 가능성
                by_id[_id - 1] = obj    # 0-based일 가능성
        except Exception:
            pass

        # correct set
        try:
            res = obj.get("result", {})
            if int(res.get("correct", 0)) == 1 and p:
                correct_set.add(_norm(p))
        except Exception:
            pass

    return by_prompt, by_id, correct_set

def main(args):
    pipe = FluxPipelineStep.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    # prompts = read_prompts(Path("geneval++.txt"))
    prompts = read_prompts(Path(f"{args.dataset}.txt"))
    cap_sys = read_txt(os.path.join(args.prompt_dir, args.caption_system_file))
    cap_usr = read_txt(os.path.join(args.prompt_dir, args.caption_user_file))
    # prop_sys = read_txt(os.path.join(args.prompt_dir, args.propose_system_file))
    # prop_usr = read_txt(os.path.join(args.prompt_dir, args.propose_user_file))
    ver_sys = read_txt(os.path.join(args.prompt_dir, args.verifier_system_file))
    ver_usr = read_txt(os.path.join(args.prompt_dir, args.verifier_user_file))

    jsonl_lines = []
    # with open("Geneval++.jsonl", "r", encoding="utf-8") as f:
    with open(f"{args.dataset}.jsonl", "r", encoding="utf-8") as f:
        jsonl_lines = [json.loads(line.strip()) for line in f if line.strip()]
    
    # correct_set = load_correct_prompts("Output_seed0.json")
    # eval_by_prompt, eval_by_id, correct_set = load_failure_index(args.eval_json)
    eval_by_prompt, eval_by_id, correct_set = load_failure_index(args.eval_json)

    for idx, prompt in enumerate(prompts):
        images_dir = Path(f"{args.save_dir}/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        correct_dir = Path(f"{args.save_dir}/corrected")
        already = any(correct_dir.glob(f"*_{slugify(prompt)}*.png"))

        if already:
            print(f"[{idx+1}] SKIP (exists in images): {prompt[:80]}")
            continue
        

        # ① (선택) 예전처럼 eval_json 기반으로 스킵하고 싶을 때만
        if args.use_eval_json_skip and _norm(prompt) in correct_set:
            print(f"[{idx+1}] SKIP by eval_json(correct): {prompt[:80]}")
            continue

        # ② 원본 이미지 경로
        orig_path = f"flux_{args.dataset}_seed0/{idx:04d}_{slugify(prompt)}.png"
        
        if not os.path.exists(orig_path):
            print(f"[{idx+1}] WARN: original image not found -> {orig_path}")
            # 원본이 없으면 기존 흐름(생성→검증)으로 진행
        else:
            # 체크리스트(있으면) 준비 — 원본이 정답이면 caption 불필요하니 뒤로 미룸
            checklist = ""
            if jsonl_lines and idx < len(jsonl_lines):
                checklist = metadata_to_explanation(jsonl_lines[idx])

            # ③ 원본 먼저 verifier로 검사 (precheck)
            if args.precheck_original:
                orig_img = Image.open(orig_path).convert("RGB")
                ver0 = call_verifier_with_files(
                    image=orig_img,
                    instruction=prompt,
                    checklist=checklist,
                    system_txt=ver_sys,
                    user_tpl=ver_usr,
                    model=args.verifier_model,
                    temperature=0.0,
                    max_retries=3
                )
                v0_correct = int(ver0.get("correct", 0))
                v0_score   = float(ver0.get("score", 0.0))
                v0_reason  = ver0.get("reason", "")

                if v0_correct == 1:
                    # 원본이 이미 정답이면 재생성 생략 + corrected로 복사
                    corrected_dir = Path(f"{args.save_dir}/corrected")
                    corrected_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = corrected_dir / Path(orig_path).name
                    try:
                        shutil.copy2(orig_path, dst_path)
                        print(f"[{idx+1}] ✅ ORIGINAL CORRECT → copied to {dst_path}")
                    except Exception as e:
                        print(f"[{idx+1}] [warn] failed to copy original to corrected: {e}")

                    # 메타 저장
                    meta_dir = Path(f"{args.save_dir}/meta"); meta_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "id": idx + 1,
                        "prompt": prompt,
                        "status": "original_correct",
                        "chosen_negative": None,
                        "chosen_image_path": str(Path(dst_path).resolve()),
                        "chosen_score": v0_score,
                        "original_verifier": ver0,
                        "params": {
                            "true_cfg_scale": args.true_cfg_scale,
                            "neg_first_n_steps": args.n_steps,
                            "seed_base": args.seed,
                            "flux_model_id": "black-forest-labs/FLUX.1-dev"
                        },
                        "attempts": []
                    }
                    with (meta_dir / f"{idx:04d}_{prompt}_verify_trace.json").open("w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    # 다음 프롬프트로
                    continue
                failure_reason_used = v0_reason
            else:
                failure_reason_used = ""  

        
        failure_rec = eval_by_prompt.get(_norm(prompt)) or eval_by_id.get(idx) or {}
        failure_for_llm = failure_rec
        print(failure_for_llm) 
        # orig_img = Image.open(f"flux_geneval_seed0/{idx:04d}_{slugify(prompt)}.png").convert("RGB")
        orig_img = Image.open(f"flux_{args.dataset}_seed0/{idx:04d}_{slugify(prompt)}.png").convert("RGB")
        caption = gpt_caption(orig_img, cap_sys, cap_usr, model=args.caption_model, temperature=0.2)
        
        negative_prompts = generate_negative_pairs(
            positive_prompt=prompt,
            caption=caption,
            # failure_json=failure_for_llm.get('result', {}).get('reason', ""),
            failure_json=failure_reason_used,
            k=5,
            prompt_dir="gpt_prompts",
            caption_model=args.caption_model,
            failure_model=args.caption_model,   # 동일 모델이면 OK
            combine_mode="zip"
        )
        print(negative_prompts)
        neg_list = (negative_prompts.get(args.ablation)
            or negative_prompts.get("combined")
            or negative_prompts.get("candidates")
            or [])
        print(f"[negatives(raw)] {neg_list}")

        # --- NEW: qty/objs 자동 추출 + proxy-SNR 정렬 ---
        qty_tokens, obj_tokens = auto_extract_qty_objs(prompt)
        neg_sorted, snr_rows = snr_rank_negatives(
            cond=prompt,
            cands=neg_list,
            qty=qty_tokens,
            objs=obj_tokens,
            repo=getattr(args, "snr_repo", "black-forest-labs/FLUX.1-dev"),
            device=getattr(args, "snr_device", "cpu"),
            dtype=getattr(args, "snr_dtype", "float32"),
        )
        print(f"[qty]  {qty_tokens}")
        print(f"[objs] {obj_tokens}")
        print(f"[negatives(sorted by proxy-SNR)] {neg_sorted}")

        # 이후 로직이 dict를 기대하므로 덮어쓰기
        negative_prompts[args.ablation] = neg_sorted
        neg_list = neg_sorted
        # 리스트 한 번에 전달 (ablation 키가 'combined'면 그대로, 없으면 'candidates' 사용)
        neg_list = negative_prompts.get(args.ablation) or negative_prompts.get("combined") or negative_prompts.get("candidates") or []
        print(f"[negatives] {neg_list}")

        checklist = ""
        if jsonl_lines and idx < len(jsonl_lines):
            checklist = metadata_to_explanation(jsonl_lines[idx])

        # 보조 디렉터리 준비
        img_dir = f"{args.save_dir}/images"
        meta_dir = Path(f"{args.save_dir}/meta")
        corrected_dir = Path(f"{args.save_dir}/corrected")
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        corrected_dir.mkdir(parents=True, exist_ok=True)

        attempts = []
        status = "failure"
        chosen_neg = None
        chosen_img_path = None
        chosen_score = None

        best_score = -1.0
        best_idx = None


        attempts = []
        status = "failure"
        chosen_neg = None
        chosen_img_path = None
        chosen_score = None

        best_score = -1.0
        best_idx = None

        for a_idx, negative_prompt in enumerate(neg_list):
            # --- 생성 ---
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                negative_prompt=negative_prompt,
                true_cfg_scale=args.true_cfg_scale,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                neg_first_n_steps=args.n_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]

            out_path = Path(f"{images_dir}/{idx:04d}_{slugify(prompt)}_{negative_prompt}_{args.seed}_{args.true_cfg_scale}.png")
            image.save(out_path)

            # --- 검증 ---
            ver = call_verifier_with_files(
                image=image,
                instruction=prompt,
                checklist=checklist,
                system_txt=ver_sys,
                user_tpl=ver_usr,
                model=args.verifier_model,
                temperature=0.0,
                max_retries=3
            )

            v_correct = int(ver.get("correct", 0))
            v_score   = float(ver.get("score", 0.0))
            v_reason  = ver.get("reason", "")

            attempts.append({
                "attempt_index": a_idx,
                "negative_prompt": negative_prompt,
                "image_path": str(out_path.resolve()),
                "verifier_result": {"correct": v_correct, "score": v_score, "reason": v_reason}
            })

            # 최고 점수 추적 (equal 시 최초 시도 유지)
            if v_score > best_score:
                best_score = v_score
                best_idx = a_idx

            # 성공 즉시 채택 & 복사
            if v_correct == 1:
                status = "success"
                chosen_neg = negative_prompt
                chosen_img_path = str(out_path.resolve())
                chosen_score = v_score

                dst_path = corrected_dir / out_path.name
                try:
                    shutil.copy2(out_path, dst_path)
                    print(f"✅ CORRECT: copied to {dst_path}")
                except Exception as e:
                    print(f"[warn] failed to copy to corrected: {e}")
                break

        # --- 여기서 fallback: 끝까지 correct=1이 없으면 최고 score 채택 ---
        if status != "success" and best_idx is not None and attempts:
            best_attempt = attempts[best_idx]
            status = "best_score_fallback"
            chosen_neg = best_attempt["negative_prompt"]
            chosen_img_path = best_attempt["image_path"]
            chosen_score = best_attempt["verifier_result"].get("score", 0.0)

            # corrected 폴더로 복사
            try:
                dst_path = corrected_dir / Path(chosen_img_path).name
                shutil.copy2(chosen_img_path, dst_path)
                print(f"🏁 FALLBACK (best score={chosen_score:.3f}): copied to {dst_path}")
            except Exception as e:
                print(f"[warn] failed to copy fallback best to corrected: {e}")


        # 5) 프롬프트별 메타 저장
        meta = {
            "id": idx + 1,
            "prompt": prompt,
            "caption": caption,
            "failure_reason_used": failure_for_llm.get("result", {}).get("reason", ""),
            "candidates": neg_list,
            "snr": {
                "qty": qty_tokens,
                "objs": obj_tokens,
                "results": snr_rows
            },
            "status": status,  # "success" or "best_score_fallback" or "failure"
            "chosen_negative": chosen_neg,
            "chosen_image_path": chosen_img_path,
            "chosen_score": chosen_score,    # << 추가
            "original_verifier": ver0 if args.precheck_original and os.path.exists(orig_path) else None,
            "params": {
                "true_cfg_scale": args.true_cfg_scale,
                "neg_first_n_steps": args.n_steps,
                "seed_base": args.seed,
                "flux_model_id": "black-forest-labs/FLUX.1-dev"
            },
            "attempts": attempts
        }

        with (meta_dir / f"{idx:04d}_{prompt}_verify_trace.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[{idx+1}/{len(prompts)}] {status.upper()} | chosen_neg={chosen_neg}")


    print(f"\n✅ Done. Images -> {img_dir}\nMeta -> {meta_dir}")

            

    
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
    parser.add_argument("--eval_json", default="Output_seed0.json", type=str,
                    help="flux_geneval_seed0 이미지 평가 JSONL (correct==1이면 스킵)")
    parser.add_argument("--ablation", type=str, default='combined')
    parser.add_argument("--save-dir", type=str, default="npc_newscore_geneval++")
    parser.add_argument("--snr_repo", default="black-forest-labs/FLUX.1-dev", type=str)
    parser.add_argument("--snr_device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--snr_dtype", choices=["float32","float16","bfloat16"], default="float32")
    parser.add_argument("--dataset", type=str, default="Geneval++", choices=["Geneval++","Imagine"])
    # argparse 옵션 추가
    parser.add_argument("--precheck_original", action="store_true", default=True,
                        help="원본 이미지를 verifier로 먼저 검사해서 correct면 재생성을 생략")
    parser.add_argument("--use_eval_json_skip", action="store_true", default=False,
                        help="Output_seed0.json의 correct=1 항목은 여전히 스킵할지 여부")

    args = parser.parse_args()
    main(args)