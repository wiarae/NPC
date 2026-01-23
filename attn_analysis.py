#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Salient-vs-Background attention (COND only) — pretty plots
- Logs per-step S_tgt(t) and S_bg(t) with softmax attention mass.
- Saves prettier overlays (ratio & fraction) across negatives (no markers).
- Also saves generated images and a tail-step summary CSV.
"""

import argparse, math, os, re
from pathlib import Path
from typing import List, Optional, Dict

import torch
from diffusers import FluxPipeline
from diffusers.models.attention import Attention

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler  # NEW: nicer color cycle

# ---------- small helpers ----------
def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-_.]", "", text)
    return (text[:max_len] or "none").strip("._")

def parse_negative(arg: str) -> Optional[str]:
    """CLI mapping: NONE -> None, ''/UNCOND -> '', else as-is."""
    if arg is None: return None
    s = str(arg).strip()
    up = s.upper()
    if up == "NONE": return None           # True-CFG off baseline
    if up in ("UNCOND", '""', "''"): return ""  # unconditional (True-CFG on)
    return s

# ---------- robust token index finder (plural/splits safe) ----------
def token_indices_for_terms(pipe: FluxPipeline, prompt: str, terms: List[str], max_window: int = 3) -> List[int]:
    tok = pipe.tokenizer_2
    enc = tok(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    pieces = tok.convert_ids_to_tokens(ids)
    norm = [p.replace("▁", "").lower() for p in pieces]

    def variants(t: str) -> set:
        t = (t or "").strip().lower()
        out = {t}
        if t.endswith("s"): out.add(t[:-1])
        else: out.add(t + "s")
        return out

    wanted = set()
    for t in terms:
        if t.strip():
            wanted |= variants(t)

    idxs = set()
    # single-piece match
    for i, p in enumerate(norm):
        if any(w in p for w in wanted):
            idxs.add(i)
    # windowed concat match (e.g., 'bowl' + 's' -> 'bowls')
    L = len(norm)
    for wlen in range(2, max_window + 1):
        for i in range(0, L - wlen + 1):
            joined = "".join(norm[i:i + wlen])
            if any(w in joined for w in wanted):
                idxs.update(range(i, i + wlen))
    return sorted(idxs)

# ---------- cond/uncond tagging ----------
class _TensorTagger:
    def __init__(self): self._tags: Dict[int, str] = {}
    def register(self, tensor, tag):
        if tensor is None or not torch.is_tensor(tensor): return
        self._tags[int(tensor.data_ptr())] = tag
    def tag_of(self, tensor):
        if tensor is None or not torch.is_tensor(tensor): return None
        return self._tags.get(int(tensor.data_ptr()))
    def clear(self): self._tags.clear()

TENSOR_TAGGER = _TensorTagger()

def wrap_encode_prompt_with_tags(pipe: FluxPipeline):
    if not hasattr(pipe, "_encode_prompt"):
        return
    orig = pipe._encode_prompt
    def _wrapped(*args, **kwargs):
        res = orig(*args, **kwargs)
        try:
            prompt_embeds = res[0]
            neg_embeds = res[2] if len(res) > 2 else None
        except Exception:
            prompt_embeds = getattr(res, "prompt_embeds", None)
            neg_embeds = getattr(res, "negative_prompt_embeds", None)
        if torch.is_tensor(prompt_embeds): TENSOR_TAGGER.register(prompt_embeds, "cond")
        if torch.is_tensor(neg_embeds):    TENSOR_TAGGER.register(neg_embeds, "uncond")
        return res
    pipe._encode_prompt = _wrapped

# ---------- store & wrapper (COND + UNCOND split; cond-only metrics) ----------
class Store:
    """Per-step attention mass on TARGET and BG. We also log uncond for optional analyses; plots use COND only."""
    def __init__(self, steps: int, target_idx: List[int]):
        self.steps = steps
        self.tgt = sorted(set(int(i) for i in target_idx))
        self.step = -1
        self.c_tgt = {i: 0.0 for i in range(steps)}
        self.c_all = {i: 0.0 for i in range(steps)}
        self.u_tgt = {i: 0.0 for i in range(steps)}
        self.u_all = {i: 0.0 for i in range(steps)}

    def next(self): self.step += 1

    def add(self, probs_BHQK: torch.Tensor, tag: Optional[str]):
        if probs_BHQK is None or self.step < 0: return
        # Sum over B,H,Q -> [K]
        vecK = probs_BHQK.sum(dim=(0,1,2)).detach().float()
        tot_all = float(vecK.sum().item())
        tot_tgt = float(vecK[self.tgt].sum().item()) if self.tgt else 0.0

        if tag == "cond":
            if self.step in self.c_tgt: self.c_tgt[self.step] += tot_tgt
            if self.step in self.c_all: self.c_all[self.step] += tot_all
        elif tag == "uncond":
            if self.step in self.u_tgt: self.u_tgt[self.step] += tot_tgt
            if self.step in self.u_all: self.u_all[self.step] += tot_all
        else:
            # default to cond if unknown
            if self.step in self.c_tgt: self.c_tgt[self.step] += tot_tgt
            if self.step in self.c_all: self.c_all[self.step] += tot_all

class Wrap:
    def __init__(self, store: Store, orig, name=""): self.store, self.orig, self.name = store, orig, name
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        try:
            if encoder_hidden_states is not None:
                B, Q, _ = hidden_states.shape
                heads = getattr(attn, "heads", getattr(attn, "num_heads", 8))
                q_lin = attn.to_q(hidden_states)
                k_lin = attn.to_k(encoder_hidden_states)
                head_dim = q_lin.shape[-1] // heads
                def shp(x, L): return x.view(B, L, heads, head_dim).permute(0,2,1,3)
                q = shp(q_lin, Q)
                k = shp(k_lin, k_lin.shape[1])
                scores = torch.matmul((q * (1.0 / math.sqrt(max(1, head_dim)))), k.transpose(-2, -1))
                if attention_mask is not None: scores = scores + attention_mask
                probs = torch.softmax(scores, dim=-1).detach()  # [B,H,Q,K]

                tag = TENSOR_TAGGER.tag_of(encoder_hidden_states)
                if tag is None and probs.shape[0] % 2 == 0:
                    half = probs.shape[0] // 2
                    self.store.add(probs[:half],  "uncond")
                    self.store.add(probs[half:], "cond")
                else:
                    self.store.add(probs, tag or "cond")
        except Exception:
            pass
        return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)

def attach_logger(pipe: FluxPipeline, store: Store) -> bool:
    try:
        procs = getattr(pipe.transformer, "attn_processors", None)
        setp  = getattr(pipe.transformer, "set_attn_processor", None)
        if isinstance(procs, dict) and callable(setp):
            if not hasattr(pipe.transformer, "_base_attn_procs"):
                pipe.transformer._base_attn_procs = {n: p for n, p in procs.items()}
            setp({n: Wrap(store, p, name=n) for n, p in pipe.transformer._base_attn_procs.items()})
            return True
    except Exception:
        pass
    # fallback
    cnt = 0
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, Attention) and hasattr(m, "processor"):
            if not hasattr(m, "_base_processor"):
                m._base_processor = m.processor
            m.processor = Wrap(store, m._base_processor)
            cnt += 1
    return cnt > 0

def role_of_negative(neg: Optional[str]) -> str:
    """Map negative prompt to role label used for palette & legend."""
    if neg is None:
        return "BASE"          # True-CFG off
    if neg == "":
        return "UNTARGETED"    # unconditional (True-CFG on)
    return "TARGETED"          # any textual negative


# ---------- one run ----------
def run_one_and_save(pipe: FluxPipeline, prompt: str, negative: Optional[str], terms: List[str],
                     steps: int, height: int, width: int, guidance: float, true_cfg: float,
                     seed: int, out_img_dir: Path, tag: str, role: str = ""):
    idxs = token_indices_for_terms(pipe, prompt, terms)
    if not idxs:
        print(f"[warn] no token matched for {terms} -> skip {tag}")
        return None

    store = Store(steps=steps, target_idx=idxs)
    TENSOR_TAGGER.clear()
    wrap_encode_prompt_with_tags(pipe)
    if not attach_logger(pipe, store):
        print("[warn] failed to attach logger")
        return None

    def on_step_end(_pipe, step, timestep, kwargs):
        store.next()
        return kwargs

    gen = torch.Generator("cpu").manual_seed(int(seed))
    call = dict(
        prompt=prompt,
        negative_prompt=negative,
        height=height, width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        max_sequence_length=512,
        generator=gen,
        callback_on_step_end=on_step_end,
    )
    with torch.inference_mode():
        try:
            out = pipe(true_cfg_scale=float(true_cfg), **call)
        except TypeError:
            out = pipe(**call)

    # save image
    out_img_dir.mkdir(parents=True, exist_ok=True)
    neg_name = "NONE" if negative is None else (f"UNCOND_{terms}" if negative=="" else negative)
    img_name = f"id0_seed{seed}_{slugify(neg_name)}.png"
    img_path = out_img_dir / img_name
    try:
        out.images[0].save(img_path)
    except Exception as e:
        print(f"[warn] failed to save image: {e}")

    # cond-only sequences
    Sc_all = [store.c_all[i] for i in range(steps)]
    Sc_tgt = [store.c_tgt[i] for i in range(steps)]
    eps = 1e-8
    Sc_bg  = [max(a - t, 0.0) for a, t in zip(Sc_all, Sc_tgt)]
    ratio  = [t / (b + eps) for t, b in zip(Sc_tgt, Sc_bg)]
    frac   = [t / (a + eps) for t, a in zip(Sc_tgt, Sc_all)]

    return dict(role=role,                     
                negative=neg_name, image=str(img_path),
                Sc_tgt=Sc_tgt, Sc_bg=Sc_bg, ratio=ratio, fraction=frac)

# ---------- pretty plotting ----------
def _apply_style(style: str):
    plt.style.use("default")
    # Pleasant color cycles (Tableau-like)
    palette_light = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
                     "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    palette_dark  = ["#6BA3D6", "#F7A046", "#FF7B80", "#8CD0CB", "#7BC67B",
                     "#F3D657", "#C89BC3", "#FFB3BB", "#B89B7A", "#D2D6D9"]
    if style == "dark":
        plt.rcParams.update({
            "figure.facecolor": "#0f1115",
            "axes.facecolor": "#0f1115",
            "axes.edgecolor":  "#d0d7de",
            "axes.labelcolor": "#e6edf3",
            "text.color":      "#e6edf3",
            "xtick.color":     "#c9d1d9",
            "ytick.color":     "#c9d1d9",
            "grid.color":      "#30363d",
            "legend.facecolor":"#0f1115",
            "legend.edgecolor":"#0f1115",
            "axes.prop_cycle": cycler(color=palette_dark),
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "axes.edgecolor":   "#222222",
            "axes.labelcolor":  "#111111",
            "text.color":       "#111111",
            "xtick.color":      "#222222",
            "ytick.color":      "#222222",
            "grid.color":       "#DDDDDD",
            "legend.facecolor": "white",
            "legend.edgecolor": "white",
            "axes.prop_cycle":  cycler(color=palette_light),
        })
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "lines.linewidth": 2.4,   # a bit bolder
        "savefig.dpi": 300,
    })

# ---- put near your plotting helpers ----
PALETTE = {
    "BASE":       "#111111",
    "UNTARGETED": "#5EB0E5",
    "UNTARGETED": "#5EB0E5",   
    "TARGETED":   "#E6AA00",
}

def _color_for(label: str, idx: int) -> str:
    key = (label or "").upper().strip()
    if key in PALETTE:
        return PALETTE[key]
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % 10]

def save_pretty_overlay(xs, series_list, labels, title, ylabel, out_path,
                        baseline=None, tail=None, style="light", transparent=False,
                        color_keys=None, legend_mode="categories"):
    _apply_style(style)
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    n = len(series_list)
    lw = 3.2

    mus = []
    for ys in series_list:
        mu = (sum(ys) / max(1, len(ys))) if ys else float("nan")
        mus.append(mu)

    # --- draw lines: color by role (color_keys), label text is free-form (labels)
    for i, ys in enumerate(series_list):
        key_for_color = (color_keys[i] if color_keys and i < len(color_keys) else labels[i])
        color = _color_for(key_for_color, i)
        ax.plot(xs, ys, linewidth=lw, color=color, alpha=0.98)

    if tail and tail > 0 and len(xs) >= tail:
        ax.axvspan(xs[-tail], xs[-1], alpha=0.06, color="#888888")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1.4, alpha=0.6, color="#888888")

    ax.set_xlabel("Denoise step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # ---- legends
    if legend_mode in ("categories", "both"):
        cat_order = ["BASE", "TARGETED", "UNTARGETED"]
        cat_handles = [plt.Line2D([0], [0], color=_color_for(k, i), lw=lw) for i, k in enumerate(cat_order)]
        cat_leg = ax.legend(cat_handles, cat_order,
                            loc="lower left", bbox_to_anchor=(0.06, 0.10),
                            frameon=True, fancybox=True, framealpha=0.85,
                            facecolor="#F2F4F7", edgecolor="#D0D7DE",
                            fontsize=14, ncol=1, handlelength=3.2)
        ax.add_artist(cat_leg)  # keep category legend

    if legend_mode in ("series", "both"):
        series_labels = [f"{labels[i]} (μ={mus[i]:.3f})" for i in range(n)]
        series_handles = []
        for i in range(n):
            key_for_color = (color_keys[i] if color_keys and i < len(color_keys) else labels[i])
            h = plt.Line2D([0], [0], color=_color_for(key_for_color, i), lw=lw)
            series_handles.append(h)
        ax.legend(series_handles, series_labels,
                  loc="upper right",
                  frameon=True, fancybox=True, framealpha=0.85,
                  facecolor="#F2F4F7", edgecolor="#D0D7DE",
                  fontsize=10, ncol=1, handlelength=3.0)

    fig.savefig(out_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none",
                transparent=transparent)
    plt.close(fig)



# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--terms", required=True, help="comma-separated (e.g., 'three,apple,red')")
    ap.add_argument("--negatives", nargs="+", required=True,
                    help='List of negatives. Use NONE for no-negative (True-CFG off), "" or UNCOND for unconditional.')
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--true-cfg", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="salient_vs_bg_results")
    ap.add_argument("--tail", type=int, default=10, help="average overlay hint over last N steps")
    ap.add_argument("--style", choices=["light","dark"], default="light")
    ap.add_argument("--transparent", action="store_true", help="save plots with transparent background")
    ap.add_argument("--save-svg", action="store_true", help="also save .svg alongside .png")
    ap.add_argument("--annotate", choices=["tail","overall","none"], default="tail",
                    help="number shown at the right: tail mean, overall mean, or none")
    ap.add_argument("--labels", type=str, default="",
                help="Comma-separated legend labels aligned with --negatives (e.g., 'BASE,UNTARGETED,TARGETED')")
    ap.add_argument("--roles", type=str, default="BASE,TARGETED,UNTARGETED")

    args = ap.parse_args()

    out_dir  = Path(args.out_dir)
    img_dir  = out_dir / "images"
    plot_dir = out_dir / "plots"
    sum_dir  = out_dir / "summaries"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16)
    pipe.to(args.device)

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]

    # run
    results = []
    for i, neg_arg in enumerate(args.negatives):
        neg = parse_negative(neg_arg)
        role = role_of_negative(neg)  
        res = run_one_and_save(pipe, args.prompt, neg, terms, args.steps, args.height, args.width,
                            args.guidance, args.true_cfg, args.seed, img_dir, f"neg{i:02d}",
                            role=role)      
        if res is not None:
            results.append(res)

    if not results:
        print("[error] nothing to plot.")
        return

    default_roles = [s.strip().upper() for s in (args.roles or "").split(",") if s.strip()]
    if not default_roles:
        default_roles = ["BASE", "TARGETED", "UNTARGETED"]

    roles = []
    for i in range(len(results)):
        roles.append(default_roles[i % len(default_roles)])

    labels = []
    for i, r in enumerate(results):
        role = roles[i]
        if role == "TARGETED":
            labels.append(f"TARGETED: {r['negative']}")
        elif role == "UNTARGETED":
            labels.append("UNTARGETED (uncond)")
        else:
            labels.append("BASE")
 
    xs = list(range(1, args.steps+1))


    # 1) Ratio overlay
    ratio_series = [r["ratio"] for r in results]
    ratio_pdf = plot_dir / "cond_ratio_overlay.pdf"
    save_pretty_overlay(
        xs, ratio_series, labels,
        title=f"Salient vs Background (COND only) — s={args.guidance}, true_cfg={args.true_cfg}",
        ylabel="S_tgt / S_bg",
        out_path=ratio_pdf,
        baseline=1.0,  # ratio=1 baseline
        tail=max(1, min(args.tail, args.steps)),
        style=args.style,
        transparent=args.transparent,
        color_keys=roles,              
        legend_mode="both",
        # annotate=(None if args.annotate=="none" else args.annotate),
    )
    if args.save_svg:
        save_pretty_overlay(
            xs, ratio_series, labels,
            title=f"Salient vs Background (COND only) — s={args.guidance}, true_cfg={args.true_cfg}",
            ylabel="S_tgt / S_bg",
            out_path=ratio_pdf.with_suffix(".svg"),
            baseline=1.0,
            tail=max(1, min(args.tail, args.steps)),
            style=args.style,
            transparent=args.transparent,
            color_keys=roles,                
            legend_mode="both",
            # annotate=(None if args.annotate=="none" else args.annotate),
        )
    print(f"[done] plot -> {ratio_pdf}")

    # 2) Fraction overlay
    frac_series = [r["fraction"] for r in results]
    frac_png = plot_dir / f"cond_fraction_overlay_{args.terms}.pdf"
    save_pretty_overlay(
        xs, frac_series, labels,
        title=f"Salient fraction (COND only) — s={args.guidance}, true_cfg={args.true_cfg}",
        ylabel="S_tgt / (S_tgt + S_bg)",
        out_path=frac_png,
        baseline=None,  # fraction baseline is contextual
        tail=max(1, min(args.tail, args.steps)),
        style=args.style,
        transparent=args.transparent,
        color_keys=roles,                 
        legend_mode="both",
        # annotate=(None if args.annotate=="none" else args.annotate),
    )
    if args.save_svg:
        save_pretty_overlay(
            xs, frac_series, labels,
            title=f"Salient fraction (COND only) — s={args.guidance}, true_cfg={args.true_cfg}",
            ylabel="S_tgt / (S_tgt + S_bg)",
            out_path=frac_png.with_suffix(".svg"),
            baseline=None,
            tail=max(1, min(args.tail, args.steps)),
            style=args.style,
            transparent=args.transparent,
            color_keys=roles,                 
            legend_mode="both",
            # annotate=(None if args.annotate=="none" else args.annotate),
        )
    print(f"[done] plot -> {frac_png}")

    # 3) Summary CSV (tail means)
    tail = max(1, min(args.tail, args.steps))
    import csv
    csv_path = sum_dir / "summary_tail.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["negative", "mean_ratio_tail", "mean_fraction_tail", "image_path"])
        for r in results:
            mr = sum(r["ratio"][-tail:]) / tail
            mf = sum(r["fraction"][-tail:]) / tail
            w.writerow([r["negative"], f"{mr:.4f}", f"{mf:.4f}", r["image"]])
    print(f"[done] summary -> {csv_path}")

if __name__ == "__main__":
    main()
