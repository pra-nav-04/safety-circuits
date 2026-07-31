"""Generate the Act-2 (editing extension) figures for the FINAL presentation.

The Findings E–G numbers live only in CSVs under results/editing_bender/<model>/;
the midterm figures (scripts/make_figures.py) cover Act 1 only. This script mirrors
that script's style (pure pandas/matplotlib, family colour palette) and writes PNGs
straight into slides/figures/ for the Beamer deck.

    python scripts/make_final_figures.py

Writes:
  fig_transfer_asr    — Finding G headline: benign-only training transfers to harmful
                        compliance (baseline -> benign-content-edit, per model, ±95% CI)
  fig_transfer_arms   — Llama-3.2-3B, all five arms: the opener edit is FAKE (0%),
                        the benign edit is REAL (0.41), and the two routes interfere
  fig_transfer_bycat  — where the Llama-3.2-3B transfer concentrates (per HarmBench cat)
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "editing_bender"
OUT = ROOT / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Same display order + family palette as scripts/make_figures.py.
MODELS = [
    ("qwen1.5-1.8b", "Qwen1.5-1.8B", "Qwen"),
    ("qwen2-1.5b",   "Qwen2-1.5B",   "Qwen"),
    ("qwen2.5",      "Qwen2.5-1.5B", "Qwen"),
    ("qwen3",        "Qwen3-1.7B",   "Qwen"),
    ("gemma1-2b",    "Gemma1-2B",    "Gemma"),
    ("gemma2-2b",    "Gemma2-2B",    "Gemma"),
    ("gemma3-1b",    "Gemma3-1B",    "Gemma"),
    ("llama3.2-1b",  "Llama-3.2-1B", "Llama"),
    ("llama3.2-3b",  "Llama-3.2-3B", "Llama"),
]
FAMILY_COLOR = {"Qwen": "#d1495b", "Gemma": "#2e7d32", "Llama": "#1565c0"}
THEME_BLUE, THEME_ORANGE, THEME_GRAY = "#35519F", "#FFBB00", "#949388"
STAR = "llama3.2-3b"


def _agg(key: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / key / f"{key}_edit_harmful_transfer_agg.csv")
    return df.set_index("arm")


def _val(df: pd.DataFrame, arm: str, col: str = "substantive_asr") -> float:
    return float(df.loc[arm, col]) if arm in df.index else float("nan")


# ─────────────────── fig 1: the headline transfer (baseline -> benign edit) ──────────
def fig_transfer_asr() -> None:
    rows = []
    for key, label, fam in MODELS:
        d = _agg(key)
        base = _val(d, "baseline")
        edit = _val(d, "benign_content_edit")
        lo, hi = _val(d, "benign_content_edit", "subst_asr_lo"), _val(d, "benign_content_edit", "subst_asr_hi")
        rows.append((key, label, fam, base, edit, edit - lo, hi - edit))
    # sort by the edited compliance, high to low — Llama-3.2-3B leads.
    rows.sort(key=lambda r: r[4], reverse=True)

    labels = [r[1] for r in rows]
    x = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - w / 2 for i in x], [r[3] for r in rows], w,
           color=THEME_GRAY, alpha=0.75, label="Baseline (aligned model)")
    ax.bar([i + w / 2 for i in x], [r[4] for r in rows], w,
           yerr=[[r[5] for r in rows], [r[6] for r in rows]], capsize=3,
           color=[FAMILY_COLOR[r[2]] for r in rows],
           edgecolor=["black" if r[0] == STAR else "none" for r in rows],
           linewidth=[2.2 if r[0] == STAR else 0 for r in rows],
           label="After benign-only content edit")
    for i, r in zip(x, rows):
        ax.annotate(f"{r[4]:.2f}", (i + w / 2, r[4] + r[6] + 0.012),
                    ha="center", fontsize=8,
                    fontweight="bold" if r[0] == STAR else "normal")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Substantive-ASR  (harmful AND ≥200 chars)")
    ax.set_ylim(0, 0.58)
    ax.set_title("Finding G — benign-only training transfers to harmful compliance\n"
                 "largest where alignment was strongest (Llama-3.2-3B: 0.08 → 0.41, no harmful data)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer_asr.png", dpi=150)
    plt.close(fig)


# ─────────────────── fig 2: the five arms on the star model ──────────────────────────
def fig_transfer_arms() -> None:
    d = _agg(STAR)
    arms = [
        ("baseline",                     "Baseline",              THEME_GRAY),
        ("opener_edit",                  "Opener edit\n(naive)",  "#c0392b"),
        ("steering_alone",               "Steering\nalone",       "#8e44ad"),
        ("benign_content_edit",          "Benign\ncontent edit",  FAMILY_COLOR["Llama"]),
        ("benign_content_edit+steering", "Benign +\nsteering",    "#5dade2"),
    ]
    raw = [_val(d, a, "asr") for a, _, _ in arms]
    sub = [_val(d, a, "substantive_asr") for a, _, _ in arms]
    los = [_val(d, a, "subst_asr_lo") for a, _, _ in arms]
    his = [_val(d, a, "subst_asr_hi") for a, _, _ in arms]
    x = range(len(arms))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    # ghost bars = raw ASR (the confounded metric), solid = substantive ASR (honest).
    ax.bar(x, raw, 0.6, color="none", edgecolor=THEME_GRAY, ls="--", linewidth=1.3,
           label="Raw ASR (opener-confounded)")
    ax.bar(x, sub, 0.6,
           yerr=[[s - lo for s, lo in zip(sub, los)], [hi - s for s, hi in zip(sub, his)]],
           capsize=4, color=[c for _, _, c in arms],
           label="Substantive-ASR (honest metric)")
    for i, (r, s) in enumerate(zip(raw, sub)):
        ax.annotate(f"raw {r:.2f}", (i, r + 0.02), ha="center", fontsize=7.5, color=THEME_GRAY)
        ax.annotate(f"{s:.2f}", (i, s + 0.045), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(0.5, color="k", lw=0.5, alpha=0.3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab, _ in arms], fontsize=9)
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Llama-3.2-3B — the obvious jailbreak is fake, the benign one is real\n"
                 "opener edit: raw 0.90 but 0% substantive · benign edit: 0.41 · routes interfere",
                 fontsize=11.5)
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer_arms.png", dpi=150)
    plt.close(fig)


# ─────────────────── fig 3: where the transfer concentrates (by category) ────────────
def fig_transfer_bycat() -> None:
    df = pd.read_csv(RESULTS / STAR / f"{STAR}_edit_harmful_transfer_bycat_agg.csv")
    base = df[df.arm == "baseline"].set_index("category")
    edit = df[df.arm == "benign_content_edit"].set_index("category")
    pretty = {
        "cybercrime_intrusion": "cybercrime / intrusion",
        "illegal": "illegal (non-violent)",
        "harmful": "general harmful",
        "harassment_bullying": "harassment / bullying",
        "misinformation_disinformation": "misinformation",
    }
    cats = edit["substantive_asr"].sort_values(ascending=True).index.tolist()
    labels = [f"{pretty.get(c, c)}  (n={int(edit.loc[c, 'n'])})" for c in cats]
    base_v = [float(base.loc[c, "substantive_asr"]) for c in cats]
    edit_v = [float(edit.loc[c, "substantive_asr"]) for c in cats]
    y = range(len(cats))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh([i + h / 2 for i in y], base_v, h, color=THEME_GRAY, alpha=0.75, label="Baseline")
    ax.barh([i - h / 2 for i in y], edit_v, h, color=FAMILY_COLOR["Llama"],
            label="Benign-only content edit")
    for i, v in zip(y, edit_v):
        ax.annotate(f"{v:.2f}", (v + 0.015, i - h / 2), va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Substantive-ASR")
    ax.set_xlim(0, 0.95)
    ax.set_title("Llama-3.2-3B — the transfer concentrates in procedural harm\n"
                 "cybercrime how-tos 0.84 · illegal 0.47", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer_bycat.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_transfer_asr()
    fig_transfer_arms()
    fig_transfer_bycat()
    print("wrote fig_transfer_asr.png, fig_transfer_arms.png, fig_transfer_bycat.png ->", OUT)
