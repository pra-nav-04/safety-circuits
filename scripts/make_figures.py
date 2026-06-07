"""Generate paper figures from the per-model artifacts in results/kaggle_neo/.

Standalone: pure pandas/matplotlib (no torch / TransformerLens), reads the CSV/JSON
each Kaggle run produced. Run from the repo root:

    pip install matplotlib    # numpy/pandas already required by the project
    python scripts/make_figures.py

Writes PNGs + a summary CSV to paper/figures/. Each figure backs one finding:
  fig_sparsity   — H1/H2: a few heads dominate (top-head |Δ| ± CI)
  fig_heatmaps   — where each model's safety heads sit (3×3 per-head |Δ| panel)
  fig_coupling   — A: refusal-removal vs capability damage (ΔPPL) are coupled
  fig_migration  — C: top-head depth migrates across generations (per family)
  fig_jailbreak  — D: refusal margin plain→jailbreak (some flip negative)
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "kaggle_neo"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Display order, grouped by family/generation, with a stable colour per family.
MODELS = [
    ("qwen1.5-1.8b", "Qwen1.5-1.8B", "Qwen"),
    ("qwen2-1.5b",   "Qwen2-1.5B",   "Qwen"),
    ("qwen2.5",      "Qwen2.5-1.5B", "Qwen"),
    ("qwen3",        "Qwen3-1.7B",   "Qwen"),
    ("gemma1-2b",    "Gemma1-2B",    "Gemma"),
    ("gemma2-2b",    "Gemma2-2B",    "Gemma"),
    ("gemma3-1b",    "Gemma3-1B",    "Gemma"),
    ("llama3.2-1b",  "Llama-3.2-1B", "Llama"),
    ("llama3-3b",    "Llama-3.2-3B", "Llama"),
]
FAMILY_COLOR = {"Qwen": "#d1495b", "Gemma": "#2e7d32", "Llama": "#1565c0"}


def _done(key: str) -> dict:
    return json.loads((RESULTS / key / "_DONE.json").read_text())


def _patch_z(key: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / key / f"{key}_patch_z.csv")
    return df[df["component"] == "z"].copy()


def load_all() -> list[dict]:
    rows = []
    for key, label, fam in MODELS:
        d = _done(key)
        jb, rtp = d.get("jailbreak", {}), d.get("rtp", {})
        rows.append({
            "key": key, "label": label, "family": fam,
            "n_layers": d["n_layers"], "n_heads": d["n_heads"],
            "top_layer": d["top_heads"][0][0], "top_head": d["top_heads"][0][1],
            "refusal_clean": d["refusal_clean"],
            "refusal_zero": d["refusal_ablated_zero"],
            "refusal_mean": d.get("refusal_ablated_mean"),
            "ppl_pct": d["perplexity_pct_change"],
            "jb_plain": jb.get("refusal_rate_clean_plain"),
            "jb_jail": jb.get("refusal_rate_clean_jailbreak"),
            "margin_plain": jb.get("mean_margin_plain"),
            "margin_jail": jb.get("mean_margin_jailbreak"),
            "rtp_delta": rtp.get("delta_tox"),
        })
    return rows


# ───────────────────────────── figures ─────────────────────────────
def fig_sparsity(rows):
    """Top-head |Δ refusal-margin| ± 95% CI per model — refusal is sparse/causal."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals, cis, colors, labels = [], [], [], []
    for r in rows:
        df = _patch_z(r["key"]).sort_values("abs_delta", ascending=False)
        top = df.iloc[0]
        vals.append(top["abs_delta"]); cis.append(top["ci95"])
        colors.append(FAMILY_COLOR[r["family"]]); labels.append(r["label"])
    x = np.arange(len(rows))
    ax.bar(x, vals, yerr=cis, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("top-head |Δ refusal-margin|")
    ax.set_title("A single head dominates refusal in every model (±95% CI)")
    _family_legend(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_sparsity.png", dpi=150); plt.close(fig)


def fig_heatmaps(rows):
    """3×3 panel of per-head |Δ| heatmaps (layer × head) — circuit location."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, r in zip(axes.flat, rows):
        df = _patch_z(r["key"])
        grid = np.zeros((r["n_layers"], r["n_heads"]))
        for _, row in df.iterrows():
            grid[int(row["layer"]), int(row["head"])] = row["abs_delta"]
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(r["label"], fontsize=10)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-head |Δ refusal-margin| — where each model's safety heads sit", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "fig_heatmaps.png", dpi=150); plt.close(fig)


def fig_coupling(rows):
    """Finding A: refusal removed (pp) vs capability damage (ΔPPL%, log). Coupled."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        removed = 100 * (r["refusal_clean"] - r["refusal_zero"])   # percentage points
        dppl = max(r["ppl_pct"], 0.1)
        ax.scatter(removed, dppl, s=90, color=FAMILY_COLOR[r["family"]],
                   edgecolor="black", zorder=3)
        ax.annotate(r["label"], (removed, dppl), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.axhline(5, ls=":", color="grey", lw=1)
    ax.text(1, 6, "H3b target: ΔPPL ≤ 5%", fontsize=8, color="grey")
    ax.set_xlabel("refusal removed by top-10 ablation (percentage points)")
    ax.set_ylabel("Δ WikiText-2 perplexity (%, log scale)")
    ax.set_title("Finding A — you can't remove refusal without breaking the model")
    _family_legend(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_coupling.png", dpi=150); plt.close(fig)


def fig_migration(rows):
    """Finding C: normalized top-head depth across generations, per family."""
    gen_order = {
        "Qwen":  [("qwen1.5-1.8b", "1.5"), ("qwen2-1.5b", "2"), ("qwen2.5", "2.5"), ("qwen3", "3")],
        "Gemma": [("gemma1-2b", "1"), ("gemma2-2b", "2"), ("gemma3-1b", "3")],
        "Llama": [("llama3.2-1b", "3.2-1B"), ("llama3-3b", "3.2-3B")],
    }
    by_key = {r["key"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fam, seq in gen_order.items():
        xs, ys, labs = [], [], []
        for i, (key, gen) in enumerate(seq):
            r = by_key[key]
            depth = r["top_layer"] / max(r["n_layers"] - 1, 1)
            xs.append(i); ys.append(depth); labs.append(gen)
        ax.plot(xs, ys, "-o", color=FAMILY_COLOR[fam], label=fam, lw=2, ms=8)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 5), textcoords="offset points")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("generation / size  (left → right = newer / larger)")
    ax.set_ylabel("top-head depth  (0 = first layer, 1 = last)")
    ax.set_title("Finding C — safety-circuit location migrates across generations")
    ax.legend(title="family")
    fig.tight_layout(); fig.savefig(OUT / "fig_migration.png", dpi=150); plt.close(fig)


def fig_jailbreak(rows):
    """Finding D: refusal margin plain → jailbreak (slope chart); some cross < 0."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        if r["margin_plain"] is None:
            continue
        c = FAMILY_COLOR[r["family"]]
        ax.plot([0, 1], [r["margin_plain"], r["margin_jail"]], "-o", color=c, alpha=0.85)
        ax.annotate(r["label"], (1, r["margin_jail"]), fontsize=8,
                    xytext=(6, 0), textcoords="offset points", va="center")
    ax.axhline(0, ls="--", color="red", lw=1)
    ax.text(0.02, 0.3, "margin < 0 ⇒ inclined to comply", color="red", fontsize=8)
    ax.set_xlim(-0.1, 1.4); ax.set_xticks([0, 1])
    ax.set_xticklabels(["plain (AdvBench)", "jailbreak (HarmBench)"])
    ax.set_ylabel("mean refusal-logit margin")
    ax.set_title("Finding D — jailbreaks weaken refusal; Qwen3 / Qwen1.5 flip negative")
    _family_legend(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_jailbreak.png", dpi=150); plt.close(fig)


def _family_legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()],
              title="family", loc="best")


def summary_table(rows):
    df = pd.DataFrame([{
        "model": r["label"], "family": r["family"],
        "layers": r["n_layers"], "heads": r["n_heads"],
        "top_head": f"L{r['top_layer']}H{r['top_head']}",
        "refusal_clean": r["refusal_clean"], "refusal_zero_abl": r["refusal_zero"],
        "ppl_pct_change": r["ppl_pct"],
        "jb_refusal_plain": r["jb_plain"], "jb_refusal_jail": r["jb_jail"],
        "margin_plain": r["margin_plain"], "margin_jail": r["margin_jail"],
        "rtp_delta_tox": r["rtp_delta"],
    } for r in rows])
    df.to_csv(OUT / "summary_table.csv", index=False)


def main():
    rows = load_all()
    fig_sparsity(rows)
    fig_heatmaps(rows)
    fig_coupling(rows)
    fig_migration(rows)
    fig_jailbreak(rows)
    summary_table(rows)
    print(f"Wrote figures + summary_table.csv to {OUT}")
    for p in sorted(OUT.glob("*")):
        print("  ", p.relative_to(ROOT), f"({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
