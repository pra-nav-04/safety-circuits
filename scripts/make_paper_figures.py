"""Generate the IEEE report's figures and result tables from the canonical Bender run.

Supersedes scripts/make_figures.py (which reads results/kaggle_neo/) and
scripts/make_final_figures.py (which writes slide-sized PNGs into slides/figures/).
Three differences, all needed for print:

  1. Both acts read the *same* run — results/bender_neo/ for the mapping study and
     results/editing_bender/ for the editing extension. The Kaggle-T4 trees are kept
     only as the cross-hardware replication and are never plotted here.
  2. Output is vector PDF sized for an IEEE two-column page (3.4 in column,
     7.0 in full width) with 7 pt type, not 150-dpi slide bitmaps.
  3. Every results table is emitted as LaTeX from the same read as the figures, so
     no number in the report is hand-transcribed.

    python scripts/make_paper_figures.py

Writes  ../report/pics/*.pdf  and  ../report/tables/*.tex.
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
MAP_RESULTS = ROOT / "results" / "bender_neo"
EDIT_RESULTS = ROOT / "results" / "editing_bender"
REPORT = ROOT.parent / "report"
PICS = REPORT / "pics"
TABLES = REPORT / "tables"
PICS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

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
    ("llama3.2-3b",  "Llama-3.2-3B", "Llama"),
]
FAMILY_COLOR = {"Qwen": "#d1495b", "Gemma": "#2e7d32", "Llama": "#1565c0"}
GRAY = "#949388"
CI_COLOR = "#1a1a1a"    # 95% interval whiskers, drawn above the bars
STAR = "llama3.2-3b"

COL, FULL = 3.4, 7.0          # inches: \columnwidth and \textwidth in IEEEtran

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# ───────────────────────────── loading ─────────────────────────────
def _done(key: str) -> dict:
    return json.loads((MAP_RESULTS / key / "_DONE.json").read_text())


def _patch_z(key: str) -> pd.DataFrame:
    df = pd.read_csv(MAP_RESULTS / key / f"{key}_patch_z.csv")
    return df[df["component"] == "z"].copy()


def _edit_summary(key: str) -> pd.DataFrame:
    return pd.read_csv(EDIT_RESULTS / key / f"{key}_edit_summary.csv").set_index("label")


def _transfer(key: str) -> pd.DataFrame:
    return pd.read_csv(
        EDIT_RESULTS / key / f"{key}_edit_harmful_transfer_agg.csv").set_index("arm")


def load_all() -> list[dict]:
    rows = []
    for key, label, fam in MODELS:
        d = _done(key)
        jb, rtp = d.get("jailbreak", {}), d.get("rtp", {})
        top = _patch_z(key).sort_values("abs_delta", ascending=False).iloc[0]
        rows.append({
            "key": key, "label": label, "family": fam,
            "n_layers": d["n_layers"], "n_heads": d["n_heads"],
            "top_layer": d["top_heads"][0][0], "top_head": d["top_heads"][0][1],
            "top_delta": float(top["abs_delta"]), "top_ci": float(top["ci95"]),
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


def _family_legend(ax, ci_label=None, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    if ci_label:
        handles.append(_ci_handle(ci_label))
    ax.legend(handles=handles, **kw)


def _ci_handle(label):
    """Proxy artist so the interval names itself in the legend."""
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=CI_COLOR, lw=0.9, label=label)


def _draw_ci(ax, x, y, yerr):
    """Capless 95% interval with a white halo, drawn above the bars.

    Drawn as its own errorbar call rather than through bar(yerr=...) so that zorder
    and the halo are controllable: the halo is what keeps the descending half of the
    whisker reading as an overlay instead of merging into the bar fill.
    """
    import matplotlib.patheffects as pe
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=CI_COLOR, elinewidth=0.9,
                capsize=0, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])


def _save(fig, name):
    fig.savefig(PICS / f"{name}.pdf")
    plt.close(fig)
    print("  pics/", name + ".pdf")


# ───────────────────────────── Act 1 figures ─────────────────────────────
def fig_sparsity(rows):
    """H1/H2 — top-head |Delta refusal-margin| +/- 95% CI, one bar per model."""
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    x = np.arange(len(rows))
    ax.bar(x, [r["top_delta"] for r in rows],
           color=[FAMILY_COLOR[r["family"]] for r in rows], alpha=0.9)
    _draw_ci(ax, x, [r["top_delta"] for r in rows], [r["top_ci"] for r in rows])
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel(r"top-head $|\Delta$ refusal-margin$|$")
    # Headroom above the tallest whisker (Qwen3, ~11.0) so the two-column legend
    # clears it; raising ylim keeps the figure height fixed, unlike moving the
    # legend outside the axes.
    ax.set_ylim(0, 14.2)
    _family_legend(ax, ci_label="95% CI (normal approx.)", loc="upper left",
                   frameon=False, handlelength=1.0, ncol=2, columnspacing=1.2)
    _save(fig, "fig_sparsity")


# Per-point label placement, tuned by hand so no annotation overlaps another.
_COUPLING_OFFSETS = {
    "qwen1.5-1.8b": (-4, -2, "right"),
    "qwen2-1.5b":   (-4, -9, "right"),
    "qwen2.5":      (-4, 4, "right"),
    "qwen3":        (-4, 4, "right"),
    "gemma1-2b":    (4, -9, "left"),
    "gemma2-2b":    (4, 3, "left"),
    "gemma3-1b":    (-4, 4, "right"),
    "llama3.2-1b":  (4, -9, "left"),
    "llama3.2-3b":  (4, 3, "left"),
}


def fig_coupling(rows):
    """Finding A — refusal removed (pp) vs capability damage (Delta PPL %, log)."""
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for r in rows:
        removed = 100 * (r["refusal_clean"] - r["refusal_zero"])
        dppl = max(r["ppl_pct"], 0.1)
        dx, dy, ha = _COUPLING_OFFSETS[r["key"]]
        ax.scatter(removed, dppl, s=30, color=FAMILY_COLOR[r["family"]],
                   edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(r["label"], (removed, dppl), fontsize=5.6, ha=ha,
                    xytext=(dx, dy), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 5e7)
    ax.set_xlim(-8, 118)
    ax.axhline(5, ls=":", color="grey", lw=0.8)
    ax.text(0, 1.6, r"H3b target: $\Delta$PPL $\leq$ 5%", fontsize=5.8, color="grey",
            ha="left")
    ax.set_xlabel("refusal removed by top-10 ablation (pp)")
    ax.set_ylabel(r"$\Delta$ WikiText-2 perplexity (%, log)")
    _family_legend(ax, loc="upper left", frameon=False, handlelength=1.0,
                   borderpad=0.1, labelspacing=0.25)
    _save(fig, "fig_coupling")


def fig_migration(rows):
    """Finding C — normalized top-head depth across generations, per family."""
    gen_order = {
        "Qwen":  [("qwen1.5-1.8b", "1.5"), ("qwen2-1.5b", "2"),
                  ("qwen2.5", "2.5"), ("qwen3", "3")],
        "Gemma": [("gemma1-2b", "1"), ("gemma2-2b", "2"), ("gemma3-1b", "3")],
        "Llama": [("llama3.2-1b", "3.2-1B"), ("llama3.2-3b", "3.2-3B")],
    }
    # label offsets keep the three families' generation tags from colliding at x=0
    off = {"Qwen": (2, -10), "Gemma": (2, 5), "Llama": (2, 5)}
    by_key = {r["key"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for fam, seq in gen_order.items():
        xs, ys, labs = [], [], []
        for i, (key, gen) in enumerate(seq):
            r = by_key[key]
            xs.append(i)
            ys.append(r["top_layer"] / max(r["n_layers"] - 1, 1))
            labs.append(gen)
        ax.plot(xs, ys, "-o", color=FAMILY_COLOR[fam], label=fam, lw=1.4, ms=4)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=6, color=FAMILY_COLOR[fam],
                        xytext=off[fam], textcoords="offset points")
    ax.set_ylim(-0.12, 1.10)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xticks(range(4))
    ax.set_xlabel(r"generation / size (left $\rightarrow$ right = newer / larger)")
    ax.set_ylabel("normalized top-head depth")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=1.2)
    _save(fig, "fig_migration")


def fig_jailbreak(rows):
    """Finding D — refusal margin plain -> jailbreak; two models cross zero."""
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for r in rows:
        if r["margin_plain"] is None:
            continue
        ax.plot([0, 1], [r["margin_plain"], r["margin_jail"]], "-o",
                color=FAMILY_COLOR[r["family"]], alpha=0.9, lw=1.1, ms=3)
    # right-hand labels: greedy de-overlap, since several models land within
    # a few tenths of a nat of each other at the jailbreak end
    lab = sorted((r for r in rows if r["margin_plain"] is not None),
                 key=lambda r: r["margin_jail"], reverse=True)
    min_gap, prev = 0.85, None
    for r in lab:
        y = r["margin_jail"] if prev is None else min(r["margin_jail"], prev - min_gap)
        prev = y
        ax.annotate(r["label"], (1, y), fontsize=5.6, color=FAMILY_COLOR[r["family"]],
                    xytext=(5, 0), textcoords="offset points", va="center")
    ax.axhline(0, ls="--", color="red", lw=0.8)
    ax.text(0.02, -1.35, r"margin $<0\;\Rightarrow$ inclined to comply",
            color="red", fontsize=5.8)
    ax.set_xlim(-0.08, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["plain (AdvBench)", "jailbreak (HarmBench)"])
    ax.set_ylabel("mean refusal-logit margin")
    _save(fig, "fig_jailbreak")


def fig_heatmaps(rows):
    """Appendix — 3x3 panel of per-head |Delta| heatmaps (layer x head)."""
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 4.6))
    for ax, r in zip(axes.flat, rows):
        grid = np.zeros((r["n_layers"], r["n_heads"]))
        for _, row in _patch_z(r["key"]).iterrows():
            grid[int(row["layer"]), int(row["head"])] = row["abs_delta"]
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"{r['label']}  (L{r['top_layer']}H{r['top_head']})", fontsize=7)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    _save(fig, "fig_heatmaps")


# ───────────────────────────── Act 2 figures ─────────────────────────────
def fig_scalpel(rows):
    """Finding E — the scalpel-sharpness axis, all nine models on one plane.

    Each intervention is one marker class: blunt zero-ablation of the top-10 heads
    (mapping study), the no-train steering baseline, and head-restricted LoRA at the
    val-chosen k. Refusal on the y-axis, capability cost on the x-axis; the clean
    corner is bottom-left (refusal removed, capability kept).
    """
    by_key = {r["key"]: r for r in rows}
    abl_x, abl_y, st_x, st_y, lo_x, lo_y = [], [], [], [], [], []
    for key, label, fam in MODELS:
        s = _edit_summary(key)
        d = by_key[key]
        k = int(s[s["chosen"] == True].iloc[0]["k"])             # noqa: E712
        abl_x.append(d["ppl_pct"]);  abl_y.append(100 * d["refusal_zero"])
        st_x.append(s.loc["steering", "perplexity_pct_change"])
        st_y.append(100 * s.loc["steering", "refusal_rate"])
        lo_x.append(s.loc[f"lora_k{k}", "perplexity_pct_change"])
        lo_y.append(100 * s.loc[f"lora_k{k}", "refusal_rate"])

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.scatter(abl_x, abl_y, marker="X", s=34, color="#c0392b", zorder=4,
               label="zero-ablation (blunt)")
    ax.scatter(st_x, st_y, marker="s", s=24, color="#8e44ad", zorder=4,
               label="steering (no train)")
    ax.scatter(lo_x, lo_y, marker="o", s=26, color="#2e7d32", zorder=5,
               label="head-restricted LoRA")

    # the one LoRA point that is not in the clean corner
    i = [m[0] for m in MODELS].index("gemma1-2b")
    ax.annotate("Gemma1-2B", (lo_x[i], lo_y[i]), fontsize=5.8, color="#2e7d32",
                xytext=(0, 8), textcoords="offset points", ha="center")

    ax.set_xscale("symlog", linthresh=1)
    ticks = [-10, 0, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["-10", "0", "10", "100", "1k", "10k", "100k", "1M", "10M"])
    ax.set_xlim(-30, 3e7)
    ax.set_ylim(-9, 118)
    ax.axvline(0, ls=":", color="grey", lw=0.7)
    ax.set_xlabel(r"$\Delta$ WikiText-2 perplexity (%, symlog) $\rightarrow$ cost")
    ax.set_ylabel(r"refusal rate (%, test split)")
    ax.annotate("clean corner", (-25, 6), fontsize=6, color="#2e7d32")
    ax.legend(loc="center right", frameon=False,
              handlelength=1.0, borderpad=0.1, labelspacing=0.25)
    _save(fig, "fig_scalpel")


def fig_transfer_arms():
    """Finding F — raw ASR is opener-confounded; substantive-ASR is not (Llama-3.2-3B)."""
    d = _transfer(STAR)
    arms = [
        ("baseline",                     "base",           GRAY),
        ("opener_edit",                  "opener\n(naive)", "#c0392b"),
        ("steering_alone",               "steer\nalone",   "#8e44ad"),
        ("benign_content_edit",          "benign\ncontent", FAMILY_COLOR["Llama"]),
        ("benign_content_edit+steering", "benign\n+steer", "#5dade2"),
    ]
    raw = [float(d.loc[a, "asr"]) for a, _, _ in arms]
    sub = [float(d.loc[a, "substantive_asr"]) for a, _, _ in arms]
    lo = [float(d.loc[a, "subst_asr_lo"]) for a, _, _ in arms]
    hi = [float(d.loc[a, "subst_asr_hi"]) for a, _, _ in arms]
    x = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.bar(x, raw, 0.62, color="none", edgecolor=GRAY, ls="--", lw=1.0,
           label="raw ASR")
    ax.bar(x, sub, 0.62, color=[c for _, _, c in arms], label="substantive-ASR")
    _draw_ci(ax, x, sub, [np.array(sub) - lo, np.array(hi) - sub])
    for i, (r, s) in enumerate(zip(raw, sub)):
        if r - s > 0.06:
            ax.annotate(f"{r:.2f}", (i, r + 0.02), ha="center", fontsize=5.8, color=GRAY)
        # value inside the bar where it fits, above it otherwise
        if s >= 0.15:
            ax.annotate(f"{s:.2f}", (i, s / 2), ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color="white")
        else:
            ax.annotate(f"{s:.2f}", (i, s + 0.075), ha="center",
                        fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in arms])
    ax.set_ylabel("attack success rate (N=150)")
    ax.set_ylim(0, 1.02)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [_ci_handle("95% CI (bootstrap)")], l + ["95% CI (bootstrap)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="y", ls=":", alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "fig_transfer_arms")


def fig_transfer_asr():
    """Finding G headline — benign-only training transfers to harmful compliance."""
    rows = []
    for key, label, fam in MODELS:
        d = _transfer(key)
        base = float(d.loc["baseline", "substantive_asr"])
        edit = float(d.loc["benign_content_edit", "substantive_asr"])
        lo = float(d.loc["benign_content_edit", "subst_asr_lo"])
        hi = float(d.loc["benign_content_edit", "subst_asr_hi"])
        rows.append((key, label, fam, base, edit, edit - lo, hi - edit))
    rows.sort(key=lambda r: r[4], reverse=True)

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(FULL, 2.05))
    ax.bar(x - w / 2, [r[3] for r in rows], w, color=GRAY, alpha=0.8,
           label="baseline (aligned model)")
    ax.bar(x + w / 2, [r[4] for r in rows], w,
           color=[FAMILY_COLOR[r[2]] for r in rows],
           edgecolor=["black" if r[0] == STAR else "none" for r in rows],
           linewidth=[1.4 if r[0] == STAR else 0 for r in rows],
           label="after benign-only content edit")
    _draw_ci(ax, x + w / 2, [r[4] for r in rows],
             [[r[5] for r in rows], [r[6] for r in rows]])
    for i, r in zip(x, rows):
        ax.annotate(f"{r[4]:.2f}", (i + w / 2, r[4] + r[6] + 0.015), ha="center",
                    fontsize=6.5, fontweight="bold" if r[0] == STAR else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], fontsize=7, rotation=12, ha="right")
    ax.set_ylabel(r"substantive-ASR")
    ax.set_ylim(0, 0.56)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [_ci_handle("95% CI (bootstrap)")], l + ["95% CI (bootstrap)"],
              loc="upper right", frameon=False, ncol=3, handlelength=1.2,
              columnspacing=1.2)
    ax.grid(axis="y", ls=":", alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "fig_transfer_asr")


def fig_transfer_bycat():
    """Appendix — where the Llama-3.2-3B transfer concentrates, per HarmBench category."""
    df = pd.read_csv(EDIT_RESULTS / STAR / f"{STAR}_edit_harmful_transfer_bycat_agg.csv")
    base = df[df.arm == "baseline"].set_index("category")
    edit = df[df.arm == "benign_content_edit"].set_index("category")
    pretty = {
        "cybercrime_intrusion": "cybercrime / intrusion",
        "illegal": "illegal (non-violent)",
        "harmful": "general harmful",
        "harassment_bullying": "harassment / bullying",
        "misinformation_disinformation": "misinformation",
    }
    cats = edit["substantive_asr"].sort_values().index.tolist()
    y = np.arange(len(cats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.barh(y + h / 2, [float(base.loc[c, "substantive_asr"]) for c in cats], h,
            color=GRAY, alpha=0.8, label="baseline")
    ev = [float(edit.loc[c, "substantive_asr"]) for c in cats]
    ax.barh(y - h / 2, ev, h, color=FAMILY_COLOR["Llama"], label="benign-only content edit")
    for i, v in zip(y, ev):
        ax.annotate(f"{v:.2f}", (v + 0.015, i - h / 2), va="center",
                    fontsize=6, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty.get(c, c)} (n={int(edit.loc[c, 'n'])})" for c in cats],
                       fontsize=6)
    ax.set_xlabel("substantive-ASR")
    ax.set_xlim(0, 0.98)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2)
    ax.grid(axis="x", ls=":", alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "fig_transfer_bycat")


# ───────────────────────────── LaTeX tables ─────────────────────────────
def _ppl(pct: float) -> str:
    """Percent change, switched to a multiplier once it stops being readable."""
    if abs(pct) < 1000:
        return f"${pct:+.1f}$\\%"
    return f"$\\times${1 + pct / 100:,.0f}"


def _stripe(lines: list[str]) -> list[str]:
    return [(r"\rowcolor{gray!10}" + ln if i % 2 else ln) for i, ln in enumerate(lines)]


def _write(name: str, body: str) -> None:
    (TABLES / name).write_text(body)
    print("  tables/", name)


def table_mapping(rows):
    body = []
    for r in rows:
        body.append(
            f"{r['label']} & {r['n_layers']}$\\times${r['n_heads']} & "
            f"L{r['top_layer']}H{r['top_head']} & "
            f"{r['top_delta']:.2f}\\,$\\pm$\\,{r['top_ci']:.2f} & "
            f"{100*r['refusal_clean']:.0f}\\,$\\rightarrow$\\,\\textbf{{{100*r['refusal_zero']:.0f}}} & "
            f"{_ppl(r['ppl_pct'])} & "
            f"{100*r['jb_plain']:.0f}\\,$\\rightarrow$\\,{100*r['jb_jail']:.0f} & "
            f"${r['margin_plain']:+.2f}$\\,$\\rightarrow$\\,${r['margin_jail']:+.2f}$ & "
            f"${r['rtp_delta']:+.3f}$ \\\\")
    _write("table_mapping.tex", r"""\begin{table*}[t]
\centering
\caption{Mapping study: per-model circuit localization and top-10 head zero-ablation
($N=50$ matched pairs, seed 0, Bender run). Refusal and jailbreak columns are
percentages; $\Delta$PPL is the WikiText-2 capability control.}
\label{tab:mapping}
\footnotesize\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{lccccccc c}
\toprule
\textbf{Model} & \textbf{L$\times$H} & \textbf{Top head} & $\mathbf{|\Delta m|\pm}$\textbf{CI} &
\textbf{Refusal clean$\rightarrow$abl.} & $\mathbf{\Delta}$\textbf{PPL} &
\textbf{JB refusal} & \textbf{JB margin} & \textbf{RTP} $\mathbf{\Delta}$\textbf{tox} \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\parbox[t]{0.98\textwidth}{\vspace{0.1cm}\footnotesize
$|\Delta m|$ is the mean absolute shift of the refusal-logit margin when the head's
output is patched from the benign run, $\pm$ a normal-approximation 95\% CI over the
50 pairs. JB refusal / margin contrast plain AdvBench prompts with HarmBench
jailbreaks. The four models reaching 0\% refusal are exactly the four with the
largest $\Delta$PPL (Finding~A).}
\end{table*}
""")


def table_editing(rows):
    by_key = {r["key"]: r for r in rows}
    body = []
    for key, label, fam in MODELS:
        s = _edit_summary(key)
        r = by_key[key]
        depth = r["top_layer"] / max(r["n_layers"] - 1, 1)
        k = int(s[s["chosen"] == True].iloc[0]["k"])            # noqa: E712
        lo = s.loc[f"lora_k{k}"]
        st = s.loc["steering"]
        body.append(
            f"{label} & L{r['top_layer']} ({depth:.2f}) & "
            f"{100*s.loc['baseline','refusal_rate']:.0f}\\% & "
            f"{100*st['refusal_rate']:.0f}\\% / ${st['perplexity_pct_change']:+.1f}$\\% & "
            f"{k} & \\textbf{{{100*lo['refusal_rate']:.0f}\\%}} & "
            f"{100*lo['jailbreak_refusal_rate']:.0f}\\% & "
            f"{_ppl(lo['perplexity_pct_change'])} & "
            f"{_ppl(s.loc['lora_k10','perplexity_pct_change'])} \\\\")
    _write("table_editing.tex", r"""\begin{table*}[t]
\centering
\caption{Editing extension: all numbers on the held-out \textbf{test} split ($n=45$);
the head count $k$ is selected on \textbf{val} ($n=45$) and never on test.
The steering column reports the best \emph{coherent} sweep combination.
Base refusal is measured on this 45-prompt test split and is therefore not directly
comparable with the 50-prompt mapping set of Table~\ref{tab:mapping}.}
\label{tab:editing}
\footnotesize\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{lccccccc c}
\toprule
\textbf{Model} & \textbf{Top-head depth} & \textbf{Base refusal} &
\textbf{Steering (refusal / $\Delta$PPL)} & \textbf{Chosen} $k$ &
\multicolumn{3}{c}{\textbf{Head-restricted LoRA @ chosen} $k$} & \textbf{LoRA @}$k{=}10$ \\
\cmidrule(lr){6-8}
 & & & & & refusal & jailbreak & $\Delta$PPL & $\Delta$PPL \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\parbox[t]{0.98\textwidth}{\vspace{0.1cm}\footnotesize
Depth is the dominant head's layer index normalized to $[0,1]$. LoRA drives
opener-refusal to 0\% on 9/9 models and HarmBench-jailbreak refusal to 0\% on 8/9
(2\% on Gemma2-2B), at
$\Delta\text{PPL}\leq\!{\sim}10\%$ on 6/9; Gemma1-2B is the catastrophic exception.
Steering never reaches 0\% while remaining coherent (0/9).}
\end{table*}
""")


def table_transfer():
    body = []
    for key, label, fam in MODELS:
        d = _transfer(key)
        def v(arm, col="substantive_asr"):
            return float(d.loc[arm, col])
        e = v("benign_content_edit")
        body.append(
            f"{label} & {v('baseline'):.2f} & {v('opener_edit'):.2f} & "
            f"\\textbf{{{e:.2f}}} [{v('benign_content_edit','subst_asr_lo'):.2f}, "
            f"{v('benign_content_edit','subst_asr_hi'):.2f}] & "
            f"{v('steering_alone'):.2f} & {v('benign_content_edit+steering'):.2f} \\\\")
    _write("table_transfer.tex", r"""\begin{table*}[t]
\centering
\caption{Finding~G: substantive-ASR by attack arm on $N=150$ non-severe HarmBench
prompts, judged by \texttt{cais/HarmBench-Llama-2-13b-cls}. Brackets give the
percentile bootstrap 95\% CI (2000 resamples). No arm is trained on harmful data.}
\label{tab:transfer}
\footnotesize\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Baseline} & \textbf{Opener edit} &
\textbf{Benign content edit} & \textbf{Steering alone} & \textbf{Benign + steering} \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\parbox[t]{0.98\textwidth}{\vspace{0.1cm}\footnotesize
substantive-ASR = judged harmful \emph{and} not a regex refusal \emph{and}
$\geq$200 characters. The opener edit scores 0.00 everywhere despite a raw ASR of
0.87--0.92 (Finding~F). The benign content edit, trained only on benign answers,
lifts Llama-3.2-3B from 0.08 to 0.41 with non-overlapping CIs. Combining
the two removal routes is never better than the better route alone.}
\end{table*}
""")


def table_bycat():
    df = pd.read_csv(EDIT_RESULTS / STAR / f"{STAR}_edit_harmful_transfer_bycat_agg.csv")
    base = df[df.arm == "baseline"].set_index("category")
    edit = df[df.arm == "benign_content_edit"].set_index("category")
    pretty = {
        "cybercrime_intrusion": "cybercrime / intrusion",
        "illegal": "illegal (non-violent)",
        "harmful": "general harmful",
        "harassment_bullying": "harassment / bullying",
        "misinformation_disinformation": "misinformation / disinformation",
    }
    cats = edit["substantive_asr"].sort_values(ascending=False).index.tolist()
    body = [f"{pretty.get(c, c)} & {int(edit.loc[c,'n'])} & "
            f"{float(base.loc[c,'substantive_asr']):.2f} & "
            f"\\textbf{{{float(edit.loc[c,'substantive_asr']):.2f}}} \\\\" for c in cats]
    _write("table_bycat.tex", r"""\begin{table}[t]
\centering
\caption{Where the Llama-3.2-3B transfer concentrates: substantive-ASR by HarmBench
category, baseline vs.\ benign-only content edit.}
\label{tab:bycat}
\begin{tabular}{lccc}
\toprule
\textbf{Category} & $\mathbf{n}$ & \textbf{Baseline} & \textbf{Benign edit} \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_sweep():
    """Appendix D2 --- benign-only tuning sweep, the three models it was run on."""
    body = []
    for key, label, _ in MODELS:
        p = EDIT_RESULTS / key / f"{key}_edit_transfer_sweep_agg.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        best = df.loc[df["substantive_asr"].idxmax()]
        k10 = df[df.k == 10]["substantive_asr"]
        k20 = df[df.k == 20]["substantive_asr"]
        body.append(
            f"{label} & {int(best['k'])} & {int(best['benign_target_tokens'])} & "
            f"{best['decoding']} & \\textbf{{{best['substantive_asr']:.2f}}} & "
            f"{k10.min():.2f}--{k10.max():.2f} & {k20.min():.2f}--{k20.max():.2f} \\\\")
    _write("table_sweep.tex", r"""\begin{table}[H]
\centering
\caption{Benign-only tuning sweep (rank 16, 600 steps; $k\in\{10,20\}$ $\times$ target
length $\in\{256,512\}$ $\times$ \{greedy, sampling\}). On both Llama models a
\emph{tighter} edit is the decisive lever: the top-10 range dominates the top-20 range.
Qwen2.5 is the exception, peaking at $k{=}20$.}
\label{tab:sweep}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & $\mathbf{k}$ & \textbf{Target} & \textbf{Decoding} &
\textbf{Best sASR} & $\mathbf{k{=}10}$ \textbf{range} & $\mathbf{k{=}20}$ \textbf{range} \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_substance():
    """Finding F mechanism --- benign substance unlock, short vs long target."""
    body = []
    for key, label, _ in MODELS:
        p = EDIT_RESULTS / key / f"{key}_edit_benign_substance_agg.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        x = df[df.category == "xstest"].set_index("label")
        if not {"baseline", "benign_short", "benign_long"} <= set(x.index):
            continue
        body.append(
            f"{label} & {100*float(x.loc['baseline','substantive']):.0f}\\% & "
            f"{100*float(x.loc['benign_short','substantive']):.0f}\\% & "
            f"\\textbf{{{100*float(x.loc['benign_long','substantive']):.0f}\\%}} & "
            f"{float(x.loc['benign_short','mean_len']):.0f} & "
            f"{float(x.loc['benign_long','mean_len']):.0f} \\\\")
    _write("table_substance.tex", r"""\begin{table}[H]
\centering
\caption{Finding~F mechanism, measured on benign data only (XSTest over-refusal
prompts, $n=150$). Retraining the same heads with a 24-token affirmative target
collapses substantive answering to 0--22\%, reproducing opener-stop; a 256-token
target partially restores it (up to 47\%) and roughly doubles generation length, so
the shallow jailbreak is a training-target artifact rather than a limit of the heads.}
\label{tab:substance}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Substantive answers}} & \multicolumn{2}{c}{\textbf{Mean length (chars)}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
\textbf{Model} & base & 24-tok & 256-tok & 24-tok & 256-tok \\
\midrule
""" + "\n".join(_stripe(body)) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def main():
    rows = load_all()
    print("figures:")
    fig_sparsity(rows)
    fig_coupling(rows)
    fig_migration(rows)
    fig_jailbreak(rows)
    fig_heatmaps(rows)
    fig_scalpel(rows)
    fig_transfer_arms()
    fig_transfer_asr()
    print("tables:")
    table_mapping(rows)
    table_editing(rows)
    table_transfer()
    table_bycat()
    table_sweep()
    table_substance()
    print(f"\nwrote {len(list(PICS.glob('*.pdf')))} PDFs and "
          f"{len(list(TABLES.glob('*.tex')))} tables under {REPORT}")


if __name__ == "__main__":
    main()
