"""Aggregation and plotting for patching / ablation results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid importing patching (→ torch / transformer_lens) at module load
    from .patching import PatchResult


def patch_results_to_df(results: "Iterable[PatchResult]") -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def aggregate_pairs(per_pair: list[list[PatchResult]]) -> pd.DataFrame:
    """Aggregate |Δmargin| across matched pairs for each (component, layer, head).

    Reports the mean plus dispersion across pairs so rankings come with error
    bars rather than bare point estimates:

    - ``abs_delta`` — mean |Δmargin| over pairs (the ranking key)
    - ``std``       — sample standard deviation across pairs (NaN if a single pair)
    - ``n``         — number of pairs contributing
    - ``sem``       — standard error of the mean (std / sqrt(n))
    - ``ci95``      — 95% confidence half-width (1.96 · sem)

    Sorted by ``abs_delta`` descending.
    """
    dfs = [patch_results_to_df(r) for r in per_pair]
    long = pd.concat(dfs, ignore_index=True)
    long["abs_delta"] = long["delta_margin"].abs()
    out = (
        long.groupby(["component", "layer", "head"], dropna=False)["abs_delta"]
        .agg(abs_delta="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out["std"] / out["n"].pow(0.5)
    out["ci95"] = 1.96 * out["sem"]
    return out.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def head_heatmap(df: pd.DataFrame, n_layers: int, n_heads: int, component: str = "z") -> np.ndarray:
    """Build an (n_layers × n_heads) matrix of mean |Δmargin| for `component`-patching.

    `component` is "z" for the standard head-output sweep or "pattern" for the
    attention-pattern sweep (G8).
    """
    grid = np.zeros((n_layers, n_heads))
    sub = df[df["component"] == component]
    for _, row in sub.iterrows():
        grid[int(row["layer"]), int(row["head"])] = row["abs_delta"]
    return grid


def plot_heatmap(grid: np.ndarray, title: str = "Δrefusal-margin per head", save_to: str | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, grid.shape[1] * 0.4), max(4, grid.shape[0] * 0.3)))
    sns.heatmap(grid, ax=ax, cmap="viridis", cbar_kws={"label": "|Δ refusal margin|"})
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


def plot_k_sweep(
    ks: list[int],
    refusal_ablated: list[float],
    refusal_clean: float,
    title: str = "Refusal rate vs #heads ablated",
    save_to: str | None = None,
):
    """Refusal-rate-vs-K curve for the ablation K-sweep (G2).

    Shows how many top heads must be ablated before refusal collapses — flat-near-zero
    early means a concentrated circuit; a slow decline means a distributed one.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [100 * r for r in refusal_ablated], marker="o", label="ablated")
    ax.axhline(100 * refusal_clean, ls="--", color="grey", label=f"clean ({100*refusal_clean:.0f}%)")
    ax.axhline(30, ls=":", color="red", label="H3 target (30%)")
    ax.set_xlabel("# top heads ablated (K)")
    ax.set_ylabel("Refusal rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig
