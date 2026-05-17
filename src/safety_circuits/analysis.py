"""Aggregation and plotting for patching / ablation results."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .patching import PatchResult


def patch_results_to_df(results: Iterable[PatchResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def aggregate_pairs(per_pair: list[list[PatchResult]]) -> pd.DataFrame:
    """Average |Δmargin| across pairs for each (component, layer, head)."""
    dfs = [patch_results_to_df(r) for r in per_pair]
    long = pd.concat(dfs, ignore_index=True)
    long["abs_delta"] = long["delta_margin"].abs()
    return (
        long.groupby(["component", "layer", "head"], dropna=False)["abs_delta"]
        .mean()
        .reset_index()
        .sort_values("abs_delta", ascending=False)
    )


def head_heatmap(df: pd.DataFrame, n_layers: int, n_heads: int) -> np.ndarray:
    """Build an (n_layers × n_heads) matrix of mean |Δmargin| for z-patching."""
    grid = np.zeros((n_layers, n_heads))
    z = df[df["component"] == "z"]
    for _, row in z.iterrows():
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
