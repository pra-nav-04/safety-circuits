"""Model-free tests for aggregation statistics + heatmap (G10, G8)."""

import math
from dataclasses import dataclass

from safety_circuits.analysis import aggregate_pairs, head_heatmap


@dataclass
class FakePatch:
    component: str
    layer: int
    head: int
    delta_margin: float


def test_aggregate_pairs_reports_mean_and_dispersion():
    # one head across two pairs: |Δ| = 1.0 and 3.0  → mean 2.0, std sqrt(2), sem 1.0
    pair1 = [FakePatch("z", 0, 0, 1.0)]
    pair2 = [FakePatch("z", 0, 0, -3.0)]
    df = aggregate_pairs([pair1, pair2])

    row = df.iloc[0]
    assert row["component"] == "z"
    assert row["abs_delta"] == 2.0
    assert int(row["n"]) == 2
    assert math.isclose(row["std"], math.sqrt(2), rel_tol=1e-6)
    assert math.isclose(row["sem"], 1.0, rel_tol=1e-6)
    assert math.isclose(row["ci95"], 1.96, rel_tol=1e-6)


def test_aggregate_pairs_sorted_descending():
    pair = [FakePatch("z", 0, 0, 0.1), FakePatch("z", 1, 1, 5.0)]
    df = aggregate_pairs([pair])
    # strongest |Δ| first
    assert df.iloc[0]["layer"] == 1 and df.iloc[0]["head"] == 1
    assert df.iloc[0]["abs_delta"] == 5.0


def test_head_heatmap_places_values_and_selects_component():
    pair = [
        FakePatch("z", 0, 1, 2.0),
        FakePatch("pattern", 0, 1, 9.0),   # different component, must be ignored for "z"
    ]
    df = aggregate_pairs([pair])

    grid_z = head_heatmap(df, n_layers=2, n_heads=2, component="z")
    assert grid_z.shape == (2, 2)
    assert grid_z[0, 1] == 2.0
    assert grid_z[1, 1] == 0.0  # untouched cell

    grid_pat = head_heatmap(df, n_layers=2, n_heads=2, component="pattern")
    assert grid_pat[0, 1] == 9.0
