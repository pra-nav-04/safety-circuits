"""Model-free tests for the refusal-metric audit agreement math (G5)."""

import csv

import pytest

from safety_circuits.audit import compute_agreement


def _write_sheet(path, rows):
    fields = ["prompt", "continuation", "margin",
              "auto_logit_refusal", "auto_regex_refusal", "human_label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_perfect_agreement(tmp_path):
    p = tmp_path / "audit.csv"
    _write_sheet(p, [
        {"prompt": "a", "continuation": "I cannot", "margin": 2.0,
         "auto_logit_refusal": 1, "auto_regex_refusal": 1, "human_label": 1},
        {"prompt": "b", "continuation": "Sure, here", "margin": -2.0,
         "auto_logit_refusal": 0, "auto_regex_refusal": 0, "human_label": 0},
    ])
    rep = compute_agreement(p)
    assert rep.n == 2
    assert rep.logit_agreement == 1.0
    assert rep.regex_agreement == 1.0
    assert rep.combined_agreement == 1.0


def test_combined_is_or_of_signals(tmp_path):
    # regex misses a refusal that the logit catches → combined should still agree.
    p = tmp_path / "audit.csv"
    _write_sheet(p, [
        {"prompt": "a", "continuation": "As a model I won't", "margin": 1.0,
         "auto_logit_refusal": 1, "auto_regex_refusal": 0, "human_label": 1},
    ])
    rep = compute_agreement(p)
    assert rep.logit_agreement == 1.0      # logit matched human=1
    assert rep.regex_agreement == 0.0      # regex said 0, human 1
    assert rep.combined_agreement == 1.0   # (1 OR 0) == 1 == human


def test_only_labelled_rows_count(tmp_path):
    p = tmp_path / "audit.csv"
    _write_sheet(p, [
        {"prompt": "a", "continuation": "I cannot", "margin": 2.0,
         "auto_logit_refusal": 1, "auto_regex_refusal": 1, "human_label": 1},
        {"prompt": "b", "continuation": "", "margin": 0.0,
         "auto_logit_refusal": 0, "auto_regex_refusal": 0, "human_label": ""},  # unlabelled
    ])
    rep = compute_agreement(p)
    assert rep.n == 1


def test_raises_when_nothing_labelled(tmp_path):
    p = tmp_path / "audit.csv"
    _write_sheet(p, [
        {"prompt": "a", "continuation": "x", "margin": 0.0,
         "auto_logit_refusal": 0, "auto_regex_refusal": 0, "human_label": ""},
    ])
    with pytest.raises(ValueError):
        compute_agreement(p)
