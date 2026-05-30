"""Refusal-metric audit harness (G5).

Success criterion (RESEARCH_PLAN §4): the automatic refusal metric must agree
with human judgement on ≥ 90% of a 50-prompt audit. Workflow:

1. `build_audit_sheet()` — run the model on a sample of prompts, record the
   automatic labels (logit-margin sign + regex on the continuation) and the
   continuation text, and write a CSV with an empty `human_label` column.
2. A human fills `human_label` with 1 = refused / 0 = complied.
3. `compute_agreement()` — read the filled CSV and report how often each automatic
   signal (and their OR-combination) matches the human label.

`compute_agreement` is pure-stdlib (no torch / TL) so it can run anywhere the
labelled CSV lives; only `build_audit_sheet` needs a loaded model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing models/transformer_lens at module load
    from .models import LoadedModel


def build_audit_sheet(
    loaded: "LoadedModel",
    prompts: list[str],
    out_csv: str | Path,
    margin_threshold: float = 0.0,
) -> Path:
    """Run `prompts`, dump prompt/continuation/auto-labels to a CSV for human review.

    The `human_label` column is left blank for a human to fill (1 refused /
    0 complied). `margin_threshold` is the logit-margin decision boundary for the
    automatic logit label (default 0.0 — refusal mass exceeds non-refusal mass).
    """
    from .refusal import score_refusal  # lazy: pulls in torch

    rows: list[dict] = []
    for p in prompts:
        s = score_refusal(loaded, p, generate_continuation=True)
        rows.append(
            {
                "prompt": p,
                "continuation": s.continuation.replace("\n", " ").strip(),
                "margin": round(s.margin, 4),
                "auto_logit_refusal": int(s.margin > margin_threshold),
                "auto_regex_refusal": int(s.refused_regex),
                "human_label": "",  # fill: 1 = refused, 0 = complied
            }
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv


@dataclass
class AgreementReport:
    n: int                      # number of human-labelled rows
    logit_agreement: float      # auto_logit_refusal vs human
    regex_agreement: float      # auto_regex_refusal vs human
    combined_agreement: float   # (logit OR regex) vs human


def compute_agreement(csv_path: str | Path) -> AgreementReport:
    """Agreement of each automatic refusal signal vs the human labels in `csv_path`.

    Only rows with a non-empty `human_label` are scored. Raises if none are filled.
    """
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human_label") or "").strip() != ""]

    n = len(rows)
    if n == 0:
        raise ValueError("No human_label values filled in — label the sheet first.")

    def agreement(kind: str) -> float:
        hits = 0
        for r in rows:
            human = int(r["human_label"])
            if kind == "combined":
                auto = int(bool(int(r["auto_logit_refusal"])) or bool(int(r["auto_regex_refusal"])))
            else:
                auto = int(r[kind])
            hits += int(auto == human)
        return hits / n

    return AgreementReport(
        n=n,
        logit_agreement=agreement("auto_logit_refusal"),
        regex_agreement=agreement("auto_regex_refusal"),
        combined_agreement=agreement("combined"),
    )
