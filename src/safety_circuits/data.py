"""Unified loaders for safety datasets.

Each loader yields `Prompt` records with a common schema so downstream code
doesn't have to special-case AdvBench vs HarmBench vs HH-RLHF vs RTP.

Sources:
- AdvBench:               github.com/llm-attacks/llm-attacks  (`harmful_behaviors.csv`)
- HarmBench:              github.com/centerforaisafety/HarmBench
- HH-RLHF:                Anthropic/hh-rlhf  on HF
- RealToxicityPrompts:    allenai/real-toxicity-prompts  on HF

For Colab convenience all loaders prefer HF `datasets` and fall back to a raw
URL only if absolutely necessary.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


Category = Literal["harm", "safe"]


@dataclass
class Prompt:
    text: str
    category: Category
    source: str          # advbench | harmbench | hh-rlhf | rtp
    expected_refusal: bool
    meta: dict | None = None


# --------------------------------------------------------------------- AdvBench
def load_advbench(limit: int | None = None) -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset("walledai/AdvBench", split="train")
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        out.append(
            Prompt(
                text=row["prompt"],
                category="harm",
                source="advbench",
                expected_refusal=True,
                meta={"target": row.get("target")},
            )
        )
    return out


# -------------------------------------------------------------------- HarmBench
def load_harmbench(limit: int | None = None) -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        out.append(
            Prompt(
                text=row["prompt"],
                category="harm",
                source="harmbench",
                expected_refusal=True,
                meta={"category": row.get("category")},
            )
        )
    return out


# ----------------------------------------------------------- RealToxicityPrompts
def load_rtp(limit: int | None = None, toxicity_threshold: float = 0.5) -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    out = []
    for row in ds:
        tox = (row.get("prompt") or {}).get("toxicity")
        if tox is None or tox < toxicity_threshold:
            continue
        out.append(
            Prompt(
                text=row["prompt"]["text"],
                category="harm",
                source="rtp",
                expected_refusal=True,
                meta={"toxicity": tox},
            )
        )
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------- HH-RLHF
def load_hh_harmless(limit: int | None = None) -> list[Prompt]:
    """Benign / harmless instructions — our negative-class anchor for patching."""
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        # `chosen` is the preferred completion; we extract the human turn before it.
        human_turn = _extract_first_human(row["chosen"])
        if human_turn is None:
            continue
        out.append(
            Prompt(
                text=human_turn,
                category="safe",
                source="hh-rlhf",
                expected_refusal=False,
            )
        )
    return out


def _extract_first_human(transcript: str) -> str | None:
    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    if "Human:" not in transcript:
        return None
    after = transcript.split("Human:", 1)[1]
    return after.split("Assistant:")[0].strip() or None


# ---------------------------------------------------- pair construction for patching
def build_matched_pairs(
    harm: list[Prompt],
    safe: list[Prompt],
    n_pairs: int,
    max_len_chars: int = 400,
    seed: int = 0,
) -> list[tuple[Prompt, Prompt]]:
    """Greedy length-matched pairing.

    For clean causal tracing we want each pair to share token length as closely
    as possible — same number of positions to patch, no length-mismatch artefacts.
    """
    rng = random.Random(seed)
    harm = [p for p in harm if len(p.text) <= max_len_chars]
    safe = [p for p in safe if len(p.text) <= max_len_chars]
    rng.shuffle(harm)
    safe_sorted = sorted(safe, key=lambda p: len(p.text))
    pairs = []
    for h in harm[:n_pairs]:
        # binary search-ish: find the safe prompt with closest char-length
        target = len(h.text)
        best_idx = min(range(len(safe_sorted)), key=lambda i: abs(len(safe_sorted[i].text) - target))
        pairs.append((h, safe_sorted.pop(best_idx)))
        if not safe_sorted:
            break
    return pairs


# ----------------------------------------------------------------- jsonl helpers
def save_jsonl(records: Iterable, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]
