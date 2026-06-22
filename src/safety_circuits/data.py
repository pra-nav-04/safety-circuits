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
    # walledai/AdvBench is gated on HF; load directly from the original public CSV.
    import csv, io, urllib.request

    url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
        "/main/data/advbench/harmful_behaviors.csv"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    out = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        if limit and i >= limit:
            break
        out.append(
            Prompt(
                text=row["goal"],
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

    # walledai/HarmBench is gated: needs HF_TOKEN whose account accepted the terms
    # at hf.co/datasets/walledai/HarmBench. Try the "standard" config, fall back to
    # the default config, and surface a clear message on auth/config failure.
    try:
        ds = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception as e:  # noqa: BLE001
        try:
            ds = load_dataset("walledai/HarmBench", split="train")
        except Exception:
            raise RuntimeError(
                "Could not load walledai/HarmBench (jailbreak stress test). It is gated — "
                "accept the terms at https://huggingface.co/datasets/walledai/HarmBench and "
                "ensure HF_TOKEN is set for this run. Original error: " + repr(e)
            ) from e
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
    """Benign / harmless instructions — our negative-class anchor for patching.

    Also captures the dataset's *chosen* assistant answer in ``meta["response"]`` — a full
    benign reply, used as a long benign training target for the substance-unlock experiment
    (T1.1b)."""
    from datasets import load_dataset

    # streaming=True avoids downloading the full ~4GB dataset.
    ds = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base", split="train", streaming=True
    )
    out = []
    for row in ds:
        human_turn = _extract_first_human(row["chosen"])
        if human_turn is None:
            continue
        out.append(
            Prompt(
                text=human_turn,
                category="safe",
                source="hh-rlhf",
                expected_refusal=False,
                meta={"response": _extract_first_assistant(row["chosen"])},
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def _extract_first_human(transcript: str) -> str | None:
    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    if "Human:" not in transcript:
        return None
    after = transcript.split("Human:", 1)[1]
    return after.split("Assistant:")[0].strip() or None


def _extract_first_assistant(transcript: str) -> str | None:
    """The first Assistant turn (up to the next Human turn) — a full benign answer."""
    if "Assistant:" not in transcript:
        return None
    after = transcript.split("Assistant:", 1)[1]
    return after.split("Human:")[0].strip() or None


# ------------------------------------------------------------ XSTest (over-refused-benign)
def load_xstest(limit: int | None = None, safe_only: bool = True) -> list[Prompt]:
    """XSTest — prompts that *sound* harmful but are benign (e.g. "how do I kill a Python
    process?"); a well-aligned model should NOT refuse them. We use the **safe** subset as
    the over-refused-benign eval for the substance-unlock experiment (T1.1b).

    Tries HF ``walledai/XSTest`` (cols ``prompt``/``label``), then falls back to the public
    CSV from the original repo (filtering safe = ``type`` not containing "contrast").
    """
    def _is_safe(row: dict) -> bool:
        # prefer an explicit label; else infer from `type` (unsafe types contain "contrast")
        if row.get("label") is not None:
            return str(row["label"]).lower() == "safe"
        return "contrast" not in (row.get("type") or "").lower()

    # 1) HF (split name varies across mirrors) — swallow failures and fall back to the CSV.
    try:
        from datasets import load_dataset

        ds = load_dataset("walledai/XSTest")
        split = "test" if "test" in ds else ("train" if "train" in ds else next(iter(ds)))
        out: list[Prompt] = []
        for row in ds[split]:
            if safe_only and not _is_safe(row):
                continue
            out.append(Prompt(text=row["prompt"], category="safe" if _is_safe(row) else "harm",
                              source="xstest", expected_refusal=not _is_safe(row),
                              meta={"label": row.get("label"), "type": row.get("type")}))
            if limit and len(out) >= limit:
                break
        if out:
            return out
    except Exception:  # noqa: BLE001
        pass

    # 2) Raw CSV from the original repo (root file is `xstest_prompts.csv`; cols prompt, type).
    import csv, io, urllib.request
    url = ("https://raw.githubusercontent.com/paul-rottger/exaggerated-safety"
           "/main/xstest_prompts.csv")
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if safe_only and not _is_safe(row):
            continue
        out.append(Prompt(text=row["prompt"], category="safe" if _is_safe(row) else "harm",
                          source="xstest", expected_refusal=not _is_safe(row),
                          meta={"type": row.get("type")}))
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------ WikiText-2 (capability control)
def load_wikitext2(limit: int = 64, min_chars: int = 200) -> list[str]:
    """Plain-text snippets for the perplexity / capability-preservation control.

    Returns WikiText-2 paragraphs (test split). Perplexity on these, measured
    clean vs ablated, is our check that ablating safety heads doesn't just break
    the model (H3: refusal should collapse while perplexity barely moves).
    """
    from datasets import load_dataset

    # The bare canonical "wikitext" id is rejected by newer datasets/huggingface_hub
    # ("Repository id must be 'namespace/name'"). Use the namespaced mirror, fall back
    # to the bare id for older datasets versions.
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for row in ds:
        t = row["text"].strip()
        # skip blank lines and section headers like " = = Heading = = "
        if len(t) < min_chars or t.startswith("="):
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


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
