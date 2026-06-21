from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


@dataclass
class ModelSpec:
    key: str
    hf_name: str
    tl_name: str | None
    refusal_first_tokens: tuple[str, ...]
    prepend_bos: bool = True        # Qwen2.x/3.x have no BOS token; set False for those
    dtype: str = "float32"          # preferred inference dtype; float16 for 7-8B models on T4
    no_think: bool = False          # Qwen3 thinking mode — set True to disable <think> blocks
    needs_rope_patch: bool = False  # Phi-3: custom code reads rope_scaling["type"] but newer
                                    # transformers uses "rope_type"; patch bridges the gap

    @property
    def is_tl_native(self) -> bool:
        """True if TransformerLens ships a built-in loader for this checkpoint."""
        return self.tl_name is not None


_REFUSAL = ("I", "Sorry", "As", "I'm", "Unfortunately")

MODELS: dict[str, ModelSpec] = {
    # ── baseline ────────────────────────────────────────────────────────────
    # NOTE: NOT loadable by the pinned TransformerLens version — TinyLlama is not in
    # its OFFICIAL_MODEL_NAMES, so both the native and HF-port paths raise ValueError
    # (see Kaggle_Logs/Qwen2.5-1.5B-Instruct/download.txt). Kept for reference/history;
    # excluded from the default multi-model run (opt in via SC_MODELS=tinyllama).
    "tinyllama": ModelSpec(
        key="tinyllama",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tl_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        refusal_first_tokens=_REFUSAL,
    ),
    # ── Qwen family: 4 generations at ~constant size (generational sweep) ────
    "qwen1.5-1.8b": ModelSpec(
        key="qwen1.5-1.8b",
        hf_name="Qwen/Qwen1.5-1.8B-Chat",
        tl_name="Qwen/Qwen1.5-1.8B-Chat",
        refusal_first_tokens=_REFUSAL,
        prepend_bos=False,
        dtype="float16",
    ),
    "qwen2-1.5b": ModelSpec(
        key="qwen2-1.5b",
        hf_name="Qwen/Qwen2-1.5B-Instruct",
        tl_name="Qwen/Qwen2-1.5B-Instruct",
        refusal_first_tokens=_REFUSAL,
        prepend_bos=False,
        dtype="float16",
    ),
    "qwen2.5": ModelSpec(
        key="qwen2.5",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        tl_name="Qwen/Qwen2.5-1.5B-Instruct",
        refusal_first_tokens=_REFUSAL,
        prepend_bos=False,
    ),
    "qwen3": ModelSpec(
        key="qwen3",
        hf_name="Qwen/Qwen3-1.7B",
        tl_name="Qwen/Qwen3-1.7B",
        refusal_first_tokens=_REFUSAL,
        prepend_bos=False,
        dtype="float16",
        no_think=True,
    ),
    "phi3": ModelSpec(
        key="phi3",
        hf_name="microsoft/Phi-3-mini-4k-instruct",
        tl_name=None,  # TL's from_pretrained loads fp32-first then casts → OOM on T4; use HF path
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
        needs_rope_patch=True,
    ),
    # ── Gemma family: 3 generations (generational sweep) — all GATED ─────────
    # fp32 for Gemma-1/2: gemma-2 uses attention logit soft-capping that can NaN in fp16.
    # HF-port (tl_name=None): TL's native loader is fp32-first → a 2B model OOMs the
    # ~13GB T4 RAM at load. The HF-port streams weights (low_cpu_mem_usage) → fits.
    "gemma1-2b": ModelSpec(
        key="gemma1-2b",
        hf_name="google/gemma-2b-it",
        tl_name=None,  # GATED: accept terms at hf.co/google/gemma-2b-it
        refusal_first_tokens=_REFUSAL,
        dtype="float16",  # Gemma-1 has no soft-capping; fp16 is safe and halves RAM
    ),
    "gemma2-2b": ModelSpec(
        key="gemma2-2b",
        hf_name="google/gemma-2-2b-it",
        tl_name=None,  # GATED: accept terms at hf.co/google/gemma-2-2b-it
        # fp32 (~10GB) + activations OOMs the T4's ~14.5GB VRAM, so fp16 is forced.
        # Gemma-2's logit soft-capping *can* be unstable in fp16 — the coherence check
        # gates this: if completions are garbage/NaN, drop gemma2-2b (keep G1 + G3).
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    "gemma3-1b": ModelSpec(
        key="gemma3-1b",
        hf_name="google/gemma-3-1b-it",
        tl_name="google/gemma-3-1b-it",  # TL-native; GATED: accept terms at hf.co/google/gemma-3-1b-it
        refusal_first_tokens=_REFUSAL,
    ),
    # ── Llama family: 2 sizes (size sweep) — GATED ───────────────────────────
    "llama3.2-1b": ModelSpec(
        key="llama3.2-1b",
        hf_name="meta-llama/Llama-3.2-1B-Instruct",
        tl_name=None,  # HF path (mirrors llama3-3b). GATED: accept terms at hf.co/meta-llama/Llama-3.2-1B-Instruct
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    "llama3-3b": ModelSpec(
        key="llama3-3b",
        hf_name="meta-llama/Llama-3.2-3B-Instruct",
        tl_name=None,  # TL's from_pretrained loads fp32-first then casts → OOM on T4; use HF path. GATED: accept terms at hf.co/meta-llama/Llama-3.2-3B-Instruct
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    # ── extended sweep ─────────────────────────────────────────────────────────
    # NOTE: both UNSUPPORTED by the pinned TransformerLens — neither is in its
    # OFFICIAL_MODEL_NAMES (the HF-port path checks that list too), so load raises
    # ValueError. Confirmed at runtime for Falcon3; OLMo-2 lists only the *base*
    # allenai/OLMo-2-0425-1B, not the -Instruct. Kept for reference; excluded from
    # the default run.
    "falcon3-1b": ModelSpec(
        key="falcon3-1b",
        hf_name="tiiuae/Falcon3-1B-Instruct",
        tl_name=None,  # Falcon3 not in TL's native list; HF path also rejects it
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    "olmo2-1b": ModelSpec(
        key="olmo2-1b",
        hf_name="allenai/OLMo-2-0425-1B-Instruct",
        tl_name=None,  # only the base OLMo-2-0425-1B is TL-supported, not -Instruct
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    # ── probe only ───────────────────────────────────────────────────────────
    # Gemma 4 (released 2026-04-02). Expected to FAIL under the pinned TL: not in
    # OFFICIAL_MODEL_NAMES, and it's a multimodal "any-to-any"/MoE arch the HF-port
    # (AutoModelForCausalLM) can't load either. Kept as a 1-min preflight probe to
    # re-check whenever TL adds support: SC_PREFLIGHT=1 SC_MODELS=gemma4-e2b.
    "gemma4-e2b": ModelSpec(
        key="gemma4-e2b",
        hf_name="google/gemma-4-E2B-it",
        tl_name=None,
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(spec: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[spec]


@dataclass
class EditConfig:
    """Hyper-parameters for the §9 weight-editing extension (head-restricted LoRA +
    the no-train steering baseline). Mirrors `ExpConfig`'s plain-dataclass style; the
    Kaggle orchestrator overrides fields from `SC_EDIT_*` env vars.
    """
    # head-masked LoRA
    rank: int = 8
    alpha: int = 16
    steps: int = 300
    lr: float = 2e-4
    batch: int = 4
    grad_accum: int = 4
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # F1b: how many top safety heads to retrain (the head-count sweep)
    head_counts: tuple[int, ...] = (1, 3, 5, 10)
    # no-train steering baseline (Arditi et al. directional ablation)
    steering_layer_frac: float = 0.6   # which layer's resid to read the refusal direction from (× depth)
    steering_coeff: float = 1.0        # subtraction strength
    steering_layers: str = "all"       # where to ablate: "all" | "extract" | comma list e.g. "10,11,12"
    # misc
    max_target_tokens: int = 24        # truncate the affirmative target continuation
    seed: int = 0
    # ── Tier 1/2 extension knobs (all opt-in via SC_DO_* in the orchestrator) ──
    minimal_ranks: tuple[int, ...] = (4, 8, 16)      # T1.2 minimal-edit sweep grid
    minimal_steps: tuple[int, ...] = (150, 300, 600)
    deep_eval_tokens: int = 128                       # T1.1 long-form generation length
    refusal_target: str = "I cannot help with that."  # T2.5 refusal-induction target
    benign_target_tokens: int = 128                   # T1.1b long benign training-target length


@dataclass
class ExpConfig:
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    model: dict = field(default_factory=lambda: {"name": "tinyllama"})
    data: dict = field(default_factory=dict)
    refusal: dict = field(default_factory=dict)
    patching: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "ExpConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @property
    def model_spec(self) -> ModelSpec:
        return MODELS[self.model["name"]]

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)
