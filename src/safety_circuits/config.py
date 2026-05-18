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
    # ── baseline (completed) ────────────────────────────────────────────────
    "tinyllama": ModelSpec(
        key="tinyllama",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tl_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        refusal_first_tokens=_REFUSAL,
    ),
    "qwen": ModelSpec(
        key="qwen",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        tl_name="Qwen/Qwen2.5-1.5B-Instruct",
        refusal_first_tokens=_REFUSAL,
        prepend_bos=False,
    ),
    # ── planned multi-model sweep ───────────────────────────────────────────
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
    "gemma3-1b": ModelSpec(
        key="gemma3-1b",
        hf_name="google/gemma-3-1b-it",
        tl_name="google/gemma-3-1b-it",  # TL-native; GATED: accept terms at hf.co/google/gemma-3-1b-it
        refusal_first_tokens=_REFUSAL,
    ),
    "llama3-3b": ModelSpec(
        key="llama3-3b",
        hf_name="meta-llama/Llama-3.2-3B-Instruct",
        tl_name=None,  # TL's from_pretrained loads fp32-first then casts → OOM on T4; use HF path. GATED: accept terms at hf.co/meta-llama/Llama-3.2-3B-Instruct
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    # ── extended sweep ─────────────────────────────────────────────────────────
    "falcon3-1b": ModelSpec(
        key="falcon3-1b",
        hf_name="tiiuae/Falcon3-1B-Instruct",
        tl_name=None,  # Falcon3 not in TL's native list; use HF path
        refusal_first_tokens=_REFUSAL,
        dtype="float16",
    ),
    "olmo2-1b": ModelSpec(
        key="olmo2-1b",
        hf_name="allenai/OLMo-2-0425-1B-Instruct",
        tl_name=None,  # OLMo-2 not in TL's native list; use HF path
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
