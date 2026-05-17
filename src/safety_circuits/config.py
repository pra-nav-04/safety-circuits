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

    @property
    def is_tl_native(self) -> bool:
        """True if TransformerLens ships a built-in loader for this checkpoint."""
        return self.tl_name is not None


MODELS: dict[str, ModelSpec] = {
    "tinyllama": ModelSpec(
        key="tinyllama",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tl_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        refusal_first_tokens=("I", "Sorry", "As", "I'm", "Unfortunately"),
    ),
    "phi3": ModelSpec(
        key="phi3",
        hf_name="microsoft/Phi-3-mini-4k-instruct",
        tl_name=None,  # loaded via HF + manual port (see models.py)
        refusal_first_tokens=("I", "Sorry", "As", "I'm", "Unfortunately"),
    ),
    "qwen": ModelSpec(
        key="qwen",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        tl_name="Qwen/Qwen2.5-1.5B-Instruct",
        refusal_first_tokens=("I", "Sorry", "As", "I'm", "Unfortunately"),
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
