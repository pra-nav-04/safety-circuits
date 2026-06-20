"""Head-restricted LoRA — the scientific core of the §9 editing extension.

The main study localized refusal to a few attention heads but could only *delete*
them (zero-ablation), which broke the model. Here we instead **retrain** just those
heads: attach a low-rank adapter whose delta is *masked to only the target heads'
dimension slices* of the `q/k/v/o_proj` matrices, train it to suppress refusal, then
merge the delta back into exactly those slices — leaving the rest of the model
bit-for-bit identical. If refusal flips at small ΔPPL, the circuit is a cleanly
*editable* switch (F1a) even though it was not cleanly *ablatable*.

Why custom (not vanilla PEFT): PEFT applies LoRA to a whole `nn.Linear`. We need
sub-projection restriction — only the rows/cols that belong to a specific head — which
requires masking the low-rank factors. The mask is GQA-aware: under grouped-query
attention (Qwen2/2.5/3, Gemma-2/3) `k_proj`/`v_proj` have fewer heads than `q_proj`,
so a query head maps to a key/value group.

Standard HF attention shapes (Llama / Qwen / Gemma):
    q_proj: [n_heads      * head_dim, d_model]   o_proj: [d_model, n_heads * head_dim]
    k_proj: [n_kv_heads   * head_dim, d_model]   v_proj: [n_kv_heads * head_dim, d_model]

For LoRA `ΔW = scaling · B @ A`  (A:[r,in], B:[out,r]):
    q/k/v  → restrict the *output* rows  → mask rows of B  (head owns out-slice)
    o      → restrict the *input* cols   → mask cols of A  (head owns in-slice)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # HeadRef lives in ablation.py, which imports TransformerLens; keep this
    from .ablation import HeadRef  # module import torch-only so the unit tests need no TL.


# --------------------------------------------------------------------- arch access
def _decoder_layers(hf_model: nn.Module) -> nn.ModuleList:
    """Return the decoder-layer ModuleList across the arches in our roster."""
    m = hf_model
    # text-only causal LMs: Llama/Qwen → .model.layers ; some wrappers nest .model.model
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"Could not locate decoder layers on {type(hf_model).__name__}")


def _attn_module(hf_model: nn.Module, layer: int) -> nn.Module:
    attn = getattr(_decoder_layers(hf_model)[layer], "self_attn", None)
    if attn is None:  # GPT-NeoX-style naming, just in case
        attn = getattr(_decoder_layers(hf_model)[layer], "attention")
    return attn


@dataclass
class GQAInfo:
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads


def gqa_info(hf_model: nn.Module) -> GQAInfo:
    cfg = hf_model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    return GQAInfo(n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim)


def kv_group(head: int, info: GQAInfo) -> int:
    """Map a query-head index (TL/`safety_heads.json` indexing) to its kv group."""
    return head // info.group_size


# --------------------------------------------------------------------- the adapter
class HeadMaskedLoRALinear(nn.Module):
    """Wrap a frozen `nn.Linear` with a LoRA delta masked to specific head slices.

    `role` ∈ {"q","k","v","o"} selects whether the head slice lives on the output
    rows (q/k/v) or the input columns (o). `heads` are the *query-head* indices to
    edit (kv slices are derived via `kv_group`).
    """

    def __init__(
        self,
        base: nn.Linear,
        role: str,
        heads: list[int],
        info: GQAInfo,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.role = role
        self.rank = rank
        self.scaling = alpha / rank
        out_features, in_features = base.weight.shape

        # Co-locate all new tensors with the base weight, so the adapter works whether
        # it is injected before or after `hf_model.to(device)` (otherwise the LoRA params
        # land on CPU while the base is on CUDA → device-mismatch in forward).
        device = base.weight.device

        # LoRA factors in fp32 for stable T4 training; base may be fp16.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # lora_B stays zero → initial delta is exactly zero → edited model == base before training.

        out_mask, in_mask = self._build_masks(role, heads, info, out_features, in_features)
        self.register_buffer("out_mask", out_mask.to(device))  # [out, 1]
        self.register_buffer("in_mask", in_mask.to(device))     # [1, in]

    @staticmethod
    def _build_masks(role, heads, info, out_features, in_features):
        out_mask = torch.zeros(out_features, 1)
        in_mask = torch.zeros(1, in_features)
        d = info.head_dim
        if role in ("q", "k", "v"):
            # restrict output rows to the head/group slices; full input
            in_mask[:] = 1.0
            slots = (
                sorted({kv_group(h, info) for h in heads}) if role in ("k", "v") else sorted(set(heads))
            )
            for s in slots:
                out_mask[s * d : (s + 1) * d, 0] = 1.0
        elif role == "o":
            # restrict input cols to the head slices; full output
            out_mask[:] = 1.0
            for h in sorted(set(heads)):
                in_mask[0, h * d : (h + 1) * d] = 1.0
        else:
            raise ValueError(f"unknown role {role!r}")
        return out_mask, in_mask

    def delta_weight(self) -> torch.Tensor:
        """The masked low-rank weight delta `scaling · (mask∘B) @ (A∘mask)` in fp32."""
        b = self.lora_B * self.out_mask
        a = self.lora_A * self.in_mask
        return self.scaling * (b @ a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x.to(torch.float32) @ self.delta_weight().t()).to(out.dtype)
        return out + delta


# --------------------------------------------------------------------- inject / merge
_ROLE_BY_PROJ = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}


def inject_head_lora(
    hf_model: nn.Module,
    heads: list[HeadRef],
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> list[HeadMaskedLoRALinear]:
    """Replace the targeted projections (only in layers that contain a safety head)
    with head-masked LoRA wrappers; freeze everything else. Returns the adapters
    (with `_parent`/`_attr` set so `merge_head_lora` can restore plain Linears).
    """
    info = gqa_info(hf_model)
    for p in hf_model.parameters():
        p.requires_grad_(False)

    by_layer: dict[int, list[int]] = {}
    for h in heads:
        by_layer.setdefault(h.layer, []).append(h.head)

    adapters: list[HeadMaskedLoRALinear] = []
    for layer, head_idx in by_layer.items():
        attn = _attn_module(hf_model, layer)
        for proj in targets:
            base = getattr(attn, proj, None)
            if not isinstance(base, nn.Linear):
                continue  # arch without this exact name — skip rather than crash
            adapter = HeadMaskedLoRALinear(
                base, role=_ROLE_BY_PROJ[proj], heads=head_idx, info=info, rank=rank, alpha=alpha
            )
            # Store the parent ref via object.__setattr__ so nn.Module does NOT register
            # `attn` as a submodule of the adapter — that would create a cycle
            # (attn → adapter → attn) and blow the stack on any module-tree walk
            # (e.g. hf_model.train()). `_attr` is a str so it's a plain attribute.
            object.__setattr__(adapter, "_parent", attn)
            adapter._attr = proj
            adapter.lora_A.requires_grad_(True)
            adapter.lora_B.requires_grad_(True)
            setattr(attn, proj, adapter)
            adapters.append(adapter)
    return adapters


@torch.no_grad()
def merge_head_lora(adapters: list[HeadMaskedLoRALinear]) -> None:
    """Fold each (masked) delta into its base weight in-place and restore the plain
    `nn.Linear` at the original attribute — so the merged HF model has standard module
    names and ports cleanly into TransformerLens, with only the target head slices changed.
    """
    for ad in adapters:
        base = ad.base
        before = base.weight.detach().clone()
        delta = ad.delta_weight().to(base.weight.dtype)
        base.weight.add_(delta)
        # sanity: rows/cols outside the head mask must be untouched
        changed = (base.weight - before).abs() > 0
        allowed = (ad.out_mask.bool() & ad.in_mask.bool())
        assert not bool((changed & ~allowed).any()), "LoRA merge touched non-target slices"
        setattr(ad._parent, ad._attr, base)
