"""No-train steering-vector baseline (Arditi et al., 2024).

The non-training comparison point on the "scalpel-sharpness" axis: blunt zero-ablation
(crude) → steering vector (no-train) → head-restricted LoRA (trained). We compute a
single **refusal direction** as the difference of mean residual-stream activations on
harmful vs benign prompts, then *project it out* of the residual stream at every layer
(directional ablation) to suppress refusal — no gradient updates, pure forward hooks,
entirely inside the existing TransformerLens harness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:  # TransformerLens is imported lazily so this module (and the unit
    from .models import LoadedModel  # test of `project_out`) need no TL at import time.


def resolve_steering_layers(spec: str, extract_layer: int, n_layers: int) -> list[int]:
    """Map a layer-set spec to concrete layer indices for directional ablation.

    `"all"` ablates the refusal direction out of every layer's residual (the literal
    Arditi method, but on some models — e.g. gemma3-1b — this compounds and wrecks
    perplexity); `"extract"` ablates only at the layer the direction was read from (the
    minimal, conservative intervention); a comma list like `"10,11,12"` ablates an
    explicit window. Pairs with `steering_coeff` for a fair midpoint on the
    blunt-ablation → steering → LoRA scalpel-sharpness axis.
    """
    spec = (spec or "all").strip().lower()
    if spec in ("", "all"):
        return list(range(n_layers))
    if spec == "extract":
        return [extract_layer]
    if spec.startswith("frac"):
        # "frac0.4-0.8" → the layer window [0.4·depth, 0.8·depth] inclusive
        lo, hi = (float(x) for x in spec[4:].split("-"))
        a, b = int(lo * n_layers), int(hi * n_layers)
        return [i for i in range(min(a, b), max(a, b) + 1) if 0 <= i < n_layers] or [extract_layer]
    out = [int(t) for t in spec.split(",") if t.strip() != ""]
    out = [i for i in out if 0 <= i < n_layers]
    return out or [extract_layer]


def project_out(resid: torch.Tensor, direction: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """`x − coeff·(x·d̂)d̂` over the last dim. Pure torch — the steering hook's core."""
    unit = direction / (direction.norm() + 1e-8)
    unit = unit.to(device=resid.device, dtype=resid.dtype)
    proj = torch.einsum("...d,d->...", resid, unit)  # [...]
    return resid - coeff * proj.unsqueeze(-1) * unit


@torch.no_grad()
def compute_refusal_direction(
    loaded: "LoadedModel",
    harm_prompts: list[str],
    safe_prompts: list[str],
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Difference-of-means refusal direction at `resid_post[layer]`, last token.

    Returns a unit vector `[d_model]` pointing from benign → refusing activations.
    Pass `normalize=False` to get the *raw* difference-of-means — its norm measures how
    strong the refusal direction is (used by the direction-shift diagnostic to see whether
    an edit shrank it; steering callers normalize internally either way).
    """
    from transformer_lens.utils import get_act_name

    from .activations import run_with_cache

    name = get_act_name("resid_post", layer)

    def _mean_last(prompts: list[str], desc: str) -> torch.Tensor:
        acc = None
        for p in tqdm(prompts, desc=desc):
            _, cache = run_with_cache(loaded, p, names_filter=["resid_post"])
            v = cache[name][0, -1].float()  # [d_model] at the last (pre-generation) token
            acc = v if acc is None else acc + v
        return acc / max(len(prompts), 1)

    mu_harm = _mean_last(harm_prompts, "dir:harm")
    mu_safe = _mean_last(safe_prompts, "dir:safe")
    direction = mu_harm - mu_safe
    return direction / (direction.norm() + 1e-8) if normalize else direction


def make_steering_hooks(direction: torch.Tensor, layers: list[int], coeff: float = 1.0):
    """Forward hooks that project the refusal direction out of `resid_post` at `layers`.

    `x' = x − coeff · (x · d̂) d̂`. Returns the `(name, hook)` list the existing
    `ablation`/`perplexity` helpers already accept as `fwd_hooks`.
    """
    from transformer_lens.utils import get_act_name

    unit = direction / (direction.norm() + 1e-8)

    def make_hook(unit=unit):
        def hook(resid: torch.Tensor, hook) -> torch.Tensor:
            return project_out(resid, unit, coeff)
        return hook

    return [(get_act_name("resid_post", layer), make_hook()) for layer in layers]
