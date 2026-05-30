"""Zero / mean ablation of candidate components.

Once patching has identified candidates, ablation is the confirmatory test:
*remove* the head and check whether refusal rate collapses on a held-out set,
without breaking general perplexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name

from .activations import run_with_cache
from .models import LoadedModel, apply_chat_template
from .refusal import score_refusal


AblationMode = Literal["zero", "mean"]


@dataclass
class HeadRef:
    layer: int
    head: int


@dataclass
class AblationReport:
    refusal_rate_clean: float
    refusal_rate_ablated: float
    perplexity_clean: float | None
    perplexity_ablated: float | None

    @property
    def perplexity_pct_change(self) -> float | None:
        """Δ perplexity as a percentage of clean (H3 target: ≤ 5%). None if unmeasured."""
        if self.perplexity_clean is None or self.perplexity_ablated is None:
            return None
        return 100.0 * (self.perplexity_ablated - self.perplexity_clean) / self.perplexity_clean


def _build_head_ablation_hooks(
    heads: list[HeadRef],
    mode: AblationMode,
    mean_cache: dict[str, torch.Tensor] | None,
):
    """Group hooks by layer so we register one per layer (TL needs that)."""
    by_layer: dict[int, list[HeadRef]] = {}
    for h in heads:
        by_layer.setdefault(h.layer, []).append(h)

    hooks = []
    for layer, head_refs in by_layer.items():
        hook_name = get_act_name("z", layer)
        head_idx = [hr.head for hr in head_refs]

        def make_hook(layer=layer, head_idx=head_idx, hook_name=hook_name):
            def hook(z: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                # z: [batch, seq, head, d_head]
                if mode == "zero":
                    z[:, :, head_idx, :] = 0.0
                elif mode == "mean":
                    assert mean_cache is not None
                    mean = mean_cache[hook_name].to(z.device, dtype=z.dtype)
                    z[:, :, head_idx, :] = mean[:, head_idx, :]
                return z
            return hook

        hooks.append((hook_name, make_hook()))
    return hooks


@torch.no_grad()
def compute_mean_z_cache(
    loaded: LoadedModel, benign_prompts: list[str]
) -> dict[str, torch.Tensor]:
    """Mean head outputs over a benign batch — used by mean-ablation.

    Returns a dict `{hook_name: [seq, head, d_head]}` averaged over prompts.
    """
    totals: dict[str, torch.Tensor] = {}
    counts = 0
    for p in tqdm(benign_prompts, desc="mean-cache"):
        _, cache = run_with_cache(loaded, p, names_filter=["z"])
        for name, val in cache.items():
            v = val[0].detach()
            if name not in totals:
                totals[name] = torch.zeros_like(v)
            # truncate or pad to a common length by mean over seq
            totals[name] = totals[name] + v.mean(dim=0, keepdim=True).expand_as(totals[name])
        counts += 1
    return {k: v / counts for k, v in totals.items()}


@torch.no_grad()
def evaluate_ablation(
    loaded: LoadedModel,
    heads: list[HeadRef],
    eval_prompts: list[str],
    mode: AblationMode = "zero",
    mean_cache: dict[str, torch.Tensor] | None = None,
    perplexity_texts: list[str] | None = None,
) -> AblationReport:
    """Refusal rate, clean vs ablated, on `eval_prompts`.

    If `perplexity_texts` is given (e.g. a WikiText-2 slice from
    `data.load_wikitext2`), also reports token-level perplexity clean vs ablated
    on those texts — the H3 capability-preservation control: refusal should
    collapse while general-language perplexity barely moves.
    """
    # clean
    clean_refused = sum(
        score_refusal(loaded, p, generate_continuation=True).refused_regex
        for p in tqdm(eval_prompts, desc="clean")
    )

    hooks = _build_head_ablation_hooks(heads, mode, mean_cache)
    abl_refused = 0
    for p in tqdm(eval_prompts, desc="ablated"):
        chat = apply_chat_template(loaded, p)
        tokens = loaded.model.to_tokens(chat, prepend_bos=True).to(loaded.device)
        # generate with hooks active
        loaded.model.reset_hooks()
        for name, hk in hooks:
            loaded.model.add_hook(name, hk)
        try:
            out = loaded.model.generate(tokens, max_new_tokens=30, temperature=0.0, verbose=False)
        finally:
            loaded.model.reset_hooks()
        continuation = loaded.model.to_string(out[0, tokens.shape[1]:])
        from .refusal import regex_refusal
        abl_refused += int(regex_refusal(continuation))

    # capability-preservation control (optional)
    ppl_clean = ppl_ablated = None
    if perplexity_texts:
        ppl_clean = perplexity(loaded, perplexity_texts)
        ppl_ablated = perplexity(loaded, perplexity_texts, fwd_hooks=hooks)

    n = len(eval_prompts)
    return AblationReport(
        refusal_rate_clean=clean_refused / n,
        refusal_rate_ablated=abl_refused / n,
        perplexity_clean=ppl_clean,
        perplexity_ablated=ppl_ablated,
    )


@torch.no_grad()
def perplexity(
    loaded: LoadedModel,
    texts: Iterable[str],
    fwd_hooks: list | None = None,
) -> float:
    """Token-level perplexity on a stream of texts. Cheap proxy for capability loss.

    Pass `fwd_hooks` (the same `(name, hook)` list produced for ablation) to measure
    perplexity with those hooks active — i.e. perplexity *under ablation*.
    """
    nll, n = 0.0, 0
    for t in texts:
        tokens = loaded.model.to_tokens(t, prepend_bos=True).to(loaded.device)
        if fwd_hooks:
            logits = loaded.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        else:
            logits = loaded.model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target = tokens[0, 1:]
        nll += -log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
        n += target.numel()
    return float(torch.exp(torch.tensor(nll / max(n, 1))).item())
