"""Load small instruct LMs into TransformerLens.

TinyLlama and Qwen2.5 are natively supported by `HookedTransformer.from_pretrained`.
Phi-3 is *not* in the supported-models list (as of TL 2.x) so we go through
`HookedTransformer.from_pretrained_no_processing` after loading the HF weights
ourselves, then port the state dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from .config import ModelSpec, resolve_device, resolve_dtype


@dataclass
class LoadedModel:
    model: HookedTransformer
    tokenizer: object
    spec: ModelSpec
    device: torch.device
    dtype: torch.dtype

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def n_heads(self) -> int:
        return self.model.cfg.n_heads


def load_model(
    spec: ModelSpec,
    device: str = "auto",
    dtype: str | None = None,
) -> LoadedModel:
    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype or spec.dtype)

    if spec.is_tl_native:
        extra: dict = {}
        if not spec.prepend_bos:
            # Qwen2.x has no BOS token; TL's internal loader passes add_bos_token=True
            # regardless of default_prepend_bos, crashing the tokenizer init.
            # Pre-loading without that flag and handing it to TL avoids the issue.
            from transformers import AutoTokenizer as _AT
            extra["tokenizer"] = _AT.from_pretrained(spec.tl_name)
        model = HookedTransformer.from_pretrained(
            spec.tl_name,
            device=str(torch_device),
            dtype=torch_dtype,
            default_prepend_bos=spec.prepend_bos,
            **extra,
        )
    else:
        model = _load_via_hf_port(spec, torch_device, torch_dtype)

    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=model.tokenizer,
        spec=spec,
        device=torch_device,
        dtype=torch_dtype,  # dtype actually used (may differ from spec.dtype if caller overrides)
    )


def _load_via_hf_port(spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> HookedTransformer:
    """Fallback path for checkpoints (e.g., Phi-3, Llama-3.2) not natively in TL.

    We load the HF model + tokenizer, hand them to TL's `from_pretrained_no_processing`,
    which keeps the weights as-is and just wires the hooks. Folding / processing flags
    are intentionally off — we want the geometry untouched so patching stays interpretable.

    KNOWN LIMITATION (G3): this works for standard-attention checkpoints (Llama-3.2
    ports cleanly) but is unreliable for Phi-3-mini, whose *combined* `qkv_proj` /
    `gate_up_proj` projections are not split into TL's separate q/k/v/gate/up weights
    by this path — the model loads but produces garbage logits (0% baseline refusal).
    Detect this with `quick_coherence_check()` after loading; the real fix is a manual
    state-dict remap (split the fused projections) that needs GPU iteration to verify.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(spec.hf_name, trust_remote_code=True)
    if spec.needs_rope_patch and hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        # Phi-3's custom modeling code reads rope_scaling["type"], but transformers ≥4.43
        # renamed that key to "rope_type" and uses "default" to mean standard RoPE.
        # Old code only handles None (standard) or "longrope"; patch accordingly.
        rs = config.rope_scaling
        rope_type = rs.get("rope_type", rs.get("type", "default"))
        if rope_type == "default":
            config.rope_scaling = None  # old code uses None to mean standard RoPE
        elif "type" not in rs:
            rs["type"] = rope_type

    hf_model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, config=config, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, trust_remote_code=True)

    model = HookedTransformer.from_pretrained_no_processing(
        spec.hf_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=str(device),
        dtype=dtype,
        trust_remote_code=True,
    )
    # Free the transient HF copy now (TL has its own weights). Lowers the peak that
    # OOM-kills the kernel on larger ports (Phi-3, Llama-3.2) and helps the next model.
    import gc
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


_COHERENCE_PROBES = (
    "The capital of France is",
    "Water is made of hydrogen and",
    "2 + 2 =",
)


@torch.no_grad()
def quick_coherence_check(
    loaded: LoadedModel,
    probes: tuple[str, ...] = _COHERENCE_PROBES,
    max_new_tokens: int = 12,
) -> list[tuple[str, str]]:
    """Cheap single-model sanity check that a (ported) model produces sane logits.

    A misaligned state-dict port (e.g. Phi-3's combined QKV projection through
    TransformerLens) loads without error but yields garbage continuations. Greedy-
    decoding a few trivial factual probes makes that obvious *before* an expensive
    sweep — if the completions are incoherent, the port geometry is wrong and the
    model should be excluded (see FINDINGS.md / PROJECT_PLAN.md G3).

    Returns a list of (probe, continuation). No assertion — caller eyeballs or logs.
    """
    out = []
    for probe in probes:
        tokens = loaded.model.to_tokens(probe, prepend_bos=True).to(loaded.device)
        gen = loaded.model.generate(
            tokens, max_new_tokens=max_new_tokens, temperature=0.0, verbose=False
        )
        out.append((probe, loaded.model.to_string(gen[0, tokens.shape[1]:])))
    return out


def apply_chat_template(loaded: LoadedModel, user_msg: str) -> str:
    """Wrap a raw user message in the model's chat template."""
    tok = loaded.tokenizer
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if loaded.spec.no_think:
            # Qwen3 defaults to <think>...</think> blocks; disable for clean logit reads.
            kwargs["enable_thinking"] = False
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            **kwargs,
        )
    return user_msg
