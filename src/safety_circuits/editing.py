"""§9 editing extension — train a head-restricted LoRA, merge it, port to TransformerLens.

The pipeline (`edit_and_load`):

    load HF model → inject head-masked LoRA on the safety heads → train on a
    refusal-suppression objective → merge the delta into just those head slices →
    port the merged HF model into HookedTransformer → return a `LoadedModel`.

Because the result is a plain `LoadedModel`, every existing metric (refusal margin,
WikiText-2 perplexity, HarmBench jailbreak, per-head patching) runs on it unchanged
(`edit_eval.py`). Training is the one deliberate departure from the read-only main
study; it lives entirely here.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from .config import EditConfig, ModelSpec, resolve_device, resolve_dtype
from .lora import HeadMaskedLoRALinear, inject_head_lora, merge_head_lora

if TYPE_CHECKING:  # these pull in TransformerLens; import lazily so `build_suppression_examples`
    from .ablation import HeadRef    # (and the unit tests) need no TL at import time.
    from .models import LoadedModel


# --------------------------------------------------------------- training data
def _as_id_list(x) -> list[int]:
    """Normalise a tokenizer return to a flat python list[int].

    Different transformers versions return a bare list, a tensor, or a
    dict/BatchEncoding (especially `apply_chat_template(tokenize=True)`), sometimes with
    a leading batch dim — coerce them all to `list[int]` so collation never sees a str
    (the `apply_chat_template` dict keys) or an extra dimension.
    """
    if hasattr(x, "input_ids"):          # BatchEncoding
        x = x.input_ids
    elif isinstance(x, dict):
        x = x["input_ids"]
    if hasattr(x, "tolist"):             # torch tensor / numpy array
        x = x.tolist()
    if x and isinstance(x[0], (list, tuple)):  # leading batch dim
        x = x[0]
    return list(x)


def build_suppression_examples(
    tokenizer,
    spec: ModelSpec,
    pairs: list[tuple[str, str]],
    max_target_tokens: int = 24,
) -> list[dict]:
    """Build SFT examples for refusal suppression: chat-templated harmful prompt →
    affirmative continuation (AdvBench's `target`, e.g. "Sure, here is how to ...").

    Loss is supervised on the *target* tokens only — the prompt is masked with -100,
    so the adapter learns to begin complying rather than to reproduce the instruction.
    Returns dicts of `{input_ids, labels}` (python lists).
    """
    examples: list[dict] = []
    chat_kwargs: dict = {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
    if spec.no_think:
        chat_kwargs["enable_thinking"] = False  # Qwen3: no <think> block

    eos = tokenizer.eos_token_id
    has_template = getattr(tokenizer, "chat_template", None) is not None

    for user_msg, target in pairs:
        if has_template:
            prompt_ids = _as_id_list(tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], **chat_kwargs
            ))
        else:
            prompt_ids = _as_id_list(tokenizer(user_msg, add_special_tokens=True))

        target_ids = _as_id_list(tokenizer(target, add_special_tokens=False))[:max_target_tokens]
        if eos is not None:
            target_ids = target_ids + [eos]

        input_ids = list(prompt_ids) + list(target_ids)
        labels = [-100] * len(prompt_ids) + list(target_ids)
        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


def _collate(batch: list[dict], pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
    )


# --------------------------------------------------------------- training loop
def train_head_lora(
    hf_model,
    tokenizer,
    adapters: list[HeadMaskedLoRALinear],
    examples: list[dict],
    cfg: EditConfig,
    device: torch.device,
    log=print,
) -> None:
    """AdamW over the LoRA params only (base frozen). T4-friendly: fp16 base, fp32 LoRA,
    GradScaler, gradient accumulation. Deterministic (seed from `cfg`). In-place."""
    import random

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    params = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    try:  # torch>=2.4 API; fall back for older torch
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    hf_model.train()
    step = 0
    opt.zero_grad()
    while step < cfg.steps:
        order = list(range(len(examples)))
        rng.shuffle(order)
        for bstart in range(0, len(order), cfg.batch):
            idx = order[bstart : bstart + cfg.batch]
            batch = [examples[i] for i in idx]
            input_ids, labels, attn = _collate(batch, pad_id)
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)

            out = hf_model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / cfg.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            if step % 20 == 0:
                log(f"  lora step {step}/{cfg.steps} loss {out.loss.item():.4f}")
            step += 1
            if step >= cfg.steps:
                break

    hf_model.eval()
    del opt, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------- port to TL
def port_hf_to_hooked(hf_model, tokenizer, spec: ModelSpec, device, dtype) -> "LoadedModel":
    """Wrap a (possibly merged) HF model as a TransformerLens `LoadedModel` via the
    same `from_pretrained_no_processing` path the HF-port models already use."""
    from transformer_lens import HookedTransformer

    from .models import LoadedModel

    model = HookedTransformer.from_pretrained_no_processing(
        spec.hf_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=str(device),
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return LoadedModel(model=model, tokenizer=model.tokenizer, spec=spec, device=device, dtype=dtype)


def load_via_port(spec: ModelSpec, device: str = "auto", dtype: str | None = None) -> "LoadedModel":
    """Baseline loader for the edit experiment: HF → port, no training. Used so the
    edit experiment's baseline goes through the *same* `no_processing` geometry as the
    edited models (the fairness note in the plan)."""
    from .models import load_hf_model

    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype or spec.dtype)
    hf_model, tokenizer = load_hf_model(spec, torch_dtype)
    loaded = port_hf_to_hooked(hf_model, tokenizer, spec, torch_device, torch_dtype)
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loaded


# --------------------------------------------------------------- full pipeline
def edit_and_load(
    spec: ModelSpec,
    heads: "list[HeadRef]",
    cfg: EditConfig,
    train_pairs: list[tuple[str, str]],
    device: str = "auto",
    dtype: str | None = None,
    log=print,
) -> "LoadedModel":
    """Full F1 pipeline: load → inject head-masked LoRA on `heads` → train → merge →
    port → return the edited `LoadedModel`."""
    from .models import load_hf_model

    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype or spec.dtype)

    hf_model, tokenizer = load_hf_model(spec, torch_dtype)
    hf_model.to(torch_device)

    adapters = inject_head_lora(hf_model, heads, rank=cfg.rank, alpha=cfg.alpha, targets=cfg.targets)
    log(f"  injected head-masked LoRA on {len(adapters)} projections "
        f"({len({h.layer for h in heads})} layers, {len(heads)} heads)")

    examples = build_suppression_examples(tokenizer, spec, train_pairs, cfg.max_target_tokens)
    train_head_lora(hf_model, tokenizer, adapters, examples, cfg, torch_device, log=log)

    merge_head_lora(adapters)
    log("  merged LoRA delta into target head slices")

    loaded = port_hf_to_hooked(hf_model, tokenizer, spec, torch_device, torch_dtype)
    del hf_model, adapters
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loaded
