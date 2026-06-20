"""Head-masked LoRA: head→slice math (GQA-aware), mask restriction, clean merge.

CPU/torch-only — no TransformerLens, no model downloads (matches the existing suite).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from safety_circuits.lora import (
    GQAInfo,
    HeadMaskedLoRALinear,
    gqa_info,
    inject_head_lora,
    kv_group,
    merge_head_lora,
)


INFO = GQAInfo(n_heads=4, n_kv_heads=2, head_dim=3)  # group_size = 2; q/o dim = 12, kv dim = 6
DMODEL = 8


def _fill(adapter):
    # non-trivial, deterministic LoRA params so the delta is dense before masking
    torch.manual_seed(0)
    adapter.lora_A.data.normal_()
    adapter.lora_B.data.normal_()


def test_gqa_group_mapping():
    assert INFO.group_size == 2
    assert [kv_group(h, INFO) for h in range(4)] == [0, 0, 1, 1]


def test_q_role_masks_output_rows():
    base = nn.Linear(DMODEL, INFO.n_heads * INFO.head_dim)  # 8 -> 12
    ad = HeadMaskedLoRALinear(base, role="q", heads=[1], info=INFO, rank=2, alpha=2)
    _fill(ad)
    dw = ad.delta_weight()                  # [12, 8]
    rows = dw.abs().sum(dim=1) > 0
    expected = torch.zeros(12, dtype=torch.bool)
    expected[3:6] = True                    # head 1 owns rows [3:6]
    assert torch.equal(rows, expected)


def test_o_role_masks_input_cols():
    base = nn.Linear(INFO.n_heads * INFO.head_dim, DMODEL)  # 12 -> 8
    ad = HeadMaskedLoRALinear(base, role="o", heads=[1], info=INFO, rank=2, alpha=2)
    _fill(ad)
    dw = ad.delta_weight()                  # [8, 12]
    cols = dw.abs().sum(dim=0) > 0
    expected = torch.zeros(12, dtype=torch.bool)
    expected[3:6] = True                    # head 1 owns input cols [3:6]
    assert torch.equal(cols, expected)


def test_kv_role_uses_group_slice():
    base = nn.Linear(DMODEL, INFO.n_kv_heads * INFO.head_dim)  # 8 -> 6
    ad = HeadMaskedLoRALinear(base, role="k", heads=[2], info=INFO, rank=2, alpha=2)  # head 2 -> group 1
    _fill(ad)
    rows = ad.delta_weight().abs().sum(dim=1) > 0
    expected = torch.zeros(6, dtype=torch.bool)
    expected[3:6] = True                    # kv group 1 owns rows [3:6]
    assert torch.equal(rows, expected)


def test_initial_delta_is_zero():
    # lora_B is zero-initialised → edited model == base before any training
    base = nn.Linear(DMODEL, INFO.n_heads * INFO.head_dim)
    ad = HeadMaskedLoRALinear(base, role="q", heads=[0], info=INFO, rank=2, alpha=2)
    assert torch.count_nonzero(ad.delta_weight()) == 0


def test_merge_only_touches_target_slice():
    base = nn.Linear(DMODEL, INFO.n_heads * INFO.head_dim)
    before = base.weight.detach().clone()
    parent = SimpleNamespace()
    ad = HeadMaskedLoRALinear(base, role="q", heads=[1], info=INFO, rank=2, alpha=2)
    ad._parent, ad._attr = parent, "q_proj"
    _fill(ad)
    expected_delta = ad.delta_weight().clone()

    merge_head_lora([ad])

    # parent attribute restored to a plain Linear with merged weights
    assert isinstance(parent.q_proj, nn.Linear)
    merged = parent.q_proj.weight.detach()
    changed = (merged - before).abs() > 0
    target = torch.zeros_like(changed)
    target[3:6, :] = True
    assert torch.equal(changed, target)                         # only head-1 rows changed
    assert torch.allclose(merged - before, expected_delta, atol=1e-6)


def _fake_hf_model():
    cfg = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=3, hidden_size=8)

    def _layer():
        attn = nn.Module()
        attn.q_proj = nn.Linear(8, 12)
        attn.k_proj = nn.Linear(8, 6)
        attn.v_proj = nn.Linear(8, 6)
        attn.o_proj = nn.Linear(12, 8)
        layer = nn.Module()
        layer.self_attn = attn
        return layer

    inner = nn.Module()
    inner.layers = nn.ModuleList([_layer(), _layer()])
    model = nn.Module()
    model.model = inner
    model.config = cfg
    return model


def test_gqa_info_reads_config():
    info = gqa_info(_fake_hf_model())
    assert (info.n_heads, info.n_kv_heads, info.head_dim) == (4, 2, 3)


def test_inject_targets_only_safety_layers():
    model = _fake_hf_model()
    head = SimpleNamespace(layer=1, head=2)            # duck-typed HeadRef
    adapters = inject_head_lora(model, [head], rank=2, alpha=4)

    assert len(adapters) == 4                            # q/k/v/o on the one targeted layer
    edited = model.model.layers[1].self_attn
    assert all(isinstance(getattr(edited, p), HeadMaskedLoRALinear)
               for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
    # untouched layer stays plain Linear
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
    # only LoRA params are trainable
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
