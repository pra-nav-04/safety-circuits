"""Steering baseline: the directional-ablation math (project the refusal direction out).

CPU/torch-only — no TransformerLens.
"""

import torch

from safety_circuits.steering import project_out, resolve_steering_layers


def test_resolve_steering_layers():
    assert resolve_steering_layers("all", 15, 26) == list(range(26))
    assert resolve_steering_layers("extract", 15, 26) == [15]
    assert resolve_steering_layers("10,11,12", 15, 26) == [10, 11, 12]
    assert resolve_steering_layers("10, 99, 12", 15, 26) == [10, 12]   # out-of-range dropped
    assert resolve_steering_layers("", 15, 26) == list(range(26))      # default -> all
    assert resolve_steering_layers("99", 15, 26) == [15]               # all invalid -> fallback


def test_project_out_removes_component():
    torch.manual_seed(0)
    resid = torch.randn(2, 5, 8)      # [batch, seq, d_model]
    direction = torch.randn(8)
    out = project_out(resid, direction, coeff=1.0)

    assert out.shape == resid.shape
    unit = direction / direction.norm()
    comp = torch.einsum("bsd,d->bs", out, unit)   # component along the direction
    assert torch.allclose(comp, torch.zeros_like(comp), atol=1e-5)


def test_coeff_zero_is_identity():
    resid = torch.randn(3, 4)
    out = project_out(resid, torch.randn(4), coeff=0.0)
    assert torch.allclose(out, resid)


def test_direction_normalisation_invariant():
    # scaling the direction must not change the result (it is unit-normalised internally)
    resid = torch.randn(2, 6)
    d = torch.randn(6)
    a = project_out(resid, d, coeff=1.0)
    b = project_out(resid, d * 7.3, coeff=1.0)
    assert torch.allclose(a, b, atol=1e-5)
