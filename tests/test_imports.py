"""Cheapest smoke test — does the package import without optional heavy deps?"""

def test_package_imports():
    import safety_circuits  # noqa: F401
    from safety_circuits import config  # noqa: F401


def test_lightweight_modules_import():
    """These modules must not pull in torch/transformer_lens at import time."""
    from safety_circuits import data, refusal  # noqa: F401
