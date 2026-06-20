"""Environment-compatibility shim — imported first by `safety_circuits/__init__.py`.

Kaggle's image periodically downgrades torch while leaving torchaudio/torchvision
pinned to the old torch (ABI mismatch), and recent `transformers` EAGERLY imports
torchaudio (via `loss_rnnt.py`) the moment TransformerLens does
`from transformers import BertForPreTraining`. Depending on whether the broken
extension is present or uninstalled, that surfaces as either an `undefined symbol`
OSError or a `ModuleNotFoundError` — both before any of our code runs.

We use no audio/vision features, so the surgical fix is to tell transformers those
optional extensions are unavailable *before* the eager import chain loads. This runs
at package import, which precedes every `import transformer_lens` in our modules, so
all entry points (orchestrators, notebooks, tests) are covered. No-op off Kaggle /
when transformers isn't installed.
"""

from __future__ import annotations


def _disable_optional_extensions() -> None:
    try:
        from transformers.utils import import_utils as iu
    except Exception:
        return  # transformers not installed (e.g. local CPU test env) — nothing to do

    for name in ("torchaudio", "torchvision"):
        flag = f"_{name}_available"
        if hasattr(iu, flag):
            setattr(iu, flag, False)
        fn = f"is_{name}_available"
        if hasattr(iu, fn):
            setattr(iu, fn, (lambda: False))


_disable_optional_extensions()
