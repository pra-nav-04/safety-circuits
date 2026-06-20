"""Environment-compatibility shim — imported first by `safety_circuits/__init__.py`.

Kaggle's image periodically downgrades torch while leaving torchaudio/torchvision
pinned to the old torch (ABI mismatch), and recent `transformers` EAGERLY imports
torchaudio (via `loss_rnnt.py`) the moment TransformerLens does
`from transformers import BertForPreTraining`. Depending on whether the broken
extension is present, absent, or ABI-mismatched, that surfaces as an `undefined symbol`
OSError or a `ModuleNotFoundError` — both before any of our code runs.

Patching transformers' `is_*_available()` flags is unreliable (the value is captured
into several namespaces before we get a chance). The robust fix is to make the bare
`import torchaudio` / `import torchvision` simply succeed: if the real package can't be
imported cleanly, register a forgiving stub in `sys.modules`. We use no audio/vision
features, so a stub is harmless. Runs at package import, before every
`import transformer_lens` in our modules → all entry points self-heal. No-op when the
real package already imports fine.
"""

from __future__ import annotations

import sys
import types


def _stub_module(name: str) -> None:
    """If `name` can't be imported cleanly, install a forgiving stub so any later
    `import name` (e.g. transformers' eager `import torchaudio`) succeeds."""
    try:
        __import__(name)
        return  # real module imports fine — leave it alone
    except BaseException:  # ModuleNotFoundError, or OSError from an ABI-mismatched .so
        sys.modules.pop(name, None)  # drop any half-initialised entry

    from unittest.mock import MagicMock

    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as a package so `import name.sub` is tolerated
    # PEP 562: any attribute access returns a MagicMock, which nests and is callable
    # indefinitely (covers e.g. `torchaudio.transforms.RNNTLoss(...)`).
    mod.__getattr__ = lambda attr: MagicMock()
    sys.modules[name] = mod


for _name in ("torchaudio", "torchvision"):
    _stub_module(_name)
