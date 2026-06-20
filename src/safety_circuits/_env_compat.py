"""Environment-compatibility shim — imported first by `safety_circuits/__init__.py`.

Background: Kaggle's image periodically downgrades torch while leaving torchaudio /
torchvision pinned to the old torch (ABI mismatch), and recent `transformers` probes /
eagerly imports those extensions the moment TransformerLens does
`from transformers import ...`.

The correct fix lives in the bootstrap (`kaggle/*` notebook): it **uninstalls**
torchaudio/torchvision, after which they are genuinely absent and transformers'
`is_torchaudio_available()` / `is_torchvision_available()` resolve to False via
`importlib.util.find_spec(...) is None` — so transformers cleanly SKIPS the optional
imports. We use no audio/vision features, so nothing is lost.

A previous version of this shim installed `sys.modules` stubs for the missing packages.
That backfired: a stub makes `find_spec` return non-None, so transformers then believed
the extension WAS present and tried real submodule imports (e.g. `torchvision.io`) that a
stub can't satisfy. So we deliberately do NOT stub — absent is exactly what we want.

This module is kept (and imported early) as the documented home for any such env fix; it
is currently a no-op.
"""

from __future__ import annotations
