"""safety-circuits: mechinterp of refusal in small instruct LMs."""

# Must run before any `import transformer_lens` (which eagerly imports transformers →
# a broken/absent torchaudio on Kaggle). Guarded; a no-op when transformers is absent.
from . import _env_compat as _env_compat  # noqa: F401

__version__ = "0.1.0"
