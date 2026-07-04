#!/usr/bin/env bash
# Pre-download the 9 models + datasets on the Bender LOGIN NODE (which has internet),
# so compute jobs read from the shared HF cache instead of hitting the network.
#
#   export HF_TOKEN=hf_...          # gated gemma/llama + walledai/HarmBench
#   bash scripts/bender/prestage.sh
#
# Requires the conda env from setup_env.sh (SC_ENV, default 'sc'). Downloading model
# weights needs no GPU — snapshot_download just fetches files.
#
# NOTE: load_advbench() and the XSTest CSV fallback in data.py fetch from raw GitHub
# URLs at RUNTIME and are never cached here — so compute nodes still need outbound
# internet for those (small) fetches, or a future local-cache fallback in data.py.
set -euo pipefail

SC_ENV="${SC_ENV:-sc}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[prestage] ERROR: HF_TOKEN is not set. Gated models (gemma/llama) + HarmBench need it."
  echo "           Accept terms once at hf.co for each google/gemma-*, meta-llama/Llama-3.2-*,"
  echo "           and hf.co/datasets/walledai/HarmBench, then: export HF_TOKEN=hf_..."
  exit 1
fi

module load Miniforge3
set +u                                    # conda's shell integration uses unbound vars
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SC_ENV"
set -u

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
echo "[prestage] HF_HOME=$HF_HOME  repo=$REPO_ROOT"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from safety_circuits.config import MODELS

# The default 9-model set (mirrors run_experiment.py's _DEFAULT_EXCLUDE).
EXCLUDE = {"tinyllama", "phi3", "falcon3-1b", "olmo2-1b", "gemma4-e2b"}
models = [s.hf_name for k, s in MODELS.items() if k not in EXCLUDE]

print(f"[prestage] downloading {len(models)} models -> {os.environ['HF_HOME']}")
for name in models:
    print(f"  - {name}")
    snapshot_download(repo_id=name, token=os.environ["HF_TOKEN"])

print("[prestage] warming datasets (cached to HF_HOME)")
from safety_circuits.data import (
    load_hh_harmless, load_wikitext2, load_harmbench, load_xstest, load_advbench, load_rtp,
)
load_hh_harmless(limit=8)      # Anthropic/hh-rlhf (streaming)
load_wikitext2(limit=8)        # Salesforce/wikitext
load_harmbench(limit=8)        # walledai/HarmBench (gated)
load_xstest(limit=8)           # walledai/XSTest
load_rtp(limit=8)              # allenai/real-toxicity-prompts
load_advbench(limit=8)         # GitHub CSV — verifies reachability (not HF-cached)
print("[prestage] DONE. Check disk with `quota -s`; run `conda clean --all` if tight.")
PY
