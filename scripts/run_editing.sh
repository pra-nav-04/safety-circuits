#!/usr/bin/env bash
# Portable §9 editing-suite runner for ANY Linux GPU box (Lightning AI, local, etc.).
# Run from inside the cloned repo. Robust to browser disconnects when run under tmux/nohup.
#
#   git clone https://github.com/pra-nav-04/safety-circuits.git && cd safety-circuits
#   export HF_TOKEN=...                      # gated gemma/llama (accept terms once on HF)
#   tmux new -s edit                         # so it survives a dropped connection
#   bash scripts/run_editing.sh gemma3-1b 2>&1 | tee run_gemma3-1b.log
#   # detach: Ctrl-b d ; reattach: tmux attach -t edit
#
# Results: $SC_OUT/editing/<model>/ (default ~/safety-circuits-editing — persists in a Studio).
# Every SC_* below is overridable from the environment. Pass a comma list to do several models.
set -euo pipefail

export SC_MODELS="${1:-${SC_MODELS:-gemma3-1b}}"
export SC_OUT="${SC_OUT:-$HOME/safety-circuits-editing}"
export SC_SKIP_EXISTING="${SC_SKIP_EXISTING:-1}"          # re-runs resume (skip finished models)

# core editing config (validated defaults)
export SC_EDIT_STEPS="${SC_EDIT_STEPS:-600}"
export SC_EDIT_RANK="${SC_EDIT_RANK:-16}"
export SC_EDIT_LR="${SC_EDIT_LR:-5e-4}"
export SC_EDIT_HEADCOUNTS="${SC_EDIT_HEADCOUNTS:-1,3,5,10}"
export SC_EDIT_METHODS="${SC_EDIT_METHODS:-steering,lora}"

# Tier 1/2 extensions (all on; override to 0 to skip)
export SC_DO_GENERALIZATION="${SC_DO_GENERALIZATION:-1}"
export SC_DO_DIRSHIFT="${SC_DO_DIRSHIFT:-1}"
export SC_DO_HARDENING="${SC_DO_HARDENING:-1}"
export SC_DO_MINIMAL_SWEEP="${SC_DO_MINIMAL_SWEEP:-1}"
export SC_EDIT_MINIMAL_RANKS="${SC_EDIT_MINIMAL_RANKS:-8,16}"
export SC_EDIT_MINIMAL_STEPS="${SC_EDIT_MINIMAL_STEPS:-300,600}"
export SC_DO_BENIGN_SUBSTANCE="${SC_DO_BENIGN_SUBSTANCE:-1}"
export SC_EDIT_MAX_TARGET_TOKENS="${SC_EDIT_MAX_TARGET_TOKENS:-128}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[run_editing] models=$SC_MODELS  SC_OUT=$SC_OUT  steps=$SC_EDIT_STEPS  repo=$REPO_ROOT"

# deps + drop ABI-mismatched torchaudio/torchvision (unused; would crash `import transformer_lens`)
pip install -q \
  "transformer-lens>=2.0.0" "transformers>=4.40" "datasets>=2.18" "accelerate>=0.27" \
  "einops>=0.7" seaborn tqdm pyyaml jaxtyping "peft>=0.10"
pip uninstall -y torchaudio torchvision || true

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
python kaggle/run_edit_experiment.py
