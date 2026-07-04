#!/bin/bash
# Smoke test: one small model end-to-end on the A40 debug queue. Validates the conda
# env, GPU visibility, HF cache, and the SC_* control surface before the full sweep.
#
#   sbatch scripts/bender/smoke.sbatch
#   squeue --me ; tail -f slurm-<jobid>.out ; seff <jobid>
#
# SC_PREFLIGHT=1 => load + coherence + one refusal, no heavy sweeps (fast).
#SBATCH --job-name=sc-smoke
#SBATCH --partition=A40devel
#SBATCH --time=0:45:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out

# NOTE: we deliberately do NOT use --export=NONE. Our env is a self-contained conda env and
# torch is a generic manylinux wheel (portable across Bender's INTEL/AMD nodes), so the
# dual-stack MODULEPATH issue doesn't apply — and --export=NONE starves the job of
# PATH/module/conda (it produced a zero-byte log). We follow the wiki's conda job pattern:
# source ~/.bashrc, then conda activate.

set -euo pipefail
SC_ENV="${SC_ENV:-sc}"
REPO_ROOT="${SC_REPO:-$HOME/safety-circuits}"

source ~/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { module load Miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"; }
conda activate "$SC_ENV"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# HF_TOKEN needed only if the chosen model is gated / not yet cached.
export SC_MODELS="${SC_MODELS:-qwen2.5}"
export SC_OUT="${SC_OUT:-$HOME/sc-out}"
export SC_PREFLIGHT="${SC_PREFLIGHT:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[smoke] node=$(hostname) model=$SC_MODELS out=$SC_OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python kaggle/run_experiment.py
echo "[smoke] done — check $SC_OUT/results/$SC_MODELS/"
