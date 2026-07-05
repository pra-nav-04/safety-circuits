#!/bin/bash
# FAST smoke for F3 directions D1/D2/D3 — one small model, forced-proxy judge, tiny grid.
# Validates: CI columns, steering arms, sweep CSV, raw jsonl path. NOT a real result.
#SBATCH --job-name=sc-smoke-tr
#SBATCH --partition=A100short
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
echo "[smoke] start on $(hostname) at $(date)"
set -euo pipefail
REPO_ROOT="$HOME/safety-circuits"
module load Miniforge3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sc; set -u
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SC_MODELS="llama3.2-1b"
export SC_OUT="$HOME/sc-out-smoke"
export SC_SKIP_EXISTING=0
export SC_HEADS_DIR="$REPO_ROOT/results/bender_neo"
export SC_EDIT_METHODS="none"
export SC_EDIT_STEPS=40
export SC_N_JB=6
export SC_DO_HARMFUL_TRANSFER=1
export SC_DO_TRANSFER_STEER=1
export SC_DO_TRANSFER_SWEEP=1
export SC_SAVE_TRANSFER_RAW=1
export SC_EDIT_MAX_TARGET_TOKENS=64
export SC_SWEEP_K=10
export SC_SWEEP_LEN=64
export SC_SWEEP_STEPS=40
export SC_JUDGE_MODEL="__smoke_force_proxy__"   # invalid -> judge falls back to proxy (fast)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python kaggle/run_edit_experiment.py
echo "[smoke] done at $(date)"
