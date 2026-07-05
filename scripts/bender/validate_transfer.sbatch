#!/bin/bash
# Real-judge validation: one model, HarmBench-13B judge, D1(CIs)+D3(steering arms).
# Confirms the judge loads + produces real labels before the full 9-model array.
#SBATCH --job-name=sc-val-tr
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
echo "[val] start on $(hostname) at $(date)"
set -euo pipefail
REPO_ROOT="$HOME/safety-circuits"
module load Miniforge3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sc; set -u
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SC_MODELS="llama3.2-1b"
export SC_OUT="$HOME/sc-out-validate"
export SC_SKIP_EXISTING=0
export SC_HEADS_DIR="$REPO_ROOT/results/bender_neo"
export SC_EDIT_METHODS="none"
export SC_EDIT_STEPS=300
export SC_N_JB=30
export SC_DO_HARMFUL_TRANSFER=1
export SC_DO_TRANSFER_STEER=1
export SC_SAVE_TRANSFER_RAW=1
export SC_EDIT_MAX_TARGET_TOKENS=256
# real judge (SC_JUDGE_MODEL unset -> cais/HarmBench-Llama-2-13b-cls, cached)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python kaggle/run_edit_experiment.py
echo "[val] done at $(date)"
