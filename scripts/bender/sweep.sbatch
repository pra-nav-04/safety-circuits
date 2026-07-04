#!/bin/bash
# Full 9-model sweep as a SLURM job array — one model per GPU, all in parallel.
# This is the main win over Kaggle (which forced the 9 models to run sequentially).
#
#   sbatch scripts/bender/sweep.sbatch                 # main study (run_experiment.py)
#   SC_ORCH=run_edit_experiment.py sbatch scripts/bender/sweep.sbatch   # §9 editing suite
#   squeue --me ; seff <jobid>_<taskid>
#
# Re-submitting resumes: SC_SKIP_EXISTING=1 skips models already finished in SC_OUT.
#SBATCH --job-name=sc-sweep
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%A_%a.out

# No --export=NONE (see smoke.sbatch). Activate conda explicitly — do NOT `source ~/.bashrc`
# (unbound vars are fatal under `set -u`).
echo "[sweep] task=${SLURM_ARRAY_TASK_ID:-?} start on $(hostname) at $(date)"
set -euo pipefail
SC_ENV="${SC_ENV:-sc}"
REPO_ROOT="${SC_REPO:-$HOME/safety-circuits}"

# Cheapest-first order (matches _COST in run_experiment.py). One entry per array index.
MODELS=(gemma3-1b llama3.2-1b qwen2.5 qwen2-1.5b qwen1.5-1.8b qwen3 gemma1-2b gemma2-2b llama3-3b)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

module load Miniforge3
set +u                                    # conda's shell integration uses unbound vars
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SC_ENV"
set -u

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# export HF_TOKEN=...  # set in your submitting shell if gated models aren't cached yet
export SC_MODELS="$MODEL"
export SC_OUT="${SC_OUT:-$HOME/sc-out}"
export SC_SKIP_EXISTING="${SC_SKIP_EXISTING:-1}"
ORCH="${SC_ORCH:-run_experiment.py}"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[sweep] task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) orch=$ORCH out=$SC_OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "kaggle/$ORCH"
echo "[sweep] task=$SLURM_ARRAY_TASK_ID model=$MODEL done"
