#!/bin/bash
# §9 editing suite (head-restricted LoRA + steering baseline) as a SLURM job array —
# one model per GPU, in parallel. Mirrors sweep.sbatch but runs run_edit_experiment.py
# and reads the Bender-localized safety heads from results/bender_neo.
#
#   export HF_TOKEN=...            # gated gemma/llama
#   sbatch scripts/bender/edit.sbatch
#   squeue --me ; seff <jobid>_<taskid>
#
# Done marker per model: $SC_OUT/editing/<model>/_EDIT_DONE.json. Re-submitting resumes
# (SC_SKIP_EXISTING=1 skips models already finished).
#SBATCH --job-name=sc-edit
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%A_%a.out

# No --export=NONE (see smoke.sbatch). Activate conda explicitly — do NOT `source ~/.bashrc`
# (unbound vars are fatal under `set -u`).
echo "[edit] task=${SLURM_ARRAY_TASK_ID:-?} start on $(hostname) at $(date)"
set -euo pipefail
SC_ENV="${SC_ENV:-sc}"
REPO_ROOT="${SC_REPO:-$HOME/safety-circuits}"

# Same index->model order as sweep.sbatch.
MODELS=(gemma3-1b llama3.2-1b qwen2.5 qwen2-1.5b qwen1.5-1.8b qwen3 gemma1-2b gemma2-2b llama3.2-3b)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

module load Miniforge3
set +u                                    # conda's shell integration uses unbound vars
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SC_ENV"
set -u

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# export HF_TOKEN=...  # set in your submitting shell (gated gemma/llama)
export SC_MODELS="$MODEL"
export SC_OUT="${SC_OUT:-$HOME/sc-out-edit}"
export SC_SKIP_EXISTING="${SC_SKIP_EXISTING:-1}"

# Use the Bender-localized safety heads (default in the orchestrator is results/kaggle_neo).
export SC_HEADS_DIR="${SC_HEADS_DIR:-$REPO_ROOT/results/bender_neo}"

# Validated core editing config (matches scripts/run_editing.sh; Tier 1/2 left default-OFF).
export SC_EDIT_METHODS="${SC_EDIT_METHODS:-steering,lora}"
export SC_EDIT_STEPS="${SC_EDIT_STEPS:-600}"
export SC_EDIT_RANK="${SC_EDIT_RANK:-16}"
export SC_EDIT_LR="${SC_EDIT_LR:-5e-4}"
export SC_EDIT_HEADCOUNTS="${SC_EDIT_HEADCOUNTS:-1,3,5,10}"

# Tier-1 add-ons (default OFF; enable to quantify the opener-stop / Finding F):
#   SC_DO_GENERALIZATION=1   harmful-side: deep_eval (opener-vs-content) -> *_edit_generalization.csv
#   SC_DO_BENIGN_SUBSTANCE=1 benign-side (weapon-free): short-vs-long target -> *_edit_benign_substance_agg.csv
#   SC_EDIT_MAX_TARGET_TOKENS  benign long-arm target + benign gen length (default 128)
export SC_DO_GENERALIZATION="${SC_DO_GENERALIZATION:-0}"
export SC_DO_BENIGN_SUBSTANCE="${SC_DO_BENIGN_SUBSTANCE:-0}"
export SC_EDIT_MAX_TARGET_TOKENS="${SC_EDIT_MAX_TARGET_TOKENS:-128}"
# F2 harmful-compliance transfer test (opt-in; aggregate ASR only, weapon-free, severe categories excluded):
export SC_DO_HARMFUL_TRANSFER="${SC_DO_HARMFUL_TRANSFER:-0}"
export SC_SAVE_TRANSFER_RAW="${SC_SAVE_TRANSFER_RAW:-0}"   # dump raw completions to a GITIGNORED jsonl (local examples)
# SC_HARM_EXCLUDE (comma list of category substrings) defaults to the severe denylist in data.py
# SC_JUDGE_MODEL defaults to cais/HarmBench-Llama-2-13b-cls (ASR judge; loads on the A100)
# F3-D3: combine benign-LoRA + steering (adds steering_alone + benign+steering arms; aggregate CSV):
export SC_DO_TRANSFER_STEER="${SC_DO_TRANSFER_STEER:-0}"
# F3-D2: benign-only hyperparameter sweep (writes *_edit_transfer_sweep_agg.csv; aggregate only):
export SC_DO_TRANSFER_SWEEP="${SC_DO_TRANSFER_SWEEP:-0}"
export SC_SWEEP_K="${SC_SWEEP_K:-10,20}"
export SC_SWEEP_RANK="${SC_SWEEP_RANK:-16}"
export SC_SWEEP_STEPS="${SC_SWEEP_STEPS:-600}"
export SC_SWEEP_LEN="${SC_SWEEP_LEN:-256,512}"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[edit] task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) heads=$SC_HEADS_DIR out=$SC_OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python kaggle/run_edit_experiment.py
echo "[edit] task=$SLURM_ARRAY_TASK_ID model=$MODEL done"
