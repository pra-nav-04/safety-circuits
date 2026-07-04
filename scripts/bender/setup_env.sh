#!/usr/bin/env bash
# One-time environment setup for the Bender GPU cluster (Uni Bonn HPC).
#
# Bender bans Anaconda/Miniconda and the `defaults` channel — conda-forge ONLY.
# The central `module load Miniforge3` already ships conda-forge as its sole channel;
# we pin it again below (belt-and-suspenders) and never pass -c defaults/-c anaconda.
#
# Run it TWICE, because the CUDA build of torch must be installed where a GPU is visible
# (installing on the login node silently gets the CPU build — a documented Bender trap):
#
#   # 1) on the LOGIN node — creates the conda env:
#   bash scripts/bender/setup_env.sh
#
#   # 2) inside an INTERACTIVE GPU job — installs CUDA torch + the package:
#   srun --partition=A40devel --gpus=1 --time=0:30:00 --pty /bin/bash
#   bash scripts/bender/setup_env.sh
#
# Idempotent: safe to re-run. Env name overridable via SC_ENV (default: sc).
set -euo pipefail

SC_ENV="${SC_ENV:-sc}"
PY_VER="${SC_PY_VER:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[setup_env] repo=$REPO_ROOT  env=$SC_ENV  python=$PY_VER"

# conda-forge-only compliance (no-op if already correct)
if [ ! -f "$HOME/.condarc" ] || ! grep -q "conda-forge" "$HOME/.condarc" 2>/dev/null; then
  echo "[setup_env] writing conda-forge-only ~/.condarc"
  cat > "$HOME/.condarc" <<'EOF'
channels:
  - conda-forge
channel_priority: strict
EOF
fi

module load Miniforge3
# Idempotent: populates ~/.bashrc so future shells / sbatch jobs get `conda activate`.
conda init bash >/dev/null 2>&1 || true
# Enable `conda activate` in THIS non-interactive shell right now.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the env once (conda-forge enforced explicitly so even python comes from it).
if ! conda env list | awk '{print $1}' | grep -qx "$SC_ENV"; then
  echo "[setup_env] creating conda env '$SC_ENV'"
  conda create -y -n "$SC_ENV" --override-channels -c conda-forge "python=$PY_VER"
else
  echo "[setup_env] conda env '$SC_ENV' already exists — skipping create"
fi

conda activate "$SC_ENV"

# The CUDA torch install + editable package must happen on a GPU node.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[setup_env] GPU visible — installing CUDA torch + package"
  # torch ONLY — we deliberately do NOT install torchvision/torchaudio: recent
  # transformers eagerly imports torchaudio on `import transformer_lens`, and an ABI
  # mismatch there crashes the import. Not installing them avoids it entirely (this is
  # the clean version of the `pip uninstall` hack in scripts/run_editing.sh).
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -e "$REPO_ROOT"
  echo "[setup_env] sanity check:"
  python - <<'PY'
import torch, transformer_lens
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformer_lens", transformer_lens.__version__)
PY
  echo "[setup_env] DONE (GPU stage). Env '$SC_ENV' is ready."
else
  cat <<EOF
[setup_env] No GPU here (login node). Conda env '$SC_ENV' created.
Next, install CUDA torch INSIDE a GPU job:
    srun --partition=A40devel --gpus=1 --time=0:30:00 --pty /bin/bash
    bash scripts/bender/setup_env.sh
EOF
fi
