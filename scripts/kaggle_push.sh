#!/usr/bin/env bash
# Push the experiment to Kaggle and tail the status.
# Usage: bash scripts/kaggle_push.sh [tinyllama|phi3]

set -euo pipefail

MODEL="${1:-tinyllama}"
USERNAME="godspeed28"

echo "── Pushing kernel as $USERNAME/safety-circuits (model=$MODEL) ──"

# Patch the model env var into the kernel script on the fly (non-destructive)
sed "s/SC_MODEL\", \"tinyllama\"/SC_MODEL\", \"$MODEL\"/" \
    kaggle/run_experiment.py > /tmp/run_experiment_patched.py
cp /tmp/run_experiment_patched.py kaggle/run_experiment.py

kaggle kernels push -p kaggle/

echo ""
echo "── Kernel submitted. Monitor with: ──"
echo "  kaggle kernels status $USERNAME/safety-circuits"
echo ""
echo "── Pull results when done: ──"
echo "  kaggle kernels output $USERNAME/safety-circuits -p results/kaggle/"
