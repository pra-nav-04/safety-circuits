#!/usr/bin/env bash
# Regenerate the thin kernel bootstrap and (optionally) push it to Kaggle.
#
# The kernel now runs the multi-model orchestrator from the pulled repo, so model
# selection is NOT baked into the script — it's controlled at run time by the
# SC_MODELS env var (default: all models, cheapest-first). See kaggle/run_experiment.py.
#
# Canonical workflow (no code pasting after first setup):
#   1. git push origin main
#   2. Open the notebook and "Save & Run All":
#        https://www.kaggle.com/code/godspeed28/safety-circuits-nb/edit
#      (The bootstrap git-pulls the latest repo on every run.)
#   3. Download results:  python scripts/kaggle_api.py output
#
# To subset/resume, add a cell ABOVE the bootstrap, e.g.:
#   import os; os.environ["SC_MODELS"] = "llama3-3b,phi3"; os.environ["SC_SKIP_EXISTING"] = "1"

set -euo pipefail

echo "── Regenerating kernel.ipynb (thin bootstrap) ──"
python3 kaggle/make_kernel_notebook.py

echo ""
echo "── Next: push code, then Save & Run All in the browser ──"
echo "  git push origin main"
echo "  https://www.kaggle.com/code/godspeed28/safety-circuits-nb/edit"
echo ""
echo "── Pull results when done: ──"
echo "  python scripts/kaggle_api.py output     # → results/kaggle/"
