"""Kaggle kernel entry point.

This script runs on the Kaggle GPU. It:
1. Installs the package from source
2. Downloads datasets via HuggingFace
3. Runs the full patching sweep
4. Saves results to /kaggle/working/ (auto-downloaded when done)

Push:    kaggle kernels push -p kaggle/
Status:  kaggle kernels status YOUR_USERNAME/safety-circuits
Output:  kaggle kernels output YOUR_USERNAME/safety-circuits -p results/kaggle/
"""

import subprocess, sys, os

# ── install deps ──────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformer-lens>=2.0.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.27",
    "einops>=0.7",
    "seaborn",
    "tqdm",
    "pyyaml",
    "jaxtyping",
], check=True)

# install safety_circuits from the kernel's working dir
sys.path.insert(0, "/kaggle/working/safety-circuits/src")

# ── copy source into working dir ───────────────────────────────────────────────
import shutil, pathlib

src = pathlib.Path("/kaggle/input/safety-circuits-src")
if src.exists():
    shutil.copytree(src, "/kaggle/working/safety-circuits", dirs_exist_ok=True)

# ── experiment config ─────────────────────────────────────────────────────────
MODEL      = os.environ.get("SC_MODEL", "tinyllama")   # tinyllama | phi3
N_PAIRS    = int(os.environ.get("SC_N_PAIRS", "64"))
TOP_K      = int(os.environ.get("SC_TOP_K", "10"))
OUT        = pathlib.Path("/kaggle/working")

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"Model: {MODEL}  |  Pairs: {N_PAIRS}  |  Top-K: {TOP_K}")

# ── run ───────────────────────────────────────────────────────────────────────
from safety_circuits.config import MODELS
from safety_circuits.models import load_model
from safety_circuits.data import load_advbench, load_hh_harmless, build_matched_pairs
from safety_circuits.patching import patch_each_head, patch_residual_stream
from safety_circuits.analysis import aggregate_pairs, head_heatmap, plot_heatmap
from safety_circuits.ablation import HeadRef, evaluate_ablation
import pandas as pd

spec   = MODELS[MODEL]
loaded = load_model(spec, device="auto", dtype="float32")
print(f"Loaded {spec.key}: {loaded.n_layers}L × {loaded.n_heads}H")

harm  = load_advbench(limit=N_PAIRS * 4)
safe  = load_hh_harmless(limit=N_PAIRS * 4)
pairs = build_matched_pairs(harm, safe, n_pairs=N_PAIRS)
print(f"Built {len(pairs)} matched pairs")

# coarse residual trace first (cheap)
r0 = patch_residual_stream(loaded, pairs[0][0].text, pairs[0][1].text)
pd.DataFrame([x.__dict__ for x in r0]).to_csv(OUT / f"{MODEL}_resid_trace.csv", index=False)
print("Residual trace saved.")

# full head sweep
per_pair = [patch_each_head(loaded, h.text, s.text) for h, s in pairs]
agg = aggregate_pairs(per_pair)
agg.to_csv(OUT / f"{MODEL}_patch_z.csv", index=False)
print("Head patching saved.")

# heatmap
grid = head_heatmap(agg, n_layers=loaded.n_layers, n_heads=loaded.n_heads)
plot_heatmap(grid, title=f"{spec.key}: per-head |Δ refusal-margin|",
             save_to=str(OUT / f"{MODEL}_heatmap.png"))
print("Heatmap saved.")

# ablation
top   = agg[agg["component"] == "z"].head(TOP_K)
heads = [HeadRef(int(r.layer), int(r.head)) for r in top.itertuples()]
eval_prompts = [h.text for h, _ in pairs[N_PAIRS//2:]]   # held-out half
report = evaluate_ablation(loaded, heads, eval_prompts, mode="zero")
pd.DataFrame([report.__dict__]).to_csv(OUT / f"{MODEL}_ablation.csv", index=False)
print(report)
print("Done. All results in /kaggle/working/")
