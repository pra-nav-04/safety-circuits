"""Kaggle kernel entry point.

Runs on Kaggle GPU (T4). Workflow:
1. Install deps
2. Clone public repo from GitHub
3. Run activation patching sweep on TinyLlama
4. Ablate top-K heads
5. Save results to /kaggle/working/ (auto-persisted as kernel output)

Push:   python3 scripts/kaggle_api.py push
Status: python3 scripts/kaggle_api.py status
Output: python3 scripts/kaggle_api.py output
"""

import os
import pathlib
import subprocess
import sys

print("=== Step 1: Install dependencies ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
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
print("Dependencies installed.")

print("=== Step 2: Clone source ===")
repo_dir = pathlib.Path("/kaggle/working/safety-circuits")
if not repo_dir.exists():
    subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/pra-nav-04/safety-circuits.git",
        str(repo_dir)
    ], check=True)
    print("Cloned.")
else:
    subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    print("Pulled latest.")

sys.path.insert(0, str(repo_dir / "src"))

print("=== Step 3: Load model & data ===")
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from safety_circuits.config import MODELS
from safety_circuits.models import load_model
from safety_circuits.data import load_advbench, load_hh_harmless, build_matched_pairs

MODEL    = os.environ.get("SC_MODEL", "tinyllama")
N_PAIRS  = int(os.environ.get("SC_N_PAIRS", "32"))
TOP_K    = int(os.environ.get("SC_TOP_K", "10"))
OUT      = pathlib.Path("/kaggle/working")

spec   = MODELS[MODEL]
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded = load_model(spec, device=device, dtype="float32")
print(f"Loaded {spec.key}: {loaded.n_layers}L × {loaded.n_heads}H  on {device}")

harm  = load_advbench(limit=N_PAIRS * 4)
safe  = load_hh_harmless(limit=N_PAIRS * 4)
pairs = build_matched_pairs(harm, safe, n_pairs=N_PAIRS)
print(f"Built {len(pairs)} matched pairs")

print("=== Step 4: Coarse residual trace ===")
from safety_circuits.patching import patch_residual_stream, patch_each_head
from safety_circuits.analysis import aggregate_pairs, head_heatmap, plot_heatmap
import pandas as pd

r0 = patch_residual_stream(loaded, pairs[0][0].text, pairs[0][1].text)
resid_df = pd.DataFrame([x.__dict__ for x in r0])
resid_df.to_csv(OUT / f"{MODEL}_resid_trace.csv", index=False)
print("Residual trace saved.")
print(resid_df.sort_values("delta_margin").tail(5))

print("=== Step 5: Per-head patching sweep ===")
per_pair = []
for i, (h, s) in enumerate(pairs):
    print(f"  pair {i+1}/{len(pairs)}", flush=True)
    per_pair.append(patch_each_head(loaded, h.text, s.text))

agg = aggregate_pairs(per_pair)
agg.to_csv(OUT / f"{MODEL}_patch_z.csv", index=False)
print("Head patching saved.")
print(agg[agg.component == "z"].head(10).to_string())

print("=== Step 6: Heatmap ===")
grid = head_heatmap(agg, n_layers=loaded.n_layers, n_heads=loaded.n_heads)
plot_heatmap(grid, title=f"{spec.key}: per-head |Δ refusal-margin|",
             save_to=str(OUT / f"{MODEL}_heatmap.png"))
print("Heatmap saved.")

print("=== Step 7: Ablation ===")
from safety_circuits.ablation import HeadRef, evaluate_ablation

top   = agg[agg["component"] == "z"].head(TOP_K)
heads = [HeadRef(int(r.layer), int(r.head)) for r in top.itertuples()]
print(f"Top-{TOP_K} candidate heads: {[(h.layer, h.head) for h in heads]}")

eval_prompts = [h.text for h, _ in pairs[N_PAIRS // 2:]]
report = evaluate_ablation(loaded, heads, eval_prompts, mode="zero")
pd.DataFrame([report.__dict__]).to_csv(OUT / f"{MODEL}_ablation.csv", index=False)
print(f"Refusal rate — clean: {report.refusal_rate_clean:.2%}  ablated: {report.refusal_rate_ablated:.2%}")

import json
top_heads = [{"layer": h.layer, "head": h.head} for h in heads]
(OUT / f"{MODEL}_safety_heads.json").write_text(json.dumps(top_heads, indent=2))

print("=== DONE. All results in /kaggle/working/ ===")
