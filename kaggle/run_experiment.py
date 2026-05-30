"""Kaggle kernel entry point.

Runs on Kaggle GPU (T4). Workflow:
1. Install deps
2. Clone / pull latest src from GitHub
3. Run activation patching sweep on the model set by SC_MODEL env var
4. Ablate top-K heads
5. Save results to /kaggle/working/ (auto-persisted as kernel output)

Update workflow: push src changes to GitHub, then click "Save & Run All" in browser.
Status / output: python3 scripts/kaggle_api.py status | output
"""

import os
import pathlib
import subprocess
import sys

# Load HF token from Kaggle secrets if available (faster downloads, higher rate limits).
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets.")
except Exception:
    pass  # No secret configured — unauthenticated HF access still works for public models.

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
from safety_circuits.data import load_advbench, load_hh_harmless, build_matched_pairs, load_wikitext2

MODEL    = os.environ.get("SC_MODEL", "qwen")
N_PAIRS  = int(os.environ.get("SC_N_PAIRS", "32"))
TOP_K    = int(os.environ.get("SC_TOP_K", "10"))
SEED     = int(os.environ.get("SC_SEED", "0"))
OUT      = pathlib.Path("/kaggle/working")

# Optional analyses (env-flag gated). Cheap ones default on; expensive ones off.
DO_COHERENCE = os.environ.get("SC_COHERENCE", "1") == "1"          # G3 port sanity check
K_SWEEP      = os.environ.get("SC_K_SWEEP", "5,10,15,20,30,40")    # G2; "" to skip
DO_MEAN      = os.environ.get("SC_MEAN_ABLATION", "0") == "1"      # G6
DO_LASTTOK   = os.environ.get("SC_LASTTOK", "0") == "1"           # G7 (doubles the sweep)
DO_PATTERN   = os.environ.get("SC_PATTERN", "0") == "1"           # G8 (doubles the sweep)
DO_JAILBREAK = os.environ.get("SC_JAILBREAK", "0") == "1"        # G9 (needs HarmBench)

# Determinism (G10): fixed seed, greedy decode everywhere.
import random
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Seed: {SEED}")

spec   = MODELS[MODEL]
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded = load_model(spec, device=device)  # dtype comes from ModelSpec.dtype
print(f"Loaded {spec.key}: {loaded.n_layers}L × {loaded.n_heads}H  on {device}")

if DO_COHERENCE:
    from safety_circuits.models import quick_coherence_check
    print("--- Port coherence check (garbage completions ⇒ broken port, e.g. Phi-3) ---")
    for probe, cont in quick_coherence_check(loaded):
        print(f"  {probe!r} -> {cont!r}")

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

print("=== Step 7: Ablation (+ perplexity capability control) ===")
from safety_circuits.ablation import HeadRef, evaluate_ablation

top   = agg[agg["component"] == "z"].head(TOP_K)
heads = [HeadRef(int(r.layer), int(r.head)) for r in top.itertuples()]
print(f"Top-{TOP_K} candidate heads: {[(h.layer, h.head) for h in heads]}")

# WikiText-2 slice for the H3 capability-preservation control: refusal should
# collapse under ablation while general-language perplexity barely moves.
ppl_texts = load_wikitext2(limit=int(os.environ.get("SC_PPL_TEXTS", "64")))
print(f"Loaded {len(ppl_texts)} WikiText-2 snippets for perplexity control")

eval_prompts = [h.text for h, _ in pairs[N_PAIRS // 2:]]
report = evaluate_ablation(loaded, heads, eval_prompts, mode="zero", perplexity_texts=ppl_texts)

row = dict(report.__dict__)
row["perplexity_pct_change"] = report.perplexity_pct_change
pd.DataFrame([row]).to_csv(OUT / f"{MODEL}_ablation.csv", index=False)
print(f"Refusal rate — clean: {report.refusal_rate_clean:.2%}  ablated: {report.refusal_rate_ablated:.2%}")
if report.perplexity_clean is not None:
    print(f"Perplexity   — clean: {report.perplexity_clean:.3f}  ablated: {report.perplexity_ablated:.3f}  "
          f"(Δ {report.perplexity_pct_change:+.2f}%)")

import json
top_heads = [{"layer": h.layer, "head": h.head} for h in heads]
(OUT / f"{MODEL}_safety_heads.json").write_text(json.dumps(top_heads, indent=2))

# Full priority-ranked head list (strongest first) — reused by the K-sweep & jailbreak.
ranked_z   = agg[agg["component"] == "z"]
ranked_all = [HeadRef(int(r.layer), int(r.head)) for r in ranked_z.itertuples()]

# ── Step 8: Ablation K-sweep (G2) — how many top heads to collapse refusal? ──────
if K_SWEEP.strip():
    print("=== Step 8: Ablation K-sweep ===")
    from safety_circuits.ablation import ablation_k_sweep
    from safety_circuits.analysis import plot_k_sweep

    ks = [k for k in (int(x) for x in K_SWEEP.split(",")) if k <= len(ranked_all)]
    clean_rate, points = ablation_k_sweep(loaded, ranked_all, eval_prompts, ks,
                                          mode="zero", perplexity_texts=ppl_texts)
    ks_rows = [{"k": p.k, "refusal_rate_clean": clean_rate,
                "refusal_rate_ablated": p.refusal_rate_ablated,
                "perplexity_ablated": p.perplexity_ablated} for p in points]
    pd.DataFrame(ks_rows).to_csv(OUT / f"{MODEL}_ksweep.csv", index=False)
    plot_k_sweep(ks, [p.refusal_rate_ablated for p in points], clean_rate,
                 title=f"{spec.key}: refusal vs #heads ablated",
                 save_to=str(OUT / f"{MODEL}_ksweep.png"))
    print(f"K-sweep — clean {clean_rate:.2%}; " +
          "  ".join(f"k={p.k}:{p.refusal_rate_ablated:.0%}" for p in points))

# ── Step 9: Mean-ablation comparison (G6) ───────────────────────────────────────
if DO_MEAN:
    print("=== Step 9: Mean-ablation comparison ===")
    from safety_circuits.ablation import compute_mean_z_cache
    benign = [s.text for _, s in pairs]
    mean_cache = compute_mean_z_cache(loaded, benign)
    report_mean = evaluate_ablation(loaded, heads, eval_prompts, mode="mean",
                                    mean_cache=mean_cache, perplexity_texts=ppl_texts)
    mrow = dict(report_mean.__dict__)
    mrow["perplexity_pct_change"] = report_mean.perplexity_pct_change
    pd.DataFrame([mrow]).to_csv(OUT / f"{MODEL}_ablation_mean.csv", index=False)
    print(f"Mean-ablation — clean: {report_mean.refusal_rate_clean:.2%}  "
          f"ablated: {report_mean.refusal_rate_ablated:.2%}  "
          f"(zero was {report.refusal_rate_ablated:.2%})")

# ── Step 10: Last-token patching (G7) ───────────────────────────────────────────
if DO_LASTTOK:
    print("=== Step 10: Last-token head patching ===")
    per_pair_lt = []
    for i, (h, s) in enumerate(pairs):
        print(f"  [last-tok] pair {i+1}/{len(pairs)}", flush=True)
        per_pair_lt.append(patch_each_head(loaded, h.text, s.text, position=-1))
    agg_lt = aggregate_pairs(per_pair_lt)
    agg_lt.to_csv(OUT / f"{MODEL}_patch_z_lasttok.csv", index=False)
    grid_lt = head_heatmap(agg_lt, n_layers=loaded.n_layers, n_heads=loaded.n_heads)
    plot_heatmap(grid_lt, title=f"{spec.key}: per-head |Δ| (last-token patch)",
                 save_to=str(OUT / f"{MODEL}_heatmap_lasttok.png"))
    print(agg_lt[agg_lt.component == "z"].head(10).to_string())

# ── Step 11: Attention-pattern patching (G8) ────────────────────────────────────
if DO_PATTERN:
    print("=== Step 11: Attention-pattern patching ===")
    from safety_circuits.patching import patch_each_head_pattern
    per_pair_pat = []
    for i, (h, s) in enumerate(pairs):
        print(f"  [pattern] pair {i+1}/{len(pairs)}", flush=True)
        per_pair_pat.append(patch_each_head_pattern(loaded, h.text, s.text))
    agg_pat = aggregate_pairs(per_pair_pat)
    agg_pat.to_csv(OUT / f"{MODEL}_patch_pattern.csv", index=False)
    grid_pat = head_heatmap(agg_pat, n_layers=loaded.n_layers, n_heads=loaded.n_heads,
                            component="pattern")
    plot_heatmap(grid_pat, title=f"{spec.key}: per-head |Δ| (attention pattern)",
                 save_to=str(OUT / f"{MODEL}_heatmap_pattern.png"))
    print(agg_pat[agg_pat.component == "pattern"].head(10).to_string())

# ── Step 12: HarmBench jailbreak stress test (G9) ───────────────────────────────
if DO_JAILBREAK:
    print("=== Step 12: HarmBench jailbreak stress test ===")
    from safety_circuits.data import load_harmbench
    from safety_circuits.jailbreak import jailbreak_stress_test

    jb = load_harmbench(limit=N_PAIRS)
    jb_prompts = [p.text for p in jb]
    jb_report = jailbreak_stress_test(loaded, heads, jb_prompts, eval_prompts, mode="zero")
    pd.DataFrame([jb_report.__dict__]).to_csv(OUT / f"{MODEL}_jailbreak.csv", index=False)
    print(f"Clean refusal — plain: {jb_report.refusal_rate_clean_plain:.2%}  "
          f"jailbreak: {jb_report.refusal_rate_clean_jailbreak:.2%}")
    print(f"Ablated refusal on jailbreaks: {jb_report.refusal_rate_ablated_jailbreak:.2%}")
    print(f"Mean margin — plain: {jb_report.mean_margin_plain:.3f}  "
          f"jailbreak: {jb_report.mean_margin_jailbreak:.3f}")

print("=== DONE. All results in /kaggle/working/ ===")
