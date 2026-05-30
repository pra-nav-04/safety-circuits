"""Multi-model experiment orchestrator (run by the thin Kaggle bootstrap).

`kaggle/kernel.ipynb` (a ~20-line bootstrap) installs deps, clones/pulls this repo,
puts `src` on the path, then `runpy`s THIS file. So every logic change — package
*and* orchestration — ships via `git push` + "Save & Run All"; the browser notebook
never needs editing again.

What it does: there is one T4 GPU per session, so models can't run in parallel — we
loop them **sequentially, cheapest first**, run the full analysis suite per model,
**skip any model that fails (logging the traceback) and continue**, flush results to
`<out>/results/<model>/` as we go, and zip everything for download.

Per-model suite: coherence check · residual trace · per-head z-patch sweep + heatmap ·
zero-ablation + perplexity · K-sweep · mean-ablation · last-token sweep ·
attention-pattern sweep · HarmBench jailbreak. Each optional analysis is isolated, so
one failing add-on never sinks a model.

Control via env vars — see SC_* below (all add-ons default ON; SC_MODELS subsets/resumes).
"""

import gc
import json
import os
import pathlib
import time
import traceback
import zipfile

import pandas as pd
import torch

from safety_circuits.config import MODELS
from safety_circuits.models import load_model, quick_coherence_check
from safety_circuits.data import (
    load_advbench, load_hh_harmless, build_matched_pairs, load_wikitext2, load_harmbench,
)
from safety_circuits.patching import (
    patch_residual_stream, patch_each_head, patch_each_head_pattern,
)
from safety_circuits.analysis import aggregate_pairs, head_heatmap, plot_heatmap, plot_k_sweep
from safety_circuits.ablation import (
    HeadRef, evaluate_ablation, ablation_k_sweep, compute_mean_z_cache,
)
from safety_circuits.jailbreak import jailbreak_stress_test


# ───────────────────────────── config (env) ─────────────────────────────
N_PAIRS     = int(os.environ.get("SC_N_PAIRS", "32"))
HEAVY_PAIRS = int(os.environ.get("SC_HEAVY_PAIRS", "8"))   # pairs for the doubler sweeps
TOP_K       = int(os.environ.get("SC_TOP_K", "10"))
K_SWEEP     = os.environ.get("SC_K_SWEEP", "5,10,15,20,30,40")
PPL_TEXTS   = int(os.environ.get("SC_PPL_TEXTS", "64"))
SEED        = int(os.environ.get("SC_SEED", "0"))
DO_COHER    = os.environ.get("SC_COHERENCE", "1") == "1"
DO_MEAN     = os.environ.get("SC_MEAN_ABLATION", "1") == "1"
DO_LASTTOK  = os.environ.get("SC_LASTTOK", "1") == "1"
DO_PATTERN  = os.environ.get("SC_PATTERN", "1") == "1"
DO_JAILBRK  = os.environ.get("SC_JAILBREAK", "1") == "1"
SKIP_EXIST  = os.environ.get("SC_SKIP_EXISTING", "0") == "1"

WORK     = pathlib.Path(os.environ.get("SC_OUT", "/kaggle/working"))
OUT_ROOT = WORK / "results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
START    = time.time()

# cheapest-first ordering (≈ n_layers × n_heads); unknown keys sort last
_COST = {"gemma3-1b": 1, "qwen": 2, "qwen3": 3, "falcon3-1b": 4,
         "olmo2-1b": 5, "tinyllama": 6, "llama3-3b": 7, "phi3": 8}

# Excluded from the DEFAULT loop (still runnable explicitly via SC_MODELS):
#   tinyllama — not in the pinned TransformerLens OFFICIAL_MODEL_NAMES → load always
#               raises ValueError (see Kaggle_Logs/Qwen2.5-1.5B-Instruct/download.txt).
#   phi3      — HF-port load tends to OOM and *kill the kernel* (uncatchable; would
#               abort the whole loop), see Kaggle_Logs/Phi3/download.txt.
_DEFAULT_EXCLUDE = {"tinyllama", "phi3"}


def _models_to_run() -> list[str]:
    env = os.environ.get("SC_MODELS", "").strip()
    if env:  # explicit request — honour it exactly (incl. tinyllama/phi3 opt-in)
        keys = [k.strip() for k in env.split(",") if k.strip()]
    else:
        keys = [k for k in MODELS if k not in _DEFAULT_EXCLUDE]
    keys = [k for k in keys if k in MODELS]
    return sorted(keys, key=lambda k: _COST.get(k, 99))


def _set_seed() -> None:
    import random
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ───────────────────────────── per-model pipeline ─────────────────────────────
def run_one(model_key: str, log) -> dict:
    """Full analysis suite for one model. Returns a summary dict; raises on *core* failure."""
    t0 = time.time()
    spec = MODELS[model_key]
    out = OUT_ROOT / model_key
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": model_key, "hf_name": spec.hf_name, "status": "running"}

    _set_seed()
    loaded = None
    try:
        loaded = load_model(spec, device=DEVICE)
        summary["n_layers"], summary["n_heads"] = loaded.n_layers, loaded.n_heads
        log(f"[{model_key}] loaded {loaded.n_layers}L x {loaded.n_heads}H on {DEVICE}")

        if DO_COHER:
            probes = quick_coherence_check(loaded)
            (out / f"{model_key}_coherence.json").write_text(
                json.dumps([{"probe": p, "continuation": c} for p, c in probes], indent=2))
            summary["coherence"] = [c for _, c in probes]
            log(f"[{model_key}] coherence: " + " | ".join(c.strip()[:40] for _, c in probes))

        # data
        harm  = load_advbench(limit=N_PAIRS * 4)
        safe  = load_hh_harmless(limit=N_PAIRS * 4)
        pairs = build_matched_pairs(harm, safe, n_pairs=N_PAIRS)
        eval_prompts = [h.text for h, _ in pairs[N_PAIRS // 2:]]
        # perplexity texts are a non-essential control — never let them sink a model
        ppl_texts = None
        if PPL_TEXTS:
            try:
                ppl_texts = load_wikitext2(limit=PPL_TEXTS)
            except Exception as e:  # noqa: BLE001
                log(f"[{model_key}] perplexity texts unavailable ({e!r}); skipping perplexity")
                summary.setdefault("addon_errors", {})["perplexity_texts"] = repr(e)

        # ── core: residual trace ───────────────────────────────────────────
        r0 = patch_residual_stream(loaded, pairs[0][0].text, pairs[0][1].text)
        pd.DataFrame([x.__dict__ for x in r0]).to_csv(out / f"{model_key}_resid_trace.csv", index=False)

        # ── core: per-head z sweep + heatmap ────────────────────────────────
        per_pair = []
        for i, (h, s) in enumerate(pairs):
            log(f"[{model_key}] z-sweep pair {i+1}/{len(pairs)}")
            per_pair.append(patch_each_head(loaded, h.text, s.text))
        agg = aggregate_pairs(per_pair)
        agg.to_csv(out / f"{model_key}_patch_z.csv", index=False)
        plot_heatmap(head_heatmap(agg, loaded.n_layers, loaded.n_heads),
                     title=f"{spec.key}: |Δ refusal-margin| per head",
                     save_to=str(out / f"{model_key}_heatmap.png"))

        ranked = [HeadRef(int(r.layer), int(r.head)) for r in agg[agg.component == "z"].itertuples()]
        heads = ranked[:TOP_K]
        (out / f"{model_key}_safety_heads.json").write_text(
            json.dumps([{"layer": h.layer, "head": h.head} for h in heads], indent=2))
        summary["top_heads"] = [(h.layer, h.head) for h in heads]

        # ── core: zero-ablation + perplexity ────────────────────────────────
        report = evaluate_ablation(loaded, heads, eval_prompts, mode="zero", perplexity_texts=ppl_texts)
        row = dict(report.__dict__); row["perplexity_pct_change"] = report.perplexity_pct_change
        pd.DataFrame([row]).to_csv(out / f"{model_key}_ablation.csv", index=False)
        summary.update(refusal_clean=report.refusal_rate_clean,
                       refusal_ablated_zero=report.refusal_rate_ablated,
                       perplexity_pct_change=report.perplexity_pct_change)
        log(f"[{model_key}] zero-ablation {report.refusal_rate_clean:.0%} -> "
            f"{report.refusal_rate_ablated:.0%}; ppl Δ {report.perplexity_pct_change}")

        # ── optional add-ons (each isolated) ────────────────────────────────
        def _try(name, fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 — one add-on must not sink the model
                log(f"[{model_key}] {name} FAILED: {e!r}")
                summary.setdefault("addon_errors", {})[name] = repr(e)

        if K_SWEEP.strip():
            def _ks():
                ks = [k for k in (int(x) for x in K_SWEEP.split(",")) if k <= len(ranked)]
                clean_rate, pts = ablation_k_sweep(loaded, ranked, eval_prompts, ks,
                                                   mode="zero", perplexity_texts=ppl_texts)
                pd.DataFrame([{"k": p.k, "refusal_rate_clean": clean_rate,
                               "refusal_rate_ablated": p.refusal_rate_ablated,
                               "perplexity_ablated": p.perplexity_ablated} for p in pts]
                             ).to_csv(out / f"{model_key}_ksweep.csv", index=False)
                plot_k_sweep(ks, [p.refusal_rate_ablated for p in pts], clean_rate,
                             title=f"{spec.key}: refusal vs #heads ablated",
                             save_to=str(out / f"{model_key}_ksweep.png"))
            _try("ksweep", _ks)

        if DO_MEAN:
            def _mean():
                mc = compute_mean_z_cache(loaded, [s.text for _, s in pairs])
                rm = evaluate_ablation(loaded, heads, eval_prompts, mode="mean",
                                       mean_cache=mc, perplexity_texts=ppl_texts)
                mrow = dict(rm.__dict__); mrow["perplexity_pct_change"] = rm.perplexity_pct_change
                pd.DataFrame([mrow]).to_csv(out / f"{model_key}_ablation_mean.csv", index=False)
                summary["refusal_ablated_mean"] = rm.refusal_rate_ablated
            _try("mean_ablation", _mean)

        if DO_LASTTOK:
            def _lt():
                pp = [patch_each_head(loaded, h.text, s.text, position=-1)
                      for h, s in pairs[:HEAVY_PAIRS]]
                a = aggregate_pairs(pp)
                a.to_csv(out / f"{model_key}_patch_z_lasttok.csv", index=False)
                plot_heatmap(head_heatmap(a, loaded.n_layers, loaded.n_heads),
                             title=f"{spec.key}: |Δ| last-token patch",
                             save_to=str(out / f"{model_key}_heatmap_lasttok.png"))
            _try("lasttok", _lt)

        if DO_PATTERN:
            def _pat():
                pp = [patch_each_head_pattern(loaded, h.text, s.text)
                      for h, s in pairs[:HEAVY_PAIRS]]
                a = aggregate_pairs(pp)
                a.to_csv(out / f"{model_key}_patch_pattern.csv", index=False)
                plot_heatmap(head_heatmap(a, loaded.n_layers, loaded.n_heads, component="pattern"),
                             title=f"{spec.key}: |Δ| attention pattern",
                             save_to=str(out / f"{model_key}_heatmap_pattern.png"))
            _try("pattern", _pat)

        if DO_JAILBRK:
            def _jb():
                jb = load_harmbench(limit=N_PAIRS)
                rep = jailbreak_stress_test(loaded, heads, [p.text for p in jb], eval_prompts, mode="zero")
                pd.DataFrame([rep.__dict__]).to_csv(out / f"{model_key}_jailbreak.csv", index=False)
                summary["jailbreak"] = rep.__dict__
            _try("jailbreak", _jb)

        summary["status"] = "ok"
        summary["seconds"] = round(time.time() - t0, 1)
        (out / "_DONE.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        if loaded is not None:
            del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ───────────────────────────── orchestration loop ─────────────────────────────
def _zip_results(zip_path: pathlib.Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_ROOT.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(WORK))


def main() -> None:
    keys = _models_to_run()
    log_path  = WORK / "_run_log.txt"
    summ_path = WORK / "_run_summary.json"
    lines: list[str] = []

    def log(msg: str) -> None:
        stamp = f"{time.time() - START:8.1f}s | {msg}"
        print(stamp, flush=True)
        lines.append(stamp)
        log_path.write_text("\n".join(lines) + "\n")

    summaries: list[dict] = []
    log(f"Device {DEVICE}; running (cheapest-first): {keys}")
    if not os.environ.get("SC_MODELS", "").strip():
        log(f"(excluded from default loop: {sorted(_DEFAULT_EXCLUDE)} — "
            f"opt in with e.g. SC_MODELS=phi3)")

    for k in keys:
        if SKIP_EXIST and (OUT_ROOT / k / "_DONE.json").exists():
            log(f"[{k}] skip — already done")
            summaries.append({"model": k, "status": "skipped-existing"})
            summ_path.write_text(json.dumps(summaries, indent=2, default=str))
            continue
        try:
            log(f"════════ {k} START ════════")
            summaries.append(run_one(k, log))
            log(f"════════ {k} OK ({summaries[-1].get('seconds', '?')}s) ════════")
        except Exception as e:  # noqa: BLE001 — skip a failed model, keep going
            log(f"[{k}] MODEL FAILED — skipping. {e!r}")
            for ln in traceback.format_exc().splitlines():
                lines.append("    " + ln)
            log_path.write_text("\n".join(lines) + "\n")
            summaries.append({"model": k, "status": "failed", "error": repr(e)})
        summ_path.write_text(json.dumps(summaries, indent=2, default=str))
        _zip_results(WORK / "safety_circuits_results.zip")

    ok = sum(s.get("status") == "ok" for s in summaries)
    log(f"ALL DONE — {ok}/{len(summaries)} models ok. "
        f"Download safety_circuits_results.zip (or results/<model>/).")


main()
