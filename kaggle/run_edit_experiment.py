"""§9 editing extension orchestrator — *editing* the localized circuit, not just mapping it.

Mirrors `kaggle/run_experiment.py`: one T4 per session, models looped sequentially
(cheapest-first), each model's run isolated (a failure is logged and skipped), results
flushed per-model and zipped. The bootstrap `runpy`s THIS file instead of
`run_experiment.py` to run the editing suite.

Per model (reads that model's localized heads from the committed
`results/kaggle_neo/<model>/<model>_safety_heads.json`):

  1. baseline   — port HF→TL (no_processing), score refusal / jailbreak / perplexity
  2. steering   — Arditi difference-of-means direction, projected out (no training)
  3. LoRA       — head-restricted LoRA transplant at each K in the head-count sweep (F1b)
  4. re-patch   — re-run the per-head sweep on the edited model (do heads still light up?)

The decisive read is the perplexity contrast: head-restricted LoRA should flip refusal
at small ΔPPL exactly where blunt ablation gave gibberish (F1a).

Control via SC_* env vars (see below). Each method/add-on is isolated so one failure
never sinks a model.
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

from safety_circuits.config import MODELS, RESULTS_DIR, EditConfig
from safety_circuits.ablation import HeadRef
from safety_circuits.data import (
    load_advbench, load_hh_harmless, build_matched_pairs, load_wikitext2, load_harmbench,
    save_jsonl,
)
from safety_circuits.editing import edit_and_load, load_via_port
from safety_circuits.steering import compute_refusal_direction, make_steering_hooks
from safety_circuits.edit_eval import evaluate_edited_model, repatch_after_edit
from safety_circuits.analysis import head_heatmap, plot_heatmap, plot_k_sweep
from safety_circuits.refusal import score_refusal


# ───────────────────────────── config (env) ─────────────────────────────
N_PAIRS     = int(os.environ.get("SC_N_PAIRS", "50"))
HEAVY_PAIRS = int(os.environ.get("SC_HEAVY_PAIRS", "8"))      # pairs for the re-patch sweep
TOP_K       = int(os.environ.get("SC_TOP_K", "10"))          # primary K (the headline LoRA edit)
PPL_TEXTS   = int(os.environ.get("SC_PPL_TEXTS", "64"))
SEED        = int(os.environ.get("SC_SEED", "0"))
SKIP_EXIST  = os.environ.get("SC_SKIP_EXISTING", "0") == "1"

METHODS     = [m.strip() for m in os.environ.get("SC_EDIT_METHODS", "steering,lora").split(",") if m.strip()]
DO_REPATCH  = os.environ.get("SC_DO_REPATCH", "1") == "1"
DO_TRANSFER = os.environ.get("SC_EDIT_TRANSFER", "0") == "1"  # F1c stretch (off by default)

HEADS_DIR   = pathlib.Path(os.environ.get("SC_HEADS_DIR", str(RESULTS_DIR / "kaggle_neo")))

WORK     = pathlib.Path(os.environ.get("SC_OUT", "/kaggle/working"))
OUT_ROOT = WORK / "results_edit"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
START    = time.time()

_COST = {
    "gemma3-1b": 1, "llama3.2-1b": 2,
    "qwen2.5": 3, "qwen2-1.5b": 4, "qwen1.5-1.8b": 5, "qwen3": 6,
    "gemma1-2b": 7, "gemma2-2b": 8, "llama3-3b": 9,
}
_DEFAULT_EXCLUDE = {"tinyllama", "phi3", "falcon3-1b", "olmo2-1b", "gemma4-e2b"}


def _edit_cfg() -> EditConfig:
    targets = os.environ.get("SC_EDIT_TARGETS", "")
    head_counts = os.environ.get("SC_EDIT_HEADCOUNTS", "")
    kw = dict(
        rank=int(os.environ.get("SC_EDIT_RANK", "8")),
        alpha=int(os.environ.get("SC_EDIT_ALPHA", "16")),
        steps=int(os.environ.get("SC_EDIT_STEPS", "300")),
        lr=float(os.environ.get("SC_EDIT_LR", "2e-4")),
        seed=SEED,
    )
    if targets.strip():
        kw["targets"] = tuple(t.strip() for t in targets.split(",") if t.strip())
    if head_counts.strip():
        kw["head_counts"] = tuple(int(x) for x in head_counts.split(",") if x.strip())
    return EditConfig(**kw)


def _models_to_run() -> list[str]:
    env = os.environ.get("SC_MODELS", "").strip()
    if env:
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


def _load_heads(model_key: str) -> list[HeadRef]:
    path = HEADS_DIR / model_key / f"{model_key}_safety_heads.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no safety-heads file at {path} — run the main study (run_experiment.py) first "
            f"or set SC_HEADS_DIR"
        )
    raw = json.loads(path.read_text())
    return [HeadRef(int(h["layer"]), int(h["head"])) for h in raw]


def _free(loaded) -> None:
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


# ───────────────────────────── per-model pipeline ─────────────────────────────
def run_one(model_key: str, cfg: EditConfig, log) -> dict:
    t0 = time.time()
    spec = MODELS[model_key]
    out = OUT_ROOT / model_key
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": model_key, "hf_name": spec.hf_name, "status": "running"}
    _set_seed()

    heads_all = _load_heads(model_key)
    log(f"[{model_key}] loaded {len(heads_all)} localized heads")

    # ── data (held-out eval split mirrors run_experiment.py) ─────────────────
    harm  = load_advbench(limit=N_PAIRS * 4)
    safe  = load_hh_harmless(limit=N_PAIRS * 4)
    pairs = build_matched_pairs(harm, safe, n_pairs=N_PAIRS, seed=SEED)
    half = N_PAIRS // 2
    train_split, eval_split = pairs[:half], pairs[half:]
    eval_prompts = [h.text for h, _ in eval_split]
    # suppression targets come from AdvBench's `target` field (Prompt.meta["target"])
    train_pairs = [(h.text, (h.meta or {}).get("target") or "Sure, here is") for h, _ in train_split]
    repatch_pairs = [(h.text, s.text) for h, s in train_split[:HEAVY_PAIRS]]

    ppl_texts = None
    if PPL_TEXTS:
        try:
            ppl_texts = load_wikitext2(limit=PPL_TEXTS)
        except Exception as e:  # noqa: BLE001
            log(f"[{model_key}] perplexity texts unavailable ({e!r}); skipping perplexity")
            summary.setdefault("addon_errors", {})["perplexity_texts"] = repr(e)

    jb_prompts = None
    try:
        jb_prompts = [p.text for p in load_harmbench(limit=N_PAIRS)]
    except Exception as e:  # noqa: BLE001
        log(f"[{model_key}] HarmBench unavailable ({e!r}); skipping jailbreak read-out")
        summary.setdefault("addon_errors", {})["harmbench"] = repr(e)

    rows: list[dict] = []
    examples_clean: dict[str, dict] = {}

    def _try(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log(f"[{model_key}] {name} FAILED: {e!r}")
            summary.setdefault("addon_errors", {})[name] = repr(e)

    # ── baseline (port HF→TL, no edit) + steering baseline (reuses same model) ──
    loaded = load_via_port(spec, device=DEVICE)
    try:
        base = evaluate_edited_model(loaded, "baseline", eval_prompts, jb_prompts, ppl_texts)
        rows.append(base.to_row())
        pd.DataFrame([base.to_row()]).to_csv(out / f"{model_key}_edit_baseline.csv", index=False)
        log(f"[{model_key}] baseline refusal {base.refusal_rate:.0%}; ppl {base.perplexity}")

        # capture baseline continuations for the examples artifact
        for ep in eval_prompts[:8]:
            s = score_refusal(loaded, ep, generate_continuation=True)
            examples_clean[ep] = {"clean_continuation": s.continuation.strip(),
                                  "clean_refused": bool(s.refused_regex)}

        if "steering" in METHODS:
            def _steer():
                layer = max(0, min(loaded.n_layers - 1, int(cfg.steering_layer_frac * loaded.n_layers)))
                direction = compute_refusal_direction(
                    loaded, [h.text for h, _ in train_split], [s.text for _, s in train_split], layer
                )
                hooks = make_steering_hooks(direction, list(range(loaded.n_layers)), cfg.steering_coeff)
                rep = evaluate_edited_model(loaded, "steering", eval_prompts, jb_prompts, ppl_texts,
                                            fwd_hooks=hooks)
                rows.append(rep.to_row())
                pd.DataFrame([rep.to_row()]).to_csv(out / f"{model_key}_edit_steering.csv", index=False)
                log(f"[{model_key}] steering refusal {rep.refusal_rate:.0%} (layer {layer})")
            _try("steering", _steer)
    finally:
        _free(loaded)

    # ── head-restricted LoRA: head-count sweep (F1b) ─────────────────────────
    if "lora" in METHODS:
        head_counts = [k for k in cfg.head_counts if k <= len(heads_all)]
        if TOP_K not in head_counts and TOP_K <= len(heads_all):
            head_counts = sorted({*head_counts, TOP_K})
        primary_k = TOP_K if TOP_K in head_counts else (head_counts[-1] if head_counts else 0)

        for k in head_counts:
            def _lora(k=k):
                log(f"[{model_key}] LoRA edit on top-{k} heads "
                    f"(rank {cfg.rank}, steps {cfg.steps})")
                edited = edit_and_load(spec, heads_all[:k], cfg, train_pairs, device=DEVICE, log=log)
                try:
                    rep = evaluate_edited_model(edited, f"lora_k{k}", eval_prompts, jb_prompts, ppl_texts)
                    rows.append(rep.to_row())
                    log(f"[{model_key}] lora_k{k} refusal {rep.refusal_rate:.0%}; ppl {rep.perplexity}")

                    if k == primary_k:
                        pd.DataFrame([rep.to_row()]).to_csv(out / f"{model_key}_edit_lora.csv", index=False)
                        # examples: edited continuations vs the captured baseline ones
                        ex = []
                        for ep in eval_prompts[:8]:
                            s = score_refusal(edited, ep, generate_continuation=True)
                            row = {"prompt": ep, **examples_clean.get(ep, {}),
                                   "edited_continuation": s.continuation.strip(),
                                   "edited_refused": bool(s.refused_regex)}
                            ex.append(row)
                        save_jsonl(ex, out / f"{model_key}_edit_examples.jsonl")
                        if DO_REPATCH:
                            def _rp():
                                agg = repatch_after_edit(edited, repatch_pairs)
                                agg.to_csv(out / f"{model_key}_edit_repatch.csv", index=False)
                                plot_heatmap(
                                    head_heatmap(agg, edited.n_layers, edited.n_heads),
                                    title=f"{spec.key}: |Δ refusal-margin| per head (after LoRA edit)",
                                    save_to=str(out / f"{model_key}_edit_repatch_heatmap.png"),
                                )
                            _try("repatch", _rp)
                finally:
                    _free(edited)
            _try(f"lora_k{k}", _lora)

        # ── F1b head-count sweep artifact (refusal-flip + ΔPPL vs #heads) ──────
        lora_rows = [r for r in rows if str(r["label"]).startswith("lora_k")]
        if lora_rows:
            sweep = pd.DataFrame(lora_rows)
            sweep["k"] = sweep["label"].str.replace("lora_k", "", regex=False).astype(int)
            sweep = sweep.sort_values("k")
            sweep.to_csv(out / f"{model_key}_edit_headcount_sweep.csv", index=False)
            base_refusal = next((r["refusal_rate"] for r in rows if r["label"] == "baseline"), float("nan"))
            _try("headcount_plot", lambda: plot_k_sweep(
                sweep["k"].tolist(), sweep["refusal_rate"].tolist(), base_refusal,
                title=f"{spec.key}: refusal vs #heads retrained (LoRA)",
                save_to=str(out / f"{model_key}_edit_headcount_sweep.png"),
            ))

    # ── combined summary (baseline / steering / lora_k*) with ΔPPL ───────────
    base_ppl = next((r["perplexity"] for r in rows if r["label"] == "baseline"), None)
    for r in rows:
        if base_ppl and r.get("perplexity"):
            r["perplexity_pct_change"] = 100.0 * (r["perplexity"] - base_ppl) / base_ppl
    pd.DataFrame(rows).to_csv(out / f"{model_key}_edit_summary.csv", index=False)

    summary["status"] = "ok"
    summary["seconds"] = round(time.time() - t0, 1)
    summary["rows"] = rows
    (out / "_EDIT_DONE.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ───────────────────────────── orchestration loop ─────────────────────────────
def _zip_results(zip_path: pathlib.Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_ROOT.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(WORK))


def main() -> None:
    cfg = _edit_cfg()
    keys = _models_to_run()
    log_path  = WORK / "_edit_run_log.txt"
    summ_path = WORK / "_edit_run_summary.json"
    lines: list[str] = []

    def log(msg: str) -> None:
        stamp = f"{time.time() - START:8.1f}s | {msg}"
        print(stamp, flush=True)
        lines.append(stamp)
        log_path.write_text("\n".join(lines) + "\n")

    summaries: list[dict] = []
    log(f"Device {DEVICE}; editing suite (cheapest-first): {keys}")
    log(f"methods={METHODS} cfg={cfg}")
    if DO_TRANSFER:
        log("note: SC_EDIT_TRANSFER set — F1c cross-model transfer is a documented stretch; "
            "not yet implemented in this orchestrator.")

    for k in keys:
        if SKIP_EXIST and (OUT_ROOT / k / "_EDIT_DONE.json").exists():
            log(f"[{k}] skip — already done")
            summaries.append({"model": k, "status": "skipped-existing"})
            summ_path.write_text(json.dumps(summaries, indent=2, default=str))
            continue
        try:
            log(f"════════ {k} START ════════")
            summaries.append(run_one(k, cfg, log))
            log(f"════════ {k} OK ({summaries[-1].get('seconds', '?')}s) ════════")
        except Exception as e:  # noqa: BLE001 — skip a failed model, keep going
            log(f"[{k}] MODEL FAILED — skipping. {e!r}")
            for ln in traceback.format_exc().splitlines():
                lines.append("    " + ln)
            log_path.write_text("\n".join(lines) + "\n")
            summaries.append({"model": k, "status": "failed", "error": repr(e)})
        summ_path.write_text(json.dumps(summaries, indent=2, default=str))
        _zip_results(WORK / "safety_circuits_edit_results.zip")

    ok = sum(s.get("status") == "ok" for s in summaries)
    log(f"ALL DONE — {ok}/{len(summaries)} models ok. "
        f"Download safety_circuits_edit_results.zip (or results_edit/<model>/).")


main()
