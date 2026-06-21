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
    load_xstest, save_jsonl,
)
from safety_circuits.editing import edit_and_load, edit_roundtrip_and_load, load_via_port
from safety_circuits.steering import compute_refusal_direction, make_steering_hooks, resolve_steering_layers
from safety_circuits.edit_eval import (
    deep_eval, evaluate_edited_model, is_substantive, refusal_direction_shift, repatch_after_edit,
)
from safety_circuits.analysis import head_heatmap, plot_heatmap, plot_k_sweep, plot_scalpel_axis
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

# ── Tier 1/2 extensions — all OPT-IN (default off); the validated pipeline is unchanged unless set ──
DO_GENERALIZATION = os.environ.get("SC_DO_GENERALIZATION", "0") == "1"  # T1.1 long-form + per-category + toxicity
DO_MINIMAL_SWEEP  = os.environ.get("SC_DO_MINIMAL_SWEEP", "0") == "1"   # T1.2 rank×steps minimal-edit grid
DO_DIRSHIFT       = os.environ.get("SC_DO_DIRSHIFT", "0") == "1"        # T2.6/2.7 refusal-direction shift
DO_HARDENING      = os.environ.get("SC_DO_HARDENING", "0") == "1"       # T2.5 comply→refuse re-patch round-trip
DO_BENIGN_SUBST   = os.environ.get("SC_DO_BENIGN_SUBSTANCE", "0") == "1" # T1.1b benign substance-unlock (weapon-free)

# Steering sweep grid — one pass tries every (extraction-frac × ablation-set:coeff) combo,
# records refusal+ppl for each in <model>_edit_steering_sweep.csv, and promotes the best
# *coherent* (ppl within tolerance) combo to the headline 'steering' row (re-eval'd with jailbreak).
STEER_FRACS   = [float(x) for x in os.environ.get("SC_STEERING_SWEEP_FRACS", "0.6,0.8").split(",") if x.strip()] or [0.6]
STEER_PPL_TOL = float(os.environ.get("SC_STEERING_PPL_TOL", "0.5"))  # "coherent" = ppl ≤ baseline×(1+tol)

def _parse_steer_grid() -> list[tuple[str, float]]:
    # format: "<layers>:<coeff>" combos separated by ';'. <layers> may contain commas
    # ("all", "extract", "10,11,12"); coeff is after the LAST colon.
    raw = os.environ.get("SC_STEERING_SWEEP",
                         "all:0.05;all:0.1;all:0.2;all:0.4;frac0.4-0.8:1.0;extract:1.0")
    grid = []
    for combo in raw.split(";"):
        combo = combo.strip()
        if not combo or ":" not in combo:
            continue
        spec, coeff = combo.rsplit(":", 1)
        grid.append((spec.strip(), float(coeff)))
    return grid or [("extract", 1.0)]

STEER_GRID = _parse_steer_grid()

HEADS_DIR   = pathlib.Path(os.environ.get("SC_HEADS_DIR", str(RESULTS_DIR / "kaggle_neo")))

WORK     = pathlib.Path(os.environ.get("SC_OUT", "/kaggle/working"))
OUT_ROOT = WORK / "editing"   # downloaded zip extracts to editing/<model>/ → repo results/editing/
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
        steering_coeff=float(os.environ.get("SC_STEERING_COEFF", "1.0")),
        seed=SEED,
    )
    if targets.strip():
        kw["targets"] = tuple(t.strip() for t in targets.split(",") if t.strip())
    if head_counts.strip():
        kw["head_counts"] = tuple(int(x) for x in head_counts.split(",") if x.strip())
    if os.environ.get("SC_STEERING_LAYERS", "").strip():
        kw["steering_layers"] = os.environ["SC_STEERING_LAYERS"].strip()
    if os.environ.get("SC_EDIT_MINIMAL_RANKS", "").strip():
        kw["minimal_ranks"] = tuple(int(x) for x in os.environ["SC_EDIT_MINIMAL_RANKS"].split(",") if x.strip())
    if os.environ.get("SC_EDIT_MINIMAL_STEPS", "").strip():
        kw["minimal_steps"] = tuple(int(x) for x in os.environ["SC_EDIT_MINIMAL_STEPS"].split(",") if x.strip())
    if os.environ.get("SC_EDIT_MAX_TARGET_TOKENS", "").strip():
        kw["benign_target_tokens"] = int(os.environ["SC_EDIT_MAX_TARGET_TOKENS"])
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


def _gpu_gc() -> None:
    """Collect + empty the CUDA cache. NOTE: the caller must drop its own reference to
    any model FIRST (e.g. `loaded = None`); a helper that takes the model as an arg can
    only delete its local copy, leaving the caller's reference alive and the VRAM pinned."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
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
    gen_prompts = gen_cats = None      # held-out harmful set WITH categories (for T1.1 deep eval)
    try:
        jb = load_harmbench(limit=N_PAIRS)
        jb_prompts = [p.text for p in jb]
        gen_prompts = [p.text for p in jb]
        gen_cats = [(p.meta or {}).get("category") for p in jb]
    except Exception as e:  # noqa: BLE001
        log(f"[{model_key}] HarmBench unavailable ({e!r}); skipping jailbreak read-out")
        summary.setdefault("addon_errors", {})["harmbench"] = repr(e)

    # text views of the train split (refusal-direction extraction, round-trip data)
    harm_txt = [h.text for h, _ in train_split]
    safe_txt = [s.text for _, s in train_split]
    edit_layers = sorted({h.layer for h in heads_all[:TOP_K]})   # layers touched by the primary edit

    rows: list[dict] = []
    examples_clean: dict[str, dict] = {}
    # state carried from the baseline block into the LoRA block (Tier 1/2 extensions)
    base_dirs: dict = {}
    gen_rows: list[dict] = []
    tox_pipe = None

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

        # T2.6/2.7: baseline per-layer refusal directions (compared on the edited model later)
        if DO_DIRSHIFT:
            def _basedirs():
                for L in edit_layers:  # raw (un-normalised) so the shift diagnostic can report strength
                    base_dirs[L] = compute_refusal_direction(loaded, harm_txt, safe_txt, L, normalize=False)
                log(f"[{model_key}] captured baseline refusal dirs @ layers {edit_layers}")
            _try("dirshift_baseline", _basedirs)

        # T1.1: baseline long-form generalization (does it produce content, not just openers?)
        if DO_GENERALIZATION and gen_prompts:
            def _genbase():
                from safety_circuits.toxicity import _get_toxicity_pipeline
                nonlocal tox_pipe
                tox_pipe = _get_toxicity_pipeline()
                gen_rows.extend(deep_eval(loaded, gen_prompts, gen_cats, label="baseline",
                                          max_new_tokens=cfg.deep_eval_tokens, tox_pipe=tox_pipe))
                log(f"[{model_key}] baseline deep-eval: {len(gen_prompts)} prompts")
            _try("generalization_baseline", _genbase)

        if "steering" in METHODS:
            def _steer():
                base_ppl = base.perplexity
                harm_txt = [h.text for h, _ in train_split]
                safe_txt = [s.text for _, s in train_split]

                # One pass collects the whole (extraction-layer × ablation-set × coeff) grid.
                # Direction depends only on the extraction layer → compute once per frac and
                # reuse across all (layers, coeff) combos to keep the sweep cheap.
                dir_cache: dict[int, "object"] = {}
                def _direction(frac):
                    layer = max(0, min(loaded.n_layers - 1, int(frac * loaded.n_layers)))
                    if layer not in dir_cache:
                        dir_cache[layer] = compute_refusal_direction(loaded, harm_txt, safe_txt, layer)
                    return layer, dir_cache[layer]

                sweep_rows = []
                for frac in STEER_FRACS:
                    layer, direction = _direction(frac)
                    for spec, coeff in STEER_GRID:
                        abl = resolve_steering_layers(spec, layer, loaded.n_layers)
                        hooks = make_steering_hooks(direction, abl, coeff)
                        # refusal + perplexity only (skip jailbreak here — cheap sweep)
                        r = evaluate_edited_model(loaded, f"steer_L{layer}_{spec}_c{coeff}",
                                                  eval_prompts, None, ppl_texts, fwd_hooks=hooks)
                        ppl_pct = (100.0 * (r.perplexity - base_ppl) / base_ppl
                                   if (base_ppl and r.perplexity) else None)
                        coherent = (r.perplexity is not None and base_ppl is not None
                                    and r.perplexity <= base_ppl * (1.0 + STEER_PPL_TOL))
                        sweep_rows.append({
                            "extract_layer": layer, "ablate_spec": spec, "n_ablate_layers": len(abl),
                            "coeff": coeff, "refusal_rate": r.refusal_rate, "perplexity": r.perplexity,
                            "perplexity_pct_change": ppl_pct, "coherent": coherent,
                        })
                        pstr = f"{r.perplexity:.1f}" if r.perplexity is not None else "n/a"
                        log(f"[{model_key}] steer L{layer} {spec}×{len(abl)} c{coeff}: "
                            f"refusal {r.refusal_rate:.0%}, ppl {pstr} "
                            f"({'coherent' if coherent else 'BROKEN'})")
                pd.DataFrame(sweep_rows).to_csv(out / f"{model_key}_edit_steering_sweep.csv", index=False)

                # Best = lowest refusal among coherent combos (tie-break: lower ppl); if none
                # stay coherent, fall back to the lowest-refusal combo overall (flagged in csv).
                coherent_rows = [r for r in sweep_rows if r["coherent"]]
                pool = coherent_rows or sweep_rows
                best = min(pool, key=lambda r: (r["refusal_rate"], r["perplexity"] or 1e9))
                bpstr = f"{best['perplexity']:.1f}" if best["perplexity"] is not None else "n/a"
                log(f"[{model_key}] best steering: L{best['extract_layer']} {best['ablate_spec']} "
                    f"c{best['coeff']} → refusal {best['refusal_rate']:.0%}, ppl {bpstr}"
                    f"{'' if coherent_rows else '  (NO coherent combo — steering cannot cleanly remove refusal)'}")

                # Full eval of the winner (now WITH jailbreak) → the headline 'steering' row.
                layer = best["extract_layer"]   # its direction is already in dir_cache
                hooks = make_steering_hooks(dir_cache[layer], resolve_steering_layers(
                    best["ablate_spec"], layer, loaded.n_layers), best["coeff"])
                rep = evaluate_edited_model(loaded, "steering", eval_prompts, jb_prompts, ppl_texts,
                                            fwd_hooks=hooks)
                rows.append(rep.to_row())
                pd.DataFrame([rep.to_row()]).to_csv(out / f"{model_key}_edit_steering.csv", index=False)
            _try("steering", _steer)
    finally:
        loaded = None      # drop the run_one reference BEFORE gc so the VRAM is released
        _gpu_gc()

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
                        # T1.1: long-form generalization on the edited model
                        if DO_GENERALIZATION and gen_prompts:
                            def _gen():
                                gen_rows.extend(deep_eval(edited, gen_prompts, gen_cats,
                                                          label=f"lora_k{k}",
                                                          max_new_tokens=cfg.deep_eval_tokens, tox_pipe=tox_pipe))
                                pd.DataFrame(gen_rows).to_csv(
                                    out / f"{model_key}_edit_generalization.csv", index=False)
                            _try("generalization", _gen)
                        # T2.6/2.7: how far the edit rotated the refusal direction
                        if DO_DIRSHIFT and base_dirs:
                            def _ds():
                                ds = refusal_direction_shift(base_dirs, edited, harm_txt, safe_txt,
                                                             sorted(base_dirs))
                                pd.DataFrame(ds).to_csv(
                                    out / f"{model_key}_edit_direction_shift.csv", index=False)
                                log(f"[{model_key}] direction shift: "
                                    + ", ".join(f"L{r['layer']} cos={r['cosine']:.2f}" for r in ds))
                            _try("dirshift", _ds)
                finally:
                    edited = None   # drop the reference BEFORE gc so the VRAM is released
                    _gpu_gc()
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

    base_ppl0 = next((r["perplexity"] for r in rows if r["label"] == "baseline"), None)

    # ── T1.2 minimal-edit sweep (opt-in): smallest rank×steps that still flips refusal ──
    if DO_MINIMAL_SWEEP:
        import dataclasses
        pk = min(TOP_K, len(heads_all))
        def _minimal():
            msweep = []
            for r in cfg.minimal_ranks:
                for st in cfg.minimal_steps:
                    c2 = dataclasses.replace(cfg, rank=r, steps=st)
                    log(f"[{model_key}] minimal-edit rank={r} steps={st} (k={pk})")
                    ed = edit_and_load(spec, heads_all[:pk], c2, train_pairs, device=DEVICE, log=log)
                    try:
                        rep = evaluate_edited_model(ed, f"r{r}_s{st}", eval_prompts, None, ppl_texts)
                        dppl = (100.0 * (rep.perplexity - base_ppl0) / base_ppl0
                                if (base_ppl0 and rep.perplexity) else None)
                        msweep.append({"rank": r, "steps": st, "refusal_rate": rep.refusal_rate,
                                       "perplexity": rep.perplexity, "perplexity_pct_change": dppl})
                    finally:
                        ed = None
                        _gpu_gc()
            pd.DataFrame(msweep).to_csv(out / f"{model_key}_edit_minimal_sweep.csv", index=False)
        _try("minimal_sweep", _minimal)

    # ── T2.5 safety re-patch round-trip (opt-in): comply-edit → refuse-edit ──
    if DO_HARDENING:
        pk = min(TOP_K, len(heads_all))
        def _hard():
            ed = edit_roundtrip_and_load(spec, heads_all[:pk], cfg, train_pairs, harm_txt,
                                         device=DEVICE, log=log)
            try:
                rep = evaluate_edited_model(ed, "roundtrip", eval_prompts, jb_prompts, ppl_texts)
                base_ref = next((r["refusal_rate"] for r in rows if r["label"] == "baseline"), None)
                removed = next((r["refusal_rate"] for r in rows if r["label"] == f"lora_k{pk}"), None)
                pd.DataFrame([
                    {"stage": "baseline", "refusal_rate": base_ref},
                    {"stage": "after_comply_edit(removed)", "refusal_rate": removed},
                    {"stage": "after_refuse_edit(restored)", "refusal_rate": rep.refusal_rate,
                     "jailbreak_refusal_rate": rep.jailbreak_refusal_rate, "perplexity": rep.perplexity},
                ]).to_csv(out / f"{model_key}_edit_roundtrip.csv", index=False)
                log(f"[{model_key}] round-trip refusal: base {base_ref} → removed {removed} "
                    f"→ restored {rep.refusal_rate}")
            finally:
                ed = None
                _gpu_gc()
        _try("hardening", _hard)

    # ── T1.1b benign substance-unlock (opt-in, WEAPON-FREE) ──────────────────
    # Is the "stops at the opener" behaviour a training-target-length artifact? Train the
    # safety heads to comply on BENIGN (instruction → full benign answer) data at two target
    # lengths, eval on over-refused-benign prompts (XSTest safe) + held-out HH benign. No
    # harmful content is trained on or generated — by design (see RESEARCH_PLAN §9 ethics).
    if DO_BENIGN_SUBST:
        import dataclasses
        pk = min(TOP_K, len(heads_all))

        def _benign():
            bpairs = [(p.text, (p.meta or {}).get("response") or "") for p in safe]
            bpairs = [pr for pr in bpairs if pr[1]][:max(N_PAIRS // 2, 8)]
            try:
                xs = [p.text for p in load_xstest(limit=N_PAIRS, safe_only=True)]
            except Exception as e:  # noqa: BLE001
                xs = []
                summary.setdefault("addon_errors", {})["xstest"] = repr(e)
            hh_eval = [p.text for p in safe[N_PAIRS:N_PAIRS + 15]]
            prompts, cats = xs + hh_eval, ["xstest"] * len(xs) + ["hh"] * len(hh_eval)
            if not prompts or not bpairs:
                log(f"[{model_key}] benign-substance: no prompts/pairs; skipping")
                return

            rows_b = []
            base_m = load_via_port(spec, device=DEVICE)   # re-port baseline (earlier one freed)
            try:
                rows_b += deep_eval(base_m, prompts, cats, label="baseline",
                                    max_new_tokens=cfg.benign_target_tokens)
            finally:
                base_m = None
                _gpu_gc()

            for tag, mtt in [("benign_short", 24), ("benign_long", cfg.benign_target_tokens)]:
                c2 = dataclasses.replace(cfg, max_target_tokens=mtt)
                log(f"[{model_key}] benign-substance edit {tag} (max_target_tokens={mtt}, k={pk})")
                ed = edit_and_load(spec, heads_all[:pk], c2, bpairs, device=DEVICE, log=log)
                try:
                    rows_b += deep_eval(ed, prompts, cats, label=tag,
                                        max_new_tokens=cfg.benign_target_tokens)
                finally:
                    ed = None
                    _gpu_gc()

            df = pd.DataFrame(rows_b)
            df["substantive"] = df.apply(lambda r: is_substantive(r.to_dict()), axis=1)
            df.to_csv(out / f"{model_key}_edit_benign_substance.csv", index=False)
            agg = (df.groupby(["label", "category"])
                     .agg(refused=("refused", "mean"), substantive=("substantive", "mean"),
                          mean_len=("gen_len_chars", "mean"), n=("refused", "size")).reset_index())
            agg.to_csv(out / f"{model_key}_edit_benign_substance_agg.csv", index=False)
            for _, r in agg.iterrows():
                log(f"[{model_key}] benign {r['label']}/{r['category']}: refused={r['refused']:.0%} "
                    f"substantive={r['substantive']:.0%} len={r['mean_len']:.0f}")
        _try("benign_substance", _benign)

    # ── combined summary (baseline / steering / lora_k*) with ΔPPL ───────────
    base_ppl = next((r["perplexity"] for r in rows if r["label"] == "baseline"), None)
    for r in rows:
        if base_ppl and r.get("perplexity"):
            r["perplexity_pct_change"] = 100.0 * (r["perplexity"] - base_ppl) / base_ppl
    pd.DataFrame(rows).to_csv(out / f"{model_key}_edit_summary.csv", index=False)

    # ── headline figure: refusal vs ΔPPL across all three removal methods ────
    def _scalpel():
        pts = []
        base_ref = next((r["refusal_rate"] for r in rows if r["label"] == "baseline"), None)
        if base_ref is not None:
            pts.append({"method": "baseline", "label": "base", "dppl": 0.0, "refusal": 100 * base_ref})
        for r in rows:
            lab = str(r["label"])
            if lab.startswith("lora_k") and r.get("perplexity_pct_change") is not None:
                pts.append({"method": "lora", "label": "k" + lab.split("lora_k")[1],
                            "dppl": r["perplexity_pct_change"], "refusal": 100 * r["refusal_rate"]})
        sp = out / f"{model_key}_edit_steering_sweep.csv"
        if sp.exists():
            for _, sr in pd.read_csv(sp).iterrows():
                if pd.notna(sr.get("perplexity_pct_change")):
                    pts.append({"method": "steering", "label": None,
                                "dppl": float(sr["perplexity_pct_change"]),
                                "refusal": 100 * float(sr["refusal_rate"])})
        ab = HEADS_DIR / model_key / f"{model_key}_ablation.csv"
        if ab.exists():
            adf = pd.read_csv(ab)
            if len(adf) and {"perplexity_pct_change", "refusal_rate_ablated"} <= set(adf.columns):
                a0 = adf.iloc[0]
                if pd.notna(a0.get("perplexity_pct_change")):
                    pts.append({"method": "ablation", "label": "top-10 zero",
                                "dppl": float(a0["perplexity_pct_change"]),
                                "refusal": 100 * float(a0["refusal_rate_ablated"])})
        if pts:
            plot_scalpel_axis(pts, title=f"{spec.key}: refusal vs ΔPPL (scalpel sharpness)",
                              save_to=str(out / f"{model_key}_scalpel_axis.png"))
    _try("scalpel_plot", _scalpel)

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
        _zip_results(WORK / "safety_circuits_editing_results.zip")

    ok = sum(s.get("status") == "ok" for s in summaries)
    log(f"ALL DONE — {ok}/{len(summaries)} models ok. "
        f"Download safety_circuits_editing_results.zip (or editing/<model>/).")


main()
