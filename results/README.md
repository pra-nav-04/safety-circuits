# Results layout

Two studies, one folder per model under each.

## `kaggle_neo/<model>/` — mapping study (main / midterm)

Localizing the refusal circuit (read-only: forward passes + hooks, no training).
Produced by `kaggle/run_experiment.py`. Key artifacts per model:

- `<model>_patch_z.csv`, `<model>_heatmap.png` — per-head activation-patching sweep
- `<model>_resid_trace.csv`, `<model>_patch_z_lasttok.csv`, `<model>_patch_pattern.csv`
- `<model>_ablation.csv`, `<model>_ablation_mean.csv`, `<model>_ksweep.{csv,png}` — zero/mean ablation + perplexity
- `<model>_jailbreak.csv`, `<model>_rtp_toxicity.csv`
- **`<model>_safety_heads.json`** — top-K localized causal heads (consumed by the editing study)
- `<model>_pairs.jsonl`, `<model>_examples.jsonl`, `<model>_coherence.json`, `_DONE.json`

Referenced by code (`scripts/make_figures.py`, `scripts/kaggle_api.py`,
`kaggle/run_edit_experiment.py`'s `SC_HEADS_DIR` default) and the paper docs — do not rename.

## `editing/<model>/` — §9 editing extension

*Editing* the localized circuit: head-restricted LoRA transplant + a no-train steering
baseline, on the same eval harness. Produced by `kaggle/run_edit_experiment.py`
(writes to `/kaggle/working/editing/<model>/` on Kaggle; download
`safety_circuits_editing_results.zip` and commit the `editing/<model>/` contents here).
Key artifacts per model:

- **`<model>_edit_summary.csv`** — baseline vs steering vs `lora_k{1,3,5,10}`: refusal,
  jailbreak refusal, perplexity, `perplexity_pct_change`
- **`<model>_scalpel_axis.png`** — headline figure: refusal-rate vs ΔPPL across blunt
  ablation / steering / LoRA (LoRA alone in the "clean corner")
- `<model>_edit_steering_sweep.csv` — every `(extraction-layer × ablation-set:coeff)` combo
- `<model>_edit_headcount_sweep.{csv,png}` — F1b: refusal-flip vs #heads retrained
- `<model>_edit_baseline.csv`, `<model>_edit_steering.csv`, `<model>_edit_lora.csv`
- `<model>_edit_examples.jsonl` — refuse→comply continuations (clean vs edited)
- `<model>_edit_repatch.csv`, `<model>_edit_repatch_heatmap.png` — do the heads still light up after editing?
- `_EDIT_DONE.json`
