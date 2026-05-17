# safety-circuits

**Mechanistic Interpretability of AI Safety Guardrails** — finding the attention heads and MLP layers that *causally* produce refusal of harmful prompts in small instruct-tuned language models.

> Lab project for **MA-INF 4330 — Lab Explainable AI and Applications**, University of Bonn.
> Pranav Yadav.

## What this repo does

Given a small instruct LM (TinyLlama-1.1B-Chat or Phi-3-mini), this code:

1. Runs the model on a curated set of toxic/harmful prompts and matched harmless prompts.
2. Detects whether the model refused (logit on refusal tokens + regex backstop).
3. Caches every attention-head and MLP activation with TransformerLens.
4. Does **activation patching** (Meng et al. ROME / Vig et al. causal tracing): replace activations from a harmless run into the harmful run, find the components where the patch flips behaviour.
5. **Ablates** the candidate components (zero / mean-ablation) to verify that refusal *collapses* without them.
6. Plots a heatmap of `Δrefusal_rate` over (layer, head) — a "mechanistic map of safety."

This is read-only mechinterp: **no training, no fine-tuning, no gradient steps**. Forward passes + hooks only.

## Quick start (Colab)

Open `notebooks/01_setup_and_smoke_test.ipynb` in Colab Pro and run all cells. The notebook clones this repo and installs the same pinned deps as the Docker image.

## Quick start (local / Docker)

```bash
make build         # builds the CUDA image
make shell         # drops you into a container with deps installed
# inside the container:
pytest -q          # smoke tests (CPU, < 30s)
python -m safety_circuits.scripts.run_mvp --model tinyllama --n_prompts 16
```

## Layout

```
safety-circuits/
├── src/safety_circuits/    # the importable package
│   ├── config.py           # model configs, paths, defaults
│   ├── models.py           # load TinyLlama / Phi-3 into HookedTransformer
│   ├── data.py             # unified loaders for HH-RLHF / AdvBench / HarmBench / RTP
│   ├── refusal.py          # refusal-detection metric
│   ├── activations.py      # cache hooks, residual-stream snapshots
│   ├── patching.py         # activation patching / causal tracing
│   ├── ablation.py         # zero / mean ablation of heads + MLPs
│   └── analysis.py         # aggregation, plotting
├── scripts/                # CLI entry points
├── notebooks/              # Colab-runnable experiments (01 → 05)
├── configs/                # YAML configs per experiment
├── tests/                  # fast CPU smoke tests
├── data/{raw,processed}/   # cached datasets (gitignored)
├── results/                # figures + activation dumps (gitignored)
├── docker/                 # Dockerfile, compose
├── pyproject.toml
├── Makefile
└── RESEARCH_PLAN.md        # hypotheses, experiments, success criteria
```

## Status

Scaffold. See `RESEARCH_PLAN.md` for the experiment plan and `notebooks/` for the executable thread.
