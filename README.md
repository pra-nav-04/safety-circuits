# safety-circuits

**Mechanistic Interpretability of AI Safety Guardrails** — finding the attention heads and MLP layers that *causally* produce refusal of harmful prompts in small instruct-tuned language models.

> Lab project for **MA-INF 4330 — Lab Explainable AI and Applications**, University of Bonn.
> Pranav Yadav.

## What this repo does

Given a small instruct LM (Qwen2.5-1.5B, Qwen3-1.7B, Gemma-3-1B, Llama-3.2-3B, and others — see `src/safety_circuits/config.py`), this code:

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
python -m safety_circuits.cli run-mvp --model qwen --n_pairs 16 --top_k 10
```

(`--model` takes any key from `MODELS` in `config.py`: `qwen`, `qwen3`, `gemma3-1b`, `llama3-3b`, `phi3`, `falcon3-1b`, `olmo2-1b`, `tinyllama`.)

## Layout

```
safety-circuits/
├── src/safety_circuits/    # the importable package
│   ├── config.py           # model configs, paths, defaults
│   ├── models.py           # load checkpoints into HookedTransformer (TL-native + HF-port path)
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
├── kaggle/                 # Kaggle GPU kernel runner (run_experiment.py)
├── pyproject.toml
├── Makefile
├── RESEARCH_PLAN.md        # hypotheses, experiments, success criteria
├── FINDINGS.md             # results so far (per-model + cross-model comparison)
└── PROJECT_PLAN.md         # living completion tracker (status, gaps, timeline)
```

## Status

**Pipeline complete; cross-model experiments largely done; paper + presentations in progress.**

- The full patching + ablation pipeline works end-to-end and has been run on Kaggle GPUs.
- **Core result reproduced on 4 models** (Qwen2.5-1.5B, Qwen3-1.7B, Gemma-3-1B, Llama-3.2-3B): a sparse set of attention heads causally produces refusal, and zero-ablating the top-K collapses it. See `FINDINGS.md`.
- **In flight:** perplexity/capability control, Llama K-sweep, Phi-3 port fix, Falcon3 + OLMo-2 runs, refusal-metric audit, jailbreak stress test.

See `PROJECT_PLAN.md` for the full status/gap tracker and timeline, `RESEARCH_PLAN.md` for hypotheses and method, and `notebooks/` for the executable thread.

> **Models note:** the original proposal/pitch named TinyLlama and Phi-3. During E3 calibration TinyLlama under-refused and the Phi-3 TransformerLens port was unreliable, so the study moved to the Qwen/Gemma/Llama instruct family (a swap anticipated in the `RESEARCH_PLAN.md` risk table). Phi-3 remains configured and is being repaired for cross-model coverage.
