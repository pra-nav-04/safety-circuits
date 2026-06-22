# Running the §9 editing suite on Lightning AI (or any Linux GPU box)

A Lightning **Studio** is a persistent Linux machine with a GPU and a real terminal — more reliable than
Colab for our ~3–4 h/model runs (a terminal job under `tmux` survives browser disconnects, and the Studio
drive persists across sessions). You only need a browser login — **no API key on the machine**.

> Security: if you ever pasted your Lightning API key anywhere, rotate it in Lightning settings.

## Steps

1. Create/open a **Studio**, then switch the machine to a **GPU** (T4/L4/A10 — any ≥16 GB works).
2. Open the **Terminal** and clone the repo:
   ```bash
   git clone https://github.com/pra-nav-04/safety-circuits.git
   cd safety-circuits
   ```
3. Set your HuggingFace token (gated gemma-3 / llama-3.2 need it; accept each model's terms once on HF):
   ```bash
   export HF_TOKEN=hf_xxx
   ```
4. Run under `tmux` so it survives a dropped connection:
   ```bash
   tmux new -s edit
   bash scripts/run_editing.sh gemma3-1b 2>&1 | tee run_gemma3-1b.log
   # detach: Ctrl-b then d        reattach later: tmux attach -t edit
   ```

Results land in `~/safety-circuits-editing/editing/<model>/` (persists in the Studio). Download that folder
and send it for organizing into `results/editing_v2/<model>/`.

## Notes

- **Smoke first:** `SC_EDIT_STEPS=50 SC_DO_MINIMAL_SWEEP=0 SC_DO_HARDENING=0 bash scripts/run_editing.sh gemma3-1b`
  (~10 min) to confirm the GPU, token, and downloads work before the full run.
- **All models:** pass a comma list — `bash scripts/run_editing.sh "gemma3-1b,qwen3,qwen2.5"` — or loop:
  `for m in gemma3-1b qwen3 qwen2.5 qwen2-1.5b qwen1.5-1.8b gemma1-2b gemma2-2b llama3.2-1b llama3-3b; do bash scripts/run_editing.sh "$m"; done`
- **Resume:** `SC_SKIP_EXISTING=1` is the default, so re-running skips any model already finished in `SC_OUT`.
- **Budget:** ~3–4 h/model × 9 ≈ 30 GPU-h (well under the 80 h free tier). **Stop the machine when idle** to
  conserve hours. Every `SC_*` knob from the notebook is overridable as an env var here.
