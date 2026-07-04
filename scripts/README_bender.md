# Running `safety-circuits` on the Bender GPU cluster (Uni Bonn HPC)

Bender is the Uni Bonn educational GPU cluster (A40 48GB / A100 80GB, SLURM-scheduled).
Unlike Kaggle's single T4, Bender runs the 9 models **in parallel** as a job array — the
whole sweep finishes in roughly the time of the slowest single model.

Reference docs are saved as PDFs in `../bender_documentation/`. Full plan:
`~/.claude/plans/okay-so-last-month-wild-gray.md`.

## 0. Connect (one-time)

Username = your Uni-ID (e.g. `s52pyada`). Bender is reachable **only via Uni Bonn VPN /
campus network**. Use SSH keys, not a stored password:

```bash
ssh-keygen -t ed25519 -a 100 -C "s52pyada@uni-bonn.de"   # if you don't have a key
ssh-copy-id s52pyada@bender.hpc.uni-bonn.de              # type your Uni password once
# ~/.ssh/config:  Host bender / Hostname bender.hpc.uni-bonn.de / User s52pyada / IdentityFile ~/.ssh/id_ed25519
ssh bender
```

## 1. Get the code

```bash
git clone https://github.com/pra-nav-04/safety-circuits.git   # login node has internet
cd safety-circuits
```

## 2. Build the environment  (conda-forge only — Anaconda/`defaults` are banned)

```bash
bash scripts/bender/setup_env.sh          # on the login node: creates the conda env
srun --partition=A40devel --gpus=1 --time=0:30:00 --pty /bin/bash
bash scripts/bender/setup_env.sh          # inside the GPU job: installs CUDA torch + `pip install -e .`
exit
```

Torch must be installed **inside a GPU job** — on the login node conda/pip pick CPU
builds (no GPU there). We install `torch` only (no torchvision/torchaudio) to avoid the
`import transformer_lens` ABI crash.

## 3. Pre-stage models + datasets (login node)

```bash
export HF_TOKEN=hf_...                     # accept terms once on HF for gated gemma/llama + HarmBench
bash scripts/bender/prestage.sh
quota -s                                   # home is 100GB, not backed up
```

> **Network note:** `load_advbench()` and the XSTest CSV fallback fetch from raw GitHub
> at runtime (not HF-cached). If the smoke test shows compute nodes have no outbound
> internet, we'll add a local-file fallback to `data.py`.

## 4. Smoke test (one model, debug queue)

```bash
sbatch scripts/bender/smoke.sbatch
squeue --me
tail -f slurm-sc-smoke-<jobid>.out
seff <jobid>                               # confirm the GPU was actually used
ls sc-out/results/qwen2.5/                 # expect _DONE.json
```

## 5. Full sweep (parallel job array)

```bash
sbatch scripts/bender/sweep.sbatch                             # main study
SC_ORCH=run_edit_experiment.py sbatch scripts/bender/sweep.sbatch   # §9 editing suite
squeue --me
```

- One array task per model (`--array=0-8`), each on its own GPU (`A40short`, 8h).
- **Resume/checkpoint:** re-submitting skips finished models (`SC_SKIP_EXISTING=1`).
- Outputs go to `$SC_OUT/results/<model>/` (default `~/sc-out`) — set via `SC_OUT`, no
  code change needed. The §9 suite writes to `$SC_OUT/editing/<model>/`.

## 6. Retrieve results

```bash
# from your local machine:
rsync -azv bender:~/sc-out ./                # or scp -r
```

## SLURM cheat-sheet (Bender)

| Need | Command |
|---|---|
| Submit | `sbatch <script>` |
| Interactive GPU shell | `srun --partition=A40devel --gpus=1 --time=0:30:00 --pty /bin/bash` |
| My jobs | `squeue --me` |
| Cancel | `scancel <jobid>` / `scancel --me` |
| Efficiency (post-run) | `seff <jobid>` |
| Queues/limits | `sinfo` / `scontrol show partitions` |
| Disk usage | `quota -s` |

Partitions: `A40devel`/`A100devel` (1h debug), `A40short`/`A100short` (8h),
`A40medium`/`A100medium` (1 day, 1 concurrent/user). No `--account` needed on Bender.
Email notifications don't work on Bender. Node-local scratch `/local/nvme/$USER_$SLURM_JOB_ID`
is fast but wiped at job end.
