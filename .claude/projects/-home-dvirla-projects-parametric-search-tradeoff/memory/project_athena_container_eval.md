---
name: project_athena_container_eval
description: "How to run this repo's eval jobs on the Athena SLURM cluster inside the agents container (working recipe + gotchas)."
metadata:
  node_type: memory
  type: project
---

Run eval jobs on **Athena** (`ssh athen` = athena.technion.ac.il) inside `~/agents_container.sqsh`.
Full guide: **docs/athena_container_eval.md**; ready template: **scripts/athena_qwen122b.job**
(`sbatch` it). Repo on Athena: `~/parametric_search_tradeoff`. Account `reichart_prj`.

**venv-wipe pitfall (cost hours 2026-07-23):** inside the container, `uv`'s default cache lands on
the small `--writable-tmpfs`. If a `uv run`/`uv sync` ever triggers an actual sync (lock/pyproject
drift, or a scancel that leaves uv metadata inconsistent), the tmpfs fills → the sync fails
mid-install and leaves `.venv/lib/pythonX/site-packages` **empty** (only `_virtualenv.pth/.py` +
`__pycache__`). Every later job then fails (`ModuleNotFoundError: httpx`, missing `certifi/cacert.pem`
→ also breaks logfire's HTTPS export). Fixes: (1) **`export UV_CACHE_DIR=/workspace/.uv_cache`** in
jobs so downloads hit the persistent bind-mount, not tmpfs (cache now populated there); (2) repair by
a full `uv sync` WITH that cache set. After repair, plain `uv run` is a 5ms no-op ("already installed"),
so normal jobs are safe. Verified green 2026-07-23: torch 2.10.0+cu128 (cuda True on L40S),
transformers 5.2.0.dev0, ollama serves gpt-oss:20b. Consider adding UV_CACHE_DIR to the job templates.

**NEVER create new venvs on Athena (`uv venv`, `--python 3.x`).** The container already has the
project venv at `/workspace/.venv` — USE IT: `uv pip install <pkg>` into it (with
`UV_CACHE_DIR=/workspace/.uv_cache`) then run `uv run --no-sync python ...`. Creating a venv or
forcing a `--python` interpreter makes uv download an interpreter to `~/.local/share/uv` which inside
the container is the small writable-tmpfs → `No space left on device (os error 28)`. This bit the
vLLM setup repeatedly. Same rule for heavy deps (vllm, peft/trl): install into the existing venv, run
with `--no-sync`.

**The working recipe (each bullet was a debugging cost):**
- Use **apptainer**, NOT pyxis — `srun --container-image=…sqsh` fails (`unrecognized option`), even
  though `example.job` lists it. Form: `srun apptainer exec --nv --bind $REPO:/workspace --bind
  ~/work:/work agents_container.sqsh /bin/bash -c '…'`.
- **`export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin` as the FIRST line
  inside** the container — its internal PATH is empty (else `bash/ollama/curl` not found).
  apptainer's `--env PATH=` does NOT stick (srun overrides it); call `/bin/bash` by absolute path.
- **ollama models are HOST-side, not in the 20GB image**: store is `~/work/ollama` (318GB). Must
  `--bind ~/work:/work` and `export OLLAMA_MODELS=/work/ollama/models`, else apptainer mounts host
  `$HOME` and `ollama list` is empty. Present: qwen3.5(122b), gemma4, nemotron-3-nano/-super, gpt-oss, qwen3.
- **`--account=reichart_prj` required**; only shared/public partitions (`h200-shared` 141GB,
  `rtx6k-shared` ~98GB, `a100-public` 80GB, `l40s-*` 48GB) — NOT `*-dds`/`*-shocher` (other groups →
  `Invalid account/partition combination`). A 122B (~81GB) fits on ONE h200 or rtx6k GPU.
- QoS options: `12h_4g,24h_1g,24h_4g,2h_2g,4d_1g,4h_0g,72h_8g`.
  **⚠️ Concurrency is capped PER QOS by `MaxJobsPU`, and this — not GPU availability — is usually
  the binding constraint for a many-single-GPU-job sweep.** Measured 2026-09-03
  (`sacctmgr show qos format=Name,MaxJobsPU`): **12h_4g -> 3**, 24h_4g -> 3, 2h_2g -> 3,
  24h_1g -> 4, 72h_8g -> 1, 24h_16g -> 1, **4d_1g -> 8**, 4h_0g -> 10 (but 0 GPUs, useless here).
  So for jobs that each request `--gres=gpu:1`, **`4d_1g` is strictly better than `12h_4g`**:
  2.7x the concurrent jobs for identical per-job resources, plus a 4-day wall that removes the
  12h timeout risk on slow models. Switching a queued HotpotQA sweep from 12h_4g to 4d_1g took
  running jobs from **3 to 11** immediately. Change already-queued jobs in place with
  `scontrol update jobid=<id> qos=4d_1g` — no cancel/resubmit needed, queue position is kept, and
  the pending reason flips from `(QOSMaxJobsPerUserLimit)` to `(None)`. `4d_1g` (1 GPU, 4 days) + `--resume`
  = no wall-time risk for a long single-GPU eval.
- **Compute-node internet works** → the `gemini-3-flash-preview` grader (GOOGLE_API_KEY in the
  mounted `/workspace/.env`, already current — don't overwrite) runs fine from jobs.

**Setup for a fresh offload:** git ff-only to origin; rsync gitignored data
(`data/medqa_index`, `data/medqa_500.jsonl`, `data/medqa_terse.jsonl`) to
`athen:parametric_search_tradeoff/data/`; `uv run` auto-syncs deps (has internet); sbatch the template.

**Handoff/resume:** eval drivers write per-condition JSON and honor `--resume`. To avoid redoing,
rsync a model's `results/medqa_grid/<slug>/` to the target before submitting — it continues
(prints "Loaded N existing results … Considering N completed"). Used this to move qwen3.5:122b's
partial from nlp-srv3 → Athena with zero redo.

**SSH caveat:** DNS/VPN to Technion hosts (athen, nlp-srv3) is flaky (`Could not resolve hostname`,
`kex_exchange_identification: Connection reset`). Space out connections; drive multi-step remote
actions through one idempotent script, not many live SSH calls. See [[project_search_backends]] for
the parallel nlp-srv3 setup.
