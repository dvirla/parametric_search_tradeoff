# Running eval jobs on Athena via the agents container

How to run this repo's eval scripts (e.g. `run_qa_eval_experiment.py`, the MedQA/FRAMES grid
drivers) on the Technion **Athena** SLURM cluster inside the `agents_container.sqsh` container.
This captures the working recipe + the non-obvious gotchas discovered 2026-07-11. For generic
SLURM/Athena background see [athena_workflow.md](athena_workflow.md).

## TL;DR working recipe

The container **must** be run with **apptainer** (not pyxis), with `PATH` and `OLLAMA_MODELS`
exported *inside* the container, and the host ollama store + repo bind-mounted:

```bash
#SBATCH --account=reichart_prj
#SBATCH --partition=h200-shared,rtx6k-shared     # 141GB / ~98GB GPUs — a 122B model fits on ONE
#SBATCH --qos=4d_1g                              # 1 GPU, 4-day wall (no chunking needed)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/home/dvirla/parametric_search_tradeoff/results/athena_%j.log

srun apptainer exec --nv \
     --bind /home/dvirla/parametric_search_tradeoff:/workspace \
     --bind /home/dvirla/work:/work \
     /home/dvirla/agents_container.sqsh \
     /bin/bash -c '
       export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
       export OLLAMA_MODELS=/work/ollama/models
       cd /workspace
       OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=6 \
         ollama serve > /workspace/results/athena_ollama_${SLURM_JOB_ID}.log 2>&1 &
       for i in $(seq 1 90); do curl -s 127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 2; done
       ollama list      # confirm the model is visible
       PARALLEL=0 NUM_WORKERS=6 bash scripts/run_medqa_grid_experiment.sh qwen3.5:122b
     '
```

Submit with `sbatch the_job.job`. A live copy is on Athena at
`~/parametric_search_tradeoff/athena_qwen122b.job`.

## The gotchas (each of these cost real time)

1. **pyxis `--container-image` is NOT installed here.** `srun --container-image=…sqsh
   --container-mounts=…` fails with `unrecognized option '--container-image'`. The `example.job`
   lines that use it are dead — use the **apptainer** form (`example.job` also shows that one).

2. **The container's internal `PATH` is empty.** Without setting it, `bash`/`ollama`/`curl`/`seq`
   aren't found (`FATAL: "bash": executable file not found`). Export it as the **first line inside**
   the container shell: `export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin`.

3. **apptainer's `--env PATH=…` does NOT stick** — srun overrides it with the Slurm PATH
   (`/opt/slurm/.../bin:…`). You must `export PATH` *inside* the `bash -c`, not via `--env`.
   Call bash by absolute path (`/bin/bash -c`) so it's found before PATH is set.

4. **The ollama models live on the HOST, not in the container.** The image is only ~20GB — far too
   small for an 81GB model. The store is `~/work/ollama` (318GB), and you must:
   - `--bind /home/dvirla/work:/work`
   - `export OLLAMA_MODELS=/work/ollama/models`
   Otherwise apptainer mounts host `$HOME` and ollama looks at the empty host `~/.ollama` →
   `ollama list` shows nothing. Models present in the store: `qwen3.5` (incl. `122b`), `gemma4`,
   `nemotron-3-nano`, `nemotron-3-super`, `gpt-oss`, `qwen3`.

5. **`--account=reichart_prj` is required**, and only shared/public partitions are allowed. Including
   another group's partition (`h200-dds`, `rtx6k-shocher`) → `Invalid account/partition combination`.
   Usable: `h200-shared` (141GB), `rtx6k-shared` (~98GB), `a100-public` (80GB), `l40s-*` (48GB).
   A 122B GGUF (~81GB) fits on ONE h200 or rtx6k GPU; needs 2 A100/L40S.

6. **Compute-node internet works** — the Gemini grader (`gemini-3-flash-preview` via
   `GOOGLE_API_KEY` in the mounted `/workspace/.env`) is reachable from inside jobs (verified). No
   need to decouple grading.

## Setup for a fresh offload

1. `ssh athen` (host alias; = `athena.technion.ac.il`). Repo at `~/parametric_search_tradeoff`.
2. Sync code: `git fetch origin && git merge --ff-only origin/<branch>` (only fast-forward; the
   Athena checkout may lag origin).
3. Transfer gitignored data with rsync from wherever it lives, e.g.
   `rsync -az data/medqa_index data/medqa_500.jsonl data/medqa_terse.jsonl athen:parametric_search_tradeoff/data/`.
   The `.env` on Athena is already current — **do not overwrite it**.
4. `uv run` inside the container auto-syncs deps (compute node has internet); no manual `uv sync` needed.
5. Write + `sbatch` a job from the template above.

## QoS / partitions (account reichart_prj)

Allowed QoS: `12h_4g, 24h_1g, 24h_4g, 2h_2g, 4d_1g, 4h_0g, 72h_8g`. `4d_1g` (1 GPU, 4 days) is the
safe choice for a long single-GPU eval — combined with `--resume` there's no wall-time risk.
`72h_8g` for multi-GPU. Always check live availability with `sinfo` / `squeue`; verify actual limits
with `sacctmgr -n show assoc user=$USER format=Account,QOS%50` (partition column is blank = any
allowed partition).

## Resumability & handoff

All eval drivers write per-condition JSON and honor `--resume` (skips completed rows). To hand a
partially-run model between machines, rsync its `results/medqa_grid/<slug>/` dir to the target and
resubmit — the driver's `--resume` continues instead of redoing (verified: it prints
`Loaded N existing results … Considering N problems as completed`).

## SSH note

DNS/VPN from the dev box to Technion hosts (`athen`, `nlp-srv3`) is intermittently flaky
(`Could not resolve hostname …`, `kex_exchange_identification: Connection reset`). Space out
connections, retry after a short backoff, and drive multi-step remote actions through a single
idempotent script rather than many live SSH calls.
