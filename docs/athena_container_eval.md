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

7. **The container's ollama is 0.18.2 and cannot be upgraded in place** (read-only squashfs, no
   sudo). Newer ollama trees live on the group share instead — see the next section.

## Upgrading ollama without sudo (required for gemma4:31b + tools)

The image ships `/usr/bin/ollama` **0.18.2** with its runtime libs in `/usr/lib/ollama`. It has
**zero gemma4 support**: `strings /usr/bin/ollama | grep gemma4` is empty, and `gemma4:31b`'s
manifest declares

```json
{"model_family":"gemma4", "renderer":"gemma4", "parser":"gemma4", "requires":"0.20.0"}
```

Note there is **no `application/vnd.ollama.image.template` layer** in that manifest (unlike
`gpt-oss:20b`, which ships a 7 KB harmony template). gemma4's chat template *and* its tool-call
parser are compiled into the ollama binary as the named `renderer`/`parser` — so an old binary
can neither load the architecture nor emit/parse tool calls, and no Modelfile trick fixes it.

**The fix does not touch the image.** Ollama's official tarball is relocatable — the binary
resolves its runtime libs relative to itself (`<prefix>/lib/ollama`) — so an extracted tree on the
group share works when put first on `PATH` inside the container:

```bash
# one-time, from the login node (has internet + zstd)
mkdir -p ~/work/opt/ollama && cd ~/work/opt/ollama
curl -fL -o ollama-linux-amd64-0.32.5.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-amd64.tar.zst
mkdir -p 0.32.5 && tar --zstd -xf ollama-linux-amd64-0.32.5.tar.zst -C 0.32.5
```

Then inside the container, **after** the mandatory `export PATH=…` line:

```bash
export PATH=/work/opt/ollama/0.32.5/bin:$PATH
```

`athena_frames_cue_eval.job`, `athena_frames_sft.job` and `athena_parametric.job` take an
**`OLLAMA_VER`** env var that does exactly this. It defaults to **empty = the container's 0.18.2**,
so the completed gpt-oss comparisons stay reproducible; opt in per submission:

```bash
sbatch --export=ALL,MODEL=gemma4:31b,OLLAMA_VER=0.32.5 scripts/athena_frames_cue_eval.job
```

Verified end-to-end by `scripts/athena_gemma4_tools_smoke.job` (job 126672, L40S, driver 570.124.06
= CUDA 12.8 → ollama picks its bundled `cuda_v12`):

- `ollama --version` → 0.32.5 inside the container, no missing libs (image is Ubuntu 22.04 /
  glibc 2.35; the tarball is built well below that).
- `ollama show gemma4:31b` → `Capabilities: completion, vision, tools, thinking`, `requires 0.20.0`.
- `/api/chat` with a `tools` array → `"tool_calls":[{"function":{"name":"search",
  "arguments":{"query":"population of Reykjavik"}}}]`, `done_reason: "stop"`, plus a `thinking`
  field. **No `PARAMETER stop` hacks needed** (unlike the converted gpt-oss GGUFs).
- `/v1/chat/completions` — the OpenAI-compatible path the eval client actually uses — returns
  `finish_reason: "tool_calls"` with well-formed `tool_calls` and a `reasoning` field.

Version notes: an older userspace install at `~/.local/{bin,lib}/ollama` is **0.21.0**, which also
satisfies `requires 0.20.0` and has the gemma4 renderer — but it sits on the personal `/home` quota
and was not what the smoke test validated. Use `0.32.5`. The shared model store
(`/work/ollama/models`) is read fine by both; 0.32.5 did not migrate or rewrite anything.

## Setup for a fresh offload

1. `ssh athen` (host alias; = `athena.technion.ac.il`). Repo at `~/parametric_search_tradeoff`.
2. Sync code: `git fetch origin && git merge --ff-only origin/<branch>` (only fast-forward; the
   Athena checkout may lag origin).
3. Transfer gitignored data with rsync from wherever it lives, e.g.
   `rsync -az data/medqa_index data/medqa_500.jsonl data/medqa_terse.jsonl athen:parametric_search_tradeoff/data/`.
   The `.env` on Athena is already current — **do not overwrite it**.
4. `uv run` inside the container auto-syncs deps (compute node has internet); no manual `uv sync` needed.
5. Write + `sbatch` a job from the template above.

## Storage: both quotas are full (measured 2026-07-28)

| Store | Quota | Used | Free |
|---|---|---|---|
| `/home/dvirla` (personal, `hpc-nfs1:/home`) | 300 G soft / **330 G hard** | 296 G | 4 G to soft |
| `/rg/reichart_prj` (group, `hpc-nfs2:/rg`) — `~/work` symlinks here | **4.0 T** | 4.0 T | **~16 G** |

**`df` lies here, in both directions.** `df -h /rg` intermittently reports the underlying
filesystem (`200T … 35T avail`) instead of the project quota; the number that governs writes is
the 4.0 T one from `quota-g`. Likewise `df /home` shows the 50 T filesystem, not the personal
quota — use `quota -s`. Overrunning either returns `OSError: … 0 written` mid-write rather than
`ENOSPC`, which reads like a mysterious bug.

Of the group's 4.0 T, `dvirla` holds **1007 G**; the rest belongs to other lab members, so the
group share is *not* a free dumping ground — offloading `/home` onto it currently fails with
`No space left on device`.

Where `dvirla`'s group space goes, and what is reclaimable:

| Path | Size | Note |
|---|---|---|
| `models/nemotron-musique-lora-v2{,.2}/checkpoint-*` | ~297 G | 11 intermediate training checkpoints × 27 G (optimizer state). The final `adapter_model.safetensors` (3.4 G) already sits at the top level of each dir. |
| `models/nemotron-musique-lora-v*_merged` | 180 G | 3 × 60 G merged HF checkpoints — regenerable from adapter + base. |
| `ollama/` `nemotron-3-super` | 86.8 G | dropped from the roster 2026-07-20. |
| `ollama/` `gpt-oss-frames-robust-full` | 41.9 G | BF16 GGUF, only ever a `llama-quantize` source. |
| `ollama/` `nemotron-3-nano-musique-{v2.240,v3-aug}` | 48.6 G | completed MusiQue project. |

The ollama blob store is otherwise healthy: **0 unreferenced blobs**, and the duplicate model
*names* (`…-q4km`, `…-q4km-ts`, `…-q4zkzm`) all share one blob — renaming with `ollama cp` costs
nothing.

Reclaimable on `/home` with **zero loss** (verified byte-for-byte duplicates elsewhere):

| Path | Size | Why it's redundant |
|---|---|---|
| `~/.cache/huggingface/…models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 63.2 G | the same 18 blobs already exist in `/work/hf_cache`. |
| `models/gpt-oss-{frames-robust,vanilla}-Q4_K_{M,S}.gguf` | 61 G | all four are already registered blobs in `/work/ollama`. |
| `.uv_cache` + `~/.cache/uv` | ~31 G net | regenerable; note the `.venv` trees **hardlink** into them, so `du` double-counts until the cache is removed. |
| `.merge_venv` | 6.8 G | one-off venv from the vLLM/merge experiments. |

There are **three** HF caches (`repo/.hf_cache` 55.6 G, `~/.cache/huggingface` 65.4 G,
`/work/hf_cache` 78.3 G) totalling 199 G for **136 G** of distinct content. The repo one holds the
gpt-oss BF16/MXFP4 bases and overlaps neither of the others; the other two overlap on 63.2 G.

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
