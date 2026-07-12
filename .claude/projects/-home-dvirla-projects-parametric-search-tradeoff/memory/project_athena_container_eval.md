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
- QoS options: `12h_4g,24h_1g,24h_4g,2h_2g,4d_1g,4h_0g,72h_8g`. `4d_1g` (1 GPU, 4 days) + `--resume`
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
