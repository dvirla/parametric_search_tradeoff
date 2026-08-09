# Archived Athena one-off job drivers

These SLURM job files lived **untracked in the repo root on Athena** (`~/parametric_search_tradeoff/`)
and were written ad hoc between 2026-07-11 and 2026-07-23. They are archived here for provenance —
they record exactly which model ran on which partition/QoS with which flags — and are **not**
maintained. For current work use the parameterised drivers in `scripts/athena_*.job`.

| Prefix | What it did |
|---|---|
| `rr_<model>.job` | re-ran / resumed a model's grid |
| `fill_<model>.job` | filled gaps in a partially-complete grid |
| `grid_cascade_*.job` | FRAMES / MedQA grids for `nemotron-cascade-2` |
| `athena_gptoss_*.job` | gpt-oss 20b/120b grids (FRAMES, MedQA) |
| `athena_frames_brave_*.job` | FRAMES with the **Brave** backend (not the local BM25 index) |
| `athena_q122b_finish.job` | finished an interrupted qwen3.5:122b run |
| `pull_cascade.job` | `ollama pull` of `nemotron-cascade-2` on a compute node |
| `run_tuned_analysis.sh` | end-to-end analysis for the tuned `nemotron-3-nano-musique-v3-aug` |

Two files from that root were deliberately **not** archived:

- `athena_qwen122b.job` — an older copy of the tracked `scripts/athena_qwen122b.job`, differing
  only in comments.
- `base_gptoss.modelfile` and the `Modelfile.*` family — generated artifacts, reproducible with
  `ollama show --modelfile gpt-oss:20b`.

Paths inside these files are absolute to `/home/dvirla/...` and the pre-2026-07-28 layout; several
also predate the `--bind /rg/reichart_prj/dvirla` and `OLLAMA_VER` conventions documented in
[docs/athena_container_eval.md](../../../docs/athena_container_eval.md).
