# Archived nlp-srv3 one-off drivers

These shell scripts lived **untracked in the repo root on nlp-srv3**
(`/data/home/dvirla/parametric_search_tradeoff/`) and were written ad hoc between 2026-07-07 and
2026-07-22. Archived for provenance — they record which model/dataset/condition combinations were
run there and with what concurrency — and **not** maintained. Current work uses the parameterised
drivers in `scripts/` (`run_frames_grid_experiment.sh`, `run_medqa_grid_experiment.sh`,
`run_parametric_probe_experiment.sh`).

| Script | What it did |
|---|---|
| `nlp_bootstrap.sh`, `bootstrap_remote_grid.sh` | first-time setup of the srv3 checkout + env |
| `nlp_swap.sh` | swapped which model was resident in the shared ollama |
| `run_medqa_cue.sh`, `run_medqa_cue_200_remote.sh` | early MedQA cue smoke runs |
| `run_medqa_grid_parallel.sh`, `run_medqa_grid_srv3.sh` | MedQA grid launches |
| `run_medqa_gemma_gemini.sh` | MedQA with the Gemini grader |
| `run_facts100_gemma.sh`, `run_facts100_qwen.sh` | 100-example facts runs |
| `srv3_phase2.sh`, `srv3_rerun.sh`, `fill_srv3_queue.sh` | queue/refill/rerun helpers for the July grids |
| `srv3_frames_brave.sh`, `srv3_frames_brave2.sh` | FRAMES with the **Brave** backend (not the local BM25 index) |
| `gemma_retry.sh` | retried the gemma4 legs that had failed |

Paths inside are absolute to `/data/home/dvirla/...`. See
[docs/athena_container_eval.md](../../../docs/athena_container_eval.md) for the Athena counterpart
and `scripts/archive/athena/` for the SLURM jobs from that machine.
