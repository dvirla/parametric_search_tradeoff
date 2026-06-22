---
name: project_dataset_registration
description: "How to add a new eval dataset/phrasing variant — register inside qa_eval.py, not via path overrides"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6f701ead-9cc4-4d49-816c-4afb3a0e88ee
---

New eval datasets (including alternate MuSiQue phrasings) must be registered as a named variant inside `EvaluationService._load_dataset` in `src/services/qa_eval.py` (its own `elif dataset_name.lower() == "..."` branch), and added to the `--dataset` choices list in `scripts/run_qa_eval_experiment.py`. Do NOT add a generic `--dataset_path` passthrough to inject arbitrary files.

**Why:** the project deliberately keeps the dataset registry centralized in qa_eval.py so every dataset is a discoverable, named entity that flows consistently through eval → interplay → commitment-locus → make_paper_figures. A path override would bypass that registry.

**How to apply:** to add e.g. a 2nd natural phrasing, add `musique-natural2` (or similar) to qa_eval.py pointing at its JSONL, add it to the runner's choices, and give it its own results dir mirroring `results/musique-natural/`. Relates to [[project_missed_hop_paradox]].
