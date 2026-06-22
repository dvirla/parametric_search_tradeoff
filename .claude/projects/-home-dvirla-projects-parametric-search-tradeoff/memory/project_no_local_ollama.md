---
name: project-no-local-ollama
description: "parametric_search_tradeoff project — Ollama runs on a remote machine, not locally. Do not try `curl localhost:11434` or invoke ollama-dependent scripts on this dev box."
metadata: 
  node_type: memory
  type: project
  originSessionId: c8672eac-fcc4-4a07-b1be-3bdac1135515
---

For the `parametric_search_tradeoff` project on this dev box: there is no local Ollama instance.

**Why:** All ollama-backed scripts (SFT data generation, uncertainty probes, LLM judges) run on a remote GPU machine. The local box is for code editing and analysis only.

**How to apply:**
- Do not run `curl http://localhost:11434/...` or any ollama-CLI commands here.
- When verifying scripts that default to `provider="ollama"`, only run the heuristic / non-LLM code paths locally.
- For LLM-mode end-to-end verification, trust the user to run on the remote machine; do not assume local availability.
- Scripts accepting `--attribution-ollama-url` / `OLLAMA_BASE_URL` are fine — the user supplies the remote URL when running there.
