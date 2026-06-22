---
name: gpu-topology-and-nccl-constraints-per-machine
description: GPU topology and NCCL communication requirements for multi-GPU training on nlp-srv3 and the Athena cluster
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d39fc16-6c75-4929-be40-6e7f904d1e8c
---

## nlp-srv3 (local GPU server)
4x NVIDIA RTX PRO 6000 Black Edition GPUs, split across two NUMA nodes, no NVLink:
- GPU0 + GPU1: PIX-connected, NUMA node 0 (CPUs 0-23, 48-71) — fast P2P
- GPU2 + GPU3: PIX-connected, NUMA node 1 (CPUs 24-47, 72-95) — fast P2P
- GPU0/1 ↔ GPU2/3: SYS (QPI/UPI cross-NUMA) — NO P2P support, NCCL hangs

**How to apply:**
- 2-GPU training: `CUDA_VISIBLE_DEVICES=0,1` or `2,3` (same NUMA)
- 4-GPU training: add `NCCL_P2P_DISABLE=1`
- Never mix cross-NUMA pairs without `NCCL_P2P_DISABLE=1`

## Athena cluster (remote training machine, 2x 96GB GPUs)
2x GPUs, PIX-connected, both on NUMA node 0 (CPUs 0-15, 128-143), no NVLink.
Work directory: `/home/dvirla/work` (group storage, ~3TB free as of 2026-05-13).

**How to apply:**
- P2P works fine — no `NCCL_P2P_DISABLE` needed
- Use `CUDA_VISIBLE_DEVICES=0,1` + `configs/deepspeed_zero3_2gpu.json` (no CPU offload — 192GB VRAM is sufficient for 30B MoE models)
- Route HF cache and model outputs to `/home/dvirla/work` via `--hf-cache-dir` and `--output-dir`
