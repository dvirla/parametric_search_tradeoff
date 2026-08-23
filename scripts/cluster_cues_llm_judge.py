"""
LLM-judge semantic-entropy clustering for the CUE-condition no_search probes
(elaborate, direct, confident_parametric, multiturn, searchmulti), 3 runs each.

Deliberately reuses the EXACT same clusterer (gpt-oss:120b), prompt, and entropy
formula (Shannon entropy over N=3 samples) as the original baseline plain-condition
3-run clustering (results/*_parametric/<model>/<prefix>_<tag>_llm_clusters.json), so
cue-condition entropy is directly comparable to the baseline -- same discrete entropy
levels (0, 0.918, 1.585 bits), same judge, same instructions. Do NOT change workers/
prompt/model here without also noting it breaks comparability.

Resumable in two ways:
  1. Per-combo: scans results/ for every (dataset, model, cue) with all 3 runs present
     and *no* existing output file yet -- already-clustered combos are skipped
     automatically, so re-running this script later (once more cues/models finish)
     only processes what's new.
  2. Per-example within a combo: a .gradecache.jsonl cache next to the output makes
     each combo itself resumable if interrupted mid-run.

Usage (on Athena, inside the apptainer with gpt-oss:120b pulled):
  uv run python scripts/_scratch_cluster_cues_llm_judge.py --workers 8
  uv run python scripts/_scratch_cluster_cues_llm_judge.py --workers 8 --dry-run   # just list what's ready
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import Counter

REPO = os.environ.get("REPO_ROOT", "/home/dvirla/parametric_search_tradeoff")
sys.path.append(REPO)

from pydantic import BaseModel, Field  # noqa: E402
from src.services.base_agent import BaseAgent  # noqa: E402

N_RUNS = 3  # matches the cue-sweep generation convention -- keep in sync with baseline 3-run clustering

CUES = ["elaborate", "direct", "confident_parametric", "multiturn", "searchmulti"]

DATASETS = {
    "frames": dict(result_dir="results/frames_parametric", file_prefix="frames-cues_no_search", target=501),
    "medqa": dict(result_dir="results/medqa_parametric", file_prefix="medqa-500_no_search", target=500),
}

# model_slug (dir name) -> ollama tag used in filenames
SLUG_TO_TAG = {
    "gemma4_31b": "gemma4:31b",
    "gpt-oss_120b": "gpt-oss:120b",
    "gpt-oss_20b": "gpt-oss:20b",
    "nemotron-3-nano_30b": "nemotron-3-nano:30b",
}


class AnswerClustering(BaseModel):
    cluster_ids: list[int] = Field(
        description=(
            "Cluster ID for each answer in the same order as provided. "
            "Answers that are semantically equivalent get the same integer ID. "
            "Use consecutive integers starting from 0."
        )
    )


_CLUSTER_PROMPT = """\
Question: {question}

The following {n} answers were given independently to this question:
{answers}

Group them by semantic equivalence. Two answers belong to the same cluster if they \
express the same fact (minor wording differences are fine; "United States" and "USA" are the same). \
Use consecutive integer cluster IDs starting from 0.\
"""


def build_clusterer(max_chars: int = 3000):
    return BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=AnswerClustering,
        agent_name="cue_cluster_judge",
    ), max_chars


def discover_ready_combos(min_frac: float = 0.95):
    """Scan results/ for (dataset, model_slug, cue) combos with all N_RUNS present
    (each >= min_frac of target rows) and no existing output file yet."""
    ready = []
    for ds, cfg in DATASETS.items():
        base = os.path.join(REPO, cfg["result_dir"])
        if not os.path.isdir(base):
            continue
        for model_slug in sorted(os.listdir(base)):
            tag = SLUG_TO_TAG.get(model_slug)
            if not tag:
                continue
            model_dir = os.path.join(base, model_slug)
            for cue in CUES:
                out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters.json")
                if os.path.exists(out_path):
                    continue  # already clustered -- resumability across invocations
                files = [os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_run_{i}.json")
                         for i in range(1, N_RUNS + 1)]
                if not all(os.path.exists(f) for f in files):
                    continue
                try:
                    counts = [len(json.load(open(f))) for f in files]
                except Exception:
                    continue
                if all(c >= cfg["target"] * min_frac for c in counts):
                    ready.append((ds, model_slug, tag, cue, counts))
    return ready


def load_runs(model_dir, file_prefix, tag, cue):
    files = [os.path.join(model_dir, f"{file_prefix}_{tag}_{cue}_run_{i}.json") for i in range(1, N_RUNS + 1)]
    runs = [json.load(open(f)) for f in files]
    by_id = {}
    for r_idx, run in enumerate(runs):
        for ex in run:
            eid = ex["example_id"]
            by_id.setdefault(eid, {"problem": ex["problem"], "correct_answer": ex["correct_answer"], "responses": {}})
            by_id[eid]["responses"][r_idx] = ex["sampler_response"]
    ids = sorted(by_id.keys(), key=lambda x: str(x))
    ids = [i for i in ids if len(by_id[i]["responses"]) == N_RUNS]
    return by_id, ids


def _prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


async def cluster_all(clusterer, max_chars, examples, cache_path, workers):
    cache: dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["k"]] = rec["v"]
                except Exception:
                    continue
        print(f"    loaded {len(cache)} cached clusterings from {os.path.basename(cache_path)}")

    prompts = {}
    for ex in examples:
        answer_lines = "\n".join(f"{i+1}. {a[:max_chars]}" for i, a in enumerate(ex["answers"]))
        prompt = _CLUSTER_PROMPT.format(question=ex["question"], n=len(ex["answers"]), answers=answer_lines)
        prompts[ex["id"]] = prompt

    todo = {eid: p for eid, p in prompts.items() if _prompt_key(p) not in cache}
    print(f"    {len(prompts)} examples: {len(prompts) - len(todo)} cached, {len(todo)} to run (workers={workers})")

    if todo:
        sem = asyncio.Semaphore(workers)
        cf = open(cache_path, "a")
        cf_lock = asyncio.Lock()
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(todo))
        except Exception:
            bar = None

        async def worker(eid, prompt):
            async with sem:
                for attempt in range(3):
                    try:
                        resp = await clusterer.arun(prompt)
                        cids = resp.output.cluster_ids
                        break
                    except Exception:
                        if attempt == 2:
                            cids = None
                        else:
                            await asyncio.sleep(2 ** attempt)
            k = _prompt_key(prompt)
            rec = {"k": k, "v": {"eid": eid, "cluster_ids": cids}}
            cache[k] = rec["v"]
            async with cf_lock:
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
            if bar:
                bar.update(1)

        try:
            await asyncio.gather(*(worker(eid, p) for eid, p in todo.items()))
        finally:
            if bar:
                bar.close()
            cf.close()

    out = {}
    for ex in examples:
        p = prompts[ex["id"]]
        k = _prompt_key(p)
        rec = cache.get(k)
        n = len(ex["answers"])
        if rec is None or rec.get("cluster_ids") is None or len(rec["cluster_ids"]) != n:
            cids = list(range(n))
        else:
            cids = rec["cluster_ids"]
        counts = list(Counter(cids).values())
        entropy = -sum((c / n) * math.log2(c / n) for c in counts)
        out[ex["id"]] = (entropy, cids)
    return out


async def process_combo(ds, model_slug, tag, cue, workers):
    cfg = DATASETS[ds]
    model_dir = os.path.join(REPO, cfg["result_dir"], model_slug)
    by_id, ids = load_runs(model_dir, cfg["file_prefix"], tag, cue)
    print(f"\n=== {ds} / {model_slug} / {cue}: {len(ids)} examples with complete {N_RUNS}/{N_RUNS} runs ===")

    examples = [
        {"id": eid, "question": by_id[eid]["problem"],
         "answers": [by_id[eid]["responses"][i] for i in range(N_RUNS)]}
        for eid in ids
    ]

    clusterer, max_chars = build_clusterer()
    cache_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters.gradecache.jsonl")
    results = await cluster_all(clusterer, max_chars, examples, cache_path, workers)

    out = []
    for eid in ids:
        ent, cids = results[eid]
        out.append({
            "example_id": eid,
            "problem": by_id[eid]["problem"],
            "correct_answer": by_id[eid]["correct_answer"],
            "semantic_entropy": ent,
            "cluster_ids": cids,
        })
    out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    mean_ent = sum(r["semantic_entropy"] for r in out) / max(len(out), 1)
    n_zero = sum(1 for r in out if r["semantic_entropy"] == 0.0)
    print(f"    wrote {out_path}  (mean entropy={mean_ent:.3f}, {n_zero}/{len(out)} zero-entropy)")


async def main_async(args):
    ready = discover_ready_combos()
    print(f"Found {len(ready)} ready (dataset, model, cue) combos not yet clustered:")
    for ds, model_slug, tag, cue, counts in ready:
        print(f"  {ds:8s} {model_slug:20s} {cue:22s} {counts}")
    if args.dry_run:
        return
    for ds, model_slug, tag, cue, counts in ready:
        try:
            await process_combo(ds, model_slug, tag, cue, args.workers)
        except Exception as e:
            print(f"  ! FAILED {ds}/{model_slug}/{cue}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
