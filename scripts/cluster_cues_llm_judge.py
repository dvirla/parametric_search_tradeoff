"""
LLM-judge semantic-entropy clustering for the CUE-condition no_search probes
(elaborate, direct, confident_parametric, multiturn, searchmulti), within a
single model's own N repeated runs per example (never across models).

Deliberately reuses the EXACT same clusterer (gpt-oss:120b), prompt, and entropy
formula (Shannon entropy over N samples) as the original baseline plain-condition
clustering (results/*_parametric/<model>/<prefix>_<tag>_llm_clusters.json), so
cue-condition entropy is directly comparable to the baseline -- same judge, same
instructions. Do NOT change workers/prompt/model here without also noting it
breaks comparability.

--n-runs controls how many repeated runs are clustered per example (default 3,
matching the original cue-sweep scope and producing the unsuffixed
`_<cue>_llm_clusters.json` filename for backward compatibility). Passing
--n-runs 5 clusters the fuller 5-run backfill instead, writing to a separate
`_<cue>_llm_clusters_5run.json` file -- mirroring the baseline plain-condition's
existing 3-run/5-run split -- so the two are never overwritten by each other and
can be compared directly for consistency once a combo has grown from 3 to 5 runs.

Resumable in two ways:
  1. Per-combo: scans results/ for every (dataset, model, cue) with all N runs
     present and *no* existing output file for that N yet -- already-clustered
     combos are skipped automatically, so re-running this script later (once
     more cues/models/runs finish) only processes what's new.
  2. Per-example within a combo: a .gradecache.jsonl cache next to the output makes
     each combo itself resumable if interrupted mid-run.

Usage:
  uv run python scripts/cluster_cues_llm_judge.py --workers 8                # 3-run (default)
  uv run python scripts/cluster_cues_llm_judge.py --workers 8 --n-runs 5     # 5-run
  uv run python scripts/cluster_cues_llm_judge.py --workers 8 --dry-run      # just list what's ready
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
    "qwen3.5_122b": "qwen3.5:122b",
    "nemotron-3-nano_30b": "nemotron-3-nano:30b",
    "nemotron-cascade-2_30b": "nemotron-cascade-2:30b",
}


def _out_suffix(n_runs: int) -> str:
    return "" if n_runs == 3 else f"_{n_runs}run"


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


def discover_ready_combos(n_runs: int, min_frac: float = 0.95):
    """Scan results/ for (dataset, model_slug, cue) combos with all n_runs present
    (each >= min_frac of target rows) and no existing output file yet for this n_runs."""
    ready = []
    suffix = _out_suffix(n_runs)
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
                out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters{suffix}.json")
                if os.path.exists(out_path):
                    continue  # already clustered -- resumability across invocations
                files = [os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_run_{i}.json")
                         for i in range(1, n_runs + 1)]
                if not all(os.path.exists(f) for f in files):
                    continue
                try:
                    counts = [len(json.load(open(f))) for f in files]
                except Exception:
                    continue
                if all(c >= cfg["target"] * min_frac for c in counts):
                    ready.append((ds, model_slug, tag, cue, counts))
    return ready


def _combo_ready(model_dir, file_prefix, tag, cue, n_runs, target, min_frac):
    """True if this combo has n_runs complete files (each >= min_frac of target) and
    no existing output for n_runs yet. Returns the per-run counts, or None if not ready."""
    suffix = _out_suffix(n_runs)
    out_path = os.path.join(model_dir, f"{file_prefix}_{tag}_{cue}_llm_clusters{suffix}.json")
    if os.path.exists(out_path):
        return None
    files = [os.path.join(model_dir, f"{file_prefix}_{tag}_{cue}_run_{i}.json") for i in range(1, n_runs + 1)]
    if not all(os.path.exists(f) for f in files):
        return None
    try:
        counts = [len(json.load(open(f))) for f in files]
    except Exception:
        return None
    if all(c >= target * min_frac for c in counts):
        return counts
    return None


def discover_ready_combos_best_available(min_frac: float = 0.95):
    """Per (dataset, model, cue): prefer 5-run clustering if ready and not yet done;
    otherwise fall back to 3-run if that's ready and not yet done. Never both -- if a
    5-run output already exists, the 3-run fallback is not considered for that combo,
    and vice versa is naturally moot since 5-run is always tried first."""
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
                # Skip entirely if the OTHER run-count's output already exists, so we
                # never cluster both 3-run and 5-run for the same combo under this mode.
                other_5run_done = os.path.exists(
                    os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters_5run.json"))
                other_3run_done = os.path.exists(
                    os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters.json"))
                if other_5run_done or other_3run_done:
                    continue
                counts5 = _combo_ready(model_dir, cfg["file_prefix"], tag, cue, 5, cfg["target"], min_frac)
                if counts5 is not None:
                    ready.append((ds, model_slug, tag, cue, 5, counts5))
                    continue
                counts3 = _combo_ready(model_dir, cfg["file_prefix"], tag, cue, 3, cfg["target"], min_frac)
                if counts3 is not None:
                    ready.append((ds, model_slug, tag, cue, 3, counts3))
    return ready


def load_runs(model_dir, file_prefix, tag, cue, n_runs: int):
    files = [os.path.join(model_dir, f"{file_prefix}_{tag}_{cue}_run_{i}.json") for i in range(1, n_runs + 1)]
    runs = [json.load(open(f)) for f in files]
    by_id = {}
    for r_idx, run in enumerate(runs):
        for ex in run:
            eid = ex["example_id"]
            by_id.setdefault(eid, {"problem": ex["problem"], "correct_answer": ex["correct_answer"], "responses": {}})
            by_id[eid]["responses"][r_idx] = ex["sampler_response"]
    ids = sorted(by_id.keys(), key=lambda x: str(x))
    ids = [i for i in ids if len(by_id[i]["responses"]) == n_runs]
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


async def process_combo(ds, model_slug, tag, cue, workers, n_runs: int):
    cfg = DATASETS[ds]
    model_dir = os.path.join(REPO, cfg["result_dir"], model_slug)
    by_id, ids = load_runs(model_dir, cfg["file_prefix"], tag, cue, n_runs)
    print(f"\n=== {ds} / {model_slug} / {cue}: {len(ids)} examples with complete {n_runs}/{n_runs} runs ===")

    examples = [
        {"id": eid, "question": by_id[eid]["problem"],
         "answers": [by_id[eid]["responses"][i] for i in range(n_runs)]}
        for eid in ids
    ]

    suffix = _out_suffix(n_runs)
    clusterer, max_chars = build_clusterer()
    cache_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters{suffix}.gradecache.jsonl")
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
    out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    mean_ent = sum(r["semantic_entropy"] for r in out) / max(len(out), 1)
    n_zero = sum(1 for r in out if r["semantic_entropy"] == 0.0)
    print(f"    wrote {out_path}  (mean entropy={mean_ent:.3f}, {n_zero}/{len(out)} zero-entropy)")


async def main_async(args):
    if args.best_available:
        ready = discover_ready_combos_best_available()
        print(f"Found {len(ready)} ready (dataset, model, cue) combos not yet clustered (best-available: 5-run else 3-run):")
        for ds, model_slug, tag, cue, n_runs, counts in ready:
            print(f"  {ds:8s} {model_slug:20s} {cue:22s} n_runs={n_runs} {counts}")
        if args.dry_run:
            return
        for ds, model_slug, tag, cue, n_runs, counts in ready:
            try:
                await process_combo(ds, model_slug, tag, cue, args.workers, n_runs)
            except Exception as e:
                print(f"  ! FAILED {ds}/{model_slug}/{cue} (n_runs={n_runs}): {e}")
        return

    ready = discover_ready_combos(args.n_runs)
    print(f"Found {len(ready)} ready (dataset, model, cue) combos not yet clustered (n_runs={args.n_runs}):")
    for ds, model_slug, tag, cue, counts in ready:
        print(f"  {ds:8s} {model_slug:20s} {cue:22s} {counts}")
    if args.dry_run:
        return
    for ds, model_slug, tag, cue, counts in ready:
        try:
            await process_combo(ds, model_slug, tag, cue, args.workers, args.n_runs)
        except Exception as e:
            print(f"  ! FAILED {ds}/{model_slug}/{cue}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=3, help="repeated runs to cluster per example (default 3; 5 for the fuller backfill); ignored if --best-available is set")
    ap.add_argument("--best-available", action="store_true",
                     help="per combo, use 5-run clustering if ready, else fall back to 3-run; never both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
