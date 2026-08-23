"""
For each example, compare the PLAIN baseline's modal (majority) no-search answer against
each CUE's modal answer, using an LLM-judge pairwise equivalence check to decide whether
the model's answer "changed" under the cue.

Modal-answer selection (per condition, per example): among the 3 no-search runs, take the
run with the lowest index that belongs to the BIGGEST semantic cluster (from the existing
3-run LLM-judge clustering). Examples where all 3 runs land in 3 different clusters (a
3-way tie, no well-defined majority) are EXCLUDED from this analysis for that condition.

Reuses the EXACT same judge config as the clustering scripts (gpt-oss:120b via Ollama,
same AnswerClustering schema/prompt) -- here with n=2 answers (plain-modal, cue-modal); the
two land in the same cluster ("unchanged") or different clusters ("changed").

Resumable: skips any (dataset, model, cue) combo whose output file already exists, and
caches individual judge calls in a .gradecache.jsonl next to the output.

Usage:
  uv run python scripts/compare_modal_plain_vs_cue.py --workers 8
  uv run python scripts/compare_modal_plain_vs_cue.py --workers 8 --dry-run
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter

REPO = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO)

from pydantic import BaseModel, Field  # noqa: E402
from src.services.base_agent import BaseAgent  # noqa: E402

N_RUNS = 3
CUES = ["elaborate", "direct", "confident_parametric", "multiturn", "searchmulti"]

DATASETS = {
    "frames": dict(result_dir="results/frames_parametric", file_prefix="frames-cues_no_search", target=501),
    "medqa": dict(result_dir="results/medqa_parametric", file_prefix="medqa-500_no_search", target=500),
}

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


def build_judge():
    return BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=AnswerClustering,
        agent_name="modal_change_judge",
    )


def modal_index(cluster_ids: list[int]) -> int | None:
    """Lowest-index run in the biggest cluster; None if a 3-way tie (all distinct)."""
    counts = Counter(cluster_ids)
    if len(counts) == len(cluster_ids):  # every run in its own cluster -> 3-way tie
        return None
    majority_cid = max(counts, key=lambda c: counts[c])
    member_idxs = [i for i, c in enumerate(cluster_ids) if c == majority_cid]
    return min(member_idxs)


def load_condition(model_dir, file_prefix, tag, cue=None):
    """Returns {example_id: {"problem", "correct_answer", "responses": [r1,r2,r3], "cluster_ids": [...]}}."""
    suffix = f"_{cue}" if cue else ""
    cluster_path = os.path.join(model_dir, f"{file_prefix}_{tag}{suffix}_llm_clusters.json")
    if not os.path.exists(cluster_path):
        return None
    clusters = {row["example_id"]: row["cluster_ids"] for row in json.load(open(cluster_path))}

    run_files = [os.path.join(model_dir, f"{file_prefix}_{tag}{suffix}_run_{i}.json") for i in range(1, N_RUNS + 1)]
    if not all(os.path.exists(f) for f in run_files):
        return None
    runs = [json.load(open(f)) for f in run_files]
    by_id = {}
    for r_idx, run in enumerate(runs):
        for ex in run:
            eid = ex["example_id"]
            by_id.setdefault(eid, {"problem": ex["problem"], "correct_answer": ex["correct_answer"], "responses": {}})
            by_id[eid]["responses"][r_idx] = ex["sampler_response"]

    out = {}
    for eid, cids in clusters.items():
        if eid not in by_id or len(by_id[eid]["responses"]) != N_RUNS:
            continue
        out[eid] = {
            "problem": by_id[eid]["problem"],
            "correct_answer": by_id[eid]["correct_answer"],
            "responses": [by_id[eid]["responses"][i] for i in range(N_RUNS)],
            "cluster_ids": cids,
        }
    return out


def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


async def judge_pairs(judge, pairs: list[dict], cache_path: str, workers: int, max_chars: int = 3000):
    """pairs: list of {id, question, a, b}. Returns {id: same_bool}."""
    cache: dict[str, bool] = {}
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
        print(f"    loaded {len(cache)} cached judge calls")

    prompts = {}
    for p in pairs:
        answers = f"1. {p['a'][:max_chars]}\n2. {p['b'][:max_chars]}"
        prompts[p["id"]] = _CLUSTER_PROMPT.format(question=p["question"], n=2, answers=answers)

    todo = {pid: pr for pid, pr in prompts.items() if _key(pr) not in cache}
    print(f"    {len(prompts)} pairs: {len(prompts) - len(todo)} cached, {len(todo)} to run (workers={workers})")

    if todo:
        sem = asyncio.Semaphore(workers)
        cf = open(cache_path, "a")
        cf_lock = asyncio.Lock()
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(todo))
        except Exception:
            bar = None

        async def worker(pid, prompt):
            async with sem:
                for attempt in range(3):
                    try:
                        resp = await judge.arun(prompt)
                        cids = resp.output.cluster_ids
                        if len(cids) != 2:
                            raise ValueError(f"expected 2 cluster_ids, got {cids}")
                        same = cids[0] == cids[1]
                        break
                    except Exception:
                        if attempt == 2:
                            same = None  # judge failure -- excluded downstream
                        else:
                            await asyncio.sleep(2 ** attempt)
            k = _key(prompt)
            rec = {"k": k, "v": same}
            cache[k] = same
            async with cf_lock:
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
            if bar:
                bar.update(1)

        try:
            await asyncio.gather(*(worker(pid, pr) for pid, pr in todo.items()))
        finally:
            if bar:
                bar.close()
            cf.close()

    return {pid: cache.get(_key(pr)) for pid, pr in prompts.items()}


def discover_ready_combos():
    """(dataset, model, cue) where BOTH the plain baseline and the cue have a completed
    3-run cluster file, and no modal-change output exists yet."""
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
            plain_cluster = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_llm_clusters.json")
            if not os.path.exists(plain_cluster):
                continue
            for cue in CUES:
                out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_vs_plain_modal_change.json")
                if os.path.exists(out_path):
                    continue
                cue_cluster = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_llm_clusters.json")
                if not os.path.exists(cue_cluster):
                    continue
                ready.append((ds, model_slug, tag, cue))
    return ready


async def process_combo(ds, model_slug, tag, cue, workers):
    cfg = DATASETS[ds]
    model_dir = os.path.join(REPO, cfg["result_dir"], model_slug)
    plain = load_condition(model_dir, cfg["file_prefix"], tag, cue=None)
    cued = load_condition(model_dir, cfg["file_prefix"], tag, cue=cue)
    if plain is None or cued is None:
        print(f"  ! skipping {ds}/{model_slug}/{cue}: missing plain or cue data")
        return

    common_ids = sorted(set(plain) & set(cued), key=str)
    pairs = []
    excluded_ties = 0
    rows_meta = {}
    for eid in common_ids:
        p_idx = modal_index(plain[eid]["cluster_ids"])
        c_idx = modal_index(cued[eid]["cluster_ids"])
        if p_idx is None or c_idx is None:
            excluded_ties += 1
            continue
        p_resp = plain[eid]["responses"][p_idx]
        c_resp = cued[eid]["responses"][c_idx]
        pairs.append({"id": eid, "question": plain[eid]["problem"], "a": p_resp, "b": c_resp})
        rows_meta[eid] = {
            "example_id": eid,
            "correct_answer": plain[eid]["correct_answer"],
            "plain_modal_run": p_idx + 1,
            "cue_modal_run": c_idx + 1,
            "plain_modal_response": p_resp,
            "cue_modal_response": c_resp,
        }

    print(f"\n=== {ds} / {model_slug} / {cue}: {len(common_ids)} common examples, "
          f"{excluded_ties} excluded (3-way tie in plain and/or cue), {len(pairs)} to judge ===")

    judge = build_judge()
    cache_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_vs_plain_modal_change.gradecache.jsonl")
    same_by_id = await judge_pairs(judge, pairs, cache_path, workers)

    out = []
    n_changed = n_unchanged = n_failed = 0
    for eid in common_ids:
        if eid not in rows_meta:
            continue
        same = same_by_id.get(eid)
        row = dict(rows_meta[eid])
        if same is None:
            row["changed"] = None
            n_failed += 1
        else:
            row["changed"] = (not same)
            n_changed += int(not same)
            n_unchanged += int(same)
        out.append(row)

    out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_{cue}_vs_plain_modal_change.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    n_scored = n_changed + n_unchanged
    pct = 100 * n_changed / n_scored if n_scored else float("nan")
    print(f"    wrote {out_path}")
    print(f"    changed={n_changed} unchanged={n_unchanged} failed={n_failed} "
          f"({pct:.1f}% changed among scored)")


async def main_async(args):
    ready = discover_ready_combos()
    print(f"Found {len(ready)} (dataset, model, cue) combos ready for modal-change comparison:")
    for ds, model_slug, tag, cue in ready:
        print(f"  {ds:8s} {model_slug:20s} {cue}")
    if args.dry_run:
        return
    for ds, model_slug, tag, cue in ready:
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
