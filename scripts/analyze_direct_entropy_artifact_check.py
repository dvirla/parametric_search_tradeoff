"""
Is the `direct` cue's entropy increase (accuracy_revision.md Sec 1.3c: 8/12
cells significant, MedQA 6/6 models) a real belief-instability effect, or an
artifact of the LLM-judge clusterer being over-sensitive to short/terse text
lacking the disambiguating context that longer `plain` answers provide?

Manual inspection of a few high-contrast examples (plain H=0, direct H>1.3)
already found BOTH phenomena co-occurring in the same small sample:
  - medqa_test_0027: judge split 5 near-identical short answers ("induction of
    [hepatic] cytochrome P450 enzymes by rifampin [(specifically CYP3A4)]")
    into 3 clusters over trivial wording -- textbook over-splitting artifact.
  - medqa_test_0044: judge correctly separated "uniformly/symmetrically
    enlarged uterus" from "irregularly enlarged uterus" -- a real, clinically
    opposite finding (uniform enlargement -> adenomyosis; irregular ->
    fibroids), i.e. genuine belief instability, not an artifact.

This script scales that manual read into a cheap, LLM-free heuristic across
every example where `direct` produced more than one cluster: for every pair
of distinct-cluster representative responses, normalize (SQuAD-style, reusing
regrade_regex.normalize) and compute word-set Jaccard overlap. High overlap
(shares most content words, differs mainly in wording/qualifiers) is scored
as "likely trivial split" (judge-artifact candidate); low overlap ("names a
different entity/fact entirely) is scored as "likely real disagreement".
This is a heuristic, not a ground truth -- report the proportion, don't treat
individual classifications as certain, and spot-check both tails before
citing a percentage in the paper.

Usage:
    uv run python scripts/analyze_direct_entropy_artifact_check.py
    uv run python scripts/analyze_direct_entropy_artifact_check.py --cue elaborate
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from itertools import combinations

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from regrade_regex import normalize  # noqa: E402

TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b",
        "nemotron-cascade-2_30b": "nemotron-cascade-2:30b", "qwen3.5_122b": "qwen3.5:122b"}

DATASETS = {
    "frames": dict(dir="results/frames_parametric", prefix="frames-cues"),
    "medqa": dict(dir="results/medqa_parametric", prefix="medqa-500"),
}

# High-frequency words that carry no content signal for this comparison --
# strip before computing overlap so two responses aren't judged "similar"
# just because they're both fluent English.
STOPWORDS = set("""a an the is are was were be been being of to in on at by for with
from as and or but if then so this that these those it its it's most likely
diagnosis mechanism best explains explain based clinical presentation patient
her his their there here what which who how why""".split())

TRIVIAL_THRESHOLD = 0.5


def load_clusters(path):
    return {r["example_id"]: r["cluster_ids"] for r in json.load(open(path))}


def load_responses(model_dir, prefix, tag, cue, n_runs=5):
    out = {}
    for run in range(1, n_runs + 1):
        fname = f"{prefix}_no_search_{tag}_{cue}_run_{run}.json" if cue else f"{prefix}_no_search_{tag}_run_{run}.json"
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            return None
        for row in json.load(open(path)):
            out.setdefault(row["example_id"], []).append(row.get("sampler_response") or "")
    return out


def cluster_representatives(cluster_ids, responses):
    """One representative response text per distinct cluster id (first member)."""
    reps = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in reps:
            reps[cid] = responses[idx]
    return list(reps.values())


def word_set(text):
    norm = normalize(text)
    return {w for w in norm.split() if w not in STOPWORDS and len(w) > 1}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", default="direct")
    ap.add_argument("--min-clusters", type=int, default=2,
                     help="only consider examples where the cue condition split into at least this many "
                          "distinct clusters (2=any split, 5=fully split/max disagreement -- the population "
                          "that disproportionately drives the aggregate entropy shift)")
    ap.add_argument("--require-plain-consensus", action="store_true",
                     help="also require the plain baseline to be FULLY consistent (1 cluster) for this example "
                          "-- isolates the exact population behind 'entropy increases under this cue' claims")
    args = ap.parse_args()
    cue = args.cue

    plain_clusters_by_cell = {}

    rows = []
    examples_dump = []
    real_dump = []
    for ds, cfg in DATASETS.items():
        for model, tag in TAGS.items():
            model_dir = os.path.join(REPO, cfg["dir"], model)
            cue_path = os.path.join(model_dir, f"{cfg['prefix']}_no_search_{tag}_{cue}_llm_clusters_5run.json")
            if not os.path.exists(cue_path):
                continue
            clusters = load_clusters(cue_path)
            responses = load_responses(model_dir, cfg["prefix"], tag, cue)
            if responses is None:
                continue

            plain_clusters = None
            if args.require_plain_consensus:
                plain_path = os.path.join(model_dir, f"{cfg['prefix']}_no_search_{tag}_llm_clusters_5run.json")
                if os.path.exists(plain_path):
                    plain_clusters = load_clusters(plain_path)

            n_split_examples = 0
            n_pairs_trivial = 0
            n_pairs_real = 0
            overlaps = []
            for eid, cids in clusters.items():
                if len(set(cids)) < args.min_clusters or eid not in responses:
                    continue
                if plain_clusters is not None:
                    if eid not in plain_clusters or len(set(plain_clusters[eid])) != 1:
                        continue
                n_split_examples += 1
                reps = cluster_representatives(cids, responses[eid])
                word_sets = [word_set(r) for r in reps]
                for (i, j) in combinations(range(len(word_sets)), 2):
                    ov = jaccard(word_sets[i], word_sets[j])
                    overlaps.append(ov)
                    if ov >= TRIVIAL_THRESHOLD:
                        n_pairs_trivial += 1
                        if len(examples_dump) < 40:
                            examples_dump.append(dict(dataset=ds, model=model, example_id=eid, jaccard=round(ov, 3),
                                                       rep_a=reps[i][:150], rep_b=reps[j][:150], verdict="trivial"))
                    else:
                        n_pairs_real += 1
                        if len(real_dump) < 40:
                            real_dump.append(dict(dataset=ds, model=model, example_id=eid, jaccard=round(ov, 3),
                                                   rep_a=reps[i][:150], rep_b=reps[j][:150], verdict="real"))

            n_pairs = n_pairs_trivial + n_pairs_real
            if n_pairs == 0:
                continue
            rows.append(dict(
                dataset=ds, model=model, n_split_examples=n_split_examples,
                n_pairs=n_pairs, n_pairs_trivial=n_pairs_trivial, n_pairs_real=n_pairs_real,
                pct_trivial=round(100 * n_pairs_trivial / n_pairs, 1),
                mean_jaccard=round(sum(overlaps) / len(overlaps), 3),
            ))

    if not rows:
        print(f"No cells found for cue={cue!r}.")
        return

    out_dir = os.path.join(REPO, "results", "entropy_under_cue")
    out_path = os.path.join(out_dir, f"direct_artifact_check_{cue}.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")

    dump_path = os.path.join(out_dir, f"direct_artifact_check_{cue}_trivial_examples.json")
    with open(dump_path, "w") as f:
        json.dump(examples_dump, f, indent=2)
    real_path = os.path.join(out_dir, f"direct_artifact_check_{cue}_real_examples.json")
    with open(real_path, "w") as f:
        json.dump(real_dump, f, indent=2)
    print(f"wrote {dump_path} ({len(examples_dump)} pairs flagged 'trivial')")
    print(f"wrote {real_path} ({len(real_dump)} pairs flagged 'real', for spot-checking both tails)\n")

    print(f"=== cluster-split pairs classified trivial (Jaccard>={TRIVIAL_THRESHOLD}) vs. real, cue={cue!r} ===")
    total_trivial = sum(r["n_pairs_trivial"] for r in rows)
    total_pairs = sum(r["n_pairs"] for r in rows)
    for r in sorted(rows, key=lambda r: (r["dataset"], r["model"])):
        print(f"  {r['dataset']:6s} {r['model']:20s} split_examples={r['n_split_examples']:4d}  "
              f"pairs={r['n_pairs']:4d}  trivial={r['n_pairs_trivial']:4d} ({r['pct_trivial']:5.1f}%)  "
              f"mean_jaccard={r['mean_jaccard']}")
    print(f"\nOVERALL: {total_trivial}/{total_pairs} ({100*total_trivial/total_pairs:.1f}%) of cluster-split pairs "
          f"under '{cue}' look like trivial rewording (Jaccard>={TRIVIAL_THRESHOLD}), not a real content disagreement.")
    print("\nThis is a heuristic on word overlap, not a semantic judgment -- spot-check "
          f"{dump_path} before citing this percentage anywhere.")


if __name__ == "__main__":
    main()
