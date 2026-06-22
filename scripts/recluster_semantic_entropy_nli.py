"""
Recompute semantic entropy / cluster_ids for MuSiQue parametric-uncertainty JSONs
using bidirectional NLI entailment instead of an LLM judge.

This is the canonical "semantic uncertainty" clustering (Kuhn et al., 2023): two
sampled answers belong to the same meaning cluster iff each *entails* the other,
given the question as shared context. A DeBERTa-MNLI cross-encoder makes the
entailment calls — deterministic, local, and free of the LLM-judge subjectivity
that mis-clustered hedged/refusal answers (e.g. splitting three identical
"I can't identify X" responses into three clusters).

Reads the existing probe JSONs (no probes are re-run) and writes new files with a
suffix (default ``_nli``), preserving every other field. The original entropy and
cluster_ids are kept as ``semantic_entropy_orig`` / ``cluster_ids_orig`` for
side-by-side comparison.

Usage:
  uv run python scripts/recluster_semantic_entropy_nli.py \
      --input results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json \
              results/musique_parametric/musique_parametric_uncertainty_qwen3.5_122b.json \
              results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
      --nli-model cross-encoder/nli-deberta-v3-base

  # quick smoke test on the first 20 examples of one file:
  uv run python scripts/recluster_semantic_entropy_nli.py --input <file> --limit 20
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ── NLI entailment backend ──────────────────────────────────────────────────────

class NLIEntailment:
    """Bidirectional-entailment judge backed by a DeBERTa-MNLI cross-encoder."""

    symmetric = False  # NLI is directional: (a,b) and (b,a) can differ.

    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 64, max_chars: int = 512):
        from sentence_transformers import CrossEncoder

        self.batch_size = batch_size
        self.max_chars = max_chars
        print(f"Loading NLI cross-encoder '{model_name}' (device={device or 'auto'})...")
        self.model = CrossEncoder(model_name, device=device)
        # Resolve the 'entailment' label index from the model config (label order
        # varies across NLI checkpoints).
        id2label = self.model.model.config.id2label
        self.entail_idx = next(i for i, lbl in id2label.items()
                               if "entail" in str(lbl).lower())
        self.contra_idx = next(i for i, lbl in id2label.items()
                               if "contradic" in str(lbl).lower())
        print(f"  labels={id2label}  entailment={self.entail_idx} contradiction={self.contra_idx}")

    def _ctx(self, question: str, answer: str) -> str:
        a = (answer or "").strip().replace("\n", " ")
        if len(a) > self.max_chars:
            a = a[: self.max_chars]
        return f"{question.strip()} {a}"

    def predict_labels(self, question: str, pairs: list[tuple[str, str]]) -> list[int]:
        """For each (premise_ans, hyp_ans) pair, return the argmax NLI class index."""
        if not pairs:
            return []
        inputs = [(self._ctx(question, p), self._ctx(question, h)) for p, h in pairs]
        return self.predict_ctx(inputs, progress=False)

    def predict_ctx(self, inputs: list[tuple[str, str]], progress: bool = True) -> list[int]:
        """Argmax NLI class for already-built (premise_ctx, hypothesis_ctx) inputs."""
        if not inputs:
            return []
        import numpy as np

        scores = self.model.predict(inputs, batch_size=self.batch_size,
                                    apply_softmax=True, show_progress_bar=progress)
        return [int(p) for p in np.asarray(scores).argmax(axis=1)]


_JUDGE_PROMPT = """\
You are judging whether two independent answers to the same question state the SAME core fact.

Question: {question}

Answer A: {a}
Answer B: {b}

Rules:
- Compare only the core claimed answer. Ignore phrasing, verbosity, and extra detail \
(e.g. "Honda" and "Honda, the maker of Acura" are the SAME).
- Treat non-answers (refusals, "I can't determine", "please specify which X") as the same \
bucket as each other, but DIFFERENT from any committed answer.
- Different named entities, places, dates, or numbers are DIFFERENT \
(e.g. "1403" vs "1372"; "Marseille" vs "Nice"; "Caucasus" vs "Iran").
Respond with exactly one word: SAME or DIFFERENT."""


# BaseAgent provider names are case-specific; accept friendly aliases.
_PROVIDER_ALIASES = {"google": "Google", "gemini": "Google", "openai": "OpenAI",
                     "anthropic": "Anthropic", "claude": "Anthropic", "ollama": "ollama"}


class LLMPairwiseJudge:
    """Pairwise semantic-equivalence judge backed by any BaseAgent provider
    (Ollama, Google/Gemini, OpenAI, Anthropic). Drop-in alternative to
    NLIEntailment.

    Exposes ``entail_idx``/``contra_idx`` sentinels and ``predict_labels`` so the
    same ``cluster_hop`` / batched path can drive it: a pair judged SAME maps to
    'entailment' in BOTH directions, DIFFERENT maps to 'contradiction', which makes
    every rule (strict/moderate/lenient) collapse to the symmetric SAME/DIFFERENT
    decision. ``symmetric=True`` so each unordered pair is judged once.
    """

    entail_idx = 1
    contra_idx = 0
    symmetric = True

    def __init__(self, model: str, provider: str = "ollama", ollama_base_url: str | None = None,
                 max_chars: int = 600, temperature: float = 0.0, use_thinking: bool = False):
        from src.services.base_agent import BaseAgent

        self.max_chars = max_chars
        prov = _PROVIDER_ALIASES.get(provider.lower(), provider)
        kwargs = dict(provider_name=prov, model_name=model, agent_name="clusterer_judge",
                      temperature=temperature, use_thinking=use_thinking)
        if prov == "ollama" and ollama_base_url:
            kwargs["ollama_base_url"] = ollama_base_url
        self.agent = BaseAgent(**kwargs)
        print(f"LLM pairwise judge: provider={prov} model={model} "
              f"(temp={temperature}, thinking={use_thinking})")

    def _trunc(self, s: str) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s[: self.max_chars]

    def predict_labels(self, question: str, pairs: list[tuple[str, str]]) -> list[int]:
        out = []
        for a, b in pairs:
            prompt = _JUDGE_PROMPT.format(question=question, a=self._trunc(a), b=self._trunc(b))
            try:
                resp = self.agent.run(prompt)
                txt = str(getattr(resp, "output", resp)).strip().upper()
                same = "SAME" in txt and "DIFFERENT" not in txt
            except Exception:
                same = False
            out.append(self.entail_idx if same else self.contra_idx)
        return out


class OllamaPairwiseJudge(LLMPairwiseJudge):
    """Back-compat shim: pairwise judge pinned to the Ollama provider."""

    def __init__(self, model: str, base_url: str | None = None, max_chars: int = 600):
        super().__init__(model, provider="ollama", ollama_base_url=base_url, max_chars=max_chars)


class GraderAccuracyJudge:
    """Clusterer that reuses the QA *accuracy* grader as the equivalence judge.

    Instead of asking "do A and B state the same fact" (NLI / SAME-DIFFERENT),
    it treats one sample as the ``[correct_answer]`` and the other as the
    ``[response]`` and runs the exact ``STANDARD_GRADER_TEMPLATE`` used to score
    model accuracy — i.e. a pairwise "is this sample correct w.r.t. that sample"
    comparison. Two samples land in the same cluster under ``combine``:

      - ``single`` (cheapest): ONE directed grade per unordered pair — is the
        2nd sample correct w.r.t. the 1st as gold. Halves the call budget.
      - ``or``  (lenient):     merge if EITHER ordering grades 'yes' (2 calls).
      - ``and`` (strict):      merge only if BOTH orderings grade 'yes' (2 calls).

    Symmetric like ``LLMPairwiseJudge`` so each unordered pair is judged once;
    the strict/moderate/lenient ``rule`` of the NLI path is therefore ignored.
    For full-file reclustering use ``_process_file_grader`` (below), which pools
    every directed grade call across all hops and runs them with bounded
    ``max_workers`` async concurrency — essential for cloud providers where each
    call is ~seconds of latency.
    """

    entail_idx = 1
    contra_idx = 0
    symmetric = True

    def __init__(self, model: str, provider: str = "ollama", ollama_base_url: str | None = None,
                 max_chars: int = 1200, temperature: float = 1.0, use_thinking: bool = True,
                 combine: str = "single", max_workers: int = 16, grader_template: str | None = None):
        from src.services.base_agent import BaseAgent
        from src.services.qa_eval import STANDARD_GRADER_TEMPLATE

        assert combine in ("single", "or", "and"), "combine must be 'single', 'or' or 'and'"
        self.model = model
        self.combine = combine
        self.max_workers = max(1, int(max_workers))
        self.max_chars = max_chars
        self.template = grader_template or STANDARD_GRADER_TEMPLATE
        prov = _PROVIDER_ALIASES.get(provider.lower(), provider)
        kwargs = dict(provider_name=prov, model_name=model, agent_name="grader_accuracy_judge",
                      temperature=temperature, use_thinking=use_thinking)
        if prov == "ollama" and ollama_base_url:
            kwargs["ollama_base_url"] = ollama_base_url
        self.agent = BaseAgent(**kwargs)
        print(f"Grader-accuracy judge: provider={prov} model={model} "
              f"(combine={combine}, workers={self.max_workers}, temp={temperature}, thinking={use_thinking})")

    def _trunc(self, s: str) -> str:
        s = (s or "").strip()
        return s[: self.max_chars]

    def _parse(self, resp) -> bool:
        txt = str(getattr(resp, "output", resp))
        m = re.search(r"correct:\s*(yes|no)", txt, re.IGNORECASE)
        return bool(m) and m.group(1).lower() == "yes"

    def _grade(self, question: str, correct_answer: str, response: str) -> bool:
        """One directed accuracy call: is ``response`` correct given ``correct_answer``?"""
        prompt = self.template.format(question=question,
                                      correct_answer=self._trunc(correct_answer),
                                      response=self._trunc(response))
        try:
            return self._parse(self.agent.run(prompt))
        except Exception:
            return False

    async def _agrade(self, question: str, correct_answer: str, response: str) -> bool:
        prompt = self.template.format(question=question,
                                      correct_answer=self._trunc(correct_answer),
                                      response=self._trunc(response))
        try:
            return self._parse(await self.agent.arun(prompt))
        except Exception:
            return False

    def _call_key(self, call: tuple[str, str, str]) -> str:
        """Stable hash of a grade call (on the *truncated* content actually sent)."""
        import hashlib

        q, ca, resp = call
        s = "\x1f".join([self.model, str(self.template is not None),
                         q.strip(), self._trunc(ca), self._trunc(resp)])
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def grade_all(self, calls: list[tuple[str, str, str]],
                  cache_path: str | None = None) -> list[bool]:
        """Run a flat list of (question, correct_answer, response) grades concurrently.

        If ``cache_path`` is set, each completed grade is flushed to an append-only
        JSONL cache immediately (one ``{"k": <hash>, "v": <bool>}`` line per call),
        and already-cached grades are skipped on rerun. This makes the run
        crash-safe and resumable: a re-invocation re-reads the cache and only
        issues the calls that never completed. Duplicate calls in one pass share a
        single LLM call via the key.
        """
        import asyncio
        import json as _json

        keys = [self._call_key(c) for c in calls]

        cache: dict[str, bool] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                        cache[rec["k"]] = bool(rec["v"])
                    except Exception:
                        continue
            print(f"  loaded {len(cache)} cached grades from {cache_path}")

        todo: dict[str, tuple[str, str, str]] = {}
        for c, k in zip(calls, keys):
            if k not in cache and k not in todo:
                todo[k] = c
        n_skip = len(calls) - len(todo)
        print(f"  {len(calls)} calls: {n_skip} cached/duplicate, {len(todo)} to run "
              f"(workers={self.max_workers})")

        if todo:
            todo_items = list(todo.items())

            async def _runner():
                sem = asyncio.Semaphore(self.max_workers)
                cf = open(cache_path, "a") if cache_path else None
                cf_lock = asyncio.Lock()
                try:
                    from tqdm import tqdm
                    bar = tqdm(total=len(todo_items), desc="grader calls")
                except Exception:
                    bar = None

                async def worker(k, c):
                    async with sem:
                        v = await self._agrade(*c)
                    cache[k] = v
                    if cf is not None:
                        async with cf_lock:
                            cf.write(_json.dumps({"k": k, "v": v}) + "\n")
                            cf.flush()
                    if bar:
                        bar.update(1)

                try:
                    await asyncio.gather(*(worker(k, c) for k, c in todo_items))
                finally:
                    if bar:
                        bar.close()
                    if cf is not None:
                        cf.close()

            asyncio.run(_runner())

        # any key still missing (e.g. crashed mid-call without cache) defaults False
        return [cache.get(k, False) for k in keys]

    def predict_labels(self, question: str, pairs: list[tuple[str, str]]) -> list[int]:
        """Sequential per-pair labels (used by the gold-set benchmark / cluster_hop)."""
        out = []
        for a, b in pairs:
            ab = self._grade(question, a, b)
            if self.combine == "single":
                same = ab
            elif self.combine == "or":
                same = ab or self._grade(question, b, a)
            else:  # 'and'
                same = ab and self._grade(question, b, a)
            out.append(self.entail_idx if same else self.contra_idx)
        return out


# ── clustering ──────────────────────────────────────────────────────────────────

def _union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def _dedup(answers: list[str]) -> tuple[list[str], list[int]]:
    """Return (unique answers in first-seen order, run->unique-index mapping)."""
    uniq_order: list[str] = []
    norm_to_uidx: dict[str, int] = {}
    run_to_uidx: list[int] = []
    for a in answers:
        key = (a or "").strip().lower()
        if key not in norm_to_uidx:
            norm_to_uidx[key] = len(uniq_order)
            uniq_order.append(a)
        run_to_uidx.append(norm_to_uidx[key])
    return uniq_order, run_to_uidx


def _clusters_from_labels(u: int, lab: dict, run_to_uidx: list[int], n: int,
                          rule: str, entail_idx: int, contra_idx: int) -> tuple[float, list[int]]:
    """Build (entropy, cluster_ids) from a directed NLI-label map over u uniques.

    rule:
      'strict'  — same cluster iff BOTH directions are entailment (Kuhn et al.).
      'moderate'— same cluster iff >=1 direction entails AND neither contradicts
                  (neutral-neutral pairs do NOT merge). Fixes lenient's under-
                  merging of distinct-but-not-contradictory answers.
      'lenient' — same cluster unless EITHER direction is contradiction
                  (entailment-or-neutral both ways). Robust to verbose answers
                  that merely add detail, but can under-merge.
    """
    def same_cluster(i, j) -> bool:
        a, b = lab.get((i, j)), lab.get((j, i))
        if rule == "strict":          # both directions entail (Kuhn et al.)
            return a == entail_idx and b == entail_idx
        if rule == "moderate":        # >=1 direction entails AND neither contradicts
            return (a == entail_idx or b == entail_idx) and a != contra_idx and b != contra_idx
        return a != contra_idx and b != contra_idx  # lenient: merge unless contradiction

    edges = [(i, j) for i in range(u) for j in range(i + 1, u) if same_cluster(i, j)]
    uniq_roots = _union_find(u, edges)
    root_to_cid: dict[int, int] = {}
    for r in uniq_roots:
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
    uniq_cid = [root_to_cid[r] for r in uniq_roots]
    cluster_ids = [uniq_cid[run_to_uidx[k]] for k in range(n)]
    counts = list(Counter(cluster_ids).values())
    entropy = -sum((c / n) * math.log2(c / n) for c in counts)
    return entropy, cluster_ids


def cluster_hop(nli: NLIEntailment, question: str, answers: list[str],
                rule: str = "lenient") -> tuple[float, list[int]]:
    """Single-hop clustering (used in tests). Dedups, runs NLI on the uniques,
    then builds clusters. process_file uses a batched path instead."""
    n = len(answers)
    if n == 0:
        return 0.0, []
    uniq_order, run_to_uidx = _dedup(answers)
    u = len(uniq_order)
    if u == 1:
        return 0.0, [0] * n
    if getattr(nli, "symmetric", False):
        # symmetric judge (e.g. LLM SAME/DIFFERENT): one call per unordered pair
        pairs = [(i, j) for i in range(u) for j in range(i + 1, u)]
        labels = nli.predict_labels(question, [(uniq_order[i], uniq_order[j]) for i, j in pairs])
        lab = {}
        for (i, j), l in zip(pairs, labels):
            lab[(i, j)] = lab[(j, i)] = l
    else:
        ordered_pairs = [(i, j) for i in range(u) for j in range(u) if i != j]
        flat = [(uniq_order[i], uniq_order[j]) for i, j in ordered_pairs]
        labels = nli.predict_labels(question, flat)
        lab = {(i, j): l for (i, j), l in zip(ordered_pairs, labels)}
    return _clusters_from_labels(u, lab, run_to_uidx, n, rule, nli.entail_idx, nli.contra_idx)


# ── driver ──────────────────────────────────────────────────────────────────────

def process_file(path: str, nli, suffix: str, limit: int | None,
                 rule: str = "lenient", grader_cache: str | None = "auto",
                 val_limit: int | None = 600) -> str:
    """Two-phase batched reclustering: collect every directed pair across all hops,
    run one big NLI pass (progress bar + full CPU), then cluster each hop.

    Backends without ``predict_ctx`` (e.g. the Ollama pairwise judge) fall back to
    the simple per-hop ``cluster_hop`` loop (LLM calls are sequential regardless).

    ``val_limit`` keeps only the first N examples (the validation split). Some probe
    JSONs (e.g. nemotron) have extra training-set uncertainty evaluations appended
    after the 600 validation questions; this trims them off before clustering."""
    with open(path) as f:
        data = json.load(f)
    if val_limit and len(data) > val_limit:
        print(f"  trimming {len(data)} examples to first {val_limit} (validation split)")
        data = data[:val_limit]
    if limit:
        data = data[:limit]

    if isinstance(nli, GraderAccuracyJudge):
        return _process_file_grader(data, path, nli, suffix, cache_path=grader_cache)

    if not hasattr(nli, "predict_ctx"):
        return _process_file_perhop(data, path, nli, suffix, rule)

    # ── Phase 1: collect hops + all directed pairs (deduped) ──
    hop_specs = []   # one per hop needing NLI: {hop, u, run_to_uidx, n, pair_base, pairs}
    all_inputs: list[tuple[str, str]] = []
    pair_owner: list[tuple[int, int, int]] = []  # (spec_idx, i, j) parallel to all_inputs
    n_hops = 0
    for ex in data:
        for hop in ex.get("sub_questions_results", []):
            n_hops += 1
            answers = [r.get("final_answer", "") for r in hop.get("runs", [])]
            n = len(answers)
            hop["semantic_entropy_orig"] = hop.get("semantic_entropy")
            hop["cluster_ids_orig"] = hop.get("cluster_ids")
            if n == 0:
                hop["_new"] = (0.0, [])
                continue
            uniq_order, run_to_uidx = _dedup(answers)
            u = len(uniq_order)
            if u == 1:
                hop["_new"] = (0.0, [0] * n)
                continue
            spec_idx = len(hop_specs)
            hop_specs.append({"hop": hop, "u": u, "run_to_uidx": run_to_uidx, "n": n})
            q = hop.get("question", "")
            for i in range(u):
                for j in range(u):
                    if i == j:
                        continue
                    all_inputs.append((nli._ctx(q, uniq_order[i]), nli._ctx(q, uniq_order[j])))
                    pair_owner.append((spec_idx, i, j))

    print(f"  {os.path.basename(path)}: {n_hops} hops, {len(hop_specs)} need NLI, "
          f"{len(all_inputs)} directed pairs -> running batched inference...")

    # ── Phase 2: one big inference pass ──
    labels = nli.predict_ctx(all_inputs, progress=True)

    # ── Phase 3: per-hop clustering ──
    lab_by_spec: list[dict] = [dict() for _ in hop_specs]
    for (spec_idx, i, j), l in zip(pair_owner, labels):
        lab_by_spec[spec_idx][(i, j)] = l
    for spec_idx, spec in enumerate(hop_specs):
        ent, cids = _clusters_from_labels(spec["u"], lab_by_spec[spec_idx],
                                          spec["run_to_uidx"], spec["n"], rule,
                                          nli.entail_idx, nli.contra_idx)
        spec["hop"]["_new"] = (ent, cids)

    # ── finalize + stats ──
    n_changed = 0
    ent_before_sum = ent_after_sum = 0.0
    for ex in data:
        for hop in ex.get("sub_questions_results", []):
            new_ent, new_cids = hop.pop("_new")
            old_cids = hop["cluster_ids_orig"]
            hop["semantic_entropy"] = new_ent
            hop["cluster_ids"] = new_cids
            ent_before_sum += (hop["semantic_entropy_orig"] or 0.0)
            ent_after_sum += new_ent
            if old_cids != new_cids:
                n_changed += 1

    stem, ext = os.path.splitext(path)
    out_path = f"{stem}{suffix}{ext}"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  {n_hops} hops, {n_changed} cluster-sets changed "
          f"({100*n_changed/max(n_hops,1):.1f}%); mean entropy {ent_before_sum/max(n_hops,1):.3f} "
          f"-> {ent_after_sum/max(n_hops,1):.3f}")
    print(f"  wrote {out_path}")
    return out_path


def _process_file_perhop(data, path, judge, suffix, rule) -> str:
    """Per-hop reclustering (for backends that judge pairs sequentially, e.g. Ollama)."""
    try:
        from tqdm import tqdm
        hops_iter = tqdm([(ex, hop) for ex in data for hop in ex.get("sub_questions_results", [])])
    except Exception:
        hops_iter = [(ex, hop) for ex in data for hop in ex.get("sub_questions_results", [])]
    n_hops = n_changed = 0
    eb = ea = 0.0
    for _, hop in hops_iter:
        answers = [r.get("final_answer", "") for r in hop.get("runs", [])]
        old_cids = hop.get("cluster_ids")
        hop["semantic_entropy_orig"] = hop.get("semantic_entropy")
        hop["cluster_ids_orig"] = old_cids
        ent, cids = cluster_hop(judge, hop.get("question", ""), answers, rule=rule)
        hop["semantic_entropy"] = ent
        hop["cluster_ids"] = cids
        n_hops += 1
        eb += (hop["semantic_entropy_orig"] or 0.0)
        ea += ent
        if old_cids != cids:
            n_changed += 1
    stem, ext = os.path.splitext(path)
    out_path = f"{stem}{suffix}{ext}"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {n_hops} hops, {n_changed} cluster-sets changed "
          f"({100*n_changed/max(n_hops,1):.1f}%); mean entropy {eb/max(n_hops,1):.3f} -> {ea/max(n_hops,1):.3f}")
    print(f"  wrote {out_path}")
    return out_path


def _process_file_grader(data, path, judge: "GraderAccuracyJudge", suffix: str,
                         cache_path: str | None = "auto") -> str:
    """Batched reclustering for the QA-accuracy grader judge.

    Pools every directed grade call across all hops into one flat list, runs them
    with bounded async concurrency (``judge.max_workers``), then combines per pair
    (``single``/``or``/``and``) into a symmetric label map and clusters each hop.

    ``cache_path`` makes the run crash-safe/resumable (see ``grade_all``):
      - ``"auto"`` (default) -> ``<stem><suffix>.gradecache.jsonl`` next to output.
      - an explicit path     -> use that file.
      - ``None``             -> no cache (not crash-safe).
    """
    hop_specs = []   # {hop, u, run_to_uidx, n, pairs:[(i,j)...]}
    calls: list[tuple[str, str, str]] = []          # (question, correct_answer, response)
    call_owner: list[tuple[int, int, int]] = []      # (spec_idx, pair_idx, slot 0|1)
    n_hops = 0
    for ex in data:
        for hop in ex.get("sub_questions_results", []):
            n_hops += 1
            answers = [r.get("final_answer", "") for r in hop.get("runs", [])]
            n = len(answers)
            hop["semantic_entropy_orig"] = hop.get("semantic_entropy")
            hop["cluster_ids_orig"] = hop.get("cluster_ids")
            if n == 0:
                hop["_new"] = (0.0, [])
                continue
            uniq, run_to_uidx = _dedup(answers)
            u = len(uniq)
            if u == 1:
                hop["_new"] = (0.0, [0] * n)
                continue
            spec_idx = len(hop_specs)
            pairs = [(i, j) for i in range(u) for j in range(i + 1, u)]
            hop_specs.append({"hop": hop, "u": u, "run_to_uidx": run_to_uidx, "n": n, "pairs": pairs})
            q = hop.get("question", "")
            for p_idx, (i, j) in enumerate(pairs):
                calls.append((q, uniq[i], uniq[j]))            # gold=i, response=j
                call_owner.append((spec_idx, p_idx, 0))
                if judge.combine != "single":
                    calls.append((q, uniq[j], uniq[i]))        # gold=j, response=i
                    call_owner.append((spec_idx, p_idx, 1))

    stem, ext = os.path.splitext(path)
    out_path = f"{stem}{suffix}{ext}"
    if cache_path == "auto":
        cache_path = f"{stem}{suffix}.gradecache.jsonl"

    print(f"  {os.path.basename(path)}: {n_hops} hops, {len(hop_specs)} need grading, "
          f"{len(calls)} grade calls (combine={judge.combine}, workers={judge.max_workers}) "
          f"-> running...")
    if cache_path:
        print(f"  grade cache: {cache_path}")
    results = judge.grade_all(calls, cache_path=cache_path)

    # combine directed grades per (spec, pair)
    from collections import defaultdict
    slots: dict[tuple[int, int], dict[int, bool]] = defaultdict(dict)
    for (spec_idx, p_idx, slot), val in zip(call_owner, results):
        slots[(spec_idx, p_idx)][slot] = val

    for spec_idx, spec in enumerate(hop_specs):
        lab: dict[tuple[int, int], int] = {}
        for p_idx, (i, j) in enumerate(spec["pairs"]):
            sd = slots[(spec_idx, p_idx)]
            if judge.combine == "and":
                same = bool(sd.get(0)) and bool(sd.get(1))
            else:  # 'or' or 'single'
                same = bool(sd.get(0)) or bool(sd.get(1))
            lab[(i, j)] = lab[(j, i)] = judge.entail_idx if same else judge.contra_idx
        ent, cids = _clusters_from_labels(spec["u"], lab, spec["run_to_uidx"], spec["n"],
                                          "lenient", judge.entail_idx, judge.contra_idx)
        spec["hop"]["_new"] = (ent, cids)

    # finalize + stats
    n_changed = 0
    ent_before_sum = ent_after_sum = 0.0
    for ex in data:
        for hop in ex.get("sub_questions_results", []):
            new_ent, new_cids = hop.pop("_new")
            old_cids = hop["cluster_ids_orig"]
            hop["semantic_entropy"] = new_ent
            hop["cluster_ids"] = new_cids
            ent_before_sum += (hop["semantic_entropy_orig"] or 0.0)
            ent_after_sum += new_ent
            if old_cids != new_cids:
                n_changed += 1

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {n_hops} hops, {n_changed} cluster-sets changed "
          f"({100*n_changed/max(n_hops,1):.1f}%); mean entropy {ent_before_sum/max(n_hops,1):.3f} "
          f"-> {ent_after_sum/max(n_hops,1):.3f}")
    print(f"  wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more parametric_uncertainty_*.json files to recluster.")
    ap.add_argument("--backend", choices=["nli", "ollama", "grader"], default="nli",
                    help="nli = DeBERTa cross-encoder (default); ollama = pairwise SAME/DIFFERENT "
                         "LLM judge; grader = QA accuracy grader as the equivalence judge.")
    ap.add_argument("--grader-model", default="gpt-oss:120b",
                    help="Model for --backend grader (default gpt-oss:120b, the eval grader).")
    ap.add_argument("--grader-provider", default="ollama",
                    help="Provider for --backend grader: ollama | google | openai | anthropic.")
    ap.add_argument("--grader-combine", choices=["single", "or", "and"], default="single",
                    help="grader backend: single = one directed grade per pair (cheapest, default); "
                         "or = merge if EITHER ordering grades correct; and = BOTH must.")
    ap.add_argument("--grader-workers", type=int, default=16,
                    help="grader backend: concurrent grade calls (async). Default 16.")
    ap.add_argument("--grader-cache", default="auto",
                    help="grader backend: append-only JSONL grade cache for crash-safe/resumable "
                         "runs. 'auto' (default) = <output>.gradecache.jsonl; a path = use it; "
                         "'none' = disable caching.")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base",
                    help="NLI cross-encoder (sentence-transformers).")
    ap.add_argument("--ollama-model", default="MHKetbi/nvidia_Llama-3.3-Nemotron-Super-49B-v1:q8_0",
                    help="Ollama model for --backend ollama.")
    ap.add_argument("--ollama-base-url", default=None, help="Ollama base URL (default: env/localhost).")
    ap.add_argument("--suffix", default=None,
                    help="Output filename suffix (default: _nli for nli backend, _llm for ollama).")
    ap.add_argument("--rule", choices=["lenient", "moderate", "strict"], default="lenient",
                    help="lenient = merge unless contradiction (best NLI rule on the gold set, "
                         "but under-merges ~1/4 of hops); moderate = >=1 dir entails and neither "
                         "contradicts; strict = both directions entail (Kuhn et al.). "
                         "For best quality use --backend ollama with a reasoning judge.")
    ap.add_argument("--device", default=None, help="torch device (e.g. cpu, cuda). Auto if omitted.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: all cores).")
    ap.add_argument("--max-chars", type=int, default=512,
                    help="Truncate each answer to this many chars before NLI.")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N examples (smoke test).")
    ap.add_argument("--val-limit", type=int, default=600,
                    help="Keep only the first N examples (the validation split) before clustering. "
                         "Trims appended training-set uncertainty evals (e.g. nemotron). "
                         "Default 600; pass 0 to disable.")
    args = ap.parse_args()
    _default_suffix = {"ollama": "_llm", "grader": "_grader"}.get(args.backend, "_nli")
    suffix = args.suffix or _default_suffix

    if args.backend == "grader":
        judge = GraderAccuracyJudge(args.grader_model, provider=args.grader_provider,
                                    ollama_base_url=args.ollama_base_url,
                                    combine=args.grader_combine, max_workers=args.grader_workers,
                                    max_chars=args.max_chars)
    elif args.backend == "ollama":
        judge = OllamaPairwiseJudge(args.ollama_model, base_url=args.ollama_base_url,
                                    max_chars=args.max_chars)
    else:
        import torch
        torch.set_num_threads(args.threads or (os.cpu_count() or 1))
        judge = NLIEntailment(args.nli_model, device=args.device,
                              batch_size=args.batch_size, max_chars=args.max_chars)

    grader_cache = None if str(args.grader_cache).lower() == "none" else args.grader_cache

    for path in args.input:
        if not os.path.exists(path):
            print(f"  ! missing: {path}")
            continue
        print(f"\nProcessing {path}  (backend={args.backend}, rule={args.rule})")
        process_file(path, judge, suffix, args.limit, rule=args.rule, grader_cache=grader_cache,
                     val_limit=args.val_limit or None)


if __name__ == "__main__":
    main()
