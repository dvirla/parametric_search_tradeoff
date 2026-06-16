"""Reusable LLM audit for MuSiQue natural-phrasing rewrites.

A paraphrase of a multi-hop question can fail in two ways that both confound the
phrasing experiments:

  * **leak** — the rewrite names or makes directly identifiable an intermediate
    (bridge) entity, collapsing a reasoning hop;
  * **drift** — the rewrite changes the information need, so it no longer resolves
    to the same gold answer.

`ParaphraseAudit` captures both in a single judge call (one LLM round-trip per
attempt, to stay search/compute frugal). Used by the generate→judge→regenerate
loop in paraphrase_musique_natural.py.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.services.base_agent import BaseAgent


class ParaphraseAudit(BaseModel):
    leaked_hop_indices: list[int]               # bridge hops whose answer the rewrite reveals
    equivalent: bool                            # same information need + same final gold answer
    drift: Literal["none", "narrower", "broader", "different_answer", "other"]
    reasoning: str

    @property
    def ok(self) -> bool:
        return not self.leaked_hop_indices and self.equivalent


AUDIT_PROMPT = """You are auditing a paraphrased ("natural") rewrite of a multi-hop benchmark question. Check two things.

ORIGINAL benchmark question:
{bench_q}

NATURAL rewrite:
{nat_q}

Reasoning hops (each yields a bridge entity; the final hop yields the overall answer):
{hops}

Final gold answer: {gold}

CHECK 1 — LEAK. A faithful rewrite keeps every BRIDGE entity (the non-final hop answers) IMPLICIT, referring to it only by description, never naming it or making it directly identifiable. Report the 0-based indices of any bridge hops whose answer the NATURAL rewrite reveals (empty list if the chain is fully preserved). Rephrasing, politeness, or narrative framing is NOT a leak as long as bridge entities stay implicit.

CHECK 2 — EQUIVALENCE. Does the NATURAL rewrite express the SAME information need and resolve to the SAME final gold answer above? Set equivalent=false if it asks something narrower, broader, or with a different answer. Classify the drift (none/narrower/broader/different_answer/other).

Give brief reasoning naming any revealed entity or describing the drift.
"""


def format_bridge_hops(sub_questions: list[dict]) -> str:
    """Render hops 0..n-2 (bridges) for the audit prompt. `sub_questions` items
    have `hop_index`, `question`, `gold_answer`."""
    bridges = sub_questions[:-1]
    if not bridges:
        return "(no bridge hops — single-hop question)"
    return "\n".join(
        f"- hop {s['hop_index']}: {s['question']}  → bridge entity: {s['gold_answer']}"
        for s in bridges
    )


def make_judge_agent(model_name: str = "gpt-oss:120b", provider_name: str = "ollama",
                     ollama_base_url: str | None = None,
                     output_type: type = ParaphraseAudit) -> BaseAgent:
    return BaseAgent(
        provider_name=provider_name,
        model_name=model_name,
        output_type=output_type,
        use_thinking=False,
        temperature=0.0,
        ollama_base_url=ollama_base_url,
        system_prompt="You are a meticulous QA dataset auditor.",
        agent_name=f"paraphrase_auditor_{model_name}",
    )


async def audit_paraphrase(agent: BaseAgent, bench_q: str, nat_q: str,
                           sub_questions: list[dict], gold: str, num_hops: int) -> ParaphraseAudit:
    """One judge call. Raises on agent failure (caller decides retry policy)."""
    prompt = AUDIT_PROMPT.format(
        bench_q=bench_q, nat_q=nat_q, hops=format_bridge_hops(sub_questions), gold=gold,
    )
    result = await agent.arun(prompt)
    audit: ParaphraseAudit = result.output
    # Clamp to valid bridge indices (final hop can't be a leak).
    audit.leaked_hop_indices = [h for h in audit.leaked_hop_indices if 0 <= h < num_hops - 1]
    return audit


def audit_feedback(audit: ParaphraseAudit, sub_questions: list[dict]) -> str:
    """Turn a failed audit into corrective guidance appended to the next attempt."""
    msgs = []
    if audit.leaked_hop_indices:
        named = []
        by_idx = {s["hop_index"]: s for s in sub_questions}
        for h in audit.leaked_hop_indices:
            g = by_idx.get(h, {}).get("gold_answer")
            named.append(f"hop {h}" + (f" ('{g}')" if g else ""))
        msgs.append(
            "Your previous rewrite REVEALED bridge entities: " + ", ".join(named) +
            ". Do not name or make these identifiable — refer to them only by description.")
    if not audit.equivalent:
        msgs.append(
            f"Your previous rewrite CHANGED the information need (drift: {audit.drift}). "
            "Preserve the exact same question and the same final answer.")
    if audit.reasoning:
        msgs.append(f"Auditor note: {audit.reasoning}")
    return " ".join(msgs)


# ---------------------------------------------------------------------------
# Benchmark-style audit (natural -> formal direction, e.g. FRAMES / ShareChat)
# ---------------------------------------------------------------------------
#
# When the SOURCE is already a naturalistic user question (FRAMES prompts, real
# ShareChat queries) we rewrite TOWARD terse benchmark phrasing. There are no
# bridge hops to leak-check; instead we audit (1) equivalence — same information
# need + same answer — and (2) benchmark-style compliance — that conversational
# framing was actually stripped (otherwise the "formal" condition is contaminated
# by natural framing).


class BenchmarkStyleAudit(BaseModel):
    equivalent: bool                            # same information need + same answer
    drift: Literal["none", "narrower", "broader", "different_answer", "other"]
    is_benchmark_style: bool                    # terse, impersonal, no conversational framing
    style_issues: list[str]                     # concrete framing problems if not benchmark-style
    reasoning: str

    @property
    def ok(self) -> bool:
        return self.equivalent and self.is_benchmark_style


BENCHMARK_AUDIT_PROMPT = """You are auditing a rewrite that converts a natural user question into a terse, formal BENCHMARK-style question. Check two things.

ORIGINAL natural question:
{nat_q}

BENCHMARK rewrite:
{bench_q}

Intended answer: {gold}

CHECK 1 — EQUIVALENCE. Does the BENCHMARK rewrite express the SAME information need and resolve to the SAME intended answer above? Set equivalent=false if it adds, drops, or alters any constraint, or asks something narrower, broader, or with a different answer. Classify the drift (none/narrower/broader/different_answer/other).

CHECK 2 — BENCHMARK STYLE. The rewrite must read like a formal QA-benchmark question: terse, impersonal, self-contained, stating the (possibly multi-hop) question directly, with NO conversational framing — no first person, no politeness, no "can you"/"I'm curious"/"I've been wondering", no greetings or narrative. Set is_benchmark_style=false and list concrete style_issues if any conversational/natural framing remains.

Give brief reasoning."""


async def audit_benchmark_style(agent: BaseAgent, nat_q: str, bench_q: str,
                                gold: str) -> BenchmarkStyleAudit:
    """One judge call for the natural->benchmark direction. Raises on agent failure."""
    prompt = BENCHMARK_AUDIT_PROMPT.format(nat_q=nat_q, bench_q=bench_q, gold=gold)
    result = await agent.arun(prompt)
    return result.output


def benchmark_style_feedback(audit: BenchmarkStyleAudit) -> str:
    """Turn a failed benchmark-style audit into corrective guidance for the next attempt."""
    msgs = []
    if not audit.equivalent:
        msgs.append(
            f"Your previous rewrite CHANGED the information need (drift: {audit.drift}). "
            "Preserve the exact same question and the same answer.")
    if not audit.is_benchmark_style:
        issues = "; ".join(audit.style_issues) if audit.style_issues else "still too conversational"
        msgs.append(
            f"Your previous rewrite is NOT benchmark-style: {issues}. "
            "Make it terse, impersonal, and formal — strip all conversational framing.")
    if audit.reasoning:
        msgs.append(f"Auditor note: {audit.reasoning}")
    return " ".join(msgs)
