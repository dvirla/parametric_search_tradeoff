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


# ---------------------------------------------------------------------------
# Cue-isolation audit (single-cue paraphrase, e.g. FRAMES prompt-cue experiment)
# ---------------------------------------------------------------------------
#
# For the prompt-cue sensitivity experiment we take a NEUTRAL formal anchor and
# rewrite it applying EXACTLY ONE linguistic cue (epistemic stance, first-person
# framing, politeness, structural implicitness, or output constraint). The audit
# must guarantee the variant differs from the anchor ONLY along the target cue and
# is otherwise an equivalent, non-leaking restatement of the same information need
# — otherwise the measured search-behavior difference is confounded by the question
# being genuinely easier/harder rather than just phrased differently.
#
# This invariant holds for EVERY cue, including the "need-relevant" ones
# (structural explicitness, output constraints): structural_implicit may hide the
# reasoning chain but must keep the same hops required and never name/reveal an
# intermediate entity; output_* may change only the requested output length, never
# the question content.


class CueComplianceAudit(BaseModel):
    equivalent: bool                            # same information need + same answer as the anchor
    drift: Literal["none", "narrower", "broader", "different_answer", "other"]
    leaks_intermediate: bool                    # reveals/short-circuits an intermediate reasoning step
    cue_applied: bool                           # clearly exhibits the target cue
    only_target_cue: bool                       # no OTHER cue dimension was altered
    issues: list[str]                           # concrete problems if any check fails
    reasoning: str

    @property
    def ok(self) -> bool:
        return (self.equivalent and not self.leaks_intermediate
                and self.cue_applied and self.only_target_cue)


CUE_AUDIT_PROMPT = """You are auditing a single-cue rewrite. A NEUTRAL anchor question was rewritten to apply EXACTLY ONE linguistic cue and nothing else. Verify the rewrite is a faithful, equivalent restatement that differs from the anchor ONLY along the target cue.

NEUTRAL anchor question:
{anchor}

TARGET cue to apply (and ONLY this):
{cue_description}

OTHER cues that must NOT change (the rewrite must not introduce or alter any of these):
{forbidden}

REWRITE:
{variant}

Intended answer: {gold}

CHECK 1 — EQUIVALENCE. Does the REWRITE express the SAME information need and resolve to the SAME intended answer as the anchor? Set equivalent=false if it adds, drops, or alters any constraint, entity, date, or number, or asks something narrower/broader/with a different answer. Classify drift (none/narrower/broader/different_answer/other).

CHECK 2 — NO LEAK. The rewrite must require the SAME reasoning chain as the anchor. Set leaks_intermediate=true if it reveals, names, or makes directly identifiable an INTERMEDIATE result (a bridge step), or otherwise short-circuits a reasoning hop so the question becomes easier. Merely rephrasing, hiding the chain inside a relative clause, or narrative framing is NOT a leak as long as no intermediate answer is exposed.

CHECK 3 — CUE APPLIED. Does the rewrite clearly exhibit the TARGET cue described above? Set cue_applied=false if the cue is absent or too weak.

CHECK 4 — ONLY THE TARGET CUE. Does the rewrite leave every OTHER cue dimension unchanged from the anchor? Set only_target_cue=false if it also changed a forbidden dimension (e.g. an epistemic-hedge rewrite that also became first-person or added an output-length instruction).

List concrete issues for any failed check and give brief reasoning."""


async def audit_cue_compliance(agent: BaseAgent, anchor: str, variant: str, gold: str,
                               cue_description: str, forbidden: str) -> CueComplianceAudit:
    """One judge call for a single-cue rewrite. Raises on agent failure."""
    prompt = CUE_AUDIT_PROMPT.format(
        anchor=anchor, variant=variant, gold=gold,
        cue_description=cue_description, forbidden=forbidden,
    )
    result = await agent.arun(prompt)
    return result.output


def cue_feedback(audit: CueComplianceAudit) -> str:
    """Turn a failed cue-compliance audit into corrective guidance for the next attempt."""
    msgs = []
    if not audit.equivalent:
        msgs.append(
            f"Your previous rewrite CHANGED the information need (drift: {audit.drift}). "
            "Preserve the exact same question, constraints, and final answer.")
    if audit.leaks_intermediate:
        msgs.append(
            "Your previous rewrite LEAKED or short-circuited an intermediate reasoning step. "
            "Keep every bridge entity implicit — refer to it only by description, never name it.")
    if not audit.cue_applied:
        msgs.append("Your previous rewrite did NOT clearly apply the target cue. Make it more pronounced.")
    if not audit.only_target_cue:
        msgs.append(
            "Your previous rewrite also changed OTHER cue dimensions. Change ONLY the target cue; "
            "leave framing, politeness, structure, and output instructions exactly as in the anchor.")
    if audit.issues:
        msgs.append("Issues: " + "; ".join(audit.issues) + ".")
    if audit.reasoning:
        msgs.append(f"Auditor note: {audit.reasoning}")
    return " ".join(msgs)


# ---------------------------------------------------------------------------
# Strict neutral-anchor audit (validate/rebuild the FRAMES formal "neutral" base)
# ---------------------------------------------------------------------------
#
# The cue-sensitivity experiment measures every cue variant against a NEUTRAL formal
# anchor. If that anchor leaks an intermediate answer (collapsing a reasoning hop) or
# drifts from the original information need, every cue contrast is measured against a
# contaminated control. The original natural->benchmark audit (BenchmarkStyleAudit)
# only checked equivalence + impersonal style — it never checked for leakage. This
# strict audit adds the leak check and a stricter "neutral register" check.
#
# Grounding is the ORIGINAL FRAMES prompt + gold final answer ONLY — judged
# holistically. We deliberately do NOT supply any hop decomposition: FRAMES ships no
# gold per-hop answers, and proxy decompositions are not ground truth, so they would
# only mislead the leak judgement.


class NeutralAnchorAudit(BaseModel):
    equivalent: bool                            # same information need + same final answer
    drift: Literal["none", "narrower", "broader", "different_answer", "other"]
    leaks_intermediate: bool                    # names/reveals an intermediate the original kept implicit
    is_neutral_register: bool                   # terse, impersonal, no epistemic/framing/politeness/output cues
    register_issues: list[str]                  # concrete register problems if not neutral
    reasoning: str

    @property
    def ok(self) -> bool:
        return self.equivalent and not self.leaks_intermediate and self.is_neutral_register


NEUTRAL_ANCHOR_AUDIT_PROMPT = """You are auditing a terse, formal rewrite of a natural multi-hop user question. This rewrite will be used as the NEUTRAL control in a search-behavior experiment, so it must (1) preserve the exact information need, (2) leak no intermediate answer, and (3) be register-neutral. Judge ONLY against the original question and the intended answer below — do not invent a hop decomposition.

ORIGINAL natural question:
{orig_q}

FORMAL rewrite (candidate neutral anchor):
{formal_q}

Intended final answer: {gold}

CHECK 1 — EQUIVALENCE. Does the FORMAL rewrite express the SAME information need and resolve to the SAME intended final answer? Set equivalent=false if it adds, drops, or alters any constraint, entity, date, or number, or asks something narrower, broader, combined with an extra question, or with a different answer. Classify drift (none/narrower/broader/different_answer/other).

CHECK 2 — NO LEAK. The rewrite must require the SAME reasoning chain as the original. The original keeps every INTERMEDIATE result implicit, referring to it only by description (e.g. "the director of the film adaptation of X's second novel"). Set leaks_intermediate=true if the FORMAL rewrite NAMES or makes directly identifiable any intermediate result that the original kept implicit, or otherwise short-circuits / removes a reasoning hop so the question becomes easier. Spelling the chain out explicitly with nested descriptions (each step given by its defining property, not its resolved value) is NOT a leak.

CHECK 3 — NEUTRAL REGISTER. The rewrite must be terse, impersonal, and free of ALL of these surface cues: no expressed confidence (no boosters like "exactly/surely/obviously", no hedges like "I think/maybe/I might be wrong"); no first-person goal/task framing ("I'm trying to", "for my project"); no politeness ("could you", "please"); no answer-length/format instruction ("explain in 2-4 sentences", "just the final answer"). Set is_neutral_register=false and list concrete register_issues for any cue present.

Give brief reasoning naming any leaked entity or drift."""


async def audit_neutral_anchor(agent: BaseAgent, orig_q: str, formal_q: str,
                               gold: str) -> NeutralAnchorAudit:
    """One judge call validating a formal neutral anchor against the original prompt. Raises on failure."""
    prompt = NEUTRAL_ANCHOR_AUDIT_PROMPT.format(orig_q=orig_q, formal_q=formal_q, gold=gold)
    result = await agent.arun(prompt)
    return result.output


def neutral_anchor_feedback(audit: NeutralAnchorAudit) -> str:
    """Turn a failed neutral-anchor audit into corrective guidance for the next attempt."""
    msgs = []
    if not audit.equivalent:
        msgs.append(
            f"Your previous rewrite CHANGED the information need (drift: {audit.drift}). "
            "Preserve the exact same question, constraints, and final answer — do not combine it "
            "with an extra sub-question or ask anything narrower/broader.")
    if audit.leaks_intermediate:
        msgs.append(
            "Your previous rewrite LEAKED an intermediate result that the original kept implicit. "
            "Refer to every intermediate target only by its defining description, never by its "
            "resolved name, and keep the same reasoning hops required.")
    if not audit.is_neutral_register:
        issues = "; ".join(audit.register_issues) if audit.register_issues else "register not neutral"
        msgs.append(
            f"Your previous rewrite is NOT register-neutral: {issues}. Make it terse and impersonal "
            "with no confidence markers, no first-person framing, no politeness, and no output instruction.")
    if audit.reasoning:
        msgs.append(f"Auditor note: {audit.reasoning}")
    return " ".join(msgs)
