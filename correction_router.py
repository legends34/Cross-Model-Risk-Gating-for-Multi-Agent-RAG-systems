"""
correction_router.py
Layers 4-6 of the Cross-Modal Conformal Risk Gating design:
  - Layer 4: Propose-then-verify correction (for "local_fix" claims)
  - Layer 5: Escalation (surgical rollback, then full re-verification
    as a last resort)
  - Layer 6: Provenance propagation (tagging every claim with how it
    was handled, for downstream trust-weighting)

--------------------------------------------------------------------
IMPORTANT DISTINCTION FROM attention_injector.py — read this first:

This file's corrections happen AFTER generation is complete, at the
CLAIM level (post-hoc text editing on extracted facts). This is a
DIFFERENT correction pathway from attention_injector.py's live
KV-cache injection, which happens DURING generation, at the TOKEN
level. Your original dual-engine pitch used the token-level
approach; this report's design uses the claim-level approach. Both
are legitimate — they intervene at different points and different
granularities, and your team should discuss which one is the actual
target for the paper (or whether to compare both as an ablation,
which would itself make an interesting experiment).
--------------------------------------------------------------------

HONEST SIMPLIFICATION: the report's Layer 5 "cross-agent consensus
check" assumes multiple cooperating agents genuinely disagreeing
(that's Aarna's original multi-agent framework). This codebase
doesn't yet implement real multi-agent orchestration, so
full_reverification() below is a simplified stand-in: a full,
independent regeneration of the answer, rather than a genuine
multi-agent consensus vote. Worth flagging explicitly to the team
rather than quietly pretending this is the "real" thing.
--------------------------------------------------------------------
"""

import torch

from claim_extractor import _parse_triples, triple_to_sentence
from dual_scorer import score_claim


# ---------------------------------------------------------------------
# Layer 4: Propose-then-verify correction (GraphCorrect pattern)
# ---------------------------------------------------------------------
def propose_correction(model, tokenizer, claim_sentence: str, evidence: str) -> tuple:
    """
    Two-pass correction, as specified in the report:
      Pass 1: given the evidence, what SHOULD the claim say?
      Pass 2: splice that correction back in as natural, readable text.

    Returns (corrected_triple, corrected_sentence). Either may be
    None if the model's output couldn't be parsed — callers must
    handle that (treat as a failed local fix, fall through to escalation).
    """
    # --- Pass 1: derive the corrected fact from evidence ---
    pass1_prompt = (
        f"Evidence: {evidence}\n"
        f"Original (possibly incorrect) claim: {claim_sentence}\n"
        f"Based ONLY on the evidence above, state the corrected fact "
        f"as a single line: subject|relation|object"
    )
    messages = [{"role": "user", "content": pass1_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Reusing claim_extractor's parser (an internal/private helper,
    # imported directly since this stays within the same project —
    # noting this rather than pretending it's a fully public API).
    parsed = _parse_triples(raw_output)
    if not parsed:
        return None, None

    corrected_triple = parsed[0]
    corrected_sentence = triple_to_sentence(corrected_triple)
    return corrected_triple, corrected_sentence


def verify_correction(model, tokenizer, entailment_scorer, gate, context, corrected_triple, retrieved_facts) -> dict:
    """
    "Re-score the corrected claim before accepting it — correction is
    never accepted on the first pass alone." Runs the corrected claim
    back through the SAME dual-scoring + gating pipeline used for the
    original claim, and only counts the fix as successful if the gate
    now says "accept".
    """
    rescored = score_claim(model, tokenizer, entailment_scorer, context, corrected_triple, retrieved_facts)
    decision = gate.decide(rescored["p"], rescored)
    return {"rescored": rescored, "decision": decision, "success": decision["route"] == "accept"}


# ---------------------------------------------------------------------
# Layer 5: Escalation
# ---------------------------------------------------------------------
def surgical_rollback(model, tokenizer, retriever, claim_sentence: str, original_query: str) -> list:
    """
    Cheaper escalation attempt before a full re-run: try a WIDER
    retrieval (more hops, more semantic candidates) specifically for
    this one claim, in case the original retrieval simply didn't
    look hard enough. Only if this still comes up empty does the
    caller move to full_reverification.
    """
    wider_retriever_query = f"{original_query} {claim_sentence}"
    # Reuses the existing retriever's semantic search directly with a
    # larger top_k for this one-off wider attempt, rather than
    # permanently changing the retriever's default settings.
    facts = retriever.semantic_index.search(wider_retriever_query, top_k=10)
    fact_sentences = [triple_to_sentence(t) for t in facts]
    return fact_sentences


def full_reverification(model, tokenizer, original_query: str) -> str:
    """
    Last resort: independently regenerate an answer to the original
    query from scratch, with no special conditioning. This is the
    simplified stand-in for "cross-agent consensus" noted in the
    module docstring — a genuine second opinion, just not from a
    literal second cooperating agent (yet).
    """
    messages = [{"role": "user", "content": original_query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------
# The full router — ties Layers 4, 5, and 6 together per claim
# ---------------------------------------------------------------------
def route_claim(
    model,
    tokenizer,
    retriever,
    entailment_scorer,
    gate,
    scored_claim: dict,
    context: str,
    original_query: str,
) -> dict:
    """
    Given one already-scored claim (from dual_scorer.score_claim) and
    the gate's decision, executes the full correction/escalation flow
    and returns a provenance-tagged result — this is what gets passed
    to whichever agent or step consumes the claim next (Layer 6).
    """
    decision = gate.decide(scored_claim["p"], scored_claim)
    provenance = {
        "claim": scored_claim["claim"],
        "original_sentence": scored_claim["claim_sentence"],
        "fused_risk": decision["fused_risk"],
        "lambda_used": decision["lambda_used"],
    }

    # --- accept: leave untouched ---
    if decision["route"] == "accept":
        provenance.update({"final_sentence": scored_claim["claim_sentence"], "status": "agreement"})
        # Agreement between independent sources IS free labeled data,
        # per the report — feed it to calibration immediately, no
        # need to wait for external ground truth on this one.
        if scored_claim["k_verdict"] == "entailment":
            gate.record_resolution("accept", was_actually_wrong=False)
        return provenance

    # --- local_fix: propose-then-verify ---
    if decision["route"] == "local_fix":
        evidence = scored_claim["k_evidence"] or ""
        corrected_triple, corrected_sentence = propose_correction(
            model, tokenizer, scored_claim["claim_sentence"], evidence
        )

        if corrected_triple is not None:
            retrieved_facts = [evidence] if evidence else []
            verification = verify_correction(
                model, tokenizer, entailment_scorer, gate, context, corrected_triple, retrieved_facts
            )
            if verification["success"]:
                provenance.update({"final_sentence": corrected_sentence, "status": "corrected"})
                return provenance
        # Correction proposed but failed re-verification (or couldn't
        # be parsed at all) — fall through to escalation rather than
        # silently keeping an unverified "fix".

    # --- escalate: surgical rollback, then full re-verification ---
    wider_facts = surgical_rollback(model, tokenizer, retriever, scored_claim["claim_sentence"], original_query)
    if wider_facts:
        rescored = score_claim(model, tokenizer, entailment_scorer, context, scored_claim["claim"], wider_facts)
        redecision = gate.decide(rescored["p"], rescored)
        if redecision["route"] == "accept":
            provenance.update({"final_sentence": scored_claim["claim_sentence"], "status": "escalated_resolved"})
            return provenance

    # Last resort — full independent regeneration
    reverified_answer = full_reverification(model, tokenizer, original_query)
    provenance.update({"final_sentence": reverified_answer, "status": "escalated_unresolved"})
    return provenance


# ---------------------------------------------------------------------
# Quick manual test — needs the full model + retriever + NLI scorer,
# so this is a genuine integration test, not a lightweight unit test.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from attention_injector import load_model_and_tokenizer
    from graph_engine import HybridGraphRetriever
    from dual_scorer import EntailmentScorer
    from risk_gate import RiskGate

    print("Loading model, retriever, NLI scorer, gate...")
    model, tokenizer = load_model_and_tokenizer()
    retriever = HybridGraphRetriever("data/metaqa/kb.txt", max_hops=2)
    entailment_scorer = EntailmentScorer()
    gate = RiskGate()

    query = "Who directed the movie written by John Balderston?"
    context = f"Question: {query}"

    # Deliberately wrong claim, to exercise the local_fix path
    wrong_claim = ("Kismet", "directed_by", "William Dieterle")
    retrieved_facts = retriever.retrieve_as_facts(query)

    scored = score_claim(model, tokenizer, entailment_scorer, context, wrong_claim, retrieved_facts)
    print(f"\nInitial score: {scored}")

    result = route_claim(model, tokenizer, retriever, entailment_scorer, gate, scored, context, query)
    print(f"\nRouting result: {result}")