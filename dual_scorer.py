"""
dual_scorer.py
Layer 2 of the Cross-Modal Conformal Risk Gating design: "Dual
scoring". For each extracted claim, compute TWO independent scores:

  - Parametric score (p): how confident the generating model itself
    is about this claim's text. Computed via teacher-forcing (see
    note below) rather than reusing tokens from the original
    generation loop.

  - Symbolic score (k): does independently-retrieved KG evidence
    support, contradict, or say nothing about this claim? Computed
    via an NLI (Natural Language Inference) entailment model.

These two scores are deliberately kept SEPARATE here (not fused) —
fusion + the self-tuning threshold is risk_gate.py's job, on top of
whatever this file produces. Keeping scoring and gating decoupled
means you can swap out either half independently later (e.g. try a
different NLI model) without touching the other.

--------------------------------------------------------------------
NOTE ON THE PARAMETRIC SCORE — a design choice worth understanding:

The claim text comes from claim_extractor.py's OWN separate
generation call, not the original answer's token stream. So we
don't have pre-computed per-token entropies sitting around for this
exact claim text already.

Instead, we use teacher-forcing: feed (context + claim text) through
the model in ONE forward pass, and read off the model's own
probability distribution at each position corresponding to the claim
— i.e. "if the model were generating this claim right now, how
confident would it be at each token?" This is the same underlying
technique used to compute perplexity, so it's standard practice, not
an improvised shortcut. A v2 improvement worth considering later:
directly reuse the original generation's real entropy values when
the claim maps cleanly onto a contiguous span of the original
output, and fall back to teacher-forcing only when it doesn't.
--------------------------------------------------------------------
"""

import torch
from transformers import pipeline

from config import DEVICE, NLI_MODEL_NAME, NLI_LABELS
from entropy_evaluator import compute_entropy
from claim_extractor import triple_to_sentence


# ---------------------------------------------------------------------
# Parametric score (p)
# ---------------------------------------------------------------------
def compute_parametric_score(model, tokenizer, context: str, claim_text: str) -> dict:
    """
    Teacher-forces the claim text after the context, and returns the
    model's own average entropy across the claim's tokens — lower
    means the model was confident generating this exact text, higher
    means it was uncertain (a useful signal on its own, and a
    genuinely different failure mode than the NLI check below).
    """
    context_ids = tokenizer(context, return_tensors="pt").to(DEVICE)
    full_ids = tokenizer(context + " " + claim_text, return_tensors="pt").to(DEVICE)

    context_len = context_ids["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**full_ids)

    logits = outputs.logits[0]  # shape: (seq_len, vocab_size)

    # Logits at position i predict the token at position i+1, so the
    # logits that "explain" the claim's tokens start one position
    # before the claim actually begins.
    claim_token_ids = full_ids["input_ids"][0][context_len:]
    relevant_logits = logits[context_len - 1: -1]

    if len(claim_token_ids) == 0:
        return {"parametric_score": None, "mean_entropy": None}

    entropies = [compute_entropy(relevant_logits[i]) for i in range(len(claim_token_ids))]
    mean_entropy = sum(entropies) / len(entropies)

    return {"parametric_score": mean_entropy, "mean_entropy": mean_entropy}


# ---------------------------------------------------------------------
# Symbolic score (k)
# ---------------------------------------------------------------------
class EntailmentScorer:
    """
    Wraps a cross-encoder NLI model: given a (premise, hypothesis)
    pair, classifies their relationship as entailment / contradiction
    / neutral, with a confidence value.

    Here: premise = a piece of retrieved KG evidence, hypothesis =
    the claim being checked. "Does this evidence support this claim?"
    """

    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None,  # return scores for ALL labels, not just the top one
        )

    def score_pair(self, evidence: str, claim: str) -> dict:
        """
        Returns the full label distribution for one (evidence, claim)
        pair. CAVEAT (same as flagged in config.py): verify NLI_LABELS
        order against this specific model's card — different NLI
        checkpoints don't always agree on label-to-index mapping, and
        silently trusting the wrong order would corrupt every k score
        downstream without any visible error.
        """
        result = self.pipeline({"text": evidence, "text_pair": claim})
        # result is a list of {"label": ..., "score": ...} dicts
        return {item["label"].lower(): item["score"] for item in result}

    def score_claim(self, claim_sentence: str, retrieved_facts: list[str]) -> dict:
        """
        Checks the claim against EVERY retrieved fact, and keeps the
        strongest verdict found (the fact that most clearly supports
        OR most clearly contradicts). "No evidence" is a first-class
        outcome, not an error — matches the report's explicit design
        choice, since a missing KG entry is meaningfully different
        from a KG entry that actively disagrees.
        """
        if not retrieved_facts:
            return {"verdict": "no_evidence", "confidence": None, "best_evidence": None}

        best = {"verdict": None, "confidence": -1.0, "best_evidence": None}

        for fact in retrieved_facts:
            scores = self.score_pair(fact, claim_sentence)
            entail_score = scores.get("entailment", 0.0)
            contradict_score = scores.get("contradiction", 0.0)

            # Track whichever single (evidence, verdict) pair is most
            # decisive so far — highest confidence in EITHER direction.
            top_score = max(entail_score, contradict_score)
            if top_score > best["confidence"]:
                verdict = "entailment" if entail_score >= contradict_score else "contradiction"
                best = {"verdict": verdict, "confidence": top_score, "best_evidence": fact}

        return best


# ---------------------------------------------------------------------
# Combined per-claim scoring
# ---------------------------------------------------------------------
def score_claim(
    model,
    tokenizer,
    entailment_scorer: EntailmentScorer,
    context: str,
    claim_triple: tuple[str, str, str],
    retrieved_facts: list[str],
) -> dict:
    """
    The main entry point: given one extracted claim, returns BOTH
    scores, ready to be handed to risk_gate.py's fusion step.
    """
    claim_sentence = triple_to_sentence(claim_triple)

    p_result = compute_parametric_score(model, tokenizer, context, claim_sentence)
    k_result = entailment_scorer.score_claim(claim_sentence, retrieved_facts)

    return {
        "claim": claim_triple,
        "claim_sentence": claim_sentence,
        "p": p_result["parametric_score"],
        "k_verdict": k_result["verdict"],
        "k_confidence": k_result["confidence"],
        "k_evidence": k_result["best_evidence"],
    }


# ---------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from attention_injector import load_model_and_tokenizer

    print("Loading model + NLI scorer (this downloads the NLI model on first run)...")
    model, tokenizer = load_model_and_tokenizer()
    entailment_scorer = EntailmentScorer()

    context = "Question: Who directed the movie written by John Balderston?"
    claim = ("Kismet", "directed_by", "Andrew Marton")

    # Case 1: supporting evidence
    supporting_facts = ["Kismet directed by Andrew Marton"]
    result = score_claim(model, tokenizer, entailment_scorer, context, claim, supporting_facts)
    print(f"\nWith supporting evidence:\n{result}")

    # Case 2: no evidence at all
    result_no_evidence = score_claim(model, tokenizer, entailment_scorer, context, claim, [])
    print(f"\nWith no evidence:\n{result_no_evidence}")

    # Case 3: contradicting evidence
    contradicting_facts = ["Kismet directed by William Dieterle"]
    result_contradict = score_claim(model, tokenizer, entailment_scorer, context, claim, contradicting_facts)
    print(f"\nWith contradicting evidence:\n{result_contradict}")