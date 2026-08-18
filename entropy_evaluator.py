"""
entropy_evaluator.py
The "Entropy Evaluator" — watches the Text Engine's generation in
real time and decides WHEN something has gone wrong, using three
independent signals (matches what you described in your finalized
design — entropy, drift from facts, AND correctness):

  1. Entropy decay — confidence dropping in a way that signals the
     model has slipped from real reasoning into pattern-repetition
     (NOT just "entropy is currently low", which is often normal).
  2. Repetition — literal n-gram repetition in the generated tokens,
     the visible symptom of the self-reinforcing loop entropy decay
     tends to cause.
  3. Fact drift — the generated text no longer semantically matches
     ANY of the retrieved facts. This is the fix for the blind spot
     we identified earlier: entropy alone can't catch a CONFIDENT
     hallucination, but comparing against retrieved facts can.

None of these three signals alone is reliable enough on its own —
combining them is the actual novelty-relevant design decision here,
worth stating explicitly in the paper.
"""

from collections import deque

import torch
from sentence_transformers import SentenceTransformer, util

from config import (
    DEVICE,
    ENTROPY_WINDOW,
    ENTROPY_DECAY_THRESHOLD,
    LOW_ENTROPY_FLOOR,
)


# ---------------------------------------------------------------------
# Signal 1: Entropy computation + decay detection
# ---------------------------------------------------------------------
def compute_entropy(logits: torch.Tensor) -> float:
    """
    Computes Shannon entropy of the model's next-token distribution
    for a single generation step. logits shape: (vocab_size,).

    Uses log_softmax instead of softmax-then-log for numerical
    stability (avoids log(very-small-number) precision issues).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)
    return entropy.item()


class EntropyTracker:
    """
    Maintains a rolling window of recent entropy values and flags a
    "decay pattern" — NOT just low entropy on its own (that's often
    a perfectly normal confident token, like "the" after "of").

    The condition: entropy has dropped sharply within the window
    AND the current value is below an absolute floor. Both
    conditions together are what distinguish "model is confused and
    collapsing into a rut" from "model is just being reasonably
    confident about an easy next word".
    """

    def __init__(
        self,
        window_size: int = ENTROPY_WINDOW,
        decay_threshold: float = ENTROPY_DECAY_THRESHOLD,
        low_entropy_floor: float = LOW_ENTROPY_FLOOR,
    ):
        self.history = deque(maxlen=window_size)
        self.decay_threshold = decay_threshold
        self.low_entropy_floor = low_entropy_floor

    def update(self, entropy: float) -> dict:
        self.history.append(entropy)

        if len(self.history) < self.history.maxlen:
            # Not enough history yet to judge a "pattern" — a single
            # data point can't show decay.
            return {"entropy": entropy, "decay_detected": False, "reason": None}

        window_max = max(self.history)
        relative_drop = (window_max - entropy) / (window_max + 1e-8)

        decay_detected = (
            relative_drop > self.decay_threshold
            and entropy < self.low_entropy_floor
        )

        reason = None
        if decay_detected:
            reason = (
                f"entropy dropped {relative_drop:.0%} within the last "
                f"{self.history.maxlen} tokens, now at {entropy:.2f} "
                f"(floor: {self.low_entropy_floor})"
            )

        return {"entropy": entropy, "decay_detected": decay_detected, "reason": reason}

    def reset(self):
        self.history.clear()


# ---------------------------------------------------------------------
# Signal 2: Repetition detection
# ---------------------------------------------------------------------
def detect_repetition(token_id_history: list[int], n: int = 3) -> bool:
    """
    Checks whether the last n generated tokens exactly match the n
    tokens immediately before them — i.e. the model just repeated an
    n-gram verbatim. This is the literal, visible symptom of the
    self-reinforcing loop: repeat once -> more likely to repeat
    again -> repeats again.

    n=3 is a starting point — worth tuning. Too small (n=1) will
    flag legitimate repeated words ("the the" almost never happens,
    but single-token repeats of e.g. punctuation could false-
    positive). Too large (n=6+) will miss short repetition loops.
    """
    if len(token_id_history) < 2 * n:
        return False

    last_n = token_id_history[-n:]
    previous_n = token_id_history[-2 * n:-n]
    return last_n == previous_n


# ---------------------------------------------------------------------
# Signal 3: Fact-drift detection
# ---------------------------------------------------------------------
class FactConsistencyChecker:
    """
    Catches the blind spot entropy can't see: a model that is
    CONFIDENTLY wrong. Compares the recently-generated text window
    against the retrieved facts using semantic similarity — if the
    generation has drifted away from everything that was retrieved,
    that's a red flag regardless of how "confident" the token
    probabilities look.

    Uses the same embedding model family as graph_engine.py's
    SemanticIndex for consistency, but kept as a separate instance
    here since this module shouldn't depend on graph_engine directly
    (keeps the files independently testable).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", drift_threshold: float = 0.3):
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.drift_threshold = drift_threshold

    def check(self, generated_text_window: str, retrieved_facts: list[str]) -> dict:
        if not retrieved_facts or not generated_text_window.strip():
            # Nothing to compare against — can't claim drift with no
            # facts to drift away FROM. Fail open, not closed: don't
            # falsely flag drift when we simply have no evidence.
            return {"drift_detected": False, "max_similarity": None, "reason": None}

        gen_embedding = self.model.encode(generated_text_window, convert_to_tensor=True)
        fact_embeddings = self.model.encode(retrieved_facts, convert_to_tensor=True)

        similarities = util.cos_sim(gen_embedding, fact_embeddings)[0]
        max_similarity = similarities.max().item()

        drift_detected = max_similarity < self.drift_threshold
        reason = None
        if drift_detected:
            reason = (
                f"generated text's best match to any retrieved fact is only "
                f"{max_similarity:.2f} similarity (threshold: {self.drift_threshold})"
            )

        return {
            "drift_detected": drift_detected,
            "max_similarity": max_similarity,
            "reason": reason,
        }


# ---------------------------------------------------------------------
# Combined evaluator — ties all three signals together
# ---------------------------------------------------------------------
class GenerationEvaluator:
    """
    The actual "Entropy Evaluator" component from the architecture
    diagram. Call evaluate_step() once per generated token (or every
    few tokens, for efficiency) during generation. It returns whether
    correction should trigger, and WHY — the "why" matters, since
    different failure types could eventually route to different
    corrections (this is the natural place to plug in Aarna's
    failure-classification idea later).
    """

    def __init__(
        self,
        entropy_tracker: EntropyTracker = None,
        fact_checker: FactConsistencyChecker = None,
        repetition_n: int = 3,
    ):
        self.entropy_tracker = entropy_tracker or EntropyTracker()
        self.fact_checker = fact_checker or FactConsistencyChecker()
        self.repetition_n = repetition_n
        self.token_id_history: list[int] = []

    def evaluate_step(
        self,
        logits: torch.Tensor,
        new_token_id: int,
        decoded_window_text: str,
        retrieved_facts: list[str],
    ) -> dict:
        self.token_id_history.append(new_token_id)

        entropy_result = self.entropy_tracker.update(compute_entropy(logits))
        repetition_detected = detect_repetition(self.token_id_history, n=self.repetition_n)
        drift_result = self.fact_checker.check(decoded_window_text, retrieved_facts)

        should_intervene = (
            entropy_result["decay_detected"]
            or repetition_detected
            or drift_result["drift_detected"]
        )

        # Collect whichever reasons actually fired — useful both for
        # debugging and as a "failure type" label for later routing.
        triggered_reasons = []
        if entropy_result["decay_detected"]:
            triggered_reasons.append(("entropy_decay", entropy_result["reason"]))
        if repetition_detected:
            triggered_reasons.append(("repetition", f"repeated last {self.repetition_n} tokens"))
        if drift_result["drift_detected"]:
            triggered_reasons.append(("fact_drift", drift_result["reason"]))

        return {
            "should_intervene": should_intervene,
            "entropy": entropy_result["entropy"],
            "triggered_reasons": triggered_reasons,
        }

    def reset(self):
        self.entropy_tracker.reset()
        self.token_id_history.clear()


# ---------------------------------------------------------------------
# Quick standalone test — doesn't need a GPU or MetaQA, just checks
# the decay-detection LOGIC in isolation with synthetic values.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing EntropyTracker decay detection with synthetic values...")
    tracker = EntropyTracker(window_size=5, decay_threshold=0.4, low_entropy_floor=0.5)

    # Simulate: normal confident generation, then a sharp decay
    # (mimicking the repetition-loop scenario we discussed).
    synthetic_entropies = [2.1, 1.9, 2.0, 1.8, 2.0, 1.5, 0.9, 0.3, 0.1, 0.05]

    for i, e in enumerate(synthetic_entropies):
        result = tracker.update(e)
        flag = "  <-- DECAY DETECTED" if result["decay_detected"] else ""
        print(f"step {i}: entropy={e:.2f}{flag}")
        if result["decay_detected"]:
            print(f"   reason: {result['reason']}")

    print("\nTesting repetition detection...")
    tokens = [10, 20, 30, 10, 20, 30]  # repeats [10,20,30] verbatim
    print(f"tokens={tokens} -> repetition_detected={detect_repetition(tokens, n=3)}")