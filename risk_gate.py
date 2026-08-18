"""
risk_gate.py
Layer 3 of the Cross-Modal Conformal Risk Gating design: "Cross-
modal risk fusion and gating". Takes the (p, k) scores from
dual_scorer.py, combines them into ONE fused risk number, compares
against a threshold that SELF-TUNES via Adaptive Conformal Inference
(ACI), and produces the routing decision: accept / local_fix /
escalate.

--------------------------------------------------------------------
TWO HONEST DESIGN NOTES BEFORE THE CODE:

1. BOOTSTRAPPING PROBLEM: the report suggests logistic regression to
   fuse (p, k) into one score — but a logistic regression needs
   LABELED training data (known-error vs known-correct claims),
   which you won't have on day one. This file includes a simple
   HAND-WEIGHTED heuristic as the default (used before you've
   collected any labeled resolved cases), with the logistic
   regression as an upgrade path once real labels exist. Don't
   mistake the heuristic's starting weights for a tuned result.

2. ACI SIGN CONVENTION IS GENUINELY AMBIGUOUS FROM THE REPORT ALONE:
   the formula lambda_{t+1} = lambda_t + gamma*(r_t - alpha) is
   given, but WHICH DIRECTION makes the gate stricter vs. more
   permissive depends on exactly how r_t is defined operationally,
   and the report doesn't fully pin this down. I've implemented a
   specific, reasoned interpretation below (see AdaptiveThreshold's
   docstring) — but you should EMPIRICALLY VERIFY the direction once
   you have real resolved cases: if the threshold visibly drifts the
   wrong way (getting more permissive right when it should be
   tightening), flip the sign in the update. Flagging this now is
   much better than debugging it silently later.
--------------------------------------------------------------------
"""

import math
from collections import deque

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import ACI_TARGET_ERROR_RATE, ACI_STEP_SIZE, ACI_INITIAL_LAMBDA


# ---------------------------------------------------------------------
# Risk fusion: combine (p, k) into one number
# ---------------------------------------------------------------------
class RiskFusionModel:
    """
    Combines the parametric score (p, mean entropy — higher means
    the model itself was less confident) and the symbolic score
    (k_result, an entailment verdict + confidence) into one fused
    risk score in [0, 1], where higher = riskier.

    Starts with a hand-weighted heuristic (arbitrary but reasoned
    starting weights — NOT a tuned result). Call .fit() once you've
    collected labeled (p, k_result, was_actually_wrong) examples from
    resolved cases, and it switches to the trained logistic
    regression automatically.
    """

    # Rough normalization constant for entropy — approximates
    # log(vocab_size) for a typical LLM tokenizer. This is an
    # approximation, not an exact figure for Qwen2.5's specific
    # vocabulary — fine for squashing entropy into a roughly [0, 1]
    # range, not something to treat as precise.
    APPROX_MAX_ENTROPY = 10.0

    def __init__(self):
        self.model = LogisticRegression()
        self.is_fitted = False

    def _featurize(self, p: float, k_result: dict) -> np.ndarray:
        normalized_entropy = min((p or 0.0) / self.APPROX_MAX_ENTROPY, 1.0)

        verdict = k_result.get("k_verdict") or k_result.get("verdict")
        confidence = k_result.get("k_confidence") or k_result.get("confidence") or 0.0

        contradiction_conf = confidence if verdict == "contradiction" else 0.0
        entailment_conf = confidence if verdict == "entailment" else 0.0
        no_evidence_flag = 1.0 if verdict == "no_evidence" else 0.0

        return np.array([[normalized_entropy, contradiction_conf, entailment_conf, no_evidence_flag]])

    def _heuristic_risk(self, features: np.ndarray) -> float:
        """
        Hand-weighted fallback, used before any labeled data exists.
        Weights are a reasoned starting guess, NOT a result:
          - higher entropy -> more risk (model itself unsure)
          - contradiction -> strongly increases risk
          - entailment -> strongly decreases risk
          - no evidence -> moderately increases risk (genuine
            uncertainty, distinct from active contradiction)
        """
        normalized_entropy, contradiction_conf, entailment_conf, no_evidence_flag = features[0]
        raw_score = (
            0.4 * normalized_entropy
            + 0.8 * contradiction_conf
            - 0.6 * entailment_conf
            + 0.3 * no_evidence_flag
        )
        return 1 / (1 + math.exp(-4 * (raw_score - 0.15)))  # sigmoid, centered arbitrarily

    def predict_risk(self, p: float, k_result: dict) -> float:
        features = self._featurize(p, k_result)
        if self.is_fitted:
            return float(self.model.predict_proba(features)[0][1])
        return self._heuristic_risk(features)

    def fit(self, scored_claims: list[dict], labels: list[int]):
        """
        scored_claims: list of dicts from dual_scorer.score_claim().
        labels: 1 if the claim was actually wrong (confirmed via
        resolution), 0 if it was actually correct.

        Once you have even a modest labeled set (tens of resolved
        cases), call this to upgrade from the heuristic to a real
        fitted model.
        """
        X = np.vstack([self._featurize(c["p"], c) for c in scored_claims])
        self.model.fit(X, labels)
        self.is_fitted = True


# ---------------------------------------------------------------------
# The self-tuning threshold (ACI)
# ---------------------------------------------------------------------
class AdaptiveThreshold:
    """
    Implements a self-tuning threshold, following the report's ACI
    formula IN SPIRIT — the exact sign was corrected after running
    the standalone test below and catching a real bug (see update()'s
    comment for the full story).

    r_t is defined per resolved case as an indicator of REALIZED,
    UNDETECTED risk — among claims that were ACCEPTED (fused_risk <=
    lambda_t, left untouched), was this particular one later found
    to actually be wrong? r_t = 1 if yes, 0 if no.

    Verified behavior (run this file directly to reproduce): when
    accepted claims keep turning out wrong, lambda_t correctly FALLS
    over successive updates, which — under decide()'s "accept if
    fused_risk <= lambda" rule — makes the gate stricter (fewer
    things get accepted untouched). This is the behavior you want,
    and it's now confirmed by actually running the code, not just
    reasoned about.
    """

    def __init__(
        self,
        initial_lambda: float = ACI_INITIAL_LAMBDA,
        target_error_rate: float = ACI_TARGET_ERROR_RATE,
        step_size: float = ACI_STEP_SIZE,
    ):
        self.lambda_t = initial_lambda
        self.alpha = target_error_rate
        self.gamma = step_size
        self.history = deque(maxlen=200)  # for inspecting convergence later

    def update(self, r_t: float):
        # NOTE: this uses (alpha - r_t), the OPPOSITE sign from the
        # report's literal formula (r_t - alpha). This was corrected
        # after actually running the standalone test below and
        # observing the bug directly: under decide()'s rule ("accept
        # if fused_risk <= lambda"), a HIGHER lambda is MORE
        # permissive. Using (r_t - alpha) caused lambda to rise when
        # accepted claims kept turning out wrong — exactly backwards.
        # This sign makes lambda correctly FALL (get stricter) when
        # too many accepted claims are wrong. If you change decide()'s
        # accept/flag direction, re-derive this sign — don't assume
        # it still holds.
        self.lambda_t = self.lambda_t + self.gamma * (self.alpha - r_t)
        self.lambda_t = min(max(self.lambda_t, 0.0), 1.0)  # clip to valid risk-score range
        self.history.append(self.lambda_t)

    def get_threshold(self) -> float:
        return self.lambda_t


# ---------------------------------------------------------------------
# The combined gate: fusion + threshold + routing decision
# ---------------------------------------------------------------------
class RiskGate:
    """
    Ties fusion and the adaptive threshold together into the actual
    three-way routing decision, following the report's Layer 4 rule
    that "no evidence" claims can't be locally fixed (nothing to fix
    WITH) and must go straight to escalation if flagged at all.
    """

    def __init__(self, fusion_model: RiskFusionModel = None, threshold: AdaptiveThreshold = None):
        self.fusion_model = fusion_model or RiskFusionModel()
        self.threshold = threshold or AdaptiveThreshold()

    def decide(self, p: float, k_result: dict) -> dict:
        fused_risk = self.fusion_model.predict_risk(p, k_result)
        current_lambda = self.threshold.get_threshold()

        verdict = k_result.get("k_verdict") or k_result.get("verdict")

        if fused_risk <= current_lambda:
            route = "accept"
        elif verdict == "no_evidence":
            route = "escalate"  # nothing to locally fix WITH — report's explicit Layer 4 rule
        else:
            route = "local_fix"

        return {
            "route": route,
            "fused_risk": fused_risk,
            "lambda_used": current_lambda,
        }

    def record_resolution(self, route_taken: str, was_actually_wrong: bool):
        """
        Call this once ground truth becomes known for a resolved
        case (after local_fix verification or escalation). Only
        ACCEPTED claims feed the threshold update, per this file's
        r_t definition above — a flagged claim being wrong isn't a
        threshold failure, since the gate already caught it.
        """
        if route_taken == "accept":
            r_t = 1.0 if was_actually_wrong else 0.0
            self.threshold.update(r_t)


# ---------------------------------------------------------------------
# Quick standalone test — no model/GPU needed, just checks the
# fusion + gating LOGIC with synthetic score inputs.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    gate = RiskGate()

    print("Test 1: confident model + supporting evidence -> expect accept")
    result = gate.decide(p=0.3, k_result={"k_verdict": "entailment", "k_confidence": 0.9})
    print(f"  {result}")

    print("\nTest 2: confident model + contradicting evidence -> expect local_fix")
    result = gate.decide(p=0.3, k_result={"k_verdict": "contradiction", "k_confidence": 0.9})
    print(f"  {result}")

    print("\nTest 3: uncertain model + no evidence -> expect escalate")
    result = gate.decide(p=3.0, k_result={"k_verdict": "no_evidence", "k_confidence": None})
    print(f"  {result}")

    print("\nSimulating threshold drift over 20 resolved 'accept' cases, all later found WRONG:")
    for i in range(20):
        gate.record_resolution("accept", was_actually_wrong=True)
        if i % 5 == 0:
            print(f"  after {i+1} resolutions: lambda = {gate.threshold.get_threshold():.3f}")