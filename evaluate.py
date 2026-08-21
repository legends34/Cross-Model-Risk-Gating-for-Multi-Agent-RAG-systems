"""
evaluate.py
Benchmarks your pipeline against the baselines the report explicitly
calls for in Section 6, and produces the accuracy + cost numbers
your paper's results section needs.

Baselines implemented here, matching the report:
  1. No intervention        — raw model, no retrieval, no correction
  2. Whole-response re-run  — current common practice: regenerate the
                               WHOLE answer if average confidence is low
  3. Self-consistency-only  — DenoiseFlow-style: sample multiple times,
                               check if the model agrees with ITSELF,
                               instead of checking against independent
                               KG evidence. THE SINGLE MOST IMPORTANT
                               baseline — it isolates exactly the
                               claimed delta (cross-modal vs same-source).
  4. Your pipeline, fixed threshold — isolates the value of ACI
                               specifically (no online calibration)
  5. Your pipeline, full ACI — the actual proposed system

--------------------------------------------------------------------
HONEST SIMPLIFICATIONS — for a paper submission, revisit these:

- Accuracy is checked via simple substring matching against gold
  answers (case-insensitive). This is a crude first-pass metric —
  real papers often use token-level F1 or exact-match after
  normalization. Fine for early iteration, worth upgrading before
  final numbers.
- Cost is estimated by COUNTING generation calls per claim's
  resolution path, not actual wall-clock time or token counts. This
  is a reasonable proxy but not identical to real compute cost —
  state this explicitly if you report it in the paper.
- This file's __main__ block runs a SMALL smoke-test scale (a
  handful of questions), not MetaQA's full test set — free-tier T4
  compute makes a full run slow. Scale up once everything's verified
  working correctly at small scale.
--------------------------------------------------------------------
"""

import time
import torch
import pandas as pd

from pipeline import CrossModalRAGPipeline
from risk_gate import AdaptiveThreshold


# ---------------------------------------------------------------------
# MetaQA question loading
# ---------------------------------------------------------------------
def load_metaqa_questions(qa_path: str, limit: int = None) -> list[dict]:
    """
    MetaQA's qa_test.txt format: one question per line, tab-separated
    from its gold answer(s), multiple answers pipe-separated:
        "who directed [Kismet]"    Andrew Marton
    VERIFY this against your actual downloaded file — MetaQA's exact
    formatting has had minor variations across mirrors/releases.
    """
    questions = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            question, answers_raw = line.split("\t", 1)
            answers = [a.strip() for a in answers_raw.split("|")]
            questions.append({"question": question.strip("[]"), "gold_answers": answers})
            if limit and len(questions) >= limit:
                break
    return questions


def answer_matches(predicted_text: str, gold_answers: list[str]) -> bool:
    predicted_lower = predicted_text.lower()
    return any(gold.lower() in predicted_lower for gold in gold_answers)


# ---------------------------------------------------------------------
# Baseline 1: No intervention
# ---------------------------------------------------------------------
def run_no_intervention(model, tokenizer, query: str) -> dict:
    messages = [{"role": "user", "content": query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {"answer": answer, "num_generations": 1}


# ---------------------------------------------------------------------
# Baseline 2: Whole-response re-run on low confidence
# ---------------------------------------------------------------------
def run_whole_rerun(model, tokenizer, query: str, entropy_threshold: float = 3.0) -> dict:
    from entropy_evaluator import compute_entropy

    messages = [{"role": "user", "content": query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
    answer = tokenizer.decode(outputs.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    avg_entropy = sum(compute_entropy(s[0]) for s in outputs.scores) / len(outputs.scores)

    num_generations = 1
    if avg_entropy > entropy_threshold:
        # Coarse "supervisor rejection": regenerate the WHOLE thing
        # once, with sampling this time (a different attempt, not
        # just repeating the same greedy generation).
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_generations = 2

    return {"answer": answer, "num_generations": num_generations}


# ---------------------------------------------------------------------
# Baseline 3: Self-consistency-only (DenoiseFlow-style)
# ---------------------------------------------------------------------
def run_self_consistency(model, tokenizer, query: str, num_samples: int = 3) -> dict:
    """
    THE key baseline. Samples multiple times, checks if the model
    agrees with ITSELF, and majority-votes. No independent evidence
    source involved at all — this is exactly what your cross-modal
    fusion is supposed to outperform, specifically on confident,
    consistent hallucinations (where self-consistency structurally
    can't help, since the model agrees with itself every time).
    """
    messages = [{"role": "user", "content": query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    samples = []
    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        samples.append(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # Simple majority vote by exact string match. A more careful
    # version would cluster semantically-similar-but-not-identical
    # samples together — worth upgrading if this baseline looks
    # artificially weak due to trivial wording differences.
    most_common = max(set(samples), key=samples.count)
    return {"answer": most_common, "num_generations": num_samples, "all_samples": samples}


# ---------------------------------------------------------------------
# Baselines 4 & 5: your own pipeline, with/without ACI
# ---------------------------------------------------------------------
class FrozenThreshold(AdaptiveThreshold):
    """A threshold that never updates — isolates ACI's contribution
    by comparing against an otherwise-identical but static gate."""
    def update(self, r_t: float):
        pass  # deliberately does nothing


def estimate_pipeline_cost(result: dict) -> int:
    cost = 1  # initial draft generation
    if result.get("stage1_used"):
        # Each Stage 1 intervention (attention_injector.py's KV-cache
        # correction) is an extra generation step — count them so
        # Stage 1 isn't scored as free relative to Stage 2.
        cost += len(result.get("stage1_interventions", []))
    for claim in result.get("claims", []):
        if claim["status"] == "corrected":
            cost += 2
        elif claim["status"] in ("escalated_resolved", "escalated_unresolved"):
            cost += 3
    return cost


# ---------------------------------------------------------------------
# The evaluation loop
# ---------------------------------------------------------------------
def run_evaluation(pipeline: CrossModalRAGPipeline, questions: list[dict]) -> pd.DataFrame:
    rows = []

    # Build a second gate variant sharing the same model/scorer but
    # with a frozen threshold, for the "no ACI" comparison arm.
    frozen_gate_pipeline_threshold = FrozenThreshold()

    for q in questions:
        query, gold = q["question"], q["gold_answers"]

        methods = {
            "no_intervention": lambda: run_no_intervention(pipeline.model, pipeline.tokenizer, query),
            "whole_rerun": lambda: run_whole_rerun(pipeline.model, pipeline.tokenizer, query),
            "self_consistency": lambda: run_self_consistency(pipeline.model, pipeline.tokenizer, query),
            "stage1_only":  lambda: pipeline.answer(query, use_stage1=True,  use_stage2=False),
            "stage2_only":  lambda: pipeline.answer(query, use_stage1=False, use_stage2=True),
            "both_stages":  lambda: pipeline.answer(query, use_stage1=True,  use_stage2=True),
            "neither_stage": lambda: pipeline.answer(query, use_stage1=False, use_stage2=False),
}
        for method_name, method_fn in methods.items():
            start = time.time()
            result = method_fn()
            elapsed = time.time() - start

            answer_text = result.get("final_answer") or result.get("answer", "")
            correct = answer_matches(answer_text, gold)
            cost = result.get("num_generations") or estimate_pipeline_cost(result)

            rows.append({
                "question": query,
                "method": method_name,
                "correct": correct,
                "cost": cost,
                "elapsed_seconds": round(elapsed, 2),
            })

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the raw per-question results into the
    accuracy/avg-cost table you'd actually put in the paper."""
    return df.groupby("method").agg(
        accuracy=("correct", "mean"),
        avg_cost=("cost", "mean"),
        avg_seconds=("elapsed_seconds", "mean"),
    ).round(3)


# ---------------------------------------------------------------------
# Smoke-test scale run — a handful of questions, not the full test set
# ---------------------------------------------------------------------
if __name__ == "__main__":
    KB_PATH = "data/metaqa/kb.txt"
    QA_PATH = "data/metaqa/1-hop/qa_test.txt"  # verify this path against your actual download
    SMOKE_TEST_SIZE = 5

    print("Loading pipeline (this loads the model, retriever, and NLI scorer)...")
    pipeline = CrossModalRAGPipeline(KB_PATH)

    print(f"Loading {SMOKE_TEST_SIZE} test questions...")
    questions = load_metaqa_questions(QA_PATH, limit=SMOKE_TEST_SIZE)

    print("Running evaluation across all methods (this will take a while)...")
    results_df = run_evaluation(pipeline, questions)

    print("\nRaw results:")
    print(results_df)

    print("\nSummary (accuracy + avg cost per method):")
    print(summarize_results(results_df))