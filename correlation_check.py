"""
correlation_check.py
Step 3 of the plan: checks whether the parametric score (p) and the
symbolic/KG score (k) actually fail on DIFFERENT wrong claims, or
whether they're catching the same errors — the premise Section 3.2
of the report depends on.

Run as: python correlation_check.py
"""

import pandas as pd
from scipy.stats import pearsonr

from pipeline import CrossModalRAGPipeline
from evaluate import load_metaqa_questions, answer_matches
from claim_extractor import extract_claims
from dual_scorer import score_claim


KB_PATH = "data/metaqa/kb.txt"
QA_PATH = "data/metaqa/1-hop/qa_test.txt"  # 1-hop only — see caveat below
SAMPLE_SIZE = 40


def collect_records(pipeline, questions):
    records = []
    for q in questions:
        query, gold = q["question"], q["gold_answers"]

        # Plain draft, no Stage 1/Stage 2 — we just need claims to score.
        result = pipeline.answer(query, use_stage1=False, use_stage2=False)
        draft = result["draft_answer"]

        claims = extract_claims(pipeline.model, pipeline.tokenizer, draft)
        for claim_triple in claims:
            facts = pipeline.retriever.retrieve_as_facts(" ".join(claim_triple))
            scored = score_claim(
                pipeline.model, pipeline.tokenizer, pipeline.entailment_scorer,
                f"Question: {query}", claim_triple, facts,
            )
            # Single-hop heuristic: the triple's object slot is the
            # answer-bearing slot. See caveat for multi-hop questions.
            is_wrong = not answer_matches(claim_triple[-1], gold)

            records.append({
                "question": query,
                "p": scored["p"],
                "k_verdict": scored["k_verdict"],
                "is_wrong": is_wrong,
            })
    return records


def analyze(df: pd.DataFrame):
    df["p_flags"] = df["p"] > df["p"].median()
    df["k_flags"] = df["k_verdict"] == "contradiction"

    wrong = df[df["is_wrong"]]
    both   = ((wrong.p_flags) & (wrong.k_flags)).sum()
    only_p = ((wrong.p_flags) & (~wrong.k_flags)).sum()
    only_k = ((~wrong.p_flags) & (wrong.k_flags)).sum()
    missed = ((~wrong.p_flags) & (~wrong.k_flags)).sum()

    print(f"\nAmong claims that were actually wrong ({len(wrong)} total):")
    print(f"  caught by both p and k : {both}")
    print(f"  caught by p only       : {only_p}")
    print(f"  caught by k only       : {only_k}")
    print(f"  missed by both         : {missed}")

    corr, _ = pearsonr(df.p_flags.astype(int), df.k_flags.astype(int))
    print(f"\nCorrelation between p-flag and k-flag (all claims): {corr:.3f}")
    print("(Low/near-zero correlation supports the cross-modal premise;")
    print(" high correlation means p and k are catching the same errors.)")


if __name__ == "__main__":
    print("Loading pipeline...")
    pipeline = CrossModalRAGPipeline(KB_PATH)

    print(f"Loading {SAMPLE_SIZE} test questions...")
    questions = load_metaqa_questions(QA_PATH, limit=SAMPLE_SIZE)

    print("Scoring claims (this will take a while)...")
    records = collect_records(pipeline, questions)

    df = pd.DataFrame(records)
    df.to_json("correlation_check.json", orient="records", indent=2)
    print(f"\nSaved {len(df)} claim records to correlation_check.json")

    analyze(df)