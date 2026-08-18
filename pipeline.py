"""
pipeline.py
Ties the whole system together in two stages:

  STAGE 1 (attention_injector.py): generate a draft answer live,
  with token-level entropy monitoring and KV-cache injection as a
  first line of defense during generation.

  STAGE 2 (claim_extractor.py -> dual_scorer.py -> risk_gate.py ->
  correction_router.py): audit that finished draft at the CLAIM
  level — extract checkable facts, score each one against
  independent KG evidence, gate/route through accept / local_fix /
  escalate, and produce a final, provenance-tagged answer.

This two-stage design isn't just convenient plumbing — it sets up a
genuine ablation study for the paper: run Stage 1 only, Stage 2
only, or both together, and measure which combination catches the
most errors at the lowest cost. See evaluate.py (next file) for
where that comparison actually gets measured.
"""

from graph_engine import HybridGraphRetriever
from entropy_evaluator import GenerationEvaluator
from attention_injector import load_model_and_tokenizer, run_monitored_generation
from claim_extractor import extract_claims
from dual_scorer import EntailmentScorer, score_claim
from risk_gate import RiskGate
from correction_router import route_claim


class CrossModalRAGPipeline:
    """
    One object holding every component, so you're not re-loading the
    model / retriever / NLI scorer on every query (each of those is
    expensive to load — this matters a lot in a Colab session where
    you'll be running many test queries in evaluate.py).
    """

    def __init__(self, kb_path: str):
        print("Loading model and tokenizer...")
        self.model, self.tokenizer = load_model_and_tokenizer()

        print("Loading knowledge graph retriever...")
        self.retriever = HybridGraphRetriever(kb_path, max_hops=2)

        print("Loading entailment scorer...")
        self.entailment_scorer = EntailmentScorer()

        # One shared gate per pipeline instance — its ACI threshold
        # is meant to accumulate calibration signal ACROSS queries,
        # not reset each time. Creating a new RiskGate per query
        # would defeat the entire point of online calibration.
        self.gate = RiskGate()

    def answer(self, query: str, use_stage1: bool = True, use_stage2: bool = True) -> dict:
        """
        Runs the full pipeline on one query. use_stage1/use_stage2
        let you toggle each stage off independently — this IS the
        ablation switch mentioned in the module docstring, exposed
        directly here rather than buried, so evaluate.py can call
        this with different flag combinations.
        """
        result = {
            "query": query,
            "stage1_used": use_stage1,
            "stage2_used": use_stage2,
        }

        # --- Stage 1: live monitored generation ---
        if use_stage1:
            evaluator = GenerationEvaluator()
            draft_answer, intervention_log = run_monitored_generation(
                self.model, self.tokenizer, query, evaluator, self.retriever
            )
            result["stage1_interventions"] = intervention_log
        else:
            # Plain, unmonitored baseline generation — the honest
            # "what if we skip Stage 1 entirely" comparison arm.
            messages = [{"role": "user", "content": query}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            import torch
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            draft_answer = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

        result["draft_answer"] = draft_answer

        # --- Stage 2: claim-level audit ---
        if not use_stage2:
            result["final_answer"] = draft_answer
            result["claims"] = []
            return result

        context = f"Question: {query}"
        claims = extract_claims(self.model, self.tokenizer, draft_answer)

        claim_results = []
        for claim_triple in claims:
            retrieved_facts = self.retriever.retrieve_as_facts(
                " ".join(claim_triple)  # search using the claim's own text, not just the original query
            )
            scored = score_claim(
                self.model, self.tokenizer, self.entailment_scorer,
                context, claim_triple, retrieved_facts,
            )
            routed = route_claim(
                self.model, self.tokenizer, self.retriever, self.entailment_scorer,
                self.gate, scored, context, query,
            )
            claim_results.append(routed)

        result["claims"] = claim_results

        # Build the final answer: start from the draft, and note
        # which claims were corrected/escalated. A genuinely careful
        # version would SPLICE each corrected sentence back into the
        # exact right place in the draft text — this simpler version
        # instead reports corrections alongside the original draft,
        # which is honest but not yet a polished rewritten answer.
        # Worth treating text-splicing as a real follow-up task, not
        # something to fake as "already solved" here.
        result["final_answer"] = draft_answer
        result["corrections_applied"] = [
            c for c in claim_results if c["status"] in ("corrected", "escalated_resolved")
        ]

        return result


# ---------------------------------------------------------------------
# Quick manual test — runs one query through the full pipeline.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = CrossModalRAGPipeline("data/metaqa/kb.txt")

    query = "Who directed the movie written by John Balderston?"
    result = pipeline.answer(query)

    print(f"\nQuery: {query}")
    print(f"Draft answer: {result['draft_answer']}")
    print(f"\nClaims audited: {len(result['claims'])}")
    for c in result["claims"]:
        print(f"  [{c['status']}] {c['original_sentence']} -> {c['final_sentence']}")