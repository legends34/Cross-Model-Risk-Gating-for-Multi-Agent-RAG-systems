"""
claim_extractor.py
Layer 1 of the Cross-Modal Conformal Risk Gating design: "Claim
extraction". Decomposes a generated answer into atomic, checkable
claims (subject-relation-object triples) instead of scoring a whole
response or individual tokens.

Why this matters (from the report): tokens don't align to what a
knowledge graph can verify, and whole-response rejection wastes
everything the agent got right. A claim IS a unit the Graph Engine
can actually check against — one triple, same shape as kb.txt's
own facts.

DESIGN CHOICE: no separate extraction model is used. We reuse the
same LLM already loaded for generation, and just prompt it
differently (few-shot, asking it to output triples). This matches
the report's Layer 1 note: "No new method required here." A
dedicated extraction model (like RefChecker uses) would likely be
more reliable, but this is the right amount of complexity for a
first version — genuinely worth stating as a limitation, not hiding.
"""

import torch

from config import CLAIM_EXTRACTION_MAX_CLAIMS, DEVICE


SYSTEM_PROMPT = (
    "You are a precise fact-extraction assistant. Given a passage, "
    "extract every distinct factual claim as a single line in the "
    "format: subject|relation|object. Use short, specific relation "
    "names with underscores (e.g. directed_by, capital_of, "
    "signed_deal_with). Output ONLY the triples, one per line, no "
    "numbering, no bullet points, no extra commentary. If a sentence "
    "contains no clear factual claim, skip it."
)

# One few-shot example, kept short — small instruct models tend to
# follow a strict output format much more reliably with even one
# example than with instructions alone.
FEW_SHOT_INPUT = (
    "Delhi is the capital of India. The Red Fort is located in Delhi, "
    "and it was built by Shah Jahan."
)
FEW_SHOT_OUTPUT = (
    "India|capital|Delhi\n"
    "Red Fort|located_in|Delhi\n"
    "Red Fort|built_by|Shah Jahan"
)


def _build_extraction_prompt(tokenizer, generated_text: str) -> str:
    """
    Builds a chat-formatted prompt using the model's own chat
    template (rather than raw string concatenation), since instruct-
    tuned models like Qwen2.5-Instruct are trained to expect this
    exact structure and follow formatting instructions much more
    reliably when given it.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_INPUT},
        {"role": "assistant", "content": FEW_SHOT_OUTPUT},
        {"role": "user", "content": generated_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _parse_triples(raw_output: str) -> list[tuple[str, str, str]]:
    """
    Parses the model's raw line-per-triple output. Deliberately
    permissive about malformed lines — skips them rather than
    crashing, same philosophy as graph_engine.py's load_triples().

    HONEST LIMITATION: small instruct models sometimes ignore
    formatting instructions entirely (add commentary, use numbering,
    etc.) despite the few-shot example. If this happens often in
    practice, worth switching to a stricter approach — e.g.
    constrained decoding, or a regex to strip common wrapper text
    before parsing. Flagging this now rather than after it silently
    produces empty claim lists during a real run.
    """
    triples = []
    seen = set()

    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue

        triple = tuple(parts)
        if triple not in seen:
            seen.add(triple)
            triples.append(triple)

    return triples


def extract_claims(
    model,
    tokenizer,
    generated_text: str,
    max_claims: int = CLAIM_EXTRACTION_MAX_CLAIMS,
) -> list[tuple[str, str, str]]:
    """
    The main entry point: takes a chunk of generated text, returns
    up to max_claims (subject, relation, object) triples.

    This runs its OWN separate, simple generation call — not part of
    the monitored token-by-token loop in attention_injector.py.
    Extraction happens on a finished piece of text (e.g. the last
    sentence, or the full answer so far), not per-token.
    """
    prompt = _build_extraction_prompt(tokenizer, generated_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # greedy — we want consistent, parseable output, not creativity
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    triples = _parse_triples(raw_output)

    if not triples:
        # DEBUG — added after seeing "Extracted 0 claims" with no
        # visibility into why. This print shows you EXACTLY what the
        # model produced, so you can tell whether it ignored the
        # pipe-format instruction entirely, added commentary the
        # parser correctly rejected, or something else. Remove once
        # extraction is reliable and you don't need this anymore.
        print(f"[claim_extractor DEBUG] Got 0 parseable claims. Raw model output was:\n---\n{raw_output}\n---")

    return triples[:max_claims]


def triple_to_sentence(triple: tuple[str, str, str]) -> str:
    """
    Same idea as graph_engine.py's SemanticIndex._triple_to_sentence,
    reimplemented locally rather than imported. Small duplication,
    deliberate choice — keeps this file independently testable
    without needing graph_engine's heavier dependencies (NetworkX,
    sentence-transformers) just to format a string.
    """
    subj, rel, obj = triple
    readable_rel = rel.replace("_", " ")
    return f"{subj} {readable_rel} {obj}"


# ---------------------------------------------------------------------
# Quick manual test — needs the model loaded, so run this after
# confirming attention_injector.py's model loading works.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from attention_injector import load_model_and_tokenizer

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    test_text = (
        "The movie Kismet was directed by Andrew Marton and written by "
        "John Balderston. It was released in the 1950s."
    )
    print(f"\nInput text: {test_text}")

    claims = extract_claims(model, tokenizer, test_text)

    print(f"\nExtracted {len(claims)} claims:")
    for c in claims:
        print(f"  {c}  ->  \"{triple_to_sentence(c)}\"")
