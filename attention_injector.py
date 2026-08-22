"""
attention_injector.py
The "Virtual Attention Masking" mechanism — the hardest, riskiest
part of the whole system (this is Step 4 from the feasibility spike
we planned: if this file doesn't work reliably, fall back to the
simpler generate-then-verify-then-correct design instead of this
live mid-generation version).

--------------------------------------------------------------------
HOW THIS ACTUALLY WORKS (read this before the code):

1. INJECTION: when the Entropy Evaluator flags trouble, we take the
   retrieved fact text, run it through the model ourselves to get
   its key/value tensors, and CONCATENATE those onto the existing
   KV cache. The model now "remembers" these facts as if they'd
   always been part of the context — without us editing the visible
   prompt text at all. This is the "injected into the KV cache" part.

2. ATTENTION BOOST: concatenating alone isn't enough — the model
   might still mostly ignore the newly-added positions. So we ALSO
   build a custom additive attention bias: a tensor added to the raw
   attention scores BEFORE softmax, which increases the score at the
   positions corresponding to our injected fact tokens. This is
   exactly what "attention_mask" already IS in HuggingFace
   Transformers under the hood (normally used to hide padding/future
   tokens with -inf) — we're reusing the same mechanism, but adding
   a POSITIVE bias instead of masking something out. Hence "virtual"
   attention masking: nothing in the visible prompt text changes,
   only this internal bias tensor.

--------------------------------------------------------------------
HONEST CAVEATS (read these too):

- This requires attn_implementation="eager" — HuggingFace's faster
  attention backends (SDPA, flash-attention) don't expose a clean
  way to pass a custom additive bias. Eager is slower but necessary
  here.
- CACHE API — this code now targets transformers 5.15.0's actual
  Cache structure: past_key_values.layers is a list of DynamicLayer
  objects, each with .keys/.values tensors, and .get_seq_length() /
  .update() are the correct methods to read/modify it. This was
  UPDATED after the original tuple-based/to_legacy_cache() approach
  broke on this version — verified working via a live Colab test
  before rewriting, not just reasoned from docs. If you install a
  DIFFERENT transformers version later, this exact API may have
  moved again — re-run the same kind of live diagnostic (inspect
  dir(past_key_values) and dir(past_key_values.layers[0])) rather
  than assuming this still holds.
- Adding log(boost_factor) to a logit does not translate exactly to
  "multiply that position's attention weight by boost_factor" post-
  softmax (softmax normalizes across ALL positions, not just this
  one) — but it does reliably and monotonically increase that
  position's share of attention. Treat ATTENTION_BOOST_FACTOR as a
  tunable knob, not an exact multiplier.
- ROPE POSITION FIX (added after live debugging a real degenerate-
  output failure — generation collapsing into repeated garbage
  tokens after a few injections): encode_facts_to_kv originally ran
  the fact text through the model standalone, so its rotary position
  embeddings got baked in as if the fact started at position 0. When
  those keys were then spliced into the middle of an ongoing
  generation's cache (at some large position N), the model ended up
  with keys "labeled" as position 0 sitting at position N — corrupting
  every subsequent attention computation and producing exactly the
  kind of repeated-token collapse we saw. FIX: encode_facts_to_kv now
  takes a position_offset and passes explicit position_ids so the
  fact's rotary encoding matches the ABSOLUTE position it will
  actually occupy once spliced in.
--------------------------------------------------------------------
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, DEVICE, TORCH_DTYPE, ATTENTION_BOOST_FACTOR, MAX_NEW_TOKENS


# ---------------------------------------------------------------------
# Model loading — centralized here since this file needs eager
# attention specifically, unlike a plain baseline generation call.
# ---------------------------------------------------------------------
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="eager",  # required to pass a custom additive attention bias
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------
# Step 1: Encode facts into KV cache entries
# ---------------------------------------------------------------------
def encode_facts_to_kv(model, tokenizer, fact_text: str, position_offset: int = 0):
    """
    Runs the fact text through the model on its own to get its
    key/value tensors — these are what we'll splice into the
    ongoing generation's cache.

    CRITICAL: position_offset must be the ABSOLUTE position in the
    ongoing generation's sequence where this fact will actually be
    spliced in (i.e. current_seq_len of the main cache BEFORE
    injection). Without this, the fact's rotary position embeddings
    get baked in as if it starts at position 0, which then produces
    an incorrect relative-position signal once spliced into the
    middle of a longer sequence — this was the root cause of a real
    degenerate-output failure (generation collapsing into repeated
    garbage tokens) diagnosed via live testing.
    """
    inputs = tokenizer(fact_text, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    position_ids = torch.arange(
        position_offset, position_offset + seq_len, device=DEVICE
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, position_ids=position_ids)
    return outputs.past_key_values, seq_len


def inject_facts_into_cache(current_past_key_values, fact_past_key_values):
    """
    Injects the fact's key/value tensors into the ongoing generation's
    cache, layer by layer, using cache.update() — the SAME method the
    model itself calls internally during normal generation, so this
    stays consistent with whatever internal bookkeeping the cache
    needs (not just raw tensor concatenation, which risks missing
    state the model tracks elsewhere).

    REWRITTEN after the original implementation (which relied on a
    to_legacy_cache() conversion) broke on transformers 5.15.0, which
    replaced the old tuple-based cache format entirely with a
    DynamicCache containing a .layers list of DynamicLayer objects.
    This version was verified working via a live test in Colab
    BEFORE being written here (seq length correctly grew 1 -> 3 after
    injecting a 2-token fact) — not just reasoned about from
    documentation, which is what went wrong the first time.

    Returns the (start, end) position range where the injected tokens
    now live — needed next to build the attention bias.

    NOTE: the caller MUST have used current_past_key_values.get_seq_length()
    (taken BEFORE this call) as the position_offset when it originally
    called encode_facts_to_kv — this function assumes fact_past_key_values
    was rotary-encoded for exactly the range it's about to occupy.
    """
    current_seq_len = current_past_key_values.get_seq_length()

    for layer_idx in range(len(current_past_key_values.layers)):
        fact_keys = fact_past_key_values.layers[layer_idx].keys
        fact_values = fact_past_key_values.layers[layer_idx].values
        current_past_key_values.update(fact_keys, fact_values, layer_idx)

    new_seq_len = current_past_key_values.get_seq_length()
    injected_range = (current_seq_len, new_seq_len)
    return current_past_key_values, injected_range


# ---------------------------------------------------------------------
# Step 3: Build the attention bias tensor (the actual "masking")
# ---------------------------------------------------------------------
def build_attention_bias(total_seq_len: int, injected_range: tuple, boost_factor: float = ATTENTION_BOOST_FACTOR):
    """
    Builds a (1, 1, 1, total_seq_len) additive bias to add to the
    next attention step's raw scores. Zero everywhere except the
    injected fact positions, where it's log(boost_factor) — pushing
    the softmax to allocate more weight there.

    Shape note: shaped for a single query position (the next token
    being generated attending back over the whole sequence so far),
    which is what a single decoding step needs.
    """
    bias = torch.zeros(1, 1, 1, total_seq_len, device=DEVICE, dtype=TORCH_DTYPE)
    start, end = injected_range
    bias[:, :, :, start:end] = torch.log(torch.tensor(boost_factor, dtype=TORCH_DTYPE))
    return bias


# ---------------------------------------------------------------------
# Step 4: Manual token-by-token generation loop with intervention
# ---------------------------------------------------------------------
def run_monitored_generation(
    model,
    tokenizer,
    prompt: str,
    evaluator,
    retriever,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    The full pipeline in one function: generates token-by-token,
    checks the Entropy Evaluator after every token, and injects
    facts + boosts attention when it flags trouble.

    Returns the generated text AND a log of every intervention that
    fired — the log is what you'll want for the paper's qualitative
    examples ("here's a case where injection corrected a drift").
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[0, -1, :]

    generated_ids = []
    intervention_log = []
    attention_bias = None
    injected_ranges = []  # track ALL injections this generation, not just the last
    already_injected_facts = set()  # NEW — see the guard below

    evaluator.reset()

    for step in range(max_new_tokens):
        next_token_id = torch.argmax(logits).item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        decoded_window = tokenizer.decode(generated_ids[-10:])
        retrieved_facts = retriever.retrieve_as_facts(prompt)

        eval_result = evaluator.evaluate_step(
            logits=logits,
            new_token_id=next_token_id,
            decoded_window_text=decoded_window,
            retrieved_facts=retrieved_facts,
        )

        if eval_result["should_intervene"] and retrieved_facts:
            # NEW — added after a real run showed the SAME fact
            # ("Ted directed by Seth MacFarlane") getting injected 27
            # times in one generation. If the top fact already got
            # injected and didn't stop the drift, injecting it again
            # is pure waste — try the next candidate instead, and
            # only skip the intervention entirely if EVERY retrieved
            # fact has already been tried this generation.
            fact_text = next((f for f in retrieved_facts if f not in already_injected_facts), None)

            if fact_text is not None:
                already_injected_facts.add(fact_text)

                # FIX: encode the fact with position_ids matching the
                # ABSOLUTE position it will occupy once spliced into
                # the main cache (current_seq_len, taken BEFORE
                # injection) — not position 0. See module docstring's
                # "ROPE POSITION FIX" note for why this matters.
                position_offset = past_key_values.get_seq_length()
                fact_kv, _ = encode_facts_to_kv(
                    model, tokenizer, fact_text, position_offset=position_offset
                )
                past_key_values, injected_range = inject_facts_into_cache(past_key_values, fact_kv)
                injected_ranges.append(injected_range)

                intervention_log.append({
                    "step": step,
                    "reasons": eval_result["triggered_reasons"],
                    "injected_fact": fact_text,
                })

        current_seq_len = past_key_values.get_seq_length()
        if injected_ranges:
            # +1 accounts for the new token THIS forward pass is about
            # to process — its key/value get appended to the cache
            # DURING this call, so the actual attention computation
            # happens over current_seq_len + 1 positions, not
            # current_seq_len. FIXED after a real error pinned this
            # down exactly: attn_weights was size 19, our mask was
            # built at size 18 — precisely this missing +1.
            attention_bias = torch.zeros(1, 1, 1, current_seq_len + 1, device=DEVICE, dtype=TORCH_DTYPE)
            for start, end in injected_ranges:
                attention_bias[:, :, :, start:end] = torch.log(
                    torch.tensor(ATTENTION_BOOST_FACTOR, dtype=TORCH_DTYPE)
                )

        next_input = torch.tensor([[next_token_id]], device=DEVICE)
        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                attention_mask=attention_bias,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, intervention_log


# ---------------------------------------------------------------------
# Quick manual test — run directly once you have kb.txt in place.
# Requires GPU/Colab realistically; will run on CPU but slowly.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from graph_engine import HybridGraphRetriever
    from entropy_evaluator import GenerationEvaluator

    print("Loading model (eager attention)...")
    model, tokenizer = load_model_and_tokenizer()

    print("Loading retriever...")
    retriever = HybridGraphRetriever("data/metaqa/kb.txt", max_hops=2)

    evaluator = GenerationEvaluator()

    test_prompt = "Who directed the movie written by John Balderston?"
    print(f"\nPrompt: {test_prompt}")

    text, log = run_monitored_generation(model, tokenizer, test_prompt, evaluator, retriever)

    print(f"\nGenerated: {text}")
    print(f"\nInterventions fired: {len(log)}")
    for entry in log:
        print(f"  step {entry['step']}: {entry['reasons']} -> injected '{entry['injected_fact']}'")
