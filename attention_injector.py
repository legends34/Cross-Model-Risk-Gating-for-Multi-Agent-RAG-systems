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
- The exact KV-cache object format (tuple-of-tuples vs. a Cache
  object) has changed across `transformers` versions. This code
  tries to handle both, but if you hit a cache-related error, that's
  the first place to look — check your installed `transformers`
  version against what's documented for your model.
- Adding log(boost_factor) to a logit does not translate exactly to
  "multiply that position's attention weight by boost_factor" post-
  softmax (softmax normalizes across ALL positions, not just this
  one) — but it does reliably and monotonically increase that
  position's share of attention. Treat ATTENTION_BOOST_FACTOR as a
  tunable knob, not an exact multiplier.
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
def encode_facts_to_kv(model, tokenizer, fact_text: str):
    """
    Runs the fact text through the model on its own to get its
    key/value tensors — these are what we'll splice into the
    ongoing generation's cache.
    """
    inputs = tokenizer(fact_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values, inputs["input_ids"].shape[1]


def _to_legacy_tuple_cache(past_key_values):
    """
    Normalizes whatever cache format the installed transformers
    version returns into a plain tuple-of-(key, value)-tuples, one
    per layer. Newer versions may return a Cache object instead of
    a raw tuple — this handles both so the rest of the file doesn't
    need to care which version is installed.
    """
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values  # already a plain tuple


# ---------------------------------------------------------------------
# Step 2: Inject fact KV entries into the ongoing generation's cache
# ---------------------------------------------------------------------
def inject_facts_into_cache(current_past_key_values, fact_past_key_values):
    """
    Concatenates the fact's key/value tensors onto the END of the
    current generation's cache, layer by layer, along the sequence
    dimension. Returns the new combined cache AND the (start, end)
    position range where the injected tokens now live — needed next
    to build the attention bias.
    """
    current_legacy = _to_legacy_tuple_cache(current_past_key_values)
    fact_legacy = _to_legacy_tuple_cache(fact_past_key_values)

    current_seq_len = current_legacy[0][0].shape[2]  # (batch, heads, seq_len, head_dim)
    fact_seq_len = fact_legacy[0][0].shape[2]

    new_cache = []
    for (cur_k, cur_v), (fact_k, fact_v) in zip(current_legacy, fact_legacy):
        combined_k = torch.cat([cur_k, fact_k], dim=2)
        combined_v = torch.cat([cur_v, fact_v], dim=2)
        new_cache.append((combined_k, combined_v))

    injected_range = (current_seq_len, current_seq_len + fact_seq_len)
    return tuple(new_cache), injected_range


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
            fact_text = retrieved_facts[0]  # simplest strategy: inject the top fact
            fact_kv, _ = encode_facts_to_kv(model, tokenizer, fact_text)
            past_key_values, injected_range = inject_facts_into_cache(past_key_values, fact_kv)
            injected_ranges.append(injected_range)

            intervention_log.append({
                "step": step,
                "reasons": eval_result["triggered_reasons"],
                "injected_fact": fact_text,
            })

        current_seq_len = _to_legacy_tuple_cache(past_key_values)[0][0].shape[2]
        if injected_ranges:
            attention_bias = torch.zeros(1, 1, 1, current_seq_len, device=DEVICE, dtype=TORCH_DTYPE)
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