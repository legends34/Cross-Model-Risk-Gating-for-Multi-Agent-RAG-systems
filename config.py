"""
config.py
Central configuration for the Cross-Modal Conformal Risk Gating
project (extends the original Dual-Engine Graph-RAG + Self-
Correcting Multi-Agent design with claim-level dual scoring and
online-calibrated gating).

Every other file imports settings from here instead of hardcoding
values, so you only ever need to change things in ONE place
(e.g. swapping model size, adjusting entropy thresholds).
"""

import torch

# ---------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------
# Qwen2.5-1.5B-Instruct: fits comfortably on a free-tier T4 (16GB VRAM)
# even with hook overhead, and iterates fast while we're still
# debugging the attention-injection mechanism. Swap to a 7B variant
# later once the pipeline is proven end-to-end.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIXED after a real bug caught in Colab testing: fp16 + eager
# attention (eager is REQUIRED for attention_injector.py's hooks)
# produced NaN logits — confirmed directly by a CUDA assertion
# ("probability tensor contains nan") during generation. Eager
# attention computes raw scores without the numerically-stabilized
# fused kernels that faster backends use, and fp16's limited range
# overflows easily. Switched to fp32 for correctness — a 1.5B model
# fits comfortably in fp32 on a T4's 16GB VRAM, just somewhat slower
# than fp16 would have been. If you later profile this as too slow,
# bfloat16 is worth trying (wider range than fp16, avoids this
# specific overflow issue) — but verify T4's bf16 support first,
# it's not hardware-accelerated on Turing-generation GPUs like T4.
TORCH_DTYPE = torch.float32

# Max new tokens to generate per answer. Kept short since MetaQA
# answers are typically factual, single-entity answers, not essays.
MAX_NEW_TOKENS = 64

# ---------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------
# MetaQA: a published, widely-used KGQA benchmark with an explicit
# knowledge graph (movie domain) and 1-hop / 2-hop / 3-hop questions.
# https://github.com/yuyuz/MetaQA
DATASET_NAME = "MetaQA"

# Which hop-count subset to start with. Start at 1-hop to sanity-check
# the pipeline, then move to 2-hop/3-hop once things work — that's
# where "static retrieval" (Gap 3) actually starts failing baselines.
DEFAULT_HOP_LEVEL = "1-hop"  # options: "1-hop", "2-hop", "3-hop"

# ---------------------------------------------------------------------
# Entropy Evaluator settings
# ---------------------------------------------------------------------
# Window of consecutive tokens to look at when checking for a decay
# pattern, rather than reacting to a single low/high entropy token
# (a single token being low-entropy is often just normal confidence).
ENTROPY_WINDOW = 5

# If entropy drops by more than this fraction within the window,
# treat it as a suspicious "decay" pattern rather than normal
# confidence. Tune this empirically once you have real generations
# to look at — this starting value is a reasonable guess, not a
# result.
ENTROPY_DECAY_THRESHOLD = 0.4

# Absolute entropy floor. Below this, a token is "very confident" —
# only worth flagging as suspicious if it also follows a decay
# pattern (not on its own; plenty of tokens are legitimately
# near-certain, like "the" after "of").
LOW_ENTROPY_FLOOR = 0.5

# ---------------------------------------------------------------------
# Attention Injection settings
# ---------------------------------------------------------------------
# How much to multiply attention weight toward injected fact tokens.
# 1.0 = no change. Start conservative; too high and generation
# degenerates into just repeating the injected fact verbatim.
ATTENTION_BOOST_FACTOR = 3.0

# Which transformer layer(s) to hook. Later layers tend to carry more
# "semantic" information; earlier layers more syntactic. Start with
# a middle-to-late layer and treat this as a hyperparameter to sweep.
INJECTION_LAYER_INDEX = -4  # 4th-from-last layer

# ---------------------------------------------------------------------
# Claim Extraction settings (Layer 1)
# ---------------------------------------------------------------------
# How generated text gets broken into checkable (subject, relation,
# object) claims before scoring. Starting with direct LLM-prompted
# extraction — no separate extraction model needed, we reuse the
# same model already loaded for generation.
CLAIM_EXTRACTION_MAX_CLAIMS = 5  # cap per generation, avoid runaway prompts

# ---------------------------------------------------------------------
# Symbolic Score settings (Layer 2 — entailment)
# ---------------------------------------------------------------------
# A cross-encoder NLI model: takes (claim, evidence) as a PAIR and
# classifies their relationship. This is a genuinely different tool
# from SemanticIndex's cosine similarity — similarity tells you
# "related", entailment tells you "actually agrees or disagrees".
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# NLI models conventionally output 3 classes in this order — kept
# here as documentation, since the exact index-to-label mapping can
# vary between checkpoints and is worth double-checking against
# whichever specific model you end up using.
NLI_LABELS = ["contradiction", "entailment", "neutral"]

# ---------------------------------------------------------------------
# Risk Fusion + Adaptive Conformal Inference settings (Layer 3)
# ---------------------------------------------------------------------
# Target error tolerance (alpha, from the paper's formula). "We're
# okay missing at most this fraction of real errors." Start
# conservative; this is a knob to discuss with your team, not a
# number with one correct answer.
ACI_TARGET_ERROR_RATE = 0.10

# Step size (gamma) — how fast the threshold adjusts per resolved
# case. Too high = threshold jumps around unstably; too low = takes
# forever to converge. Starting value only, needs empirical tuning.
ACI_STEP_SIZE = 0.05

# Starting threshold (lambda_0) before any calibration has happened.
# Arbitrary starting point in [0, 1] risk-score space — the whole
# point of ACI is that this self-corrects over time regardless of
# where it starts.
ACI_INITIAL_LAMBDA = 0.5

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
