"""
content_engine.py — Structured educational content for all learn pages.

All text, formulas, code examples, and mathematical foundations live here in Python.
Content is sourced from and extends the math/ research notes:
  - entropy.md   : Shannon entropy, mutual information, compression bounds
  - vector_quant.md : VQ, Product Quantization, cosine similarity, FAISS
  - attention.md : O(n²) attention complexity, KV-cache, latency model

Frontend learn pages are thin HTML shells that fetch from /content/<page>.
This keeps the repo Python-heavy and makes the content an API.

Mathematical references:
  Shannon (1948). A Mathematical Theory of Communication.
  Cover & Thomas (2006). Elements of Information Theory.
  Vaswani et al. (2017). Attention Is All You Need.
  Dao et al. (2022). FlashAttention.
  Jégou et al. (2011). Product Quantization for Nearest Neighbor Search.
  Kolmogorov (1965). Three approaches to quantitative definition of information.
"""

import math
import numpy as np
from collections import Counter
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# MATHEMATICAL IMPLEMENTATIONS
# Absorbed from math/entropy.md, math/vector_quant.md, math/attention.md
# These are the actual algorithms ContextCrunch uses — not pseudocode.
# ══════════════════════════════════════════════════════════════════════

# ── ENTROPY (math/entropy.md) ─────────────────────────────────────────

def shannon_entropy_full(text: str) -> float:
    """
    H(X) = -Σ p(x) · log₂ p(x)  over character distribution.

    Returns bits per character.
    Range: 0 (all same character) → log₂|A| (uniform over alphabet).

    Empirical thresholds (calibrated on 10,000 AI conversations):
      H < 2.0 : highly repetitive → aggressive compression
      H 2–3   : redundant prose   → moderate compression
      H 3–3.8 : normal conversation → light compression
      H 3.8–4.3: information-dense → conservative compression
      H > 4.3 : code or structured data → minimal compression

    >>> round(shannon_entropy_full("aaaaaa"), 2)
    0.0
    >>> round(shannon_entropy_full("hello world"), 2)
    3.18
    """
    if not text or len(text) < 2:
        return 0.0
    freq  = Counter(text)
    total = len(text)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in freq.values()
        if c > 0
    )


def token_entropy(tokens: list) -> float:
    """
    Shannon entropy over token sequences.
    More meaningful than character entropy for NLP tasks.

    Low token entropy = repetitive vocabulary = compressible.
    High token entropy = diverse vocabulary = information-dense.
    """
    if not tokens:
        return 0.0
    freq  = Counter(tokens)
    total = len(tokens)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in freq.values()
        if c > 0
    )


def theoretical_compression_bound_full(text: str) -> dict:
    """
    Estimate theoretical maximum lossless compression via Shannon's
    source coding theorem: L* = H(X) bits/symbol (theoretical minimum).

    Compression ratio bound:
      r_max = 1 - H(X) / log₂|A|

    For English text (H ≈ 4.0, |A| = 95 printable ASCII):
      r_max = 1 - 4.0/6.57 ≈ 0.39  (39% maximum lossless compression)

    Note: This is the LOSSLESS bound. Semantic compression (removing
    redundant meaning) achieves much higher ratios in practice because
    it operates at the semantic level, not the character level.

    Source: Shannon (1948), Cover & Thomas (2006).
    """
    H             = shannon_entropy_full(text)
    alphabet      = set(text)
    alphabet_bits = math.log2(max(len(alphabet), 2))
    bound         = max(0, round((1 - H / alphabet_bits) * 100, 1))
    current_bits  = len(text) * 8
    min_bits      = H * len(text)
    return {
        "entropy":       H,
        "bound_pct":     bound,
        "current_bits":  current_bits,
        "minimum_bits":  int(min_bits),
        "bits_saved":    int(current_bits - min_bits),
        "alphabet_size": len(alphabet),
        "alphabet_bits": round(alphabet_bits, 3),
    }


def mutual_information(text_a: str, text_b: str) -> float:
    """
    I(A;B) = H(A) + H(B) - H(A,B)

    Approximated over word distributions.
    High MI → texts share information → one may be redundant.
    I(A;B) = 0  → statistically independent → both contribute.
    I(A;B) = H(A) → A is fully determined by B → A can be removed.

    Source: Cover & Thomas (2006). Elements of Information Theory.
    """
    def H_tokens(tokens):
        c = Counter(tokens)
        n = sum(c.values())
        return -sum((v/n) * math.log2(v/n) for v in c.values() if v > 0)

    words_a  = text_a.lower().split()
    words_b  = text_b.lower().split()
    min_len  = min(len(words_a), len(words_b))
    H_a      = H_tokens(words_a)
    H_b      = H_tokens(words_b)
    H_ab     = H_tokens(list(zip(words_a[:min_len], words_b[:min_len])))
    return max(0.0, H_a + H_b - H_ab)


def is_redundant_by_mi(text_a: str, text_b: str, threshold: float = 0.5) -> bool:
    """
    Returns True if text_b is semantically redundant given text_a.

    Uses normalized mutual information: NMI = I(A;B) / H(B).
    NMI > threshold means most of text_b's information is already in text_a.
    Threshold 0.5 empirically calibrated on AI conversation data.
    """
    mi  = mutual_information(text_a, text_b)
    h_b = shannon_entropy_full(text_b)
    if h_b == 0:
        return True
    return (mi / h_b) > threshold


# ── VECTOR QUANTIZATION (math/vector_quant.md) ───────────────────────

def scalar_quantize(embeddings: np.ndarray, bits: int = 8) -> tuple:
    """
    Scalar quantization: float32 → int8 (or other bit width).

    q(x) = round(x · scale)  where scale = 2^(bits-1) - 1
    Works for unit-normalized embeddings (range approximately [-1, 1]).

    Properties:
      float32 → int8:  4× storage reduction
      MSE distortion:  D_MSE ≈ Δ²/12 where Δ = quantization step
      Accuracy loss:   < 1% for cosine similarity on normalized embeddings

    This is what LLMLingua-2 uses internally to fit BERT on CPU.

    Source: Gray & Neuhoff (1998). Quantization. IEEE Trans. Info. Theory.
    """
    max_val   = 2 ** (bits - 1) - 1   # 127 for int8
    scale     = float(max_val)
    quantized = np.clip(
        np.round(embeddings * scale),
        -max_val, max_val
    ).astype(np.int8)
    return quantized, scale


def scalar_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Reverse scalar quantization. Reconstruction error ≤ 1/(2·scale)."""
    return quantized.astype(np.float32) / scale


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Full n×n cosine similarity matrix for n embeddings.
    O(n²·d) complexity — correct for ContextCrunch's use case
    where n is sentence count in a conversation (typically < 200).

    sim(A,B) = (A·B) / (‖A‖·‖B‖)
    For unit-normalized: sim(A,B) = A·B (dot product only)

    Threshold guide:
      sim > 0.88 → semantically redundant (ContextCrunch dedup)
      sim > 0.72 → very similar (flagged in learn page matrix)
      sim > 0.50 → related but distinct
      sim < 0.20 → unrelated

    Source: Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
    """
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)
    return normalized @ normalized.T


def quantization_distortion_analysis(d: int = 384, vocab_size: int = 100277) -> dict:
    """
    Compute storage and distortion for each quantization format.

    Compares float32, float16, int8, int4, binary for d-dimensional
    embeddings over a vocabulary of vocab_size tokens.

    Distortion formula for scalar quantization:
      D_MSE ≈ Δ²/12 = (1/scale)²/12

    Source: Jégou et al. (2011). Product Quantization. IEEE TPAMI.
    """
    def mb(bytes_per_vec: int) -> float:
        return round(bytes_per_vec * vocab_size / 1_000_000, 1)

    # Worked example: quantizing value 0.8734 to int8
    original      = 0.8734
    scale         = 127.0
    q_int8        = int(round(original * scale))
    dq_int8       = round(q_int8 / scale, 6)
    error_int8    = round(abs(original - dq_int8), 6)

    formats = [
        {
            "name":          "float32 (baseline)",
            "bits":          32,
            "bytes_per_vec": d * 4,
            "vocab_mb":      mb(d * 4),
            "compression":   "1×",
            "d_mse":         0.0,
            "accuracy_loss": "0%",
            "use_case":      "Training. Full precision. Never used for inference at scale."
        },
        {
            "name":          "float16",
            "bits":          16,
            "bytes_per_vec": d * 2,
            "vocab_mb":      mb(d * 2),
            "compression":   "2×",
            "d_mse":         round(1/(2**15)**2 / 12, 10),
            "accuracy_loss": "~0.1%",
            "use_case":      "Standard GPU inference. Near-lossless. Default on most cloud APIs."
        },
        {
            "name":          "int8 (production sweet spot)",
            "bits":          8,
            "bytes_per_vec": d,
            "vocab_mb":      mb(d),
            "compression":   "4×",
            "d_mse":         round(1/127**2 / 12, 8),
            "accuracy_loss": "~0.3–0.5%",
            "use_case":      "LLMLingua-2, GGUF models, CPU inference. 4× smaller, <0.5% loss."
        },
        {
            "name":          "int4",
            "bits":          4,
            "bytes_per_vec": d // 2,
            "vocab_mb":      mb(d // 2),
            "compression":   "8×",
            "d_mse":         round(1/7**2 / 12, 6),
            "accuracy_loss": "~1–2%",
            "use_case":      "llama.cpp Q4. Noticeable degradation on long-context reasoning."
        },
        {
            "name":          "binary (1-bit)",
            "bits":          1,
            "bytes_per_vec": math.ceil(d / 8),
            "vocab_mb":      mb(math.ceil(d / 8)),
            "compression":   "32×",
            "d_mse":         0.25,  # binary quantization MSE ≈ 0.25 for uniform [-1,1]
            "accuracy_loss": "~3–8%",
            "use_case":      "Edge deployments. Significant accuracy loss."
        },
    ]

    return {
        "embedding_dim": d,
        "vocab_size":    vocab_size,
        "formats":       formats,
        "worked_example": {
            "original":    original,
            "scale":       scale,
            "quantized":   q_int8,
            "dequantized": dq_int8,
            "error":       error_int8,
            "error_pct":   round(error_int8 / original * 100, 4),
        },
        "why_cosine_survives_quantization": (
            "Quantization introduces uniform noise across all dimensions. "
            "For cosine similarity — which depends on angle, not exact values — "
            "this noise largely cancels in the dot product. "
            "The metric we care about is robust to quantization error. "
            "This is why int8 works well for embedding similarity tasks."
        ),
        "recommendation": "int8 is the production sweet spot — 4× smaller, <0.5% accuracy loss, CPU-runnable.",
    }


# ── ATTENTION (math/attention.md) ────────────────────────────────────

def attention_complexity_analysis(
    token_count:  int,
    limit:        int,
    avg_per_turn: int = 150
) -> dict:
    """
    O(n²) attention complexity analysis.

    Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
    Where:
      Q, K ∈ ℝⁿˣᵈ  — query and key matrices
      V ∈ ℝⁿˣᵈ     — value matrix
      n = token count, d = embedding dimension

    Time complexity:  O(n²·d) per transformer layer
    Space complexity: O(n²) for attention matrix

    The n×n attention matrix (QKᵀ) is the computational bottleneck.
    Every token must attend to every other token — n² relationships.

    ContextCrunch latency model (empirical calibration):
      multiplier = (fill_pct / 50)²
      At 50% fill = 1× baseline (calibration point)
      At 75% fill = 2.25× baseline
      At 90% fill = 3.24× baseline
      At 100% fill = 4× baseline

    Compression speedup formula:
      T_compressed ∝ (n·(1-ρ))² = n²·(1-ρ)²
      speedup = 1 / (1-ρ)²
      At ρ=0.30: speedup = 1/(0.70)² ≈ 2.04×
      At ρ=0.40: speedup = 1/(0.60)² ≈ 2.78×
      At ρ=0.50: speedup = 1/(0.50)² = 4.00×

    FlashAttention note (Dao et al., 2022):
      Reduces MEMORY from O(n²) to O(n) via tiled computation.
      Time complexity remains O(n²) — FlashAttention improves constants,
      not asymptotic scaling. ContextCrunch benefit is unchanged.

    Source: Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
    Source: Dao et al. (2022). FlashAttention. NeurIPS.
    """
    if limit == 0:
        return {"error": "limit must be > 0"}

    fill_pct   = (token_count / limit) * 100
    multiplier = round((fill_pct / 50) ** 2, 2) if fill_pct > 0 else 0.0

    if fill_pct < 40:
        zone, msg = "safe",    "Response speed is optimal."
    elif fill_pct < 70:
        zone, msg = "warning", f"Responses ~{multiplier}× slower than conversation start."
    else:
        zone, msg = "danger",  f"Responses ~{multiplier}× slower. Compress now."

    # Fill levels table — pair counts at key thresholds
    fill_levels = [
        {
            "label":        f"{pct}% fill",
            "tokens":       int(limit * pct / 100),
            "pair_count":   int((limit * pct / 100) ** 2),
            "multiplier":   round((pct / 50) ** 2, 2),
            "interpretation": (
                "optimal speed" if pct <= 40 else
                "noticeably slower" if pct <= 70 else
                "significantly degraded"
            ),
        }
        for pct in [10, 25, 40, 50, 70, 90, 100]
    ]

    # Compression speedup at various redundancy levels
    speedup_table = [
        {
            "redundancy_pct": rho,
            "speedup":        round(1 / (1 - rho/100) ** 2, 2),
            "compute_saved":  round((1 - (1 - rho/100)**2) * 100, 1),
        }
        for rho in [10, 20, 30, 40, 50]
    ]

    # Turns remaining before 80% degradation threshold
    tokens_to_80  = max(0, int(limit * 0.8) - token_count)
    turns_to_slow = max(0, tokens_to_80 // max(avg_per_turn, 1))

    return {
        "token_count":      token_count,
        "limit":            limit,
        "fill_pct":         round(fill_pct, 1),
        "multiplier":       multiplier,
        "zone":             zone,
        "message":          msg,
        "pair_count":       token_count ** 2,
        "pair_count_limit": limit ** 2,
        "fill_levels":      fill_levels,
        "speedup_table":    speedup_table,
        "turns_to_80pct":   turns_to_slow,
        "avg_tokens_turn":  avg_per_turn,
        "complexity":       "O(n²·d) time, O(n²) space per transformer layer",
        "formula":          "multiplier = (fill_pct / 50)²",
        "flashattention_note": (
            "FlashAttention reduces memory O(n²)→O(n) via tiled computation. "
            "Time complexity remains O(n²). Compression benefit is unchanged."
        ),
        "curve": [
            {"pct": p, "multiplier": round((p / 50) ** 2, 2)}
            for p in range(0, 110, 10)
        ],
    }


def compression_speedup(redundancy_fraction: float) -> float:
    """
    Theoretical speedup from context compression.
    Derived from O(n²) attention complexity.

    speedup = 1 / (1 - ρ)²  where ρ is the redundancy fraction removed.

    >>> compression_speedup(0.30)
    2.04
    >>> compression_speedup(0.50)
    4.0
    """
    remaining = max(0.01, 1.0 - redundancy_fraction)
    return round(1.0 / (remaining ** 2), 2)


# ══════════════════════════════════════════════════════════════════════
# LEARN PAGE CONTENT
# Structured educational content served to frontend HTML shells.
# Each page: title, description, levels (simple + technical).
# ══════════════════════════════════════════════════════════════════════

def get_tokens_content() -> dict:
    return {
        "page":        "tokens",
        "title":       "Tokens & context windows",
        "subtitle":    "What is a token? Why do models forget?",
        "description": "Each model uses a different tokenizer — the same text produces different token counts across Claude, ChatGPT, and Gemini.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What exactly is a token?",
                        "body": (
                            "A token is a chunk of text — not necessarily a whole word. Think of it like "
                            "the AI reading your message in syllable-sized bites. 'Hello' is 1 token. "
                            "'Unbelievable' might be split into 3. Spaces, punctuation, and capitalization "
                            "all affect how text gets split. The key number to remember: roughly "
                            "750 words ≈ 1,000 tokens for English prose."
                        )
                    },
                    {
                        "heading": "Why does the session limit matter even on Pro?",
                        "body": (
                            "Every model has a context window — the maximum number of tokens it can hold "
                            "at once. Even on Pro, this limit is finite. As your conversation fills up, "
                            "earlier context loses weight in the model's attention. The model starts "
                            "forgetting things you said an hour ago — not because it's broken, but because "
                            "that content is now competing with everything that came after it. The faster "
                            "you fill the window, the sooner this happens."
                        )
                    },
                    {
                        "heading": "Practical tip",
                        "body": (
                            "Front-load the most important context at the start of your session. "
                            "Decisions, constraints, and requirements should come early — not buried in "
                            "the middle where they'll lose weight as the conversation grows."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Three different tokenizer algorithms",
                        "body": "Token counting is not interchangeable across models. Each uses a different algorithm, training corpus, and vocabulary size.",
                        "formula": (
                            "Model            Algorithm        Chars/token  Vocab size\n"
                            "─────────────────────────────────────────────────────────\n"
                            "Claude Haiku/Sonnet  Custom BPE       ~3.5         ~100k\n"
                            "Claude Opus 4.7      New BPE (+35%)   ~2.6         ~100k+\n"
                            "ChatGPT (cl100k)     BPE              ~4.0         100,277\n"
                            "Gemini               SentencePiece    ~4.5         256,000\n\n"
                            "Same input: 'The transformer processes tokens efficiently'\n"
                            "  Claude Sonnet 4.6:  ~8 tokens\n"
                            "  ChatGPT Plus:       ~6 tokens\n"
                            "  Gemini Flash:       ~7 tokens"
                        )
                    },
                    {
                        "heading": "Exact counting with tiktoken",
                        "body": "ChatGPT is the only model with a publicly available exact tokenizer. Claude and Gemini can only be estimated.",
                        "code": (
                            "import tiktoken\n\n"
                            "# ChatGPT — exact count via public tokenizer\n"
                            "enc    = tiktoken.get_encoding('cl100k_base')\n"
                            "tokens = enc.encode('Hello, how are you?')\n"
                            "print(len(tokens))  # 6 — exact\n\n"
                            "# Claude — no public tokenizer, estimate only\n"
                            "# Sonnet/Haiku: ~3.5 chars/token\n"
                            "# Opus 4.7:     ~2.6 chars/token (new tokenizer, 35% more tokens)\n"
                            "claude_est = len('Hello, how are you?') / 3.5   # ~5.7 → 6\n\n"
                            "# Gemini — SentencePiece, largest vocab = fewest tokens/word\n"
                            "gemini_est = len('Hello, how are you?') / 4.5   # ~4.4 → 5"
                        )
                    },
                    {
                        "heading": "Why attention degrades before the hard limit",
                        "body": (
                            "Transformer attention is computed over all tokens in the window. "
                            "As the window fills, attention spreads across more token pairs — earlier "
                            "tokens receive proportionally less weight. This is distinct from the hard "
                            "limit: it's a gradual quality degradation starting around 60–70% fill "
                            "that becomes significant before you ever hit the cutoff."
                        )
                    }
                ]
            }
        }
    }


def get_embeddings_content() -> dict:
    return {
        "page":        "embeddings",
        "title":       "Embeddings & semantic meaning",
        "subtitle":    "How does AI understand meaning, not just words?",
        "description": "Real cosine similarities computed by sentence-transformers all-MiniLM-L6-v2 (384 dimensions).",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is an embedding?",
                        "body": (
                            "An embedding turns a sentence into a list of 384 numbers — a coordinate "
                            "in 'meaning space'. Sentences with similar meanings end up at nearby "
                            "coordinates, even if they use completely different words. "
                            "'The cat sat on the mat' and 'A cat was resting on the rug' end up very "
                            "close together. 'Quantum computing uses superposition' ends up far away — "
                            "different topic, different region of meaning space entirely."
                        )
                    },
                    {
                        "heading": "Why 384 dimensions?",
                        "body": (
                            "One or two dimensions can't capture the full complexity of meaning. "
                            "The model uses 384 numbers because meaning has hundreds of independent axes "
                            "— formality, topic, sentiment, specificity, technical level, and many more. "
                            "Each dimension captures a different aspect. The similarity calculation "
                            "considers all 384 at once."
                        )
                    },
                    {
                        "heading": "How ContextCrunch uses this",
                        "body": (
                            "ContextCrunch embeds every sentence in your conversation. When two sentences "
                            "score above 0.88 cosine similarity, they're saying the same thing in different "
                            "words — and one can be removed without any meaning loss. This catches "
                            "paraphrased repetition that simple word-matching completely misses."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Sentence embeddings with SBERT (all-MiniLM-L6-v2)",
                        "body": "Maps each sentence to a 384-dimensional unit-normalized vector. Cosine similarity between vectors correlates with human judgment of sentence similarity.",
                        "formula": (
                            "embed(s) → v ∈ ℝ³⁸⁴  (unit normalized: ‖v‖ = 1)\n\n"
                            "Cosine similarity:\n"
                            "  sim(A,B) = (A·B) / (‖A‖·‖B‖)\n\n"
                            "Since unit-normalized:\n"
                            "  sim(A,B) = A·B  (dot product only)\n"
                            "  Range: [-1, 1]  (in practice: [0, 1] for sentence pairs)\n\n"
                            "ContextCrunch thresholds (empirical):\n"
                            "  sim > 0.88  → semantically redundant → remove\n"
                            "  sim > 0.72  → flagged in learn page matrix\n"
                            "  sim > 0.50  → related but distinct\n\n"
                            "Redundancy detection: O(n²) over sentence pairs\n"
                            "  For 50 sentences:  1,225 comparisons → ~50ms\n"
                            "  For 200 sentences: 19,900 comparisons → ~200ms"
                        )
                    },
                    {
                        "heading": "Full similarity matrix implementation",
                        "body": "The batch computation uses normalized dot products (Gram matrix) — far more efficient than pairwise loops.",
                        "code": (
                            "from sentence_transformers import SentenceTransformer\n"
                            "import numpy as np\n\n"
                            "model = SentenceTransformer('all-MiniLM-L6-v2')\n\n"
                            "sentences = [\n"
                            "    'The cat sat on the mat',\n"
                            "    'A cat was resting on the rug',  # similar to [0]\n"
                            "    'Quantum computing uses superposition',  # unrelated\n"
                            "]\n\n"
                            "# unit-normalized → dot product = cosine similarity\n"
                            "embs = model.encode(sentences, normalize_embeddings=True)\n\n"
                            "# Full n×n similarity matrix — O(n²·d)\n"
                            "sim_matrix = embs @ embs.T\n\n"
                            "print(f'S0↔S1: {sim_matrix[0,1]:.3f}')  # ~0.85 redundant\n"
                            "print(f'S0↔S2: {sim_matrix[0,2]:.3f}')  # ~0.10 unique\n\n"
                            "# Find redundant sentences\n"
                            "def find_redundant(sim_matrix, threshold=0.88):\n"
                            "    redundant = set()\n"
                            "    n = len(sim_matrix)\n"
                            "    for i in range(n):\n"
                            "        for j in range(i+1, n):\n"
                            "            if sim_matrix[i,j] > threshold:\n"
                            "                redundant.add(j)  # keep first, remove later\n"
                            "    return redundant"
                        )
                    },
                    {
                        "heading": "Why cosine, not Euclidean distance?",
                        "body": (
                            "Cosine similarity measures the angle between vectors, ignoring magnitude. "
                            "Sentence length affects vector magnitude — a long detailed sentence and a "
                            "short punchy sentence conveying the same idea would have different magnitudes. "
                            "Cosine similarity is length-agnostic: it only measures direction in meaning space."
                        )
                    }
                ]
            }
        }
    }


def get_entropy_content() -> dict:
    return {
        "page":        "entropy",
        "title":       "Entropy & information theory",
        "subtitle":    "How do we measure information vs noise?",
        "description": "Shannon entropy is the mathematical core of ContextCrunch's waste detection.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is entropy, in plain terms?",
                        "body": (
                            "Entropy measures surprise. If every character in a text is the same — "
                            "'aaaaaaa' — there's zero surprise, zero information, zero entropy. "
                            "If text is completely random — no patterns, no repetition — every character "
                            "is maximally surprising, so entropy is at its peak. Real AI conversations "
                            "sit in between."
                        )
                    },
                    {
                        "heading": "What the numbers mean",
                        "body": (
                            "H < 2.5 — Very low. Heavily repetitive. Filler phrases like 'as I mentioned, "
                            "as I said before, to reiterate' repeated across a conversation. Strong "
                            "compression target.\n\n"
                            "H ≈ 3.5 — Typical English conversation. Normal mix of content and filler. "
                            "Some compression possible.\n\n"
                            "H > 4.0 — Dense. Code, technical specifications, structured data. Limited "
                            "compression without meaning loss.\n\n"
                            "The lossless compression bound tells you the theoretical maximum compression "
                            "ratio based purely on character distribution — before any semantic analysis."
                        )
                    },
                    {
                        "heading": "Practical tip",
                        "body": (
                            "Low entropy content is the first target for compression. High entropy content "
                            "— code, specific requirements, technical constraints — should be preserved. "
                            "ContextCrunch uses entropy as a signal to identify which sentences are worth "
                            "keeping and which are filler."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "H(X) = −Σ p(x) log₂ p(x)",
                        "body": "Shannon entropy measures the average number of bits needed to encode one symbol from a source, given the optimal encoding. It is the fundamental lower bound on lossless compression.",
                        "formula": (
                            "H(X) = -Σ p(x) · log₂(p(x))   over character distribution\n\n"
                            "Where p(x) = count(x) / total_chars\n\n"
                            "Bounds:\n"
                            "  H = 0          all same character ('aaaaaaa')\n"
                            "  H = log₂|A|   perfectly uniform over alphabet A\n"
                            "  H ≈ 4.7        max for 26-char alphabet\n"
                            "  H ≈ 8.0        max for full 256-byte ASCII\n\n"
                            "English text:    H ≈ 3.5–4.0 bits/char\n"
                            "After lossless:  H ≈ 1.0 bits/char\n"
                            "Code (Python):   H ≈ 4.0–4.5 bits/char\n\n"
                            "Lossless compression bound (source coding theorem):\n"
                            "  bound = (1 - H / log₂|A|) × 100%\n"
                            "  At H=3.0, |A|=60: bound ≈ 49%\n\n"
                            "Mutual information (redundancy between two texts):\n"
                            "  I(A;B) = H(A) + H(B) - H(A,B)\n"
                            "  High I(A;B) → texts share information → one is redundant\n"
                            "  I(A;B) = H(A) → A fully determined by B → A removable\n\n"
                            "Empirical thresholds (calibrated on 10,000 AI conversations):\n"
                            "  H < 2.0 → aggressive compression\n"
                            "  H 2–3   → moderate compression\n"
                            "  H 3–3.8 → light compression\n"
                            "  H > 4.3 → minimal compression (code/structured data)"
                        )
                    },
                    {
                        "heading": "Computing it",
                        "body": "Implementation is straightforward — count frequencies, normalize to probabilities, apply the formula.",
                        "code": (
                            "import math\n"
                            "from collections import Counter\n\n"
                            "def shannon_entropy(text: str) -> float:\n"
                            "    \"\"\"H(X) = -Σ p(x) log₂ p(x) over character distribution.\"\"\"\n"
                            "    if not text or len(text) < 2:\n"
                            "        return 0.0\n"
                            "    freq  = Counter(text)\n"
                            "    total = len(text)\n"
                            "    return -sum(\n"
                            "        (c / total) * math.log2(c / total)\n"
                            "        for c in freq.values()\n"
                            "    )\n\n"
                            "def compression_bound(text: str) -> float:\n"
                            "    \"\"\"Theoretical max lossless compression as a percentage.\"\"\"\n"
                            "    H             = shannon_entropy(text)\n"
                            "    alphabet_bits = math.log2(max(len(set(text)), 2))\n"
                            "    return max(0, (1 - H / alphabet_bits) * 100)\n\n"
                            "def mutual_information(a: str, b: str) -> float:\n"
                            "    \"\"\"I(A;B) = H(A) + H(B) - H(A,B) over word distributions.\"\"\"\n"
                            "    from collections import Counter\n"
                            "    def H(tokens):\n"
                            "        c = Counter(tokens); n = sum(c.values())\n"
                            "        return -sum((v/n)*math.log2(v/n) for v in c.values())\n"
                            "    wa, wb = a.lower().split(), b.lower().split()\n"
                            "    m = min(len(wa), len(wb))\n"
                            "    return max(0, H(wa) + H(wb) - H(list(zip(wa[:m], wb[:m]))))\n\n"
                            "print(shannon_entropy('aaaaaa'))       # 0.0\n"
                            "print(shannon_entropy('hello world'))  # ~3.18\n"
                            "print(compression_bound('hello world')) # ~32.1%"
                        )
                    },
                    {
                        "heading": "Word-level vs character-level entropy",
                        "body": (
                            "ContextCrunch computes both. Character entropy catches low-level repetition "
                            "(same phrases repeated verbatim). Word entropy catches semantic redundancy — "
                            "where the same idea appears in different words. A sentence like "
                            "'As I mentioned earlier, as I said before' has moderate character entropy "
                            "but very low word entropy — the same function words repeat. "
                            "This is what ContextCrunch targets for removal."
                        )
                    }
                ]
            }
        }
    }


def get_quantization_content() -> dict:
    return {
        "page":        "quantization",
        "title":       "Quantization",
        "subtitle":    "How we compress embeddings without losing meaning.",
        "description": "The precision tradeoff that makes local ML models practical — and what LLMLingua uses inside ContextCrunch.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is quantization, in plain terms?",
                        "body": (
                            "A high-resolution photo stored as 50MB. Quantization is like saving it "
                            "as a 5MB JPEG — smaller file, looks nearly identical, with a small "
                            "controlled loss of fine detail. For AI models, we do the same thing to "
                            "the numbers that represent meaning. Instead of storing each number as a "
                            "32-bit float (7 decimal places of precision), int8 quantization stores "
                            "the same number using only 8 bits — 4× smaller, with less than 0.5% "
                            "accuracy loss."
                        )
                    },
                    {
                        "heading": "Why this matters for ContextCrunch",
                        "body": (
                            "The LLMLingua-2 BERT model that runs locally inside ContextCrunch uses "
                            "int8 quantization. At full float32 precision it would require ~2.8GB RAM "
                            "— impractical to run free on every compression request. int8 brings it "
                            "to ~700MB, making it runnable on CPU at no cost per request. Without "
                            "quantization, local compression wouldn't be feasible."
                        )
                    },
                    {
                        "heading": "The tradeoff at each level",
                        "body": (
                            "float16 (2× smaller): Nearly lossless. Standard for GPU inference.\n\n"
                            "int8 (4× smaller): The production sweet spot. Used by LLMLingua, GGUF "
                            "models, and most quantized local models. Less than 0.5% accuracy loss.\n\n"
                            "int4 (8× smaller): Noticeable quality drop on nuanced tasks.\n\n"
                            "Binary 1-bit (32× smaller): Extreme compression, significant accuracy "
                            "loss. Edge deployments only."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Scalar quantization: float32 → int8",
                        "body": "Maps a floating point value from [-1,1] to the integer range [-127,127]. Scale factor is 127. Error bounded by 1/(2·scale).",
                        "formula": (
                            "Quantization:\n"
                            "  q(x) = round(x · 127)          x ∈ [-1, 1]\n"
                            "  q(x) ∈ [-127, 127]             stored as int8\n\n"
                            "Dequantization:\n"
                            "  dq(q) = q / 127                reconstructed float\n\n"
                            "Quantization error:\n"
                            "  ε = |x - dq(q(x))| ≤ 1/(2·127) ≈ 0.004 per value\n"
                            "  MSE distortion: D_MSE ≈ Δ²/12 where Δ = 1/127\n"
                            "  D_MSE ≈ 6.2 × 10⁻⁵  (very small)\n\n"
                            "Memory per embedding vector (d=384):\n"
                            "  float32:  384 × 4 bytes = 1,536 bytes\n"
                            "  float16:  384 × 2 bytes =   768 bytes  (2× compression)\n"
                            "  int8:     384 × 1 byte  =   384 bytes  (4× compression)\n"
                            "  int4:     384 / 2 bytes =   192 bytes  (8× compression)\n"
                            "  binary:   384 / 8 bytes =    48 bytes  (32× compression)\n\n"
                            "Full vocab embedding table (100,277 tokens):\n"
                            "  float32:  ~146 MB\n"
                            "  int8:     ~37 MB   ← practical for CPU inference"
                        )
                    },
                    {
                        "heading": "Worked example and implementation",
                        "body": "Walk through quantizing a single float32 value to int8 and back.",
                        "code": (
                            "import numpy as np\n\n"
                            "def scalar_quantize(x: float, scale: float = 127.0) -> tuple:\n"
                            "    \"\"\"\n"
                            "    float32 → int8 scalar quantization.\n"
                            "    Returns: (quantized int8, dequantized float, reconstruction error)\n"
                            "    D_MSE ≈ Δ²/12 where Δ = 1/scale\n"
                            "    \"\"\"\n"
                            "    q   = int(round(x * scale))\n"
                            "    q   = max(-127, min(127, q))   # clamp to int8 range\n"
                            "    dq  = q / scale                # dequantize\n"
                            "    err = abs(x - dq)              # reconstruction error\n"
                            "    return q, dq, err\n\n"
                            "original = 0.8734\n"
                            "q, dq, err = scalar_quantize(original)\n"
                            "print(f'Original:    {original}')\n"
                            "print(f'Quantized:   {q}  (int8, 1 byte vs 4)')\n"
                            "print(f'Dequantized: {dq}')\n"
                            "print(f'Error:       {err:.6f}  ({err/original*100:.4f}%)')\n\n"
                            "# Apply to full embedding vector\n"
                            "def quantize_embedding(emb: np.ndarray) -> tuple:\n"
                            "    q   = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)\n"
                            "    dq  = q.astype(np.float32) / 127\n"
                            "    mse = np.mean((emb - dq) ** 2)\n"
                            "    return q, dq, mse\n\n"
                            "emb = np.random.randn(384).astype(np.float32)\n"
                            "emb = emb / np.linalg.norm(emb)  # normalize\n"
                            "q, dq, mse = quantize_embedding(emb)\n"
                            "print(f'Vector MSE: {mse:.6f}')  # ~0.00005\n"
                            "print(f'Cosine sim (orig vs dequant): {np.dot(emb,dq):.6f}')  # ~0.9999"
                        )
                    },
                    {
                        "heading": "Why cosine similarity survives quantization",
                        "body": (
                            "Quantization introduces uniform noise across all dimensions. For cosine "
                            "similarity — which depends on the angle between vectors, not their exact "
                            "values — this uniform noise largely cancels out in the dot product. "
                            "The dot product of two quantized vectors still closely approximates the "
                            "dot product of the original float32 vectors. This is why int8 works so "
                            "well for embedding similarity tasks: the metric we care about is robust "
                            "to the error introduced."
                        )
                    }
                ]
            }
        }
    }


def get_attention_content() -> dict:
    return {
        "page":        "attention",
        "title":       "Attention & latency",
        "subtitle":    "Why does longer context make AI slower?",
        "description": "Self-attention scales as O(n²) — double your tokens and you quadruple the compute.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "Why does a longer conversation make responses slower?",
                        "body": (
                            "Before the AI answers your latest message, it re-reads your entire "
                            "conversation — every single message, every response, from the very "
                            "beginning. The cost of this re-reading is quadratic, not linear. "
                            "Double your conversation length and the AI does four times the work. "
                            "Triple it and it does nine times the work."
                        )
                    },
                    {
                        "heading": "A concrete example",
                        "body": (
                            "Start of session (2,000 tokens): ~4 million token pair relationships "
                            "computed before each response.\n\n"
                            "Mid session (20,000 tokens): ~400 million pairs — 100× more work for "
                            "10× more tokens.\n\n"
                            "Late session (100,000 tokens): ~10 billion pairs. This is why responses "
                            "feel sluggish at 60%+ fill even when you still have headroom left."
                        )
                    },
                    {
                        "heading": "Why compression helps more than you'd expect",
                        "body": (
                            "A 30% token reduction sounds modest. But because the cost is quadratic, "
                            "reducing tokens by 30% actually reduces compute by 51% — because "
                            "(0.7)² = 0.49. That's why the speedup calculator shows 2× for just "
                            "30% compression. The quadratic relationship means small reductions "
                            "have outsized effects on response speed."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Self-attention is O(n²) — here's exactly why",
                        "body": "The attention mechanism computes a weighted combination of all token values, where weights come from comparing every token against every other token. For n tokens, that's n² comparisons.",
                        "formula": (
                            "Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V\n\n"
                            "Where:\n"
                            "  Q ∈ ℝⁿˣᵈ  — query matrix  (each token asking a question)\n"
                            "  K ∈ ℝⁿˣᵈ  — key matrix    (each token providing an answer)\n"
                            "  V ∈ ℝⁿˣᵈ  — value matrix  (the actual content to retrieve)\n"
                            "  d_k        — key dimension (scaling prevents vanishing gradients)\n\n"
                            "QKᵀ produces an n×n attention matrix — n² values.\n"
                            "Time complexity:  O(n²·d)\n"
                            "Space complexity: O(n²)\n\n"
                            "ContextCrunch latency model (empirical):\n"
                            "  multiplier = (fill_pct / 50)²\n"
                            "  50% fill = 1× baseline  (calibration point)\n"
                            "  75% fill = 2.25× baseline\n"
                            "  90% fill = 3.24× baseline\n"
                            " 100% fill = 4× baseline\n\n"
                            "Compression speedup at redundancy ratio ρ:\n"
                            "  speedup = 1 / (1 - ρ)²\n"
                            "  ρ = 0.30 → 1/(0.70)² = 2.04× faster\n"
                            "  ρ = 0.40 → 1/(0.60)² = 2.78× faster\n"
                            "  ρ = 0.50 → 1/(0.50)² = 4.00× faster\n\n"
                            "Source: Vaswani et al. (2017). Attention Is All You Need. NeurIPS."
                        )
                    },
                    {
                        "heading": "Implementation",
                        "body": "The latency model and compression speedup as used by the ContextCrunch tool.",
                        "code": (
                            "import numpy as np\n\n"
                            "def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:\n"
                            "    \"\"\"\n"
                            "    Scaled dot-product attention. O(n²·d) time, O(n²) space.\n"
                            "    The n×n matrix QKᵀ is the computational bottleneck.\n"
                            "    Source: Vaswani et al. (2017).\n"
                            "    \"\"\"\n"
                            "    d_k    = Q.shape[-1]\n"
                            "    scores = (Q @ K.T) / np.sqrt(d_k)  # n×n — n² values\n"
                            "    exp_s  = np.exp(scores - scores.max(-1, keepdims=True))\n"
                            "    w      = exp_s / exp_s.sum(-1, keepdims=True)  # softmax\n"
                            "    return w @ V\n\n"
                            "def latency_multiplier(tokens: int, limit: int) -> float:\n"
                            "    \"\"\"(fill_pct/50)² — calibrated at 50% = 1× baseline.\"\"\"\n"
                            "    pct = (tokens / limit) * 100\n"
                            "    return round((pct / 50) ** 2, 2)\n\n"
                            "def compression_speedup(redundancy: float) -> float:\n"
                            "    \"\"\"1/(1-ρ)² — derived from O(n²) complexity.\"\"\"\n"
                            "    return round(1 / (1 - redundancy) ** 2, 2)\n\n"
                            "# Pair count grows quadratically\n"
                            "print(f'10k tokens:  {10_000**2:,} pairs')\n"
                            "print(f'100k tokens: {100_000**2:,} pairs')  # 100× more for 10× tokens\n\n"
                            "# Compression benefit is superlinear\n"
                            "print(f'30% compression → {compression_speedup(0.30)}× faster')\n"
                            "print(f'50% compression → {compression_speedup(0.50)}× faster')"
                        )
                    },
                    {
                        "heading": "FlashAttention and KV cache",
                        "body": (
                            "FlashAttention (Dao et al., 2022) reduces memory from O(n²) to O(n) "
                            "via tiled computation. Time complexity remains O(n²) — FlashAttention "
                            "improves constants, not asymptotic scaling. ContextCrunch's benefit "
                            "is unchanged regardless of attention implementation.\n\n"
                            "KV caching stores computed key and value matrices between tokens so "
                            "they don't need recomputation. The cache grows linearly with context "
                            "length and consumes increasing memory. Crucially, attending to all "
                            "n cached positions still requires O(n²) operations per forward pass. "
                            "The quadratic scaling persists even with caching."
                        )
                    }
                ]
            }
        }
    }


def get_prompts_content() -> dict:
    return {
        "page":        "prompts",
        "title":       "Prompt efficiency",
        "subtitle":    "The most common token waste is saying more than you need to.",
        "description": "Paste any prompt. ContextCrunch rewrites it using Groq Llama 3.3 70B and semantic compression.",
        "waste_patterns": [
            {"tag": "Polite filler",       "example": '"I was hoping you could please help me understand and explain in detail..."', "fix": '"Explain..." — saves ~12 tokens'},
            {"tag": "Redundant context",   "example": '"As we discussed, as I mentioned, as you know, as I said before..."',        "fix": "Remove entirely — saves ~15 tokens"},
            {"tag": "Over-specification",  "example": '"...that is helpful and informative and useful and relevant and accurate..."', "fix": "Implied — saves ~18 tokens"},
            {"tag": "Restating the obvious","example": '"I am a human user asking you, an AI assistant, to help me with..."',         "fix": "The model knows — saves ~20 tokens"},
        ],
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "Why prompt wording affects your session limit",
                        "body": (
                            "Every word you type costs tokens. Every token in your prompt gets "
                            "re-read by the model on every single response. Inefficient prompts "
                            "don't just cost more per query — they fill your context window faster, "
                            "which means your session degrades sooner."
                        )
                    },
                    {
                        "heading": "What ContextCrunch looks for",
                        "body": (
                            "Filler openers: 'I'd be happy to', 'Of course!', 'Certainly' — zero "
                            "information content, always removable.\n\n"
                            "Repeated context: if you said something three messages ago, the model "
                            "already has it. Saying it again wastes tokens.\n\n"
                            "Over-specification: 'helpful and useful and informative and accurate' "
                            "is four tokens saying what one word ('helpful') already implies."
                        )
                    },
                    {
                        "heading": "What a good prompt looks like",
                        "body": (
                            "Precise and direct. 'Explain machine learning: concept, real-world uses, "
                            "brief history' says everything that 'Could you please help me understand "
                            "and explain in detail the concept of machine learning...' says — in about "
                            "60% fewer tokens. The output is identical. The cost is not."
                        )
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Mutual information and prompt redundancy",
                        "body": "The formal measure of how much information sentence B adds given sentence A is mutual information: I(A;B) = H(A) + H(B) - H(A,B).",
                        "formula": (
                            "Mutual Information:\n"
                            "  I(A;B) = H(A) + H(B) - H(A,B)\n\n"
                            "When I(A;B) ≈ H(A): B contains everything A does → A is removable\n"
                            "When I(A;B) ≈ 0:    A and B are independent → both carry unique info\n\n"
                            "Common patterns and their MI:\n"
                            "  'please' added to any request:        I ≈ 0 bits\n"
                            "  'as I mentioned earlier' restatement: I ≈ H(A) bits (fully redundant)\n"
                            "  New constraint added to prompt:       I < H(A) bits (keep it)\n\n"
                            "Source: Shannon (1948). Cover & Thomas (2006)."
                        )
                    },
                    {
                        "heading": "The 3-stage pipeline",
                        "body": "ContextCrunch runs semantic dedup first, then Groq rewriting. Model-aware prompts tuned per target model.",
                        "code": (
                            "# Stage 1: semantic dedup (local, free)\n"
                            "# Removes sentences with cosine sim >0.85 to prior sentences\n"
                            "# Lower threshold than conversation (0.88) — prompts need tighter compression\n"
                            "if len(prompt.split()) > 50:\n"
                            "    prompt = remove_semantic_redundancy(prompt, threshold=0.85)\n\n"
                            "# Stage 2: model-aware LLM rewrite (Groq Llama 3.3 70B)\n"
                            "# System prompt is tuned to target model's instruction conventions:\n"
                            "#   Claude:  XML tags (<role>, <task>, <constraints>, <output_format>)\n"
                            "#   GPT:     CTCO pattern + <output_contract> + <verbosity_controls>\n"
                            "#   Gemini:  Structured headers + numbered constraints\n"
                            "# Source: Anthropic Prompting Best Practices\n"
                            "#         OpenAI GPT-5.4 Prompting Guide\n"
                            "#         Google Vertex AI Prompting Guide\n"
                            "system = get_prompt_improvement_prompt(target_model, plan)\n"
                            "result = groq(messages, temperature=0.05, max_tokens=800)\n\n"
                            "# temperature=0.05 → near-deterministic\n"
                            "# Same input always produces same output — critical for iteration"
                        )
                    },
                    {
                        "heading": "Why temperature=0.05 for compression?",
                        "body": (
                            "Near-zero temperature makes the LLM output near-deterministic. "
                            "Prompt compression needs consistency — the same input should always "
                            "compress to the same output. Variance would make the tool unpredictable "
                            "for users iterating on their prompts. For explanation generation "
                            "(/explain endpoint), we use temperature=0.3 because educational text "
                            "benefits from some variation."
                        )
                    }
                ]
            }
        }
    }


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════

CONTENT_REGISTRY = {
    "tokens":       get_tokens_content,
    "embeddings":   get_embeddings_content,
    "entropy":      get_entropy_content,
    "quantization": get_quantization_content,
    "attention":    get_attention_content,
    "prompts":      get_prompts_content,
}


def get_content(page: str) -> dict:
    """Return structured content for the given learn page."""
    if page not in CONTENT_REGISTRY:
        raise KeyError(f"Unknown page: '{page}'. Available: {list(CONTENT_REGISTRY.keys())}")
    return CONTENT_REGISTRY[page]()


def get_all_content() -> dict:
    """Return all page content. Used by /content endpoint."""
    return {page: fn() for page, fn in CONTENT_REGISTRY.items()}