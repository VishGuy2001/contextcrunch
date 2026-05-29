"""
math_engine.py — Shannon entropy, redundancy detection, O(n²) attention latency,
                  quantization analysis, token budget modeling, embedding analysis

Core algorithms:

1. Shannon entropy:     H(X) = -Σ p(x) log₂ p(x)
2. Redundancy:          Jaccard + TF-cosine + optional neural embeddings
3. Attention latency:   T(n) ∝ n²  —  multiplier = (context% / 50)²
4. Quantization:        float32 → int8 precision/size tradeoff with real numbers
5. Token budget:        per-model fill projection, turns remaining, cost estimate
6. Deep per-concept:    rich structured output for each learn page
"""
import math
import re
import numpy as np
from collections import Counter
from typing import List, Optional


# ── STOPWORDS ─────────────────────────────────────────────────────────
_STOP = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','i','you','he',
    'she','it','we','they','this','that','and','or','but','in','on','at','to','for',
    'of','with','by','from','as','not','just','so','what','how','when','where','who',
    'which','if','then','than','too','very','also','can','get','got','like','go',
    'going','want','need','think','know','me','my','your','our','their','its',
    'im','okay','ok','yes','no','hi','hey','hello',
}


# ── CORE MATH ─────────────────────────────────────────────────────────

def shannon_entropy(text: str) -> float:
    """H(X) = -Σ p(x) log₂ p(x) over character distribution."""
    if not text or len(text) < 2:
        return 0.0
    freq  = Counter(text)
    total = len(text)
    return round(-sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0), 3)


def word_entropy(text: str) -> float:
    """H(X) over word distribution — better signal for semantic redundancy."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:
        return 0.0
    freq  = Counter(words)
    total = len(words)
    return round(-sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0), 3)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """sim(A,B) = (A·B) / (‖A‖·‖B‖)"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(text_a: str, text_b: str) -> float:
    """Jaccard on content words after stopword removal."""
    wa = set(re.sub(r'[^a-z0-9\s]', '', text_a.lower()).split()) - _STOP
    wb = set(re.sub(r'[^a-z0-9\s]', '', text_b.lower()).split()) - _STOP
    if not wa and not wb:
        ra = set(text_a.lower().split())
        rb = set(text_b.lower().split())
        if not ra or not rb:
            return 0.0
        return len(ra & rb) / len(ra | rb)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _tf_cosine(text_a: str, text_b: str) -> float:
    """TF cosine on content words — catches semantic paraphrasing."""
    def tf(text):
        words = [w for w in re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
                 if len(w) > 1 and w not in _STOP]
        if not words:
            return {}
        n = len(words)
        v: dict = {}
        for w in words:
            v[w] = v.get(w, 0) + 1/n
        return v
    va, vb = tf(text_a), tf(text_b)
    if len(va) < 2 or len(vb) < 2:
        return 0.0
    dot = sum(va[w] * vb.get(w, 0) for w in va)
    na  = math.sqrt(sum(v*v for v in va.values()))
    nb  = math.sqrt(sum(v*v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return round(dot / (na * nb), 3)


def redundancy_score(text: str, embeddings: Optional[List[np.ndarray]] = None, threshold: float = 0.72) -> dict:
    """
    Detect what percentage of sentences are semantically redundant.
    Returns score, redundant indices, top similar pairs for visualization.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]

    if len(sentences) < 2:
        return {"score": 0, "redundant": [], "removable": 0, "method": "none",
                "sentence_count": len(sentences), "pairs": [], "top_redundant_pairs": []}

    redundant_set = set()
    method        = "word_overlap"
    pair_scores   = []

    if embeddings and len(embeddings) == len(sentences):
        method = "embedding"
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                pair_scores.append({"i": i, "j": j, "sim": round(sim, 3)})
                if j not in redundant_set and sim > threshold:
                    redundant_set.add(j)
    else:
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if j not in redundant_set:
                    j_sim = _jaccard(sentences[i], sentences[j])
                    c_sim = _tf_cosine(sentences[i], sentences[j]) if j_sim < 0.4 else 0.0
                    sim   = max(j_sim, c_sim)
                    pair_scores.append({"i": i, "j": j, "sim": round(sim, 3)})
                    if j_sim >= 0.4 or c_sim >= 0.65:
                        redundant_set.add(j)

    score     = round((len(redundant_set) / max(len(sentences), 1)) * 100, 1)
    removable = max(0, int(sum(len(sentences[i]) for i in redundant_set) / 3.8))
    top_pairs = sorted(pair_scores, key=lambda x: -x["sim"])[:5]

    return {
        "score":    min(score, 90.0),
        "redundant": list(redundant_set),
        "removable": removable,
        "method":    method,
        "sentence_count": len(sentences),
        "sentences": sentences[:20],
        "pairs":     pair_scores[:50],
        "top_redundant_pairs": [
            {
                "a":   sentences[p["i"]][:80] + ("..." if len(sentences[p["i"]]) > 80 else ""),
                "b":   sentences[p["j"]][:80] + ("..." if len(sentences[p["j"]]) > 80 else ""),
                "sim": p["sim"],
            }
            for p in top_pairs if p["sim"] > 0.3
        ],
    }


def attention_cost_multiplier(current_tokens: int, limit_tokens: int) -> dict:
    """O(n²) latency model. Returns multiplier, zone, O(n²) curve."""
    if limit_tokens == 0:
        return {"multiplier": 1.0, "percentage": 0, "zone": "safe", "message": "Response speed is optimal.", "curve": []}

    pct  = (current_tokens / limit_tokens) * 100
    mult = round((pct / 50) ** 2, 2) if pct > 0 else 0.0

    if pct < 40:   zone, msg = "safe",    "Response speed is optimal."
    elif pct < 70: zone, msg = "warning", f"Responses ~{mult}× slower than conversation start."
    else:          zone, msg = "danger",  f"Responses ~{mult}× slower. Compress now."

    curve = [{"pct": p, "multiplier": round((p / 50) ** 2, 2)} for p in range(0, 110, 10)]

    return {
        "multiplier": mult, "percentage": round(pct, 1),
        "zone": zone, "message": msg, "curve": curve,
        "pair_count": current_tokens ** 2,
        "pair_count_at_limit": limit_tokens ** 2,
    }


def theoretical_compression_bound(text: str) -> dict:
    """Theoretical maximum lossless compression ratio via Shannon entropy."""
    if not text:
        return {"bound": 0, "entropy": 0, "bits_saved": 0, "interpretation": ""}
    H             = shannon_entropy(text)
    alphabet_bits = math.log2(max(len(set(text)), 2))
    bound         = max(0, round((1 - H / alphabet_bits) * 100, 1))
    return {
        "bound":          bound,
        "entropy":        H,
        "bits_saved":     max(0, int(len(text)*8 - H*len(text))),
        "alphabet_size":  len(set(text)),
        "alphabet_bits":  round(alphabet_bits, 3),
        "interpretation": f"Lossless compression bound: {bound}%. Semantic compression typically achieves 20–60% token reduction.",
    }


# ── RICH PER-CONCEPT ANALYSIS ─────────────────────────────────────────

def analyze_entropy_deep(text: str) -> dict:
    """
    Full entropy analysis for learn/entropy page.
    Returns char distribution, word distribution, bigram entropy,
    compression bound, and compressibility rating.
    """
    if not text or len(text) < 2:
        return {"error": "Text too short"}

    H_char   = shannon_entropy(text)
    H_word   = word_entropy(text)
    bigrams  = [text[i:i+2] for i in range(len(text)-1)]
    H_bigram = 0.0
    if bigrams:
        freq_bg  = Counter(bigrams)
        n_bg     = len(bigrams)
        H_bigram = round(-sum((c/n_bg)*math.log2(c/n_bg) for c in freq_bg.values()), 3)

    freq      = Counter(text)
    total     = len(text)
    top_chars = [
        {
            "char":  (k if k not in ('\n', '\t', ' ') else {'\\n': '\\n', '\t': '\\t', ' ': '·'}.get(k, k)),
            "count": v,
            "prob":  round(v/total, 4),
            "bits":  round(-math.log2(v/total), 2),
        }
        for k, v in sorted(freq.items(), key=lambda x: -x[1])[:15]
    ]

    words     = re.findall(r'\b\w+\b', text.lower())
    freq_word = Counter(words)
    top_words = [{"word": w, "count": c, "prob": round(c/max(len(words),1), 4)}
                 for w, c in freq_word.most_common(10)]

    bound = theoretical_compression_bound(text)

    if H_char < 2.0:   rating, color = "Very high", "#1a6b4a"
    elif H_char < 3.0: rating, color = "High",      "#2d8a5e"
    elif H_char < 3.8: rating, color = "Moderate",  "#c4602a"
    elif H_char < 4.3: rating, color = "Low",        "#8b1a1a"
    else:              rating, color = "Very low",   "#5a0f0f"

    return {
        "entropy_char":    H_char,
        "entropy_word":    H_word,
        "entropy_bigram":  H_bigram,
        "compression_bound": bound,
        "compressibility": {"rating": rating, "color": color},
        "char_count":      total,
        "word_count":      len(words),
        "unique_chars":    len(freq),
        "unique_words":    len(freq_word),
        "top_chars":       top_chars,
        "top_words":       top_words,
        "interpretation": (
            "Very low entropy — highly repetitive, strong compression possible" if H_char < 2 else
            "Low entropy — redundant content, good compression candidate"       if H_char < 3 else
            "Moderate entropy — typical conversational text"                    if H_char < 3.8 else
            "High entropy — information-dense, limited compression headroom"    if H_char < 4.3 else
            "Very high entropy — code or structured data, minimal compressibility"
        ),
    }


def analyze_redundancy_deep(text: str) -> dict:
    """
    Full redundancy analysis for learn/similarity page.
    Returns per-sentence scoring, filler detection, top redundant pairs.
    """
    sentences    = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    fillers      = [
        "i'd be happy to", "of course", "great question", "certainly", "absolutely",
        "as i mentioned", "as i said", "to reiterate", "let me explain", "in other words",
        "it's worth noting", "it goes without saying", "as previously", "i hope this helps",
        "please let me know", "happy to help", "feel free to", "does that make sense",
    ]
    text_lower   = text.lower()
    filler_hits  = [f for f in fillers if f in text_lower]
    filler_count = sum(text_lower.count(f) for f in fillers)

    red = redundancy_score(text)

    sentence_analysis = [
        {
            "index":        i,
            "text":         s[:100] + ("..." if len(s) > 100 else ""),
            "entropy":      shannon_entropy(s),
            "words":        len(s.split()),
            "is_redundant": i in red["redundant"],
            "filler_risk":  shannon_entropy(s) < 2.5 and len(s.split()) < 15,
        }
        for i, s in enumerate(sentences[:20])
    ]

    return {
        **red,
        "filler_phrases_found": filler_hits[:10],
        "filler_count":         filler_count,
        "sentence_analysis":    sentence_analysis,
        "estimated_token_savings": red["removable"],
        "recommendation": (
            "High redundancy — compression strongly recommended"   if red["score"] > 40 else
            "Moderate redundancy — compression will help"          if red["score"] > 20 else
            "Low redundancy — text is reasonably efficient"        if red["score"] > 8  else
            "Minimal redundancy — text is well-optimized"
        ),
    }


def analyze_attention_deep(token_count: int, limit: int, avg_tokens_per_turn: int = 150) -> dict:
    """
    Full O(n²) attention analysis for learn/attention page.
    Returns cost curve, pair counts, fill-level table, turns remaining.
    """
    att = attention_cost_multiplier(token_count, limit)

    fill_levels = [
        {
            "label":       f"{pct}% fill",
            "tokens":      int(limit * pct / 100),
            "pairs":       int((limit * pct / 100) ** 2),
            "multiplier":  round((pct / 50) ** 2, 2),
        }
        for pct in [10, 25, 50, 75, 100]
    ]

    current_pct  = (token_count / limit * 100) if limit else 0
    tokens_to_80 = max(0, int(limit * 0.8) - token_count)
    turns_left   = max(0, tokens_to_80 // max(avg_tokens_per_turn, 1))

    return {
        **att,
        "current_tokens":      token_count,
        "limit":               limit,
        "fill_levels":         fill_levels,
        "turns_to_80pct":      turns_left,
        "avg_tokens_per_turn": avg_tokens_per_turn,
        "complexity_class":    "O(n²)",
        "formula":             "T(n) = k · n²  where n = token count",
    }


def analyze_quantization_deep(embedding_dim: int = 384, vocab_size: int = 100277) -> dict:
    """
    Quantization analysis for learn/quantization page.
    Shows float32 → int8 tradeoff with real numbers and worked example.
    """
    def mb(bytes_per): return round(bytes_per * vocab_size / 1_000_000, 1)

    formats = [
        {"name": "float32 (baseline)", "bits": 32, "bytes_per_vec": embedding_dim*4,
         "vocab_table_mb": mb(embedding_dim*4), "compression": "1×",
         "accuracy_loss": "0%", "notes": "Full precision. Used during training."},
        {"name": "float16", "bits": 16, "bytes_per_vec": embedding_dim*2,
         "vocab_table_mb": mb(embedding_dim*2), "compression": "2×",
         "accuracy_loss": "~0.1%", "notes": "Standard inference. Near-lossless."},
        {"name": "int8", "bits": 8, "bytes_per_vec": embedding_dim,
         "vocab_table_mb": mb(embedding_dim), "compression": "4×",
         "accuracy_loss": "~0.3–0.5%", "notes": "Production standard. Used by LLMLingua, GGUF models."},
        {"name": "int4", "bits": 4, "bytes_per_vec": embedding_dim//2,
         "vocab_table_mb": mb(embedding_dim//2), "compression": "8×",
         "accuracy_loss": "~1–2%", "notes": "Used by llama.cpp Q4. Noticeable quality drop on long context."},
        {"name": "binary (1-bit)", "bits": 1, "bytes_per_vec": math.ceil(embedding_dim/8),
         "vocab_table_mb": mb(math.ceil(embedding_dim/8)), "compression": "32×",
         "accuracy_loss": "~3–8%", "notes": "Extreme compression. Significant degradation."},
    ]

    original_val   = 0.8734
    scale          = 127.0
    quantized_int8 = int(round(original_val * scale))
    dequantized    = round(quantized_int8 / scale, 6)
    error          = round(abs(original_val - dequantized), 6)

    return {
        "embedding_dim": embedding_dim,
        "vocab_size":    vocab_size,
        "formats":       formats,
        "worked_example": {
            "original":    original_val,
            "scale":       scale,
            "quantized":   quantized_int8,
            "dequantized": dequantized,
            "error":       error,
            "error_pct":   round(error / original_val * 100, 4),
        },
        "recommendation": "int8 is the production sweet spot — 4× smaller, <0.5% accuracy loss, runs on CPU.",
    }


def analyze_token_budget(text: str, model: str = "claude", plan: str = "sonnet", limit: int = 200000) -> dict:
    """
    Token budget analysis for learn/tokens page.
    Per-speaker breakdown, fill projection, cost estimate, turns remaining.
    """
    lines        = text.split('\n')
    human_chars  = sum(len(l) for l in lines if l.strip().startswith(('Human:', 'User:')))
    ai_chars     = sum(len(l) for l in lines if l.strip().startswith(('Assistant:', 'AI:')))
    other_chars  = len(text) - human_chars - ai_chars
    cpt          = {"claude": 3.5, "chatgpt": 4.0, "gemini": 4.5}.get(model, 3.5)

    human_tokens = max(1, int(human_chars / cpt))
    ai_tokens    = max(1, int(ai_chars    / cpt))
    other_tokens = max(1, int(other_chars / cpt))
    total_tokens = human_tokens + ai_tokens + other_tokens
    fill_pct     = round(total_tokens / limit * 100, 2)
    msg_count    = max(1, text.count('Human:') + text.count('User:') + text.count('Assistant:') + text.count('AI:'))
    avg_per_msg  = max(1, total_tokens // msg_count)
    tokens_to_80 = max(0, int(limit * 0.8) - total_tokens)
    turns_left   = max(0, tokens_to_80 // max(avg_per_msg, 1))

    costs        = {"claude": {"sonnet": 3.0, "opus": 15.0, "haiku": 0.25},
                    "chatgpt": {"plus": 5.0, "pro": 2.5}, "gemini": {"free": 0.0, "pro": 1.25}}
    cost_per_m   = costs.get(model, {}).get(plan, 3.0)
    cost_usd     = round(total_tokens / 1_000_000 * cost_per_m, 6)

    return {
        "total_tokens": total_tokens, "limit": limit, "fill_pct": fill_pct,
        "human_tokens": human_tokens, "ai_tokens": ai_tokens, "other_tokens": other_tokens,
        "msg_count": msg_count, "avg_per_msg": avg_per_msg,
        "turns_remaining": turns_left, "cost_usd": cost_usd,
        "model": model, "plan": plan,
        "zone": "safe" if fill_pct < 40 else "warning" if fill_pct < 70 else "danger",
    }


# ── COMBINED ANALYZER ────────────────────────────────────────────────

def analyze_text(text: str, token_count: int, limit: int, embeddings=None) -> dict:
    """Run all analyses. Called by /analyze endpoint."""
    H     = shannon_entropy(text)
    red   = redundancy_score(text, embeddings)
    att   = attention_cost_multiplier(token_count, limit)
    bound = theoretical_compression_bound(text)
    return {
        "entropy": H, "redundancy": red, "attention": att, "compression_bound": bound,
        "summary": {
            "information_density": "high" if H > 4.0 else "medium" if H > 3.0 else "low",
            "compressibility":     "high" if red["score"] > 30 else "medium" if red["score"] > 15 else "low",
            "latency_impact":      att["zone"],
            "recommended_action":  "compress" if red["score"] > 20 or att["zone"] != "safe" else "monitor",
        },
    }