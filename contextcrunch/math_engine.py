"""
math_engine.py — Shannon entropy, redundancy detection, O(n²) attention latency

Three core algorithms:

1. Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
   Measures average information per character.
   Low H = repetitive/compressible. High H = information-dense.
   Range: 0 (all same character) to ~4.7 (perfectly uniform ASCII).

2. Redundancy detection: two-pass approach
   Pass 1 — Jaccard similarity: |A∩B| / |A∪B| on word sets
     Catches exact duplicates, near-duplicates, and repeated short phrases.
     Works on short sentences where cosine similarity fails.
   Pass 2 — TF cosine similarity: sim(A,B) = (A·B) / (‖A‖·‖B‖)
     Catches semantic paraphrasing on longer sentences.
   Score = redundant sentences / total sentences (intuitive %)
   If neural embeddings provided: uses cosine on 384-dim vectors.

3. Attention latency model: T(n) ∝ n²
   Self-attention computes all n² token pair relationships per layer.
   Multiplier = (context% / 50)² — calibrated so 50% fill = 1× baseline.
"""
import math
import re
import numpy as np
from collections import Counter
from typing import List, Optional


# Stopwords removed before similarity comparison — carry no semantic signal
_STOP = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','i','you','he',
    'she','it','we','they','this','that','and','or','but','in','on','at','to','for',
    'of','with','by','from','as','not','just','so','what','how','when','where','who',
    'which','if','then','than','too','very','also','can','get','got','like','go',
    'going','want','need','think','know','me','my','your','our','their','its',
    'im','okay','ok','yes','no','hi','hey','hello',
}


def shannon_entropy(text: str) -> float:
    """
    H(X) = -Σ p(x) log₂ p(x) over the character distribution of text.
    Low = repetitive/compressible. High = information-dense.
    """
    if not text or len(text) < 2:
        return 0.0
    freq  = Counter(text)
    total = len(text)
    H     = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
    return round(H, 3)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """sim(A,B) = (A·B) / (‖A‖·‖B‖) — for neural embedding vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(text_a: str, text_b: str) -> float:
    """
    Jaccard(A,B) = |A∩B| / |A∪B| on word sets after stopword removal.
    No minimum word length — catches short phrases like "how are you".
    Falls back to raw word overlap if both sentences are all stopwords.
    """
    wa = set(re.sub(r'[^a-z0-9\s]', '', text_a.lower()).split()) - _STOP
    wb = set(re.sub(r'[^a-z0-9\s]', '', text_b.lower()).split()) - _STOP

    # Fallback when all words are stopwords (e.g. "how are you")
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
    """
    TF cosine: sim(A,B) = (A·B) / (‖A‖·‖B‖) over term-frequency vectors.
    Better than Jaccard for longer sentences with semantic paraphrasing.
    Requires at least 2 content words per sentence.
    """
    def tf(text):
        words = [
            w for w in re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
            if len(w) > 1 and w not in _STOP
        ]
        if not words:
            return {}
        n = len(words)
        v = {}
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


def redundancy_score(
    text: str,
    embeddings: Optional[List[np.ndarray]] = None,
    threshold: float = 0.72
) -> dict:
    """
    Detect what percentage of sentences are semantically redundant.

    Scoring: redundant sentences / total sentences
    (not pairs — gives more intuitive % for the UI)

    Thresholds:
      Jaccard >= 0.4  — 40% word overlap = clearly the same idea
      TF cosine >= 0.65 — strong semantic overlap on content words
      Embedding cosine >= 0.72 — neural similarity (highest accuracy)
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]

    if len(sentences) < 2:
        return {
            "score": 0, "redundant": [], "removable": 0,
            "method": "none", "sentence_count": len(sentences)
        }

    redundant_set = set()
    method        = "word_overlap"

    if embeddings and len(embeddings) == len(sentences):
        method = "embedding"
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if j not in redundant_set:
                    if cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                        redundant_set.add(j)
    else:
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if j not in redundant_set:
                    j_sim = _jaccard(sentences[i], sentences[j])
                    c_sim = _tf_cosine(sentences[i], sentences[j]) if j_sim < 0.4 else 0.0
                    if j_sim >= 0.4 or c_sim >= 0.65:
                        redundant_set.add(j)

    # Score = redundant sentences / total sentences
    # More intuitive than pairs-based scoring for the UI
    score = round((len(redundant_set) / max(len(sentences), 1)) * 100, 1)

    removable = max(0, int(
        sum(len(sentences[i]) for i in redundant_set) / 3.8
    ))

    return {
        "score": min(score, 90.0),
        "redundant": list(redundant_set),
        "removable": removable,
        "method": method,
        "sentence_count": len(sentences),
    }


def attention_cost_multiplier(current_tokens: int, limit_tokens: int) -> dict:
    """
    O(n²) latency model — T(n) ∝ n²
    Multiplier = (context% / 50)²
    At 50% fill = 1× baseline. At 100% fill = 4× baseline.
    """
    if limit_tokens == 0:
        return {"multiplier": 1.0, "percentage": 0, "zone": "safe", "message": "Response speed is optimal."}
    pct  = (current_tokens / limit_tokens) * 100
    mult = round((pct / 50) ** 2, 2) if pct > 0 else 0.0
    if pct < 40:
        zone, msg = "safe",    "Response speed is optimal."
    elif pct < 70:
        zone, msg = "warning", f"Responses ~{mult}× slower than conversation start."
    else:
        zone, msg = "danger",  f"Responses ~{mult}× slower. Compress now."
    return {"multiplier": mult, "percentage": round(pct, 1), "zone": zone, "message": msg}


def theoretical_compression_bound(text: str) -> dict:
    """Theoretical maximum lossless compression ratio based on Shannon entropy."""
    if not text:
        return {"bound": 0, "entropy": 0, "bits_saved": 0, "interpretation": ""}
    H             = shannon_entropy(text)
    alphabet_bits = math.log2(max(len(set(text)), 2))
    bound         = max(0, round((1 - H / alphabet_bits) * 100, 1))
    return {
        "bound": bound, "entropy": H,
        "bits_saved": max(0, int(len(text)*8 - H*len(text))),
        "interpretation": f"Lossless compression bound: {bound}%. Semantic compression typically achieves 20-60% token reduction.",
    }


def analyze_text(text: str, token_count: int, limit: int, embeddings=None) -> dict:
    """Run all analyses and return combined results. Called by /analyze endpoint."""
    H     = shannon_entropy(text)
    red   = redundancy_score(text, embeddings)
    att   = attention_cost_multiplier(token_count, limit)
    bound = theoretical_compression_bound(text)
    return {
        "entropy":           H,
        "redundancy":        red,
        "attention":         att,
        "compression_bound": bound,
        "summary": {
            "information_density": "high"   if H > 4.0          else "medium" if H > 3.0          else "low",
            "compressibility":     "high"   if red["score"] > 30 else "medium" if red["score"] > 15 else "low",
            "latency_impact":      att["zone"],
            "recommended_action":  "compress" if red["score"] > 20 or att["zone"] != "safe" else "monitor",
        },
    }