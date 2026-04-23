"""
math_engine.py — Shannon entropy, redundancy detection, O(n²) latency model
"""
import math
import re
import numpy as np
from collections import Counter
from typing import List, Optional

# Stopwords — carry no semantic signal, removed before similarity comparison
_STOP = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','i','you','he',
    'she','it','we','they','this','that','and','or','but','in','on','at','to','for',
    'of','with','by','from','as','not','just','so','what','how','when','where','who',
    'which','if','then','than','too','very','also','can','get','got','like','go',
    'going','want','need','think','know','me','my','your','our','their','its','im',
    'its','okay','ok','yes','no','hi','hey','hello',
}


def shannon_entropy(text: str) -> float:
    """H(X) = -Σ p(x) log₂ p(x) over character distribution."""
    if not text or len(text) < 2:
        return 0.0
    freq = Counter(text)
    total = len(text)
    H = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
    return round(H, 3)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """sim(A,B) = (A·B) / (‖A‖·‖B‖)"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity: |A∩B| / |A∪B| on word sets after stopword removal.
    No minimum word length — catches short repeated phrases like "how are you".
    """
    wa = set(re.sub(r'[^a-z0-9\s]', '', text_a.lower()).split()) - _STOP
    wb = set(re.sub(r'[^a-z0-9\s]', '', text_b.lower()).split()) - _STOP
    # If both empty after stopword removal, sentences are all stopwords — likely identical
    if not wa and not wb:
        # Fall back to raw word overlap
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
    TF cosine similarity for longer sentences.
    sim(A,B) = (A·B) / (‖A‖·‖B‖) over term-frequency vectors.
    """
    def tf(text):
        words = [w for w in re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
                 if len(w) > 1 and w not in _STOP]
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
    Detect what percentage of sentence pairs are semantically redundant.

    Two-pass for plain text:
    1. Jaccard on raw word sets — catches exact/near-exact repeats, no min length
    2. TF cosine — catches semantic paraphrasing on longer sentences

    If neural embeddings provided: uses cosine similarity on 384-dim vectors.

    Thresholds:
      Jaccard >= 0.4  — 40% word overlap = clearly the same idea
      TF cosine >= 0.65 — strong semantic overlap on content words
      Embedding cosine >= threshold (default 0.72)
    """
    # Split on sentence boundaries and newlines — not commas
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]

    if len(sentences) < 2:
        return {
            "score": 0, "redundant": [], "removable": 0,
            "method": "none", "sentence_count": len(sentences)
        }

    redundant_set = set()
    total_pairs   = (len(sentences) * (len(sentences) - 1)) // 2
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

    redundant_pairs = len(redundant_set)
    score = round((redundant_pairs / max(total_pairs, 1)) * 100, 1)
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
    """O(n²) latency model — T(n) ∝ n²"""
    if limit_tokens == 0:
        return {"multiplier": 1.0, "percentage": 0, "zone": "safe", "message": "Response speed is optimal."}
    pct  = (current_tokens / limit_tokens) * 100
    mult = round((pct / 50) ** 2, 2) if pct > 0 else 0.0
    if pct < 40:
        zone, msg = "safe", "Response speed is optimal."
    elif pct < 70:
        zone, msg = "warning", f"Responses ~{mult}× slower than conversation start."
    else:
        zone, msg = "danger",  f"Responses ~{mult}× slower. Compress now."
    return {"multiplier": mult, "percentage": round(pct, 1), "zone": zone, "message": msg}


def theoretical_compression_bound(text: str) -> dict:
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
    H     = shannon_entropy(text)
    red   = redundancy_score(text, embeddings)
    att   = attention_cost_multiplier(token_count, limit)
    bound = theoretical_compression_bound(text)
    return {
        "entropy":          H,
        "redundancy":       red,
        "attention":        att,
        "compression_bound": bound,
        "summary": {
            "information_density": "high"   if H > 4.0          else "medium" if H > 3.0          else "low",
            "compressibility":     "high"   if red["score"] > 30 else "medium" if red["score"] > 15 else "low",
            "latency_impact":      att["zone"],
            "recommended_action":  "compress" if red["score"] > 20 or att["zone"] != "safe" else "monitor",
        },
    }