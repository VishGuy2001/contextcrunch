"""
math_engine.py — Shannon entropy, cosine similarity, redundancy, O(n²) latency
"""
import math
import numpy as np
from collections import Counter
from typing import List, Optional


def shannon_entropy(text: str) -> float:
    """H(X) = -Σ p(x) log₂ p(x) over character distribution."""
    if not text or len(text) < 2:
        return 0.0
    freq = Counter(text)
    total = len(text)
    H = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
    return round(H, 3)


def token_entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    return round(-sum((c/total)*math.log2(c/total) for c in freq.values()), 3)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """sim(A,B) = (A·B) / (‖A‖·‖B‖)"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity over meaningful words — fast client-side equivalent."""
    wa = {w for w in text_a.lower().split() if len(w) > 4}
    wb = {w for w in text_b.lower().split() if len(w) > 4}
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 3)


def redundancy_score(text: str, embeddings: Optional[List[np.ndarray]] = None, threshold: float = 0.72) -> dict:
    """Detect what percentage of text is semantically redundant."""
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    if len(sentences) < 2:
        return {"score": 0, "redundant": [], "removable": 0, "method": "none", "sentence_count": len(sentences)}

    redundant = set()
    method = "word_overlap"

    if embeddings and len(embeddings) == len(sentences):
        method = "embedding"
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if j not in redundant and cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                    redundant.add(j)
    else:
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if j not in redundant and word_overlap_similarity(sentences[i], sentences[j]) > 0.45:
                    redundant.add(j)

    score = round((len(redundant) / len(sentences)) * 100, 1)
    removable = max(0, int(sum(len(sentences[i]) for i in redundant) / 3.8))

    return {
        "score": min(score, 75.0),
        "redundant": list(redundant),
        "removable": removable,
        "method": method,
        "sentence_count": len(sentences),
    }


def attention_cost_multiplier(current_tokens: int, limit_tokens: int) -> dict:
    """O(n²) latency model — T(n) ∝ n²"""
    if limit_tokens == 0:
        return {"multiplier": 1.0, "percentage": 0, "zone": "safe", "message": ""}
    pct = (current_tokens / limit_tokens) * 100
    mult = round((pct / 50) ** 2, 2) if pct > 0 else 0.0
    if pct < 40:
        zone, msg = "safe", "Response speed is optimal."
    elif pct < 70:
        zone, msg = "warning", f"Responses ~{mult}× slower than conversation start."
    else:
        zone, msg = "danger", f"Responses ~{mult}× slower. Compression recommended."
    return {"multiplier": mult, "percentage": round(pct, 1), "zone": zone, "message": msg}


def theoretical_compression_bound(text: str) -> dict:
    if not text:
        return {"bound": 0, "entropy": 0, "bits_saved": 0, "interpretation": ""}
    H = shannon_entropy(text)
    alphabet_bits = math.log2(max(len(set(text)), 2))
    bound = max(0, round((1 - H / alphabet_bits) * 100, 1))
    return {
        "bound": bound, "entropy": H,
        "bits_saved": max(0, int(len(text)*8 - H*len(text))),
        "interpretation": f"Lossless compression can reduce by at most {bound}%. Semantic compression achieves 20-60% token reduction.",
    }


def analyze_text(text: str, token_count: int, limit: int, embeddings=None) -> dict:
    H = shannon_entropy(text)
    red = redundancy_score(text, embeddings)
    att = attention_cost_multiplier(token_count, limit)
    bound = theoretical_compression_bound(text)
    return {
        "entropy": H,
        "redundancy": red,
        "attention": att,
        "compression_bound": bound,
        "summary": {
            "information_density": "high" if H > 4.0 else "medium" if H > 3.0 else "low",
            "compressibility": "high" if red["score"] > 30 else "medium" if red["score"] > 15 else "low",
            "latency_impact": att["zone"],
            "recommended_action": "compress" if red["score"] > 20 or att["zone"] != "safe" else "monitor",
        },
    }
