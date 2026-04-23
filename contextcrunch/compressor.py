"""
compressor.py — sentence embedding compression pipeline
Uses sentence-transformers all-MiniLM-L6-v2 (384-dim) for semantic similarity.
Model is loaded once at startup and cached — baked into Docker image.
"""
import re
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .math_engine import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_sentences(sentences: List[str]) -> np.ndarray:
    """Convert sentences to 384-dim unit-normalized embeddings."""
    return _get_model().encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


def find_redundant_sentences(embeddings: np.ndarray, threshold: float = 0.72) -> List[int]:
    """
    Find semantically redundant sentences via cosine similarity on neural embeddings.
    Returns indices of sentences that are redundant (can be removed).
    Threshold 0.72 = strong semantic overlap on 384-dim normalized vectors.
    """
    redundant = set()
    for i in range(len(embeddings)):
        if i in redundant:
            continue
        for j in range(i+1, len(embeddings)):
            if j not in redundant:
                if cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                    redundant.add(j)
    return sorted(redundant)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on punctuation and newlines."""
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]


def compress(text: str, model: str = "claude", plan: str = "sonnet", threshold: float = 0.72) -> dict:
    """
    Compress text using sentence embedding similarity.
    Step 1: embed all sentences with all-MiniLM-L6-v2
    Step 2: find semantically redundant pairs via cosine similarity
    Step 3: remove redundant sentences
    Step 4: LLM rewrites the result (called from main.py)
    """
    from .tokenizer import count_tokens

    original_tokens = count_tokens(text, model, plan)
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return {
            "original": text,
            "compressed_math": text,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "tokens_saved": 0,
            "compression_ratio": 0,
            "removed_count": 0,
            "model": model,
            "plan": plan,
        }

    embeddings       = embed_sentences(sentences)
    redundant        = find_redundant_sentences(embeddings, threshold)
    kept             = [s for i, s in enumerate(sentences) if i not in redundant]
    compressed_text  = " ".join(kept)
    compressed_tokens = count_tokens(compressed_text, model, plan)
    ratio            = round((1 - len(kept) / len(sentences)) * 100, 1)

    return {
        "original": text,
        "compressed_math": compressed_text,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": max(0, original_tokens - compressed_tokens),
        "compression_ratio": ratio,
        "removed_count": len(redundant),
        "model": model,
        "plan": plan,
    }


def get_embeddings_for_demo(sentences: List[str]) -> dict:
    """Return embeddings and full similarity matrix for the learn page demo."""
    if not sentences:
        return {"embeddings": [], "similarities": [], "sentences": []}

    embeddings = embed_sentences(sentences)
    n = len(embeddings)

    sim_matrix = [
        [round(float(cosine_similarity(embeddings[i], embeddings[j])), 3) for j in range(n)]
        for i in range(n)
    ]

    return {
        "sentences": sentences,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_sample": embeddings[0][:10].tolist(),
        "similarities": sim_matrix,
        "redundant_pairs": [
            {"i": i, "j": j, "similarity": sim_matrix[i][j]}
            for i in range(n)
            for j in range(i+1, n)
            if sim_matrix[i][j] > 0.72
        ],
    }