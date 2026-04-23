# Vector Quantization & Product Quantization

*Mathematical foundation for ContextCrunch's embedding compression*

---

## Overview

Vector quantization (VQ) maps high-dimensional embedding vectors to a finite set of codewords (centroids), reducing storage and enabling fast similarity search. ContextCrunch uses a combination of:

1. **Scalar Quantization** — simple float32 → int8 reduction
2. **Product Quantization (PQ)** — split into subspaces, quantize independently
3. **TurboQuant** — near-optimal via polar rotation + QJL (see turboquant.md)

---

## Scalar Quantization

Simplest form: map each float coordinate to an integer.

```
q(x) = round(x · scale)  where scale = (2^(bits-1) - 1) / max(|x|)
```

**Properties:**
```
Float32 → Int8:  4× storage reduction
MSE distortion:  D_MSE ≈ Δ²/12  where Δ = quantization step size
Accuracy loss:   < 1% for cosine similarity on normalized embeddings
```

**Python:**
```python
import numpy as np

def scalar_quantize(
    embeddings: np.ndarray,
    bits: int = 8
) -> tuple[np.ndarray, float]:
    """
    Scalar quantization: float32 → intN
    
    Args:
        embeddings: (n, d) float32, unit normalized
        bits: target bit width (default 8)
    
    Returns:
        quantized: (n, d) int8
        scale: float for dequantization
    """
    max_val = 2**(bits-1) - 1  # 127 for int8
    scale = max_val  # works for unit-normalized embeddings
    quantized = np.clip(
        np.round(embeddings * scale),
        -max_val, max_val
    ).astype(np.int8)
    return quantized, scale

def scalar_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) / scale
```

---

## Cosine Similarity

The core metric for semantic redundancy detection in ContextCrunch:

```
sim(A, B) = (A · B) / (‖A‖ · ‖B‖)

For unit-normalized vectors:
  sim(A, B) = A · B  (simple dot product)

Range: [-1, 1]
  1.0  → identical direction (same meaning)
  0.7+ → very similar → likely redundant
  0.0  → orthogonal → unrelated
 -1.0  → opposite direction → contradictory
```

**Python:**
```python
import numpy as np

def cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray
) -> float:
    """
    Cosine similarity between two embedding vectors.
    For unit-normalized inputs, equals dot product.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def batch_cosine_similarity(
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Full n×n similarity matrix for n embeddings.
    O(n²·d) complexity.
    
    For large n, use FAISS approximate search instead.
    """
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)
    # Gram matrix = cosine similarity matrix
    return normalized @ normalized.T
```

---

## Product Quantization (PQ)

PQ decomposes the d-dimensional space into M disjoint subspaces, each of dimension d/M, and quantizes each subspace independently with K centroids:

```
PQ(x) = [q₁(x[0:d/M]), q₂(x[d/M:2d/M]), ..., qₘ(x[(M-1)d/M:d])]

Storage per vector: M · log₂(K) bits
Example (d=384, M=96, K=256): 96 · 8 = 768 bits = 96 bytes
vs float32: 384 · 32 = 12,288 bits = 1,536 bytes
Compression: 16×
```

**Asymmetric distance computation (ADC):**
```
For query q and database vector x with PQ codes cₘ:
  d(q, x) ≈ Σₘ dₘ(qₘ, centroids[m][cₘ])

Pre-compute lookup tables for all M subspaces:
  T[m][k] = ‖qₘ - centroid[m][k]‖²

Then: d(q, x) = Σₘ T[m][cₘ(x)]  — O(M) per query
vs brute force O(d) — significant speedup
```

**Python:**
```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ProductQuantizer:
    """
    Product Quantization for sentence embeddings.
    
    Usage:
        pq = ProductQuantizer(d=384, M=96, K=256)
        pq.fit(training_embeddings)
        codes = pq.encode(embeddings)
        approx = pq.decode(codes)
        
    Compression: 16× vs float32 (96 bytes vs 1536 bytes per vector)
    Distortion: D_MSE ≈ d/(M·K^(2/d)) — decreases with more centroids
    """
    
    def __init__(self, d: int = 384, M: int = 96, K: int = 256):
        assert d % M == 0, "d must be divisible by M"
        self.d = d
        self.M = M
        self.K = K
        self.sub_d = d // M
        self.codebooks = None
    
    def fit(self, embeddings: np.ndarray) -> 'ProductQuantizer':
        """Learn codebooks from training data."""
        self.codebooks = []
        for m in range(self.M):
            subvectors = embeddings[:, m*self.sub_d:(m+1)*self.sub_d]
            kmeans = MiniBatchKMeans(n_clusters=self.K, random_state=42)
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_)
        return self
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode embeddings to PQ codes."""
        n = embeddings.shape[0]
        codes = np.zeros((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            subvectors = embeddings[:, m*self.sub_d:(m+1)*self.sub_d]
            # Find nearest centroid for each subvector
            diffs = subvectors[:, None] - self.codebooks[m][None]
            distances = np.sum(diffs**2, axis=-1)
            codes[:, m] = np.argmin(distances, axis=-1)
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate embeddings from codes."""
        n = codes.shape[0]
        reconstructed = np.zeros((n, self.d), dtype=np.float32)
        for m in range(self.M):
            reconstructed[:, m*self.sub_d:(m+1)*self.sub_d] = \
                self.codebooks[m][codes[:, m]]
        return reconstructed
    
    def approximate_similarity(
        self,
        query: np.ndarray,
        codes: np.ndarray
    ) -> np.ndarray:
        """
        Asymmetric Distance Computation (ADC).
        O(M) per query vs O(d) for exact search.
        """
        lookup_tables = []
        for m in range(self.M):
            q_sub = query[m*self.sub_d:(m+1)*self.sub_d]
            dists = np.sum((self.codebooks[m] - q_sub)**2, axis=-1)
            lookup_tables.append(dists)
        
        n = codes.shape[0]
        distances = np.zeros(n)
        for m in range(self.M):
            distances += lookup_tables[m][codes[:, m]]
        
        # Convert distance to similarity (negate for max = most similar)
        return -distances
```

---

## Distortion Analysis

For PQ with M subspaces and K centroids:

```
Expected MSE distortion:
  D_MSE(PQ) ≈ d/(M · K^(2/d))

For d=384, M=96, K=256:
  D_MSE ≈ 384/(96 · 256^(2/384))
         ≈ 4 · 256^(-0.0052)
         ≈ 4 · 0.987
         ≈ 3.95 × 10⁻³

Compare to scalar quantization (int8):
  D_MSE(scalar) ≈ 1/(127²) ≈ 6.2 × 10⁻⁵

PQ has higher distortion but much better compression:
  PQ: 16× compression, D_MSE ≈ 3.95×10⁻³
  Int8: 4× compression, D_MSE ≈ 6.2×10⁻⁵
  TurboQuant: ~9× compression, D_prod near-optimal
```

---

## FAISS Integration

For production scale (>10,000 vectors), use Facebook AI Similarity Search:

```python
import faiss
import numpy as np

def build_faiss_index(
    embeddings: np.ndarray,
    use_pq: bool = True
) -> faiss.Index:
    """
    Build FAISS index for fast similarity search.
    
    For ContextCrunch:
      - Flat (exact) for conversations < 1000 sentences
      - PQ (approximate) for larger contexts
    """
    d = embeddings.shape[1]
    
    if not use_pq or len(embeddings) < 1000:
        # Exact flat index — cosine similarity
        index = faiss.IndexFlatIP(d)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    else:
        # Product quantization index
        M = min(d // 4, 64)  # subspaces
        nbits = 8             # bits per subspace
        index = faiss.IndexPQ(d, M, nbits)
        index.train(embeddings)
        index.add(embeddings)
    
    return index

def find_redundant_with_faiss(
    sentences: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.82
) -> list[int]:
    """
    Find redundant sentences using FAISS ANN search.
    O(n log n) vs O(n²) for brute force.
    """
    index = build_faiss_index(embeddings, use_pq=False)
    redundant = set()
    
    for i, emb in enumerate(embeddings):
        if i in redundant:
            continue
        # Find k nearest neighbors
        emb_normalized = emb / np.linalg.norm(emb)
        k = min(10, len(embeddings))
        similarities, indices = index.search(
            emb_normalized.reshape(1, -1), k
        )
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != i and idx > i and sim > threshold:
                redundant.add(int(idx))
    
    return sorted(redundant)
```

---

## References

1. Jégou, H., Douze, M., Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search*. IEEE TPAMI, 33(1), 117-128.
2. Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Big Data.
3. Gray, R.M., Neuhoff, D.L. (1998). *Quantization*. IEEE Transactions on Information Theory.
4. Babenko, A., Lempitsky, V. (2014). *The Inverted Multi-Index*. IEEE TPAMI.
5. Zandieh et al. (2025). *TurboQuant*. arXiv:2504.19874. ICLR 2026.
