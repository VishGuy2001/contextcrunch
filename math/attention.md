# Attention Complexity & Latency

*Why longer context makes AI slower — and how ContextCrunch addresses it*

---

## Overview

The self-attention mechanism in transformer models has O(n²) time and memory complexity, where n is the sequence length in tokens. This quadratic scaling is the primary reason why:

1. Claude, ChatGPT, and Gemini responses slow down in long conversations
2. Context window limits exist and matter practically, not just in theory
3. Compressing context with ContextCrunch has a superlinear effect on response speed

---

## Scaled Dot-Product Attention

The core operation of every transformer layer:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V

Where:
  Q ∈ ℝ^{n × d_k}  — query matrix
  K ∈ ℝ^{n × d_k}  — key matrix
  V ∈ ℝ^{n × d_v}  — value matrix
  n                 — sequence length (tokens)
  d_k               — key/query dimension
  d_v               — value dimension
```

**Complexity analysis:**
```
QKᵀ computation:      O(n² · d_k)   — n² pairs, each d_k dot products
Softmax over QKᵀ:     O(n²)         — normalize each row of n×n matrix
Attention · V:        O(n² · d_v)   — weighted sum over n values

Total per layer:      O(n² · d)
Total for L layers:   O(L · n² · d)

Memory:               O(n²)         — attention matrix storage
```

**Python reference implementation:**
```python
import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray,  # (n, d_k)
    K: np.ndarray,  # (n, d_k)
    V: np.ndarray,  # (n, d_v)
    mask: np.ndarray = None
) -> np.ndarray:
    """
    Scaled dot-product attention.
    
    Time:   O(n² · d)
    Memory: O(n²)
    
    This is the bottleneck for long-context inference.
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: O(n² · d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Apply causal mask (decoder-only models like GPT/Claude)
    if mask is not None:
        scores = scores + mask * -1e9
    
    # Softmax: O(n²)
    scores_max = scores.max(-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(-1, keepdims=True)
    
    # Weighted sum: O(n² · d_v)
    return weights @ V
```

---

## KV-Cache and Memory Bottleneck

During inference (generating one token at a time), the model caches K and V matrices from all previous layers to avoid recomputation:

```
KV-cache memory at context length n:
  M_KV = 2 · n · d · L · precision_bytes

For GPT-4 scale (hypothetical, d=8192, L=96, float16):
  M_KV = 2 · n · 8192 · 96 · 2 bytes
       = 3,145,728 · n bytes
       ≈ 3 MB per token in context

For n = 128,000 (ChatGPT Plus limit):
  M_KV ≈ 384 GB  (distributed across many GPUs in practice)

This is why context windows are expensive to serve —
not the parameter count, but the KV-cache memory.
```

**TurboQuant reduces KV-cache memory:**
```
After TurboQuant at 3.5 bits:
  M_KV^TQ = M_KV · (3.5 / 16) ≈ M_KV / 4.6

At n = 128,000:
  M_KV^TQ ≈ 384 / 4.6 ≈ 83 GB  — still large but manageable
```

---

## Latency Model

**Python implementation of ContextCrunch's latency estimator:**
```python
def attention_latency_model(
    current_tokens: int,
    limit_tokens: int,
    baseline_fraction: float = 0.50
) -> dict:
    """
    Model relative response latency from O(n²) attention.
    
    Baseline: conversation at 50% context fill.
    
    Args:
        current_tokens: tokens currently in context
        limit_tokens:   model's context window size
        baseline_fraction: fraction of limit to use as baseline
    
    Returns:
        multiplier: how much slower than baseline
        zone:       "safe" | "warning" | "danger"
        message:    human-readable explanation
    
    Note: This is a theoretical upper bound.
    Real speedup from compression is typically 1.5-2.5×
    due to hardware optimizations (FlashAttention, etc.)
    """
    pct = current_tokens / limit_tokens
    baseline = baseline_fraction
    
    # T(n) ∝ n² → multiplier = (n/n_baseline)²
    multiplier = round((pct / baseline) ** 2, 2)
    
    zones = {
        "safe": (0, 0.40),
        "warning": (0.40, 0.70),
        "danger": (0.70, 1.0),
    }
    zone = next(
        name for name, (lo, hi) in zones.items()
        if lo <= pct < hi
    ) if pct < 1.0 else "danger"
    
    messages = {
        "safe": f"At {pct*100:.0f}% fill — responses are fast ({multiplier}× baseline).",
        "warning": f"At {pct*100:.0f}% fill — responses are ~{multiplier}× slower than start. Compress soon.",
        "danger": f"At {pct*100:.0f}% fill — responses are ~{multiplier}× slower. Compression recommended.",
    }
    
    return {
        "multiplier": multiplier,
        "percentage": round(pct * 100, 1),
        "zone": zone,
        "message": messages[zone],
        "compression_benefit": round((1 - (0.5 / pct) ** 2) * 100, 1) if pct > 0.5 else 0,
    }


# Examples:
print(attention_latency_model(20_000, 200_000))
# multiplier=0.04, zone="safe"

print(attention_latency_model(100_000, 200_000))
# multiplier=1.0, zone="warning" (baseline)

print(attention_latency_model(180_000, 200_000))
# multiplier=3.24, zone="danger"
```

---

## Compression Benefit on Latency

When ContextCrunch reduces context from n to n·(1-ρ) tokens, where ρ is the redundancy ratio:

```
T_original ∝ n²
T_compressed ∝ (n·(1-ρ))² = n²·(1-ρ)²

Speedup factor = T_original / T_compressed = 1/(1-ρ)²

At ρ = 0.30 (30% compression):
  Speedup = 1/(0.70)² ≈ 2.04×

At ρ = 0.50 (50% compression):
  Speedup = 1/(0.50)² = 4.0×

At ρ = 0.40 (40% compression, typical ContextCrunch):
  Speedup = 1/(0.60)² ≈ 2.78×
```

**Python:**
```python
def compression_speedup(redundancy_fraction: float) -> float:
    """
    Theoretical speedup from context compression.
    
    Derived from O(n²) attention complexity.
    Assumes proportional reduction in sequence length.
    
    Args:
        redundancy_fraction: fraction removed (0.0 to 0.75)
    
    Returns:
        speedup: response time multiplier (>1.0 = faster)
    
    >>> compression_speedup(0.30)
    2.04
    >>> compression_speedup(0.50)
    4.0
    """
    remaining = 1.0 - redundancy_fraction
    return round(1.0 / (remaining ** 2), 2)
```

---

## FlashAttention Note

Modern inference uses FlashAttention (Dao et al., 2022) which reduces memory from O(n²) to O(n) via tiled computation. However:

- **Time complexity remains O(n²)** — FlashAttention improves constants, not asymptotic complexity
- **Wall-clock improvement is real** — ~2-4× faster for long sequences vs naive attention
- **ContextCrunch benefit remains** — reducing n still gives quadratic speedup regardless of attention implementation

---

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. arXiv:1706.03762.
2. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. arXiv:2205.14135.
3. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR. arXiv:2307.08691.
4. Zandieh et al. (2025). *TurboQuant*. arXiv:2504.19874. ICLR 2026.
5. Press, O. et al. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR.
