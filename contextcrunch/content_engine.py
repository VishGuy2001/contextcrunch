"""
content_engine.py — Structured educational content for all learn pages.

All text, formulas, code examples, and explanations live here in Python.
Frontend learn pages are thin HTML shells that fetch from /content/<page>.

This is what keeps the repo Python-heavy and makes the content an API.
Each function returns a dict with:
  - title, subtitle, description
  - levels: { simple: {...}, technical: {...} }
  - Each level has: paragraphs, formula (optional), code (optional)
"""


def get_tokens_content() -> dict:
    return {
        "page":     "tokens",
        "title":    "Tokens & context windows",
        "subtitle": "What is a token? Why do models forget?",
        "description": "Each model uses a different tokenizer — the same text produces different token counts across Claude, ChatGPT, and Gemini.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is a token?",
                        "body": "A token is a chunk of text — but the exact size depends on which AI you're using. Claude, ChatGPT, and Gemini each use a different tokenizer trained on different data with a different vocabulary. 'Hello' might be 1 token in all three — but 'Unbelievable' could be 2 tokens in one model and 3 in another. Roughly speaking: ChatGPT splits text into ~4 characters per token, Claude into ~3.5, and Gemini into ~4.5 because it has a larger vocabulary of 256,000 tokens."
                    },
                    {
                        "heading": "Why does the session limit matter even on Pro?",
                        "body": "Every model has a context window — the maximum number of tokens it can hold in memory at once. Even on a Pro plan, this limit is finite. As your conversation fills up, earlier context loses weight. The model starts forgetting things you said an hour ago — not because it's broken, but because that content is now competing with everything that came after it. The faster you fill the window, the sooner this happens. Efficient prompts stretch the same session further."
                    },
                    {
                        "heading": "Practical tip",
                        "body": "Before starting a long session, front-load the most important context. Decisions, constraints, and requirements should come early — not buried in the middle of a long thread where they'll lose weight as the conversation grows."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Three different tokenizer algorithms",
                        "body": "Token counting is not interchangeable across models. Each uses a different algorithm and vocabulary size.",
                        "formula": "Claude Haiku/Sonnet  Custom BPE      ~3.5 chars/token  vocab ~100k\nClaude Opus 4.7      New BPE (+35%)  ~2.6 chars/token  vocab ~100k+\nChatGPT (cl100k)     BPE             ~4.0 chars/token  vocab 100,277\nGemini               SentencePiece   ~4.5 chars/token  vocab 256,000\n\nSame sentence: \"The transformer processes tokens\"\n  Claude Sonnet:  ~8 tokens\n  ChatGPT Plus:   ~6 tokens\n  Gemini Flash:   ~7 tokens"
                    },
                    {
                        "heading": "Exact counting with tiktoken",
                        "body": "ChatGPT is the only model with a publicly available exact tokenizer. Claude and Gemini can only be estimated.",
                        "code": "import tiktoken\n\n# ChatGPT — exact count via tiktoken (public)\nenc = tiktoken.get_encoding(\"cl100k_base\")\ntokens = enc.encode(\"Hello, how are you?\")\nprint(len(tokens))  # 6 — exact\n\n# Claude — no public tokenizer, estimate only\nclaude_estimate = len(\"Hello, how are you?\") / 3.5  # ~5.4 → 6 tokens\n\n# Gemini — no public tokenizer, estimate only  \ngemini_estimate = len(\"Hello, how are you?\") / 4.5  # ~4.2 → 5 tokens\n\n# Opus 4.7 uses a new tokenizer — up to 35% more tokens than older Claude"
                    },
                    {
                        "heading": "Why context degrades quadratically",
                        "body": "Self-attention computes relationships between all token pairs. For n tokens, that's n² pairs. As the window fills, the model's attention is spread thinner across more pairs — earlier content competes with more recent content and loses weight. This is why response quality degrades at high fill levels, not just speed."
                    }
                ]
            }
        }
    }


def get_embeddings_content() -> dict:
    return {
        "page":     "embeddings",
        "title":    "Embeddings & semantic meaning",
        "subtitle": "How does AI understand meaning, not just words?",
        "description": "Enter sentences and see real cosine similarities computed by the backend using sentence-transformers all-MiniLM-L6-v2.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is an embedding?",
                        "body": "An embedding turns text into a list of numbers — a coordinate in high-dimensional space. The key insight: texts with similar meaning end up at nearby coordinates. 'The cat sat on the mat' and 'A cat was resting on the rug' end up very close together in that space, even though the words are different."
                    },
                    {
                        "heading": "How ContextCrunch uses this",
                        "body": "ContextCrunch embeds every sentence in your conversation. When two sentences are neighbors in embedding space — similarity above 0.88 — one can be removed without losing meaning. This is how it catches paraphrased repetition that simple word-matching misses entirely."
                    },
                    {
                        "heading": "Practical tip",
                        "body": "If you've explained something to an AI, don't re-explain it in different words later in the same session. The model has it — you're just burning tokens repeating it."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Sentence embeddings with SBERT",
                        "body": "ContextCrunch uses all-MiniLM-L6-v2 to map each sentence to a 384-dimensional unit-normalized vector. Cosine similarity between embeddings correlates with human judgment of sentence similarity.",
                        "formula": "embed(s) → v ∈ ℝ³⁸⁴  (unit normalized)\nsim(A,B) = (A·B) / (‖A‖·‖B‖) ∈ [-1, 1]\nsim > 0.88 → semantically redundant (ContextCrunch threshold)"
                    },
                    {
                        "heading": "Implementation",
                        "body": "The redundancy detection runs in O(n²) over sentence pairs. For typical conversations this is fast — 50 sentences = 1,225 pairs. The model is cached after first load so subsequent calls are near-instant.",
                        "code": "from sentence_transformers import SentenceTransformer\nimport numpy as np\n\nmodel = SentenceTransformer(\"all-MiniLM-L6-v2\")\nsentences = [\"The cat sat on the mat\", \"A cat was on the rug\", \"Quantum computing\"]\nembeddings = model.encode(sentences, normalize_embeddings=True)\n\n# Cosine similarity — O(n²) over all pairs\nsim_01 = np.dot(embeddings[0], embeddings[1])  # ~0.85 — redundant\nsim_02 = np.dot(embeddings[0], embeddings[2])  # ~0.12 — unique\nprint(f\"0↔1: {sim_01:.3f}, 0↔2: {sim_02:.3f}\")"
                    },
                    {
                        "heading": "Why 0.88 threshold?",
                        "body": "At 0.88, sentences must share near-identical meaning to be flagged. 'How do I install Python?' and 'How do I install Python on Windows?' score ~0.85 — below the threshold, because the second adds a constraint that changes the answer. Only truly redundant paraphrasing crosses 0.88."
                    }
                ]
            }
        }
    }


def get_entropy_content() -> dict:
    return {
        "page":     "entropy",
        "title":    "Entropy & information theory",
        "subtitle": "How do we measure information vs noise?",
        "description": "Shannon entropy is the mathematical core of ContextCrunch's waste detection. Paste any text and see real entropy calculated by the backend.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is entropy?",
                        "body": "Entropy measures surprise. 'aaaaaaa' has zero entropy — no surprises, no information. Random gibberish has maximum entropy — all surprise, no pattern. Real conversations sit in between."
                    },
                    {
                        "heading": "What low entropy looks like in practice",
                        "body": "When someone says 'as I mentioned earlier' for the third time, they're adding low-entropy content — predictable, redundant, compressible waste. Filler phrases like 'I'd be happy to help' score near zero entropy because the model has seen them millions of times and they predict nothing about what comes next."
                    },
                    {
                        "heading": "Practical tip",
                        "body": "High-entropy content is worth keeping — it carries real information. Low-entropy content is the first target for compression. The ContextCrunch entropy score tells you the ratio of signal to noise in your conversation."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "H(X) = −Σ p(x) log₂ p(x)",
                        "body": "For a discrete random variable X over alphabet A, entropy H(X) measures the average number of bits needed to encode one symbol. Low entropy means the distribution is skewed — some characters appear much more than others, so they can be encoded cheaply. High entropy means a flat distribution — maximum unpredictability.",
                        "formula": "H(X) = -Σ p(x) · log₂(p(x))\n\nBounds: 0 ≤ H(X) ≤ log₂|A|\nEnglish text: H ≈ 3.5–4.0 bits/char\nAfter lossless compression: H ≈ 1.0 bits/char\nMax (uniform over 256 ASCII): H = 8.0 bits/char"
                    },
                    {
                        "heading": "Computing it",
                        "body": "The implementation is straightforward — count character frequencies, normalize to probabilities, apply the formula.",
                        "code": "import math\nfrom collections import Counter\n\ndef shannon_entropy(text: str) -> float:\n    \"\"\"H(X) = -Σ p(x) log₂ p(x) over character distribution.\"\"\"\n    freq = Counter(text)\n    total = len(text)\n    return -sum(\n        (c / total) * math.log2(c / total)\n        for c in freq.values()\n        if c > 0\n    )\n\nprint(shannon_entropy(\"aaaaaa\"))       # 0.0   — zero information\nprint(shannon_entropy(\"hello world\"))  # ~3.18 — moderate\nprint(shannon_entropy(\"def f(x):...\")) # ~4.1  — code, high density"
                    },
                    {
                        "heading": "Compression bound",
                        "body": "Shannon's source coding theorem states H(X) is the minimum average code length achievable by any lossless compressor. ContextCrunch uses this to estimate how much of your conversation is theoretically removable before semantic compression even starts."
                    }
                ]
            }
        }
    }


def get_quantization_content() -> dict:
    return {
        "page":     "quantization",
        "title":    "Quantization",
        "subtitle": "How we compress embeddings without losing meaning.",
        "description": "From float32 to int8 — the precision tradeoff that makes local embedding models practical. This is what LLMLingua uses internally.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "What is quantization?",
                        "body": "A high-resolution photo stored as 50MB. Quantization converts it to a 5MB JPEG — smaller file, looks almost identical to the human eye. For AI embeddings, we do the same to the numbers representing meaning. Instead of storing each number as a 32-bit float (very precise), we store it as an 8-bit integer (4× smaller) with a small, controlled accuracy loss."
                    },
                    {
                        "heading": "Why ContextCrunch uses int8",
                        "body": "The LLMLingua-2 BERT model that runs locally inside ContextCrunch uses int8 quantization. This is why it fits on CPU with reasonable memory usage. Without quantization, the model would require 4× more RAM and be impractical to run for free on every compression request."
                    },
                    {
                        "heading": "The tradeoff",
                        "body": "int8 gives you 4× compression with less than 0.5% accuracy loss. That's the production sweet spot. int4 gives 8× compression but noticeable quality degradation. Binary (1-bit) gives 32× compression but significant accuracy loss — only useful in extreme memory-constrained environments."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Scalar quantization: float32 → int8",
                        "body": "The basic operation: scale the float value to fit the int8 range [-127, 127], round, store. Dequantization reverses the scale. The error is small but non-zero.",
                        "formula": "q(x) = round(x · 127)  where x ∈ [-1, 1]\ndq(q) = q / 127\nerror  = |x - dq(q(x))| ≤ 1/(2·127) ≈ 0.004\n\nMemory per embedding vector (d=384):\n  float32: 384 × 4 = 1,536 bytes\n  float16: 384 × 2 =   768 bytes\n  int8:    384 × 1 =   384 bytes  (4× smaller)\n  int4:    384 / 2 =   192 bytes  (8× smaller)"
                    },
                    {
                        "heading": "Worked example",
                        "body": "Walk through quantizing a single embedding value from float32 to int8 and back.",
                        "code": "import numpy as np\n\ndef scalar_quantize(x: float, scale: float = 127.0) -> tuple:\n    \"\"\"float32 → int8 scalar quantization.\"\"\"\n    q   = int(round(x * scale))              # quantize\n    q   = max(-127, min(127, q))             # clamp\n    dq  = q / scale                          # dequantize\n    err = abs(x - dq)                        # reconstruction error\n    return q, dq, err\n\noriginal = 0.8734\nq, dq, err = scalar_quantize(original)\nprint(f\"Original:      {original}\")\nprint(f\"Quantized:     {q}  (int8)\")\nprint(f\"Dequantized:   {dq}\")\nprint(f\"Error:         {err:.6f}  ({err/original*100:.4f}%)\")  # ~0.0004%\n\n# Apply to full embedding vector\ndef quantize_embedding(emb: np.ndarray) -> np.ndarray:\n    return np.clip(np.round(emb * 127), -127, 127).astype(np.int8)"
                    },
                    {
                        "heading": "Why LLMLingua uses int8",
                        "body": "LLMLingua-2 runs a BERT model locally to score token importance. At full float32 precision the model would use ~2.8GB RAM. int8 quantization brings this to ~700MB — practical for Cloud Run and local development. The token importance scoring accuracy drops by less than 0.5%, which is well within the tolerance for compression decisions."
                    }
                ]
            }
        }
    }


def get_attention_content() -> dict:
    return {
        "page":     "attention",
        "title":    "Attention & latency",
        "subtitle": "Why does longer context make AI slower?",
        "description": "Self-attention scales as O(n²) — double your tokens and you quadruple the compute. Use the slider to feel the math.",
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "Why does longer context = slower responses?",
                        "body": "Before answering your message, the AI reads your entire conversation from the very beginning. Every single message, every response — all of it, every time. As conversations get longer, this reading task gets exponentially harder."
                    },
                    {
                        "heading": "It's quadratic, not linear",
                        "body": "Double your conversation length and the AI does four times as much work per response. Triple it and it does nine times as much work. This is why long Claude sessions feel noticeably slower toward the end — the model is doing fundamentally more computation on every single reply."
                    },
                    {
                        "heading": "What compression does",
                        "body": "A 30% token reduction doesn't produce 30% faster responses — it produces 51% faster responses, because (0.7)² = 0.49. The quadratic relationship means compression benefits compound. ContextCrunch's compression benefit calculator on this page shows this effect on your actual token count."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Self-attention complexity: O(n²)",
                        "body": "The attention mechanism computes relationships between all token pairs. For n tokens, that's n² pairs — hence the quadratic scaling. This is fundamental to the transformer architecture, not an implementation choice.",
                        "formula": "Attention(Q,K,V) = softmax(QKᵀ/√d) · V\n\nComplexity: O(n²·d) time,  O(n²) memory\n\nLatency model (empirical):\n  T(n) ∝ n²\n  Multiplier = (fill_pct / 50)²\n  50% fill = 1× baseline\n  75% fill = 2.25× baseline\n  90% fill = 3.24× baseline\n  100% fill = 4× baseline\n\nCompression speedup at ratio ρ:\n  speedup = 1 / (1 - ρ)²\n  ρ = 0.30 → 2.04× faster\n  ρ = 0.50 → 4.00× faster"
                    },
                    {
                        "heading": "Implementation",
                        "body": "The attention cost multiplier is what ContextCrunch uses to color-code the latency zone in the analysis panel.",
                        "code": "def attention_cost_multiplier(tokens: int, limit: int) -> dict:\n    \"\"\"O(n²) latency model. Returns multiplier and zone.\"\"\"\n    pct  = (tokens / limit) * 100\n    mult = round((pct / 50) ** 2, 2)  # calibrated to 50% baseline\n\n    if pct < 40:   zone = \"safe\"     # optimal speed\n    elif pct < 70: zone = \"warning\"  # noticeably slower\n    else:          zone = \"danger\"   # compress now\n\n    return {\"multiplier\": mult, \"percentage\": pct, \"zone\": zone}\n\n# Token pair count grows quadratically\ndef pair_count(tokens: int) -> int:\n    return tokens ** 2  # every token attends to every other token\n\nprint(pair_count(10_000))   # 100,000,000 pairs\nprint(pair_count(100_000))  # 10,000,000,000 pairs — 100× more"
                    },
                    {
                        "heading": "FlashAttention note",
                        "body": "FlashAttention (Dao, 2022) reduces memory from O(n²) to O(n) by tiling the computation. But the fundamental time complexity remains O(n²) — FlashAttention is faster in practice due to fewer memory reads, not fewer operations. The latency model here still applies."
                    }
                ]
            }
        }
    }


def get_prompts_content() -> dict:
    return {
        "page":     "prompts",
        "title":    "Prompt efficiency",
        "subtitle": "The most common token waste is saying more than you need to.",
        "description": "Paste any prompt. ContextCrunch rewrites it to say the same thing in fewer tokens using Groq Llama 3.3 70B and semantic compression.",
        "waste_patterns": [
            {
                "tag":     "Polite filler",
                "example": "\"I was hoping you could please help me understand and explain in detail...\"",
                "fix":     "\"Explain...\" — saves ~12 tokens"
            },
            {
                "tag":     "Redundant context",
                "example": "\"As we discussed, as I mentioned, as you know, as I said before...\"",
                "fix":     "Remove entirely — saves ~15 tokens"
            },
            {
                "tag":     "Over-specification",
                "example": "\"...that is helpful and informative and useful and relevant and accurate...\"",
                "fix":     "Implied — saves ~18 tokens"
            },
            {
                "tag":     "Restating the obvious",
                "example": "\"I am a human user asking you, an AI assistant, to help me with...\"",
                "fix":     "The model knows — saves ~20 tokens"
            },
        ],
        "levels": {
            "simple": {
                "sections": [
                    {
                        "heading": "The most common waste: saying more than you need to",
                        "body": "Most people use 2–3× more tokens than necessary when prompting AI. Politeness filler, redundant context-setting, over-specifying obvious things — all of these cost tokens without adding information the model will act on."
                    },
                    {
                        "heading": "What good prompts look like",
                        "body": "Precise and direct. 'Explain machine learning: concept, real-world uses, brief history' says everything that 'Could you please help me understand and explain in detail the concept of machine learning...' says — in about 60% fewer tokens. The output is the same. The cost is not."
                    },
                    {
                        "heading": "Practical tip",
                        "body": "Before sending a long prompt, read it once and ask: does every sentence change what the model will output? If not, cut it. The session limit you save extends how far your conversation can go before context degrades."
                    }
                ]
            },
            "technical": {
                "sections": [
                    {
                        "heading": "Mutual information and redundancy",
                        "body": "ContextCrunch scores prompts by measuring mutual information between sentence pairs. High MI means two parts convey the same information — one can be dropped without changing the model's output.",
                        "formula": "I(A;B) = H(A) + H(B) - H(A,B)\nI(A;B) > threshold → A and B are redundant\n\nCommon flagged patterns:\n  Polite filler: \"please\", \"could you\" → I ≈ 0 bits\n  Restatements:  \"as I mentioned\"      → negative signal\n  Over-spec:     \"helpful and useful\"  → 1 concept, 2 tokens"
                    },
                    {
                        "heading": "The 3-stage pipeline",
                        "body": "ContextCrunch runs semantic dedup first, then Groq rewriting. For prompts over 50 words, sentence-transformers removes near-duplicate sentences before the LLM sees the text — the LLM receives already-cleaned input.",
                        "code": "# Stage 1: semantic dedup (local, free)\nif len(prompt.split()) > 50:\n    prompt = remove_semantic_redundancy(prompt, threshold=0.85)\n\n# Stage 2: LLM rewrite (Groq Llama 3.3 70B)\nmessages = [\n    {\"role\": \"system\", \"content\": PROMPT_IMPROVEMENT_SYSTEM},\n    {\"role\": \"user\",   \"content\": f\"Rewrite to use fewer tokens:\\n\\n{prompt}\"}\n]\nresult = groq(messages, temperature=0.05, max_tokens=800)\n\n# Structured output parsing\n# COMPRESSED: [rewritten prompt]\n# CHANGES:\n# - [specific change with reason]"
                    },
                    {
                        "heading": "Why temperature=0.05?",
                        "body": "Near-zero temperature makes the rewrite deterministic. Prompt compression needs to be consistent — the same input should always produce the same output. Higher temperature introduces variance that makes the tool unpredictable for users who are iterating on their prompts."
                    }
                ]
            }
        }
    }


# ── REGISTRY ──────────────────────────────────────────────────────────

CONTENT_REGISTRY = {
    "tokens":       get_tokens_content,
    "embeddings":   get_embeddings_content,
    "entropy":      get_entropy_content,
    "quantization": get_quantization_content,
    "attention":    get_attention_content,
    "prompts":      get_prompts_content,
}


def get_content(page: str) -> dict:
    """Return structured content for the given learn page. Raises KeyError if not found."""
    if page not in CONTENT_REGISTRY:
        raise KeyError(f"Unknown page: {page}. Available: {list(CONTENT_REGISTRY.keys())}")
    return CONTENT_REGISTRY[page]()


def get_all_content() -> dict:
    """Return all page content. Used by /content endpoint."""
    return {page: fn() for page, fn in CONTENT_REGISTRY.items()}