"""
llm_engine.py — compression pipeline with LLMLingua + sentence-transformers

Compression pipeline (3 stages):
  1. sentence-transformers  — semantic redundancy removal (cosine similarity)
  2. LLMLingua-2            — token-level compression (no LLM call needed)
  3. Groq / Gemini          — fluency rewrite of the compressed result

This means:
  - Most of the compression work happens locally, free, with no API call
  - Groq/Gemini only rewrites the already-compressed text for fluency
  - Faster, cheaper, more consistent than sending raw text to an LLM

LLM roles:
  Primary:  Groq Llama 3.3 70B — fast inference, 6000 req/day free
  Fallback: Gemini 2.0 Flash   — 1500 req/day free
  Demo:     Groq Llama 3.1 8B  — fast, cheap, good enough for short demos
"""
import os
import re
import numpy as np
from typing import Optional


# ── LAZY-LOADED MODELS ────────────────────────────────────────────────
# Both models are loaded once on first use and cached.
# Cold start on Cloud Run will be ~3-5s on first request — acceptable.

_st_model    = None  # sentence-transformers model
_ll_compressor = None  # LLMLingua-2 compressor

def _get_st_model():
    """sentence-transformers: all-MiniLM-L6-v2 — 22MB, fast, good semantic quality."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

def _get_lingua():
    """LLMLingua-2 BERT model — runs entirely locally, no API call."""
    global _ll_compressor
    if _ll_compressor is None:
        try:
            from llmlingua import PromptCompressor
            _ll_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"
            )
        except Exception as e:
            print(f"LLMLingua load failed: {e}")
            _ll_compressor = None
    return _ll_compressor


# ── SYSTEM PROMPTS ────────────────────────────────────────────────────

COMPRESSION_SYSTEM = """You are a conversation compression engine. Return a shorter version that preserves 100% of the meaning.

REMOVE:
- Filler: "I'd be happy to", "Of course!", "Great question!", "Sure,", "Certainly", "Absolutely", "Sounds good!", "That's a great point!"
- Repeated requests — if the user asks the same thing more than once, keep only the clearest version
- Repeated information — if the assistant restates the same fact or conclusion, keep only the first or clearest instance
- Verbose phrasing — rewrite long sentences into shorter ones without losing meaning
- Meta-commentary — "In this response I will...", "To summarize what I said above...", "As mentioned earlier..."
- Excessive hedging — "it's worth noting that", "it's important to understand that", "generally speaking"

KEEP:
- Every unique fact, decision, request, constraint, and piece of context
- Speaker labels on their own line: Human: / Assistant:
- Code, commands, file paths, and technical strings — verbatim, never paraphrase these
- Numbers, names, dates, and URLs — exactly as written
- The full semantic intent of every message

RULES:
- Do not merge separate messages into one
- Do not reorder the conversation
- Do not add anything that wasn't there
- Do not summarize — compress. Every sentence should still be present in some form.

Output: ONLY the compressed conversation. No preamble. Start immediately with the first speaker label."""


PROMPT_IMPROVEMENT_SYSTEM = """You are a prompt engineering expert who optimizes for token efficiency and semantic precision.

Rewrite the prompt to preserve exact meaning in fewer tokens.

REMOVE:
- Politeness filler: "please", "could you", "I was wondering if", "would you mind", "feel free to"
- Redundant context — anything the model can infer that doesn't change the output
- Repeated instructions that say the same thing twice in different words
- Weak hedges: "maybe", "perhaps", "sort of", "kind of", "if possible"
- Meta-instructions that describe the act of prompting rather than the task itself

KEEP:
- Every constraint, format requirement, and specific detail
- Examples — high-value signal, never remove them
- Negative instructions ("do not", "never") — carry meaning that cannot be dropped
- Persona or role instructions — they shape the entire output
- Output format specifications — structure, length, labels
- Technical terms, variable names, function names — verbatim, never paraphrase

RULES:
- Never change the semantic meaning
- Never drop a constraint even if it seems redundant
- Shorter is better only if meaning is fully preserved — do not compress at the cost of ambiguity
- If the prompt has multiple tasks, keep all of them clearly separated

Output format — use EXACTLY these two labels, nothing before them:
COMPRESSED: [rewritten prompt on one line]
CHANGES:
- [specific change made and why]
- [specific change made and why]"""


DEMO_SYSTEM = """You are an educational AI for ContextCrunch.
Use the user's actual input as a concrete example to explain the concept.
Be specific, clear, and under 250 words. Plain English first."""


EXPLANATION_PROMPTS = {
    "simple": "Explain this concept in plain English using a simple everyday analogy. Max 3 short paragraphs. No jargon. No equations. Focus on what it means for someone using Claude or ChatGPT daily.",
    "technical": "Explain this concept for a developer. Include the key formula with a one-line explanation of each variable. Include one short Python example under 10 lines. Explain practical implications for LLM context management. Max 5 paragraphs.",
}


# ── LLM WRAPPERS ──────────────────────────────────────────────────────

def _groq(
    messages: list,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2000,
    temperature: float = 0.1
) -> Optional[str]:
    """Groq API call. Returns stripped text or None on failure."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def _gemini(prompt: str, system: str = "", max_tokens: int = 2000) -> Optional[str]:
    """Gemini 2.0 Flash fallback. Returns stripped text or None on failure."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        m = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=system or None
        )
        resp = m.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
        )
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


# ── LOCAL COMPRESSION PIPELINE ────────────────────────────────────────

def _remove_semantic_redundancy(text: str, threshold: float = 0.92) -> str:
    """
    Stage 1: sentence-transformers redundancy removal.

    Splits text into sentences, embeds each one, then removes any sentence
    whose cosine similarity to a previous sentence exceeds the threshold.

    threshold=0.92 means near-identical meaning — conservative enough to
    avoid removing anything that actually differs.

    Speaker labels (Human: / Assistant:) are always preserved.
    Code blocks are passed through untouched.
    """
    # Extract and protect code blocks
    code_blocks = {}
    def _protect_code(m):
        key = f"__CODE_{len(code_blocks)}__"
        code_blocks[key] = m.group(0)
        return key
    text_safe = re.sub(r"```[\s\S]*?```", _protect_code, text)

    # Split on sentence boundaries, keep speaker labels as-is
    sentences = re.split(r'(?<=[.!?])\s+', text_safe)
    if len(sentences) <= 3:
        # Too short to bother — restore code blocks and return
        for k, v in code_blocks.items():
            text_safe = text_safe.replace(k, v)
        return text_safe

    model = _get_st_model()

    kept        = []
    kept_embeds = []

    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue

        # Always keep speaker labels and code placeholders
        is_label = stripped.startswith(("Human:", "Assistant:", "User:", "AI:"))
        is_code  = any(k in stripped for k in code_blocks)
        if is_label or is_code:
            kept.append(stripped)
            continue

        embed = model.encode(stripped, convert_to_numpy=True)

        if not kept_embeds:
            kept.append(stripped)
            kept_embeds.append(embed)
            continue

        # Cosine similarity against all kept sentences
        sims = np.dot(kept_embeds, embed) / (
            np.linalg.norm(kept_embeds, axis=1) * np.linalg.norm(embed) + 1e-9
        )

        if sims.max() < threshold:
            kept.append(stripped)
            kept_embeds.append(embed)
        # else: sentence is redundant — drop it

    result = " ".join(kept)

    # Restore code blocks
    for k, v in code_blocks.items():
        result = result.replace(k, v)

    return result


def _lingua_compress(text: str, rate: float = 0.7) -> str:
    """
    Stage 2: LLMLingua-2 token-level compression.

    rate=0.7 means keep 70% of tokens — aggressive but meaning-safe
    for the fluency rewrite that follows in stage 3.

    Speaker labels are force-kept so the conversation structure survives.
    Falls back to input text if LLMLingua fails to load.
    """
    lingua = _get_lingua()
    if lingua is None:
        return text  # graceful fallback

    try:
        result = lingua.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", "Human:", "Assistant:", "User:", "AI:", "?", ":"]
        )
        # compress_prompt returns a dict with 'compressed_prompt' key
        return result.get("compressed_prompt", text)
    except Exception as e:
        print(f"LLMLingua error: {e}")
        return text


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────

def generate_compression(text: str, math_compressed: str, model: str = "claude", plan: str = "sonnet") -> dict:
    """
    3-stage compression pipeline:

      Stage 1 — sentence-transformers semantic dedup (local, free)
        Removes sentences that are near-identical in meaning to a prior sentence.
        Catches paraphrased redundancy that string matching misses.

      Stage 2 — LLMLingua-2 token pruning (local, free)
        Drops low-signal tokens using a BERT-based importance scorer.
        No LLM API call. Runs on CPU in ~200ms.

      Stage 3 — Groq/Gemini fluency rewrite (API, cheap)
        Takes the locally-compressed text and rewrites it to flow naturally.
        Input is already 40-60% smaller so this call is fast and cheap.

    math_compressed is the last-resort fallback if all three stages fail.
    """
    # Stage 1: semantic dedup
    after_dedup = _remove_semantic_redundancy(text, threshold=0.88)

    # Stage 2: token pruning
    # Use a lighter compression rate here since stage 3 will clean up fluency
    after_lingua = _lingua_compress(after_dedup, rate=0.75)

    # Stage 3: fluency rewrite
    # Input is already compressed — max_tokens capped accordingly
    max_tok = min(len(after_lingua.split()) * 2, 3000)

    rewrite_prompt = f"""This conversation has already been compressed locally. 
Rewrite it so it flows naturally. Remove any remaining filler. Keep 100% of the meaning.
Output ONLY the rewritten conversation.

{after_lingua}"""

    messages = [
        {"role": "system", "content": COMPRESSION_SYSTEM},
        {"role": "user",   "content": rewrite_prompt}
    ]

    groq_result   = _groq(messages, temperature=0.05, max_tokens=max_tok)
    gemini_result = None if groq_result else _gemini(rewrite_prompt, COMPRESSION_SYSTEM, max_tok)

    result     = groq_result or gemini_result or after_lingua or math_compressed
    model_used = "groq" if groq_result else "gemini" if gemini_result else "lingua" if result == after_lingua else "math"

    return {"compressed": result, "model_used": model_used}


def improve_prompt(prompt_text: str) -> dict:
    """
    Rewrite a user prompt to use fewer tokens while keeping identical intent.

    For short prompts (<50 words): skips local pipeline, goes straight to LLM.
    For longer prompts: runs sentence-transformers dedup first, then LLM rewrite.

    Parses COMPRESSED: and CHANGES: labels from structured response.
    Falls back to full result if model ignores the format.
    """
    # For longer prompts, run semantic dedup first
    input_text = prompt_text
    if len(prompt_text.split()) > 50:
        input_text = _remove_semantic_redundancy(prompt_text, threshold=0.88)

    messages = [
        {"role": "system", "content": PROMPT_IMPROVEMENT_SYSTEM},
        {"role": "user",   "content": f"Rewrite this prompt to use fewer words:\n\n{input_text}"}
    ]

    result = (
        _groq(messages, temperature=0.1, max_tokens=600)
        or _gemini(
            f"Rewrite this prompt to use fewer words:\n\n{input_text}",
            PROMPT_IMPROVEMENT_SYSTEM,
            600
        )
        or ""
    )

    if not result:
        return {"compressed": prompt_text, "tokens_saved": 0, "changes": ["Could not compress — check API keys"]}

    compressed, changes = "", []
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("COMPRESSED:"):
            compressed = stripped[len("COMPRESSED:"):].strip()
        elif stripped.startswith("- "):
            changes.append(stripped[2:].strip())

    if not compressed:
        compressed = result.strip()

    # Strip meta-commentary prefixes
    for prefix in ["here is", "here's", "compressed:", "rewritten:", "result:"]:
        if compressed.lower().startswith(prefix):
            compressed = compressed[len(prefix):].strip().lstrip(":").strip()

    tokens_saved = max(0, len(prompt_text.split()) - len(compressed.split()))

    return {
        "compressed":   compressed,
        "tokens_saved": tokens_saved,
        "changes":      changes or ["Removed filler and shortened phrasing"],
    }


def generate_explanation(concept: str, level: str = "simple", user_data: Optional[str] = None) -> str:
    """
    Plain-English or technical explanation of an ML/LLM concept.
    Gemini first — slightly better at long-form educational content.
    """
    system   = EXPLANATION_PROMPTS.get(level, EXPLANATION_PROMPTS["simple"])
    data_ctx = f"\n\nUse this as a concrete example:\n{user_data[:300]}" if user_data else ""
    prompt   = f"Explain: {concept}{data_ctx}"

    return (
        _gemini(prompt, system, 1500)
        or _groq(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        or "Explanation unavailable — check API keys."
    )


def generate_demo_response(concept: str, user_input: str, demo_type: str = "tokenize") -> str:
    """
    Educational responses for the six learn page demos.
    Groq Llama 3.1 8B — fast and cheap for short responses.
    """
    prompts = {
        "tokenize":   f"Show how '{user_input}' gets tokenized. Show each token, why it splits there, and the total count. Compare Claude vs ChatGPT tokenization.",
        "embed":      f"Explain how '{user_input}' becomes a vector embedding. What meaning does it capture? What sentences would be close to it in vector space?",
        "entropy":    f"Calculate the Shannon entropy of '{user_input}'. Is it high or low? What does this tell us about how compressible it is?",
        "compress":   f"Compress this and explain every specific thing you removed and why: '{user_input}'",
        "attention":  f"For a conversation similar in length to '{user_input[:80]}' — explain the O(n²) attention cost. How many token pairs are computed? What does doubling the length do to speed?",
        "similarity": f"Are these ideas semantically redundant? Estimate their cosine similarity and explain: '{user_input[:300]}'",
        "quantize":   f"Walk through quantizing an embedding of '{user_input}'. Show the float32 → int8 process with real example numbers and explain the accuracy tradeoff.",
    }

    prompt   = prompts.get(demo_type, f"Demonstrate {concept} using: '{user_input}'")
    messages = [{"role": "system", "content": DEMO_SYSTEM}, {"role": "user", "content": prompt}]

    return (
        _groq(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=500)
        or _gemini(prompt, DEMO_SYSTEM, 500)
        or "Demo unavailable — check API keys."
    )