"""
llm_engine.py — Groq Llama 3.3 70B (primary) + Gemini 2.0 Flash (fallback)

Primary:  Groq Llama 3.3 70B — fast inference, 6000 req/day free
Fallback: Gemini 2.0 Flash — 1500 req/day free

All functions try Groq first, fall back to Gemini if Groq fails,
and return a safe default if both fail rather than crashing.
"""
import os
from typing import Optional


# ── SYSTEM PROMPTS ────────────────────────────────────────────────────

COMPRESSION_SYSTEM = """You are a conversation compression engine.

Your job: return a shorter version of the conversation that keeps 100% of the meaning.

What to remove:
- Filler phrases: "I'd be happy to", "Of course!", "Great question!", "Sure,", "Certainly", "Absolutely", "Great topic!"
- Repeated requests — if the user asks the same thing twice, keep it once
- Repeated information — if the AI says the same thing twice, keep it once
- Verbose phrasing — say the same thing in fewer words where possible

What to always keep:
- Every unique fact, decision, request, and piece of context
- Speaker labels on their own line: Human: / Assistant:
- The full meaning and intent of every message

Output: ONLY the compressed conversation. No explanation. No preamble. No "Here is the compressed version:". Just the conversation."""


PROMPT_IMPROVEMENT_SYSTEM = """You are a prompt engineering expert.

Rewrite the prompt to say the exact same thing in fewer words.

What to remove:
- Polite filler: "please", "could you", "I was wondering if", "would you mind"
- Redundant context that doesn't change what the AI will output
- Repeated instructions that say the same thing twice

What to keep:
- Every constraint, requirement, and specific detail
- The exact intent of the original prompt

Output format — use EXACTLY these two labels:
COMPRESSED: [rewritten prompt on one line]
CHANGES:
- [one specific thing you changed]
- [one specific thing you changed]"""


# Used by learn page demos — short educational responses using fast 8B model
DEMO_SYSTEM = """You are an educational AI for ContextCrunch.
Use the user's actual input as a concrete example to explain the concept.
Be specific, clear, and under 250 words. Plain English first."""


# Two levels only — academic was removed from the frontend
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
    """
    Call Groq API. Returns stripped response text or None on failure.
    Temperature 0.1 = near-deterministic, good for rewriting tasks.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp   = client.chat.completions.create(
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
    """
    Call Gemini 2.0 Flash. Used as fallback when Groq fails or rate limits.
    Returns stripped response text or None on failure.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        m    = genai.GenerativeModel(
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


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────

def generate_compression(text: str, math_compressed: str, model: str = "claude", plan: str = "sonnet") -> dict:
    """
    Compress a conversation using Llama 3.3 70B with Gemini fallback.

    Sends the FULL original text — not truncated, not the math_compressed version.
    The LLM needs full context to rewrite fluently and catch filler phrases
    that sentence-level embedding removal misses.

    math_compressed is kept as a last resort fallback only — if both LLMs fail.
    """
    prompt = f"""Compress this conversation. Remove filler and repeated content. Keep 100% of the meaning. Output ONLY the compressed conversation.

{text}"""

    messages = [
        {"role": "system", "content": COMPRESSION_SYSTEM},
        {"role": "user",   "content": prompt}
    ]

    # Cap max_tokens at twice the word count — compressed output should never
    # be longer than the original
    max_tok = min(len(text.split()) * 2, 4000)

    groq_result   = _groq(messages, temperature=0.05, max_tokens=max_tok)
    gemini_result = None if groq_result else _gemini(prompt, COMPRESSION_SYSTEM, max_tok)

    result     = groq_result or gemini_result or math_compressed
    model_used = "groq" if groq_result else "gemini" if gemini_result else "math"

    return {"compressed": result, "model_used": model_used}


def improve_prompt(prompt_text: str) -> dict:
    """
    Rewrite a user prompt to use fewer tokens while keeping identical intent.

    Parses COMPRESSED: and CHANGES: labels from the structured response.
    If parsing fails, uses the full result rather than returning nothing.
    tokens_saved is estimated from word count difference.
    """
    messages = [
        {"role": "system", "content": PROMPT_IMPROVEMENT_SYSTEM},
        {"role": "user",   "content": f"Rewrite this prompt to use fewer words:\n\n{prompt_text}"}
    ]

    result = (
        _groq(messages, temperature=0.1, max_tokens=600)
        or _gemini(
            f"Rewrite this prompt to use fewer words:\n\n{prompt_text}",
            PROMPT_IMPROVEMENT_SYSTEM,
            600
        )
        or ""
    )

    if not result:
        return {"compressed": prompt_text, "tokens_saved": 0, "changes": ["Could not compress — check API keys"]}

    # Parse structured output
    compressed, changes = "", []
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("COMPRESSED:"):
            compressed = stripped[len("COMPRESSED:"):].strip()
        elif stripped.startswith("- "):
            changes.append(stripped[2:].strip())

    # Fallback: if the model ignored the format, use the full result
    if not compressed:
        compressed = result.strip()

    # Strip any meta-commentary the model may have added before the actual prompt
    for prefix in ["here is", "here's", "compressed:", "rewritten:", "result:"]:
        if compressed.lower().startswith(prefix):
            compressed = compressed[len(prefix):].strip().lstrip(":").strip()

    # Estimate tokens saved from word count difference
    # Word count is a reasonable proxy — avoids calling count_tokens here
    tokens_saved = max(0, len(prompt_text.split()) - len(compressed.split()))

    return {
        "compressed":   compressed,
        "tokens_saved": tokens_saved,
        "changes":      changes or ["Removed filler and shortened phrasing"],
    }


def generate_explanation(concept: str, level: str = "simple", user_data: Optional[str] = None) -> str:
    """
    Generate a plain-English or technical explanation of a concept.
    Used by the learn pages when the user asks the AI to explain something.
    Tries Gemini first for explanations — slightly better at long-form text.
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
    Generate educational responses for the six learn page demos.
    Uses Llama 3.1 8B Instant — faster and cheaper for short responses
    where maximum reasoning depth is not needed.

    demo_type options:
      tokenize  — show how text splits into tokens, compare models
      embed     — show how text becomes a vector, what's nearby in space
      entropy   — calculate and explain Shannon entropy of the input
      compress  — show before/after with explanation of what was removed
      attention — explain O(n²) cost for the given conversation length
      similarity — compare two pieces of text for semantic overlap
      quantize  — walk through float32 → int8 quantization with numbers
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