"""
llm_engine.py — Groq Llama 3.3 70B (primary) + Gemini 2.0 Flash (fallback)
"""
import os
from typing import Optional

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────

COMPRESSION_SYSTEM = """You are ContextCrunch, an AI conversation compression engine.

Your job: rewrite the conversation to be as short as possible while keeping 100% of the meaning.

How to compress:
- Remove any sentence that repeats an idea already stated earlier
- Merge two sentences that say related things into one clear sentence
- Remove filler phrases: "of course", "certainly", "great question", "I'd be happy to help", "sure thing", "absolutely"
- Remove redundant questions — if the same thing is asked twice, keep it once
- Shorten verbose explanations — say the same thing in fewer words
- Keep all unique facts, decisions, requests, and context
- Keep speaker labels exactly as they appear (Human: / Assistant: / You: etc)

Output: ONLY the compressed conversation. No explanation. No preamble. No "Here is the compressed version:". Just the conversation."""


PROMPT_IMPROVEMENT_SYSTEM = """You are a prompt engineering expert specializing in token efficiency.

Rewrite the given prompt to convey the exact same intent in as few tokens as possible.

Techniques to use:
- Remove redundant words and filler phrases
- Use imperative voice instead of "please can you"
- Remove unnecessary context that doesn't affect the output
- Consolidate repeated instructions
- Use precise vocabulary instead of lengthy descriptions

Output format — use EXACTLY these labels, nothing else:
COMPRESSED: [your rewritten prompt on a single line]
CHANGES:
- [specific change made]
- [specific change made]"""


DEMO_SYSTEM = """You are an interactive educational AI for ContextCrunch.
Explain AI concepts clearly using the user's actual input as examples.
Be concrete, specific, and educational. Under 250 words. Plain English first, technical detail if asked."""


EXPLANATION_PROMPTS = {
    "simple": """Explain this concept in plain English. Use a simple everyday analogy.
Max 3 short paragraphs. No jargon. No equations.
Focus on: what it means for someone using Claude or ChatGPT every day.""",

    "technical": """Explain this concept for a developer.
Include: the key formula with a one-line explanation of each variable.
Include: one short Python code example (under 10 lines).
Include: practical implications for LLM context management.
Max 5 paragraphs.""",
}


# ── LLM CALLS ─────────────────────────────────────────────────────────

def _groq(
    messages: list,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2000,
    temperature: float = 0.1
) -> Optional[str]:
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


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────

def generate_compression(text: str, math_compressed: str, model: str = "claude", plan: str = "sonnet") -> dict:
    """
    Compress a conversation using Llama 3.3 70B.
    Sends the full original text — not truncated.
    Falls back to Gemini, then to math_compressed.
    """
    prompt = f"""Compress this conversation. Remove ALL repetition and filler. Keep 100% of unique meaning. Output ONLY the compressed conversation.

{text}"""

    messages = [
        {"role": "system", "content": COMPRESSION_SYSTEM},
        {"role": "user",   "content": prompt}
    ]

    result = (
        _groq(messages, temperature=0.05, max_tokens=min(len(text.split()) * 2, 4000))
        or _gemini(prompt, COMPRESSION_SYSTEM, min(len(text.split()) * 2, 4000))
        or math_compressed
    )

    model_used = "groq" if result not in (math_compressed, None) else "gemini"
    return {"compressed": result, "model_used": model_used}


def improve_prompt(prompt_text: str) -> dict:
    """
    Rewrite a prompt to be shorter while keeping identical intent.
    Returns compressed text + list of specific changes made.
    """
    messages = [
        {"role": "system", "content": PROMPT_IMPROVEMENT_SYSTEM},
        {"role": "user",   "content": f"Rewrite this prompt to use fewer tokens:\n\n{prompt_text}"}
    ]

    result = (
        _groq(messages, temperature=0.1, max_tokens=800)
        or _gemini(
            f"Rewrite this prompt to use fewer tokens:\n\n{prompt_text}",
            PROMPT_IMPROVEMENT_SYSTEM,
            800
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

    # If parsing failed, use the full result as compressed
    if not compressed:
        compressed = result.strip()

    # Remove any meta-commentary the model may have added
    for prefix in ["here is", "here's", "compressed:", "rewritten:"]:
        if compressed.lower().startswith(prefix):
            compressed = compressed[len(prefix):].strip().lstrip(":").strip()

    tokens_saved = max(0, len(prompt_text.split()) - len(compressed.split()))

    return {
        "compressed": compressed,
        "tokens_saved": tokens_saved,
        "changes": changes or ["Removed filler and shortened phrasing"],
    }


def generate_explanation(concept: str, level: str = "simple", user_data: Optional[str] = None) -> str:
    """Generate an explanation of a concept at the given level."""
    system  = EXPLANATION_PROMPTS.get(level, EXPLANATION_PROMPTS["simple"])
    data_ctx = f"\n\nUse this as an example in your explanation:\n{user_data[:300]}" if user_data else ""
    prompt  = f"Explain: {concept}{data_ctx}"

    return (
        _gemini(prompt, system, 1500)
        or _groq(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1500
        )
        or "Explanation unavailable — check API keys."
    )


def generate_demo_response(concept: str, user_input: str, demo_type: str = "tokenize") -> str:
    """Power learn page interactive demos — uses fast Llama 3.1 8B."""
    prompts = {
        "tokenize":  f"Show how '{user_input}' gets tokenized. Show each token, why it splits there, and the total count. Compare Claude vs ChatGPT tokenization.",
        "embed":     f"Explain how '{user_input}' becomes a vector embedding. What meaning does it capture? What sentences would be close to it in vector space?",
        "entropy":   f"Calculate the Shannon entropy of '{user_input}'. Is it high or low? What does this mean for how compressible it is?",
        "compress":  f"Compress this and explain every specific thing you removed and why: '{user_input}'",
        "attention": f"For a conversation of length similar to '{user_input[:80]}' — explain the O(n²) attention cost. How many token pairs are computed? What does doubling length do?",
        "similarity":f"Are these ideas semantically redundant? Estimate their cosine similarity and explain: '{user_input[:300]}'",
        "quantize":  f"Walk through quantizing an embedding of '{user_input}'. Show float32 → int8 with real numbers.",
    }

    prompt   = prompts.get(demo_type, f"Demonstrate {concept} using: '{user_input}'")
    messages = [{"role": "system", "content": DEMO_SYSTEM}, {"role": "user", "content": prompt}]

    return (
        _groq(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=500)
        or _gemini(prompt, DEMO_SYSTEM, 500)
        or "Demo unavailable — check API keys."
    )