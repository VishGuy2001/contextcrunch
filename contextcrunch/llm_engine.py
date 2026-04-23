"""
llm_engine.py — Groq (primary) + Gemini 2.0 Flash (fallback)
"""
import os
from typing import Optional

COMPRESSION_SYSTEM = """You are ContextCrunch — a precise context compression engine.
Compress AI conversations by:
1. Removing redundant statements
2. Merging related points into single clear statements  
3. Removing filler and over-explanation
4. Preserving ALL unique information and context
5. Keeping speaker labels (Human:/Assistant: etc)
Output ONLY the compressed conversation — no commentary.
Aim for 30-60% token reduction while preserving 100% of meaning."""

PROMPT_IMPROVEMENT_SYSTEM = """You are a prompt engineering expert.
Rewrite prompts to be more token-efficient while preserving exact intent.
Output format — use these exact labels:
COMPRESSED: [the compressed prompt]
TOKENS_SAVED: [rough estimate]
CHANGES:
- [change 1]
- [change 2]"""

DEMO_SYSTEM = """You are an interactive educational AI for ContextCrunch.
Help users understand how tokens, embeddings, entropy, and compression work.
When given input, demonstrate the concept clearly:
- Show the step by step process using the actual input
- Show before and after states  
- Explain what changed and why
- Keep responses focused, educational, and under 300 words
- Use simple language unless asked for technical detail"""

EXPLANATION_PROMPTS = {
    "simple": "Explain this concept in plain English using a simple everyday analogy. Maximum 3 short paragraphs. No jargon. No equations. Focus on intuition and practical meaning for someone using Claude or ChatGPT.",
    "technical": "Explain this concept for a developer. Include the key formula with brief explanation. Show a short Python code example. Explain practical implications for LLM context management. 4-6 paragraphs max.",
    "academic": "Provide a rigorous academic explanation. Include formal mathematical notation and proofs. Reference key papers. Discuss theoretical bounds and limitations. This is for ML researchers and engineers.",
}


def _groq(messages: list, model: str = "llama-3.3-70b-versatile", max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def _gemini(prompt: str, system: str = "", max_tokens: int = 2000) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        m = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system or None)
        resp = m.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        return resp.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def generate_compression(text: str, math_compressed: str, model: str = "claude", plan: str = "plus") -> dict:
    prompt = f"Original conversation (for reference):\n{text[:500]}...\n\nMathematically compressed version to rewrite naturally:\n{math_compressed}\n\nTarget model: {model} ({plan} plan). Rewrite naturally, keeping all information."
    messages = [{"role": "system", "content": COMPRESSION_SYSTEM}, {"role": "user", "content": prompt}]
    result = _groq(messages, temperature=0.1) or _gemini(prompt, COMPRESSION_SYSTEM) or math_compressed
    return {"compressed": result, "model_used": "groq" if result != math_compressed else "gemini"}


def generate_explanation(concept: str, level: str = "simple", user_data: Optional[str] = None) -> str:
    system = EXPLANATION_PROMPTS.get(level, EXPLANATION_PROMPTS["simple"])
    data_ctx = f"\nUse this user's actual data in your explanation:\n{user_data[:200]}" if user_data else ""
    prompt = f"Explain: {concept}{data_ctx}"
    return _gemini(prompt, system, 1500) or _groq([{"role": "system", "content": system}, {"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500) or "Explanation unavailable — check API keys."


def generate_demo_response(concept: str, user_input: str, demo_type: str = "tokenize") -> str:
    """Power learn page interactive demos — uses fast Llama 3.1 8B."""
    prompts = {
        "tokenize": f"Show how this text would be tokenized step by step: '{user_input}'. Show each token, explain why it was split that way, give the token count.",
        "embed": f"Explain how this text becomes a vector embedding: '{user_input}'. What semantic meaning does it capture? What would be close to it in vector space?",
        "entropy": f"Calculate and explain the Shannon entropy of: '{user_input}'. Is it high or low? Why? What does this mean for compressibility?",
        "compress": f"Show a compressed version of this prompt and explain exactly what was removed and why: '{user_input}'",
        "attention": f"If an AI model is processing text this long: '{user_input[:100]}' — explain the attention computation cost. How many token pairs are computed?",
        "similarity": f"Compare these two texts for semantic similarity and explain whether one is redundant: Text A: '{user_input[:200]}' — what is their cosine similarity approximately?",
        "quantize": f"Walk through quantizing a vector representation of: '{user_input}'. Show float32 → int8 → TurboQuant steps with example numbers.",
    }
    prompt = prompts.get(demo_type, f"Demonstrate {concept} using: '{user_input}'")
    messages = [{"role": "system", "content": DEMO_SYSTEM}, {"role": "user", "content": prompt}]
    return _groq(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=600) or _gemini(prompt, DEMO_SYSTEM, 600) or "Demo unavailable — check API keys."


def improve_prompt(prompt_text: str) -> dict:
    messages = [{"role": "system", "content": PROMPT_IMPROVEMENT_SYSTEM}, {"role": "user", "content": f"Improve this prompt:\n\n{prompt_text}"}]
    result = _groq(messages, temperature=0.15, max_tokens=1000) or _gemini(f"Improve this prompt:\n\n{prompt_text}", PROMPT_IMPROVEMENT_SYSTEM) or ""
    if not result:
        return {"compressed": prompt_text, "tokens_saved": 0, "changes": []}
    compressed, changes = "", []
    for line in result.split("\n"):
        if line.startswith("COMPRESSED:"):
            compressed = line.replace("COMPRESSED:", "").strip()
        elif line.startswith("- ") or line.startswith("* "):
            changes.append(line[2:].strip())
    return {"compressed": compressed or result, "tokens_saved": max(0, len(prompt_text)//4 - len(compressed)//4), "changes": changes, "raw": result}
