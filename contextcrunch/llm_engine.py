"""
llm_engine.py — 3-stage compression pipeline

Stage 1: sentence-transformers — semantic redundancy removal
Stage 2: LLMLingua-2           — token-level pruning (local, free, no API)
Stage 3: Groq / Gemini         — fluency rewrite of already-compressed text

The LLM in stage 3 receives text that is already 40-60% smaller,
so each API call is faster, cheaper, and more consistent.

LLM roles:
  Primary:  Groq Llama 3.3 70B — fast, 6000 req/day free
  Fallback: Gemini 2.0 Flash   — 1500 req/day free
"""
import os
import re
import numpy as np
from typing import Optional


# ── LAZY-LOADED MODELS ────────────────────────────────────────────────

_st_model      = None
_ll_compressor = None

def _get_st_model():
    """sentence-transformers all-MiniLM-L6-v2 — 22MB, fast, cached after first load."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

def _get_lingua():
    """LLMLingua-2 BERT — runs entirely on CPU, no API call needed."""
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
- Filler openers: "I'd be happy to", "Of course!", "Great question!", "Sure,", "Certainly", "Absolutely", "Sounds good!", "That's a great point!", "Happy to help!", "Great!", "Definitely!", "No problem!", "You're welcome!", "Thanks for asking!"
- Repeated requests — if the user asks the same thing more than once, keep only the clearest version
- Repeated information — if the assistant restates the same fact or conclusion, keep only the first or most complete instance
- Verbose phrasing — rewrite long sentences into shorter equivalents without losing meaning
- Meta-commentary: "In this response I will...", "To summarize...", "As I mentioned earlier...", "Let me explain...", "Allow me to..."
- Excessive hedging: "it's worth noting that", "it's important to understand that", "generally speaking", "in most cases"
- Closing filler: "I hope this helps!", "Let me know if you have questions!", "Feel free to ask!", "Is there anything else?"

KEEP — never remove or paraphrase these:
- Every unique fact, decision, constraint, requirement, and piece of context
- Speaker labels on their own line: Human: / Assistant:
- All code blocks exactly — verbatim, every character
- Commands, file paths, URLs, package names — verbatim
- Numbers, dates, names, version strings — exactly as written
- Negative instructions ("do not", "never", "avoid") — they carry meaning that cannot be dropped
- Error messages and stack traces — verbatim

RULES:
- Never merge separate messages into one turn
- Never reorder the conversation
- Never add anything that wasn't in the original
- Never summarize — compress. The compressed version must contain every piece of information.
- If you are unsure whether something is removable, keep it.

Output: ONLY the compressed conversation. No preamble, no explanation, no label. Start immediately with the first speaker label."""


PROMPT_IMPROVEMENT_SYSTEM = """You are a prompt engineering expert who optimizes for maximum token efficiency and semantic precision.

Your job: rewrite the prompt to say exactly the same thing in fewer tokens.

REMOVE:
- Politeness filler: "please", "could you", "I was wondering if", "would you mind", "feel free to", "I'd appreciate it if"
- Redundant context — anything the model can reliably infer that does not change the output
- Repeated instructions — if two sentences say the same thing, keep only the more specific one
- Weak hedges: "maybe", "perhaps", "sort of", "kind of", "if possible", "if you can"
- Meta-instructions: "I want you to", "your task is to", "I need you to"
- Unnecessary preamble: "I have a question about...", "I'd like to ask...", "Can you help me with..."

KEEP — never remove these:
- Every constraint, requirement, and boundary condition
- Format requirements: length, structure, output format, labels
- Examples — highest-signal content, never remove them
- Negative instructions ("do not", "never", "avoid", "exclude") — dropping these changes the output
- Persona or role instructions — they shape every word of the response
- Technical terms, variable names, function names — verbatim
- Multiple tasks — if the prompt asks for N things, the output must ask for all N things

RULES:
- The compressed prompt must produce identical output to the original when sent to an LLM
- Never sacrifice meaning for brevity — a shorter prompt that changes the answer is wrong
- If a constraint seems redundant but its removal could change behavior, keep it

Output format — use EXACTLY these two labels, nothing before them:
COMPRESSED: [the rewritten prompt on a single line]
CHANGES:
- [one specific change with a brief reason]
- [one specific change with a brief reason]"""


EXPLANATION_PROMPTS = {
    "simple": """Explain this concept in plain English using a concrete everyday analogy.
Max 3 short paragraphs. No jargon. No equations.
Focus entirely on what this means for someone using Claude, ChatGPT, or Gemini daily.
End with one practical tip they can apply right now.""",

    "technical": """Explain this concept for a software engineer or ML practitioner.
Structure:
1. Core definition — one precise sentence
2. The key formula — write it out, then explain each variable in one line
3. A Python code example — under 12 lines, runnable, with one comment per meaningful line
4. Practical implications — how this affects LLM performance, cost, or context management
5. One common misconception — and why it's wrong
Max 5 paragraphs total.""",
}


# ── LLM WRAPPERS ──────────────────────────────────────────────────────

def _groq(messages: list, model: str = "llama-3.3-70b-versatile", max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
    """Groq API. Returns stripped text or None on any failure."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp   = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def _gemini(prompt: str, system: str = "", max_tokens: int = 2000) -> Optional[str]:
    """Gemini 2.0 Flash fallback. Returns stripped text or None on any failure."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        m    = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system or None)
        resp = m.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": 0.1})
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


# ── LOCAL COMPRESSION PIPELINE ────────────────────────────────────────

def _remove_semantic_redundancy(text: str, threshold: float = 0.88) -> str:
    """
    Stage 1: sentence-transformers semantic dedup.
    Drops sentences whose cosine similarity to a prior kept sentence exceeds threshold.
    Always preserves: speaker labels, code blocks, short sentences (<5 words).
    """
    code_blocks: dict = {}

    def _protect_code(m):
        key = f"__CODE_{len(code_blocks)}__"
        code_blocks[key] = m.group(0)
        return key

    text_safe = re.sub(r"```[\s\S]*?```|`[^`]+`", _protect_code, text)
    sentences = re.split(r'(?<=[.!?])\s+', text_safe)

    if len(sentences) <= 3:
        for k, v in code_blocks.items():
            text_safe = text_safe.replace(k, v)
        return text_safe

    model       = _get_st_model()
    kept        = []
    kept_embeds = []

    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue

        if (stripped.startswith(("Human:", "Assistant:", "User:", "AI:"))
                or any(k in stripped for k in code_blocks)
                or len(stripped.split()) < 5):
            kept.append(stripped)
            continue

        embed = model.encode(stripped, convert_to_numpy=True)

        if not kept_embeds:
            kept.append(stripped)
            kept_embeds.append(embed)
            continue

        sims = np.dot(kept_embeds, embed) / (
            np.linalg.norm(kept_embeds, axis=1) * np.linalg.norm(embed) + 1e-9
        )

        if sims.max() < threshold:
            kept.append(stripped)
            kept_embeds.append(embed)

    result = " ".join(kept)
    for k, v in code_blocks.items():
        result = result.replace(k, v)
    return result


def _lingua_compress(text: str, rate: float = 0.75) -> str:
    """
    Stage 2: LLMLingua-2 token-level pruning.
    Keeps 75% of tokens, drops lowest-importance 25%.
    Falls back to input if LLMLingua fails to load.
    """
    lingua = _get_lingua()
    if lingua is None:
        return text
    try:
        result = lingua.compress_prompt(
            text, rate=rate,
            force_tokens=["\n", "Human:", "Assistant:", "User:", "AI:", "?", ":", "def ", "```"]
        )
        return result.get("compressed_prompt", text)
    except Exception as e:
        print(f"LLMLingua error: {e}")
        return text


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────

def generate_compression(text: str, math_compressed: str, model: str = "claude", plan: str = "sonnet") -> dict:
    """
    3-stage compression pipeline.
    Stage 1: semantic dedup (local), Stage 2: token pruning (local), Stage 3: fluency rewrite (API).
    """
    after_dedup  = _remove_semantic_redundancy(text, threshold=0.88)
    after_lingua = _lingua_compress(after_dedup, rate=0.75)
    max_tok      = min(len(after_lingua.split()) * 2, 3000)

    rewrite_prompt = (
        "This conversation has already been compressed. "
        "Rewrite it so it flows naturally. Remove any remaining filler. "
        "Keep 100% of the meaning. Output ONLY the rewritten conversation.\n\n"
        + after_lingua
    )

    messages      = [{"role": "system", "content": COMPRESSION_SYSTEM}, {"role": "user", "content": rewrite_prompt}]
    groq_result   = _groq(messages, temperature=0.05, max_tokens=max_tok)
    gemini_result = None if groq_result else _gemini(rewrite_prompt, COMPRESSION_SYSTEM, max_tok)
    result        = groq_result or gemini_result or after_lingua or math_compressed
    model_used    = "groq" if groq_result else "gemini" if gemini_result else "lingua" if result == after_lingua else "math"

    return {"compressed": result, "model_used": model_used}


def improve_prompt(prompt_text: str) -> dict:
    """
    Rewrite a prompt for maximum token efficiency while preserving exact intent.
    Runs semantic dedup first on prompts >50 words, then LLM rewrite.
    """
    input_text = prompt_text
    if len(prompt_text.split()) > 50:
        input_text = _remove_semantic_redundancy(prompt_text, threshold=0.85)

    messages = [
        {"role": "system", "content": PROMPT_IMPROVEMENT_SYSTEM},
        {"role": "user",   "content": f"Rewrite this prompt to use fewer tokens:\n\n{input_text}"}
    ]

    result = (
        _groq(messages, temperature=0.05, max_tokens=800)
        or _gemini(f"Rewrite this prompt to use fewer tokens:\n\n{input_text}", PROMPT_IMPROVEMENT_SYSTEM, 800)
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

    for prefix in ["here is", "here's", "compressed:", "rewritten:", "result:", "output:"]:
        if compressed.lower().startswith(prefix):
            compressed = compressed[len(prefix):].strip().lstrip(":").strip()

    return {
        "compressed":   compressed,
        "tokens_saved": max(0, len(prompt_text.split()) - len(compressed.split())),
        "changes":      changes or ["Removed filler and shortened phrasing"],
    }


def generate_explanation(concept: str, level: str = "simple", user_data: Optional[str] = None) -> str:
    """Generate a plain-English or technical explanation of an ML/LLM concept."""
    system   = EXPLANATION_PROMPTS.get(level, EXPLANATION_PROMPTS["simple"])
    data_ctx = f"\n\nUse this as a concrete example:\n{user_data[:400]}" if user_data else ""
    prompt   = f"Explain this concept: {concept}{data_ctx}"

    return (
        _gemini(prompt, system, 1500)
        or _groq([{"role": "system", "content": system}, {"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500)
        or "Explanation unavailable — check API keys."
    )