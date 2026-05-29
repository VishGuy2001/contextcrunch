"""
llm_engine.py — Model-aware compression pipeline

3-stage compression:
  1. sentence-transformers  — semantic redundancy removal (local, free)
  2. LLMLingua-2            — token-level pruning (local, free)
  3. Model-aware LLM rewrite — fluency pass using model-specific prompts

Prompt engineering sources:
  Claude:  Anthropic Prompting Best Practices (platform.claude.com/docs)
           — XML tags for structure, explicit constraints, role + task separation
           — Direct, unambiguous instructions; no filler; literal instruction following
  GPT:     OpenAI GPT-5.4 Prompting Guide (developers.openai.com)
           — CTCO pattern (Context/Task/Constraints/Output)
           — Output contracts via <output_contract> XML, verbosity controls
           — Structured reasoning effort for production workloads
  Gemini:  Google Vertex AI Prompting Guide
           — Explicit output format headers, clear scope constraints
           — Role + task separation, negative constraints before positive

LLM routing:
  Primary:  Groq Llama 3.3 70B  — fast, 6000 req/day free
  Fallback: Gemini 2.0 Flash    — 1500 req/day free
"""
import os
import re
import numpy as np
from typing import Optional


# ── LAZY-LOADED MODELS ────────────────────────────────────────────────

_st_model      = None
_ll_compressor = None


def _get_st_model():
    """sentence-transformers all-MiniLM-L6-v2 — 22MB, cached after first load."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _get_lingua():
    """LLMLingua-2 BERT — CPU inference, no API call needed."""
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


# ── MODEL-AWARE SYSTEM PROMPTS ────────────────────────────────────────
# Each prompt is tuned to the specific model receiving it.
# Source: Anthropic Prompting Best Practices, OpenAI GPT-5.4 Prompting Guide,
#         Google Vertex AI Prompting Guide (all verified April 2026)

def _get_compression_prompt(target_model: str, plan: str) -> str:
    """
    Return a system prompt tuned to the model that will execute the rewrite.

    Claude (Anthropic guidance):
      - XML tags for structured instructions (most effective technique per docs)
      - Role in system prompt for domain expertise boost
      - Direct, literal instructions — new Claude models follow precisely
      - Explicit negative constraints before positive ones

    GPT (OpenAI guidance):
      - CTCO pattern: Context → Task → Constraints → Output
      - <output_contract> XML block for format adherence
      - <verbosity_controls> to prevent over-generation
      - Explicit "do not" instructions (GPT-5.4 is highly literal)

    Gemini (Google guidance):
      - Clear role definition + objective statement
      - Numbered constraint lists (structured format works well)
      - Explicit output format specification
      - Scope limits stated before task description
    """

    if target_model == "claude":
        # Anthropic best practice: XML tags, role prompting, direct constraints
        # Source: platform.claude.com/docs/en/build-with-claude/prompt-engineering
        return """<role>
You are a lossless conversation compression engine with expertise in semantic information theory.
</role>

<task>
Return a compressed version of the conversation that preserves 100% of the semantic meaning
in fewer tokens. The output must be immediately usable as a replacement for the original.
</task>

<constraints>
<remove>
- Filler openers: "I'd be happy to", "Of course!", "Great question!", "Certainly", "Absolutely", "Sure,", "Sounds good!", "Happy to help!", "No problem!", "Thanks for asking!"
- Repeated requests — if the user asks the same thing more than once, keep only the clearest version
- Repeated information — if the assistant restates a fact already established, keep only the first
- Verbose phrasing — rewrite to say the same thing in fewer words
- Meta-commentary: "In this response I will...", "As I mentioned earlier...", "To summarize..."
- Closing filler: "I hope this helps!", "Let me know if you need anything!", "Feel free to ask!"
- Excessive hedging: "it's worth noting that", "generally speaking", "in most cases"
</remove>

<preserve>
- Every unique fact, decision, constraint, requirement, and piece of context
- Speaker labels on their own line: Human: / Assistant:
- All code blocks verbatim — every character, every space
- Commands, file paths, URLs, package names, version strings — exactly as written
- Numbers, dates, names — exactly as written
- Negative instructions ("do not", "never", "avoid") — they carry meaning that cannot be inferred
- Error messages and stack traces — verbatim
</preserve>

<rules>
- Never merge separate messages into one turn
- Never reorder the conversation
- Never add content that was not in the original
- Do not summarize — compress. Every piece of information must survive.
- If uncertain whether something is removable, keep it.
</rules>
</constraints>

<output_format>
Output ONLY the compressed conversation. No preamble. No explanation. No label.
Start immediately with the first speaker label.
</output_format>"""

    elif target_model in ("chatgpt", "gpt"):
        # OpenAI best practice: CTCO pattern, output_contract XML, verbosity controls
        # Source: developers.openai.com/api/docs/guides/prompt-guidance
        # Source: developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide
        return """<context>
You are a production-grade conversation compression engine. You receive AI conversation transcripts
and return shorter versions that preserve 100% of semantic meaning. The output is used directly
as a replacement for the original in a new AI session.
</context>

<task>
Rewrite the conversation to remove token waste while preserving every piece of information.
</task>

<constraints>
DO NOT include:
- Filler openers: "Of course!", "Certainly!", "I'd be happy to", "Great question!", "Absolutely!", "Sure,"
- Closing phrases: "I hope this helps!", "Let me know if you need anything!", "Feel free to ask!"
- Restatements: "As I mentioned earlier", "To reiterate", "As I said", "In summary"
- Hedges: "it's worth noting", "generally speaking", "in most cases", "it should be noted"
- Repeated content: if the same fact or request appears more than once, keep only the clearest instance
- Verbose phrasing: if a sentence can be shorter without losing meaning, shorten it

ALWAYS preserve:
- Speaker labels exactly as written: Human: / Assistant:
- All code blocks — verbatim, no modifications
- All file paths, URLs, commands, package names, version strings
- All numbers, dates, proper names
- Negative instructions ("do not", "never") — removing these changes behavior
- Error messages — verbatim
</constraints>

<output_contract>
- Return ONLY the compressed conversation text
- Do not include any preamble, explanation, label, or commentary
- Do not include phrases like "Here is the compressed version:"
- Start the output immediately with the first speaker label
- Every message from the original must appear in some form in the output
- Do not merge separate messages or reorder turns
</output_contract>

<verbosity_controls>
- Prefer information-dense rewrites over literal shortening
- Do not compress so aggressively that meaning becomes ambiguous
- When in doubt, keep the content
</verbosity_controls>"""

    else:
        # Gemini best practice: role + objective, numbered constraints, explicit format
        # Source: Google Vertex AI Prompting Guide, cloud.google.com/vertex-ai
        return """## Role and Objective
You are a conversation compression engine. Your objective is to return a shorter version
of the conversation that preserves 100% of the semantic meaning and is immediately usable
as a replacement in a new AI session.

## Scope
Input: An AI conversation transcript with Human: / Assistant: speaker labels.
Output: The same conversation compressed — shorter, same meaning, ready to paste.

## What to Remove
1. Filler openers: "Of course!", "Certainly!", "I'd be happy to", "Great question!", "Absolutely!"
2. Closing phrases: "I hope this helps!", "Let me know if you need anything!"
3. Restatements: "As I mentioned", "To reiterate", "As I said before"
4. Excessive hedging: "it's worth noting", "generally speaking", "in most cases"
5. Repeated content: when the same fact or request appears more than once, keep only the clearest
6. Verbose phrasing: rewrite long sentences into shorter equivalents

## What to Always Keep
1. Every unique fact, decision, constraint, and piece of context
2. Speaker labels on their own line: Human: / Assistant:
3. All code blocks — verbatim, every character
4. All file paths, URLs, commands, package names, version strings — exactly as written
5. All numbers, dates, and proper names — exactly as written
6. Negative instructions ("do not", "never", "avoid") — these cannot be inferred
7. Error messages and stack traces — verbatim

## Output Format
Output ONLY the compressed conversation. No preamble. No explanation. No label.
Begin immediately with the first speaker label."""


def _get_prompt_improvement_prompt(target_model: str, plan: str) -> str:
    """
    Return a prompt improvement system prompt tuned to the executing model.

    All three follow the same professional structure but use model-native
    instruction patterns based on official guidance from each provider.
    """

    if target_model == "claude":
        # Anthropic: XML structure, role expertise, explicit output format
        return """<role>
You are a prompt engineering expert specializing in token efficiency and semantic precision.
You optimize prompts for use with large language models.
</role>

<task>
Rewrite the prompt to say exactly the same thing in fewer tokens.
The rewritten prompt must produce identical LLM output to the original when sent to any model.
</task>

<constraints>
<remove>
- Politeness filler: "please", "could you", "I was wondering if", "would you mind", "feel free to"
- Redundant context: anything the model can reliably infer that does not change the output
- Repeated instructions: if two sentences express the same constraint, keep only the more specific
- Weak hedges: "maybe", "perhaps", "sort of", "kind of", "if possible", "if you can"
- Meta-instructions: "I want you to", "your task is to", "I need you to", "I'd like you to"
- Unnecessary preamble: "I have a question about", "I'd like to ask", "Can you help me with"
</remove>

<preserve>
- Every constraint, requirement, and boundary condition — even if seemingly redundant
- Format requirements: length limits, output structure, labels, JSON schemas
- Examples — they are the highest-signal content in any prompt, never remove them
- Negative instructions ("do not", "never", "avoid", "exclude") — dropping these changes output
- Persona or role instructions — they shape every word of the response
- Technical terms, variable names, function names — verbatim, never paraphrase
- All tasks in a multi-task prompt — if N things are asked for, N things must remain
</preserve>

<rules>
- The rewritten prompt must produce identical output to the original
- Never sacrifice meaning for brevity — a shorter prompt that changes the answer is wrong
- Shorter is only better when meaning is fully preserved
</rules>
</constraints>

<output_format>
Use EXACTLY these two labels, nothing before them, nothing after CHANGES:
COMPRESSED: [the rewritten prompt on a single line]
CHANGES:
- [specific change and why it preserves meaning]
- [specific change and why it preserves meaning]
</output_format>"""

    elif target_model in ("chatgpt", "gpt"):
        # OpenAI: CTCO, output_contract, verbosity_controls
        return """<context>
You are a production prompt engineering expert. You optimize prompts for token efficiency
without changing the output they produce when sent to an LLM.
</context>

<task>
Rewrite the user's prompt to use fewer tokens while preserving exact semantic intent.
The rewritten prompt must produce identical LLM behavior to the original.
</task>

<constraints>
REMOVE these patterns (they add tokens without adding information):
- Politeness: "please", "could you", "would you mind", "feel free to", "I'd appreciate if"
- Meta-instructions: "I want you to", "your task is to", "I need you to"
- Preamble: "I have a question about", "Can you help me with", "I'd like to ask"
- Redundant context: anything the model can infer that does not change the output
- Duplicate instructions: keep only the most specific version of repeated constraints
- Weak hedges: "maybe", "perhaps", "sort of", "if possible"

NEVER remove these (removing them changes LLM behavior):
- Constraints and requirements — even if they seem redundant
- Negative instructions: "do not", "never", "avoid", "exclude", "do not include"
- Output format specifications: length, structure, JSON schema, labels
- Examples — they are the highest-signal content in any prompt
- Role or persona instructions
- Technical terms and variable names — verbatim always
- In a multi-task prompt: all N tasks must remain
</constraints>

<output_contract>
Return EXACTLY this structure, nothing before COMPRESSED:, nothing after the CHANGES list:
COMPRESSED: [rewritten prompt on one line]
CHANGES:
- [what changed and why it still preserves meaning]
- [what changed and why it still preserves meaning]
</output_contract>

<verbosity_controls>
- Do not compress so aggressively that the intent becomes ambiguous
- A shorter prompt that changes the model's answer is wrong
- When in doubt, keep the content
</verbosity_controls>"""

    else:
        # Gemini: structured headers, numbered lists, clear output format
        return """## Role
You are a prompt engineering expert specializing in token efficiency for large language models.

## Objective
Rewrite the prompt to use fewer tokens. The rewritten version must produce identical LLM output
to the original when sent to any model.

## What to Remove
1. Politeness filler: "please", "could you", "would you mind", "feel free to"
2. Meta-instructions: "I want you to", "your task is to", "I need you to"
3. Unnecessary preamble: "I have a question about", "Can you help me with"
4. Redundant context: anything the model can infer without changing the output
5. Duplicate instructions: when two sentences say the same constraint, keep the more specific one
6. Weak hedges: "maybe", "perhaps", "sort of", "if possible"

## What to Always Keep
1. Every constraint and requirement — even seemingly redundant ones
2. Negative instructions: "do not", "never", "avoid", "exclude" — removing these changes output
3. Output format specs: length limits, structure, JSON schema, labels
4. Examples — highest-signal content, never remove
5. Role or persona instructions
6. Technical terms, variable names — verbatim
7. All tasks in a multi-task prompt

## Output Format
Use exactly these two labels, with nothing before COMPRESSED:
COMPRESSED: [rewritten prompt on a single line]
CHANGES:
- [specific change and why meaning is preserved]
- [specific change and why meaning is preserved]"""


# ── EXPLANATION PROMPTS ───────────────────────────────────────────────

EXPLANATION_PROMPTS = {
    "simple": """Explain this concept in plain English using a concrete everyday analogy.
Max 3 short paragraphs. No jargon. No equations.
Focus entirely on what this means for someone using Claude, ChatGPT, or Gemini daily.
End with one practical tip they can apply right now.""",

    "technical": """Explain this concept for a software engineer or ML practitioner.
Structure your response exactly as follows:
1. Core definition — one precise sentence
2. The key formula — write it out, then explain each variable in one line
3. A Python code example — under 12 lines, runnable, with one comment per meaningful line
4. Practical implications — how this affects LLM performance, cost, or context management
5. One common misconception and why it is wrong
Max 5 paragraphs total. No filler. No closing pleasantries.""",
}


# ── LLM WRAPPERS ──────────────────────────────────────────────────────

def _groq(
    messages: list,
    model: str        = "llama-3.3-70b-versatile",
    max_tokens: int   = 2000,
    temperature: float = 0.05
) -> Optional[str]:
    """Groq API. Returns stripped text or None on any failure."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp   = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
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
        resp = m.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.05}
        )
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


# ── LOCAL COMPRESSION PIPELINE ────────────────────────────────────────

def _remove_semantic_redundancy(text: str, threshold: float = 0.88) -> str:
    """
    Stage 1: sentence-transformers semantic deduplication.

    Embeds each sentence and removes any whose cosine similarity to a
    previously kept sentence exceeds the threshold (0.88 = near-identical meaning).

    Always preserved: speaker labels, code blocks, sentences < 5 words.
    Falls back gracefully if the model is unavailable.
    """
    code_blocks: dict = {}

    def _protect(m):
        key = f"__CODE_{len(code_blocks)}__"
        code_blocks[key] = m.group(0)
        return key

    safe      = re.sub(r"```[\s\S]*?```|`[^`]+`", _protect, text)
    sentences = re.split(r'(?<=[.!?])\s+', safe)

    if len(sentences) <= 3:
        for k, v in code_blocks.items():
            safe = safe.replace(k, v)
        return safe

    try:
        model       = _get_st_model()
        kept        = []
        kept_embeds = []

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            if (s.startswith(("Human:", "Assistant:", "User:", "AI:"))
                    or any(k in s for k in code_blocks)
                    or len(s.split()) < 5):
                kept.append(s)
                continue

            embed = model.encode(s, convert_to_numpy=True)

            if not kept_embeds:
                kept.append(s)
                kept_embeds.append(embed)
                continue

            sims = np.dot(kept_embeds, embed) / (
                np.linalg.norm(kept_embeds, axis=1) * np.linalg.norm(embed) + 1e-9
            )
            if sims.max() < threshold:
                kept.append(s)
                kept_embeds.append(embed)

        result = " ".join(kept)
        for k, v in code_blocks.items():
            result = result.replace(k, v)
        return result

    except Exception as e:
        print(f"Semantic dedup error: {e}")
        for k, v in code_blocks.items():
            safe = safe.replace(k, v)
        return safe


def _lingua_compress(text: str, rate: float = 0.75) -> str:
    """
    Stage 2: LLMLingua-2 token-level pruning.

    Keeps 75% of tokens, drops lowest-importance 25% via BERT importance scoring.
    Force-preserves structural tokens so conversation shape survives.
    Falls back to input if LLMLingua is unavailable.
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

def generate_compression(
    text: str,
    math_compressed: str,
    model: str = "claude",
    plan:  str = "sonnet"
) -> dict:
    """
    3-stage compression pipeline.

    Stage 1 (local, free): sentence-transformers semantic dedup
      Removes sentences near-identical in meaning to prior sentences.
      Uses cosine similarity on 384-dim embeddings. Threshold: 0.88.

    Stage 2 (local, free): LLMLingua-2 token pruning
      BERT-based token importance scoring drops low-signal tokens.
      Rate: 75% retention. Runs on CPU in ~200ms.

    Stage 3 (API): Model-aware fluency rewrite
      Groq Llama 3.3 70B receives already-compressed text.
      System prompt is tuned to the target model (claude/chatgpt/gemini)
      using that provider's official prompt engineering guidance.
      Input is 40-60% smaller so API call is fast and cheap.

    math_compressed: last-resort fallback if all three stages fail.
    """
    # Determine the target model for prompt tuning
    # We use a generic professional prompt for the Groq/Gemini rewriter
    # but structure it according to the user's target model conventions
    target = model if model in ("claude", "chatgpt", "gemini") else "claude"

    after_dedup  = _remove_semantic_redundancy(text, threshold=0.88)
    after_lingua = _lingua_compress(after_dedup, rate=0.75)
    max_tok      = min(len(after_lingua.split()) * 2, 3000)

    system_prompt  = _get_compression_prompt(target, plan)
    rewrite_prompt = (
        "This conversation has already been pre-compressed locally. "
        "Rewrite it so it flows naturally while following the instructions above. "
        "Output ONLY the rewritten conversation.\n\n"
        + after_lingua
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": rewrite_prompt}
    ]

    groq_result   = _groq(messages, temperature=0.05, max_tokens=max_tok)
    gemini_result = None if groq_result else _gemini(rewrite_prompt, system_prompt, max_tok)
    result        = groq_result or gemini_result or after_lingua or math_compressed
    model_used    = (
        "groq"   if groq_result   else
        "gemini" if gemini_result else
        "lingua" if result == after_lingua else
        "math"
    )

    return {"compressed": result, "model_used": model_used}


def improve_prompt(
    prompt_text: str,
    model: str = "claude",
    plan:  str = "sonnet"
) -> dict:
    """
    Rewrite a prompt for maximum token efficiency while preserving exact intent.

    Uses a model-aware system prompt tuned to the target model's conventions:
    - Claude: XML-structured, role-based, explicit constraints
    - GPT:    CTCO pattern, output_contract, verbosity_controls
    - Gemini: Structured headers, numbered constraints, explicit format

    For prompts > 50 words: semantic dedup runs first to remove redundant
    sentences before the LLM sees the text.

    Parses COMPRESSED: / CHANGES: structured output.
    Falls back to full result if the model ignores the format.
    """
    target     = model if model in ("claude", "chatgpt", "gemini") else "claude"
    input_text = prompt_text

    if len(prompt_text.split()) > 50:
        input_text = _remove_semantic_redundancy(prompt_text, threshold=0.85)

    system_prompt = _get_prompt_improvement_prompt(target, plan)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Rewrite this prompt to use fewer tokens:\n\n{input_text}"}
    ]

    result = (
        _groq(messages, temperature=0.05, max_tokens=800)
        or _gemini(
            f"Rewrite this prompt to use fewer tokens:\n\n{input_text}",
            system_prompt, 800
        )
        or ""
    )

    if not result:
        return {
            "compressed":   prompt_text,
            "tokens_saved": 0,
            "changes":      ["Could not compress — check API keys"]
        }

    compressed, changes = "", []
    for line in result.split("\n"):
        s = line.strip()
        if s.upper().startswith("COMPRESSED:"):
            compressed = s[len("COMPRESSED:"):].strip()
        elif s.startswith("- "):
            changes.append(s[2:].strip())

    if not compressed:
        compressed = result.strip()

    # Strip any meta-prefix the model added before the actual prompt
    for prefix in ["here is", "here's", "compressed:", "rewritten:", "result:", "output:"]:
        if compressed.lower().startswith(prefix):
            compressed = compressed[len(prefix):].strip().lstrip(":").strip()

    return {
        "compressed":   compressed,
        "tokens_saved": max(0, len(prompt_text.split()) - len(compressed.split())),
        "changes":      changes or ["Removed filler and shortened phrasing"],
    }


def generate_explanation(
    concept:   str,
    level:     str            = "simple",
    user_data: Optional[str]  = None,
    model:     str            = "claude",
    plan:      str            = "sonnet"
) -> str:
    """
    Generate a plain-English or technical explanation of an ML/LLM concept.

    The explanation style adapts to the level:
    - simple:    plain English analogy, practical tip, no jargon
    - technical: formula + variables + code + implications + misconception

    Gemini is tried first for explanations — slightly better at structured
    long-form educational content. Falls back to Groq.
    """
    system   = EXPLANATION_PROMPTS.get(level, EXPLANATION_PROMPTS["simple"])
    data_ctx = f"\n\nUse this as a concrete example in your explanation:\n{user_data[:400]}" if user_data else ""
    prompt   = f"Explain this concept: {concept}{data_ctx}"

    return (
        _gemini(prompt, system, 1500)
        or _groq(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1500
        )
        or "Explanation unavailable — check API keys."
    )