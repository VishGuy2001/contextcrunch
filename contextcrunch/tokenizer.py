"""
tokenizer.py — exact token counting per model/plan
Verified April 22, 2026 from official documentation.
"""
import tiktoken
from dataclasses import dataclass
from typing import Optional

# Verified April 22 2026:
# Claude:  platform.claude.com/docs/en/about-claude/models/overview
# ChatGPT: developers.openai.com/api/docs/models/gpt-5.4
# Gemini:  ai.google.dev/gemini-api/docs/models
MODEL_LIMITS = {
    "claude":  {
        "haiku":  200_000,   # Claude Haiku 4.5
        "sonnet": 1_000_000, # Claude Sonnet 4.6
        "opus":   1_000_000, # Claude Opus 4.7
        # legacy keys for backwards compat
        "free":   200_000,
        "plus":   1_000_000,
        "max":    1_000_000,
    },
    "chatgpt": {
        "free":   32_000,    # GPT-5.4 Mini effective window
        "plus":   272_000,   # GPT-5.4 standard window
        "pro":    1_050_000, # GPT-5.4 Pro full window
    },
    "gemini":  {
        "free":   1_048_576, # 2.5 Flash
        "pro":    1_048_576, # 2.5 Pro
        "ultra":  1_048_576, # 3.1 Pro Preview
    },
}

# Cost per token in USD (approximate, for display only)
TOKEN_COST = {
    "claude":  {"haiku": 0.000001, "sonnet": 0.000003, "opus": 0.000005, "free": 0.0, "plus": 0.000003, "max": 0.000005},
    "chatgpt": {"free": 0.0, "plus": 0.0000025, "pro": 0.00003},
    "gemini":  {"free": 0.0, "pro": 0.00000125, "ultra": 0.0000035},
}

# Chars per token by model — each uses a different tokenizer
# Claude Haiku/Sonnet: Custom BPE ~3.5 chars/token
# Claude Opus 4.7:     New BPE — up to 35% more tokens — ~2.6 chars/token
# ChatGPT:             cl100k BPE ~4.0 chars/token vocab 100,277
# Gemini:              SentencePiece unigram ~4.5 chars/token vocab 256,000
CHARS_PER_TOKEN = {
    "claude":  {"haiku": 3.5, "sonnet": 3.5, "opus": 2.6, "free": 3.5, "plus": 3.5, "max": 3.5},
    "chatgpt": {"free": 4.0, "plus": 4.0, "pro": 4.0},
    "gemini":  {"free": 4.5, "pro": 4.5, "ultra": 4.5},
}

LANGUAGE_DENSITY = {
    "python": 10, "javascript": 14, "typescript": 15,
    "java": 20, "cpp": 16, "c": 14, "rust": 15,
    "go": 12, "sql": 8, "html": 30, "css": 12,
    "json": 20, "yaml": 8, "markdown": 7, "unknown": 12,
}

MODEL_BEHAVIORS = {
    "claude": {
        "memory": "Full recall — keeps entire conversation, re-reads on every message",
        "truncation": False,
        "thinking_tokens": True,
        "warning": "Claude re-reads your full conversation on every message. Token burn accelerates fast. Opus 4.7 uses a new tokenizer — up to 35% more tokens for the same text.",
    },
    "chatgpt": {
        "memory": "Silent truncation — drops oldest messages without telling you",
        "truncation": True,
        "thinking_tokens": False,
        "warning": "ChatGPT quietly forgets older messages when context fills. You never get a hard stop — responses just quietly lose accuracy on earlier context.",
    },
    "gemini": {
        "memory": "Largest context window. Rate limits hit before token limits on free tier.",
        "truncation": False,
        "thinking_tokens": False,
        "warning": "Gemini Free: 15 req/min rate limit hits long before the 1M token window. Paid tiers may see quality degradation at very long contexts.",
    },
}


def image_tokens(width: int, height: int, model: str) -> int:
    """Official image token formulas per model."""
    if model == "claude":
        return min(int((width * height) / 750), 1600)
    elif model == "chatgpt":
        tiles = ((width + 511) // 512) * ((height + 511) // 512)
        return 85 + (tiles * 170)
    return 258  # Gemini — fixed cost per image


@dataclass
class TokenResult:
    total: int
    user_tokens: int
    ai_tokens: int
    system_tokens: int
    limit: int
    percentage: float
    cost_usd: float
    model: str
    plan: str
    warning: Optional[str] = None


def count_tokens(text: str, model: str = "claude", plan: str = "sonnet") -> int:
    """Count tokens using model-specific tokenizer."""
    if not text:
        return 0
    # ChatGPT uses cl100k BPE — exact count available via tiktoken
    if model == "chatgpt":
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Claude and Gemini: estimate using model-specific chars/token ratio
    cpt = CHARS_PER_TOKEN.get(model, {}).get(plan, 3.8)
    return max(1, int(len(text) / cpt))


def count_tokens_by_speaker(conversation: str, model: str = "claude", plan: str = "sonnet") -> TokenResult:
    """Split conversation by speaker and count tokens per speaker."""
    lines = conversation.strip().split("\n")
    user_text, ai_text, system_text = [], [], []
    user_markers   = {"human:", "user:", "you:", "me:"}
    ai_markers     = {"assistant:", "ai:", "claude:", "chatgpt:", "gemini:", "bot:", "gpt:"}
    system_markers = {"system:", "[system]", "<s>"}
    current_speaker = "user"

    for line in lines:
        lower = line.lower().strip()
        if any(lower.startswith(m) for m in user_markers):
            current_speaker = "user"
            user_text.append(line.split(":", 1)[1] if ":" in line else line)
        elif any(lower.startswith(m) for m in ai_markers):
            current_speaker = "ai"
            ai_text.append(line.split(":", 1)[1] if ":" in line else line)
        elif any(lower.startswith(m) for m in system_markers):
            current_speaker = "system"
            system_text.append(line.split(":", 1)[1] if ":" in line else line)
        else:
            (user_text if current_speaker == "user" else
             ai_text if current_speaker == "ai" else system_text).append(line)

    user_tokens   = count_tokens(" ".join(user_text), model, plan)
    ai_tokens     = count_tokens(" ".join(ai_text), model, plan)
    system_tokens = count_tokens(" ".join(system_text), model, plan)
    if system_tokens == 0:
        system_tokens = int((user_tokens + ai_tokens) * 0.05)

    total      = user_tokens + ai_tokens + system_tokens
    limit      = get_limit(model, plan)
    percentage = round((total / limit) * 100, 1)
    cost       = round((total / 1_000_000) * TOKEN_COST.get(model, {}).get(plan, 0), 6)

    warning = None
    if percentage > 90:
        warning = f"Critical: {percentage}% of {model} {plan} limit used. Compress now."
    elif percentage > 70:
        warning = f"Warning: {percentage}% of {model} {plan} context limit used."

    return TokenResult(
        total=total, user_tokens=user_tokens, ai_tokens=ai_tokens,
        system_tokens=system_tokens, limit=limit,
        percentage=min(percentage, 100.0), cost_usd=cost,
        model=model, plan=plan, warning=warning,
    )


def count_code_tokens(code: str, language: str = "unknown") -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(code))
    except Exception:
        lines = code.strip().split("\n")
        return len(lines) * LANGUAGE_DENSITY.get(language.lower(), 12)


def get_limit(model: str, plan: str) -> int:
    return MODEL_LIMITS.get(model, {}).get(plan, 200_000)


def detect_language(filename: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".rs": "rust",
        ".go": "go", ".sql": "sql", ".html": "html", ".css": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
        ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
        ".r": "r", ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
        ".ipynb": "python",
    }
    for ext, lang in ext_map.items():
        if filename.lower().endswith(ext):
            return lang
    return "unknown"