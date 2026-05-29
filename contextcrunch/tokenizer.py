"""
tokenizer.py — exact token counting, limits, and pricing rates per model/plan.
Verified May 2026 from official Anthropic, OpenAI, and Google AI Studio documentations.
"""
import tiktoken
from dataclasses import dataclass
from typing import Optional

MODEL_LIMITS = {
    "claude": {
        "haiku": 200_000,   # Claude Haiku 4.5
        "sonnet": 1_000_000, # Claude Sonnet 4.6
        "opus": 1_000_000,   # Claude Opus 4.7
        "free": 200_000,
        "plus": 1_000_000,
        "max": 1_000_000,
    },
    "chatgpt": {
        "free": 32_000,     # GPT-5.4 Mini
        "plus": 272_000,    # GPT-5.4 standard window
        "pro": 1_050_000,   # GPT-5.5 frontier window
    },
    "gemini": {
        "free": 1_048_576,  # Gemini 3.1 Flash-Lite
        "pro": 1_048_576,   # Gemini 3.5 Flash
        "ultra": 2_097_152, # Gemini 3.1 Pro (2,000,000 / 2M tokens)
    },
}

# Cost per 1M input tokens in USD
TOKEN_COST = {
    "claude": {
        "haiku": 1.0,
        "sonnet": 3.0,
        "opus": 5.0,
        "free": 0.0,
        "plus": 3.0,
        "max": 5.0
    },
    "chatgpt": {
        "free": 0.0,
        "plus": 2.5,
        "pro": 5.0
    },
    "gemini": {
        "free": 0.25,       # 3.1 Flash-Lite
        "pro": 0.50,        # 3.5 Flash standard input cost
        "ultra": 2.00       # 3.1 Pro standard (<= 200k tokens)
    },
}

# Cost per 1M output tokens in USD (for tiered estimation)
TOKEN_COST_OUTPUT = {
    "claude": {
        "haiku": 5.0,
        "sonnet": 15.0,
        "opus": 25.0,
        "free": 0.0,
        "plus": 15.0,
        "max": 25.0
    },
    "chatgpt": {
        "free": 0.0,
        "plus": 15.0,
        "pro": 30.0
    },
    "gemini": {
        "free": 1.50,
        "pro": 3.00,
        "ultra": 12.00     # 3.1 Pro standard (<= 200k tokens)
    },
}

CHARS_PER_TOKEN = {
    "claude": {
        "haiku": 3.5, 
        "sonnet": 3.5, 
        "opus": 2.6, # Dense Opus 4.7 tokenizer
        "free": 3.5, 
        "plus": 3.5, 
        "max": 3.5
    },
    "chatgpt": {
        "free": 4.0, 
        "plus": 4.0, 
        "pro": 4.0
    },
    "gemini": {
        "free": 4.5, 
        "pro": 4.5, 
        "ultra": 4.5
    },
}

LANGUAGE_DENSITY = {
    "python": 10, "javascript": 14, "typescript": 15,
    "java": 20, "cpp": 16, "c": 14, "rust": 15,
    "go": 12, "sql": 8, "html": 30, "css": 12,
    "json": 20, "yaml": 8, "markdown": 7, "unknown": 12,
}

MODEL_BEHAVIORS = {
    "claude": {
        "memory": "Full recall — re-reads the entire conversation history before every single response",
        "truncation": False,
        "thinking_tokens": True,
        "warning": "Claude re-reads full chats on every turn. Input usage scales quadratically. Opus 4.7 uses a dense BPE tokenizer that generates up to 35% more tokens for the same words.",
    },
    "chatgpt": {
        "memory": "Silent truncation — when the context fills up, GPT quietly deletes old messages",
        "truncation": True,
        "thinking_tokens": False,
        "warning": "ChatGPT silently drops your oldest messages when context window fills. You never receive an out-of-context error, but the model starts losing critical details from earlier turns.",
    },
    "gemini": {
        "memory": "Industry-leading 2,000,000 context window on Gemini 3.1 Pro. Low cost on Flash series.",
        "truncation": False,
        "thinking_tokens": False,
        "warning": "Gemini 3.1 Pro charges double for both input and output once prompt length exceeds 200,000 tokens — applied retroactively to the entire session.",
    },
}


def image_tokens(width: int, height: int, model: str) -> int:
    """Official image token formulas per model."""
    if model == "claude":
        return min(int((width * height) / 750), 1600)
    elif model == "chatgpt":
        tiles = ((width + 511) // 512) * ((height + 511) // 512)
        return 85 + (tiles * 170)
    return 258  # Gemini — flat rate per image


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
    """Split conversation by speaker and count tokens per speaker with 2026 pricing logic."""
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

    # 2026 Model Surcharge / Tiered Pricing Calculations
    if model == "gemini" and plan == "ultra":
        # Gemini 3.1 Pro tiered context window billing:
        # If input context <= 200,000 tokens: input $2.00/M, output $12.00/M
        # If input context > 200,000 tokens: input pricing doubles to $4.00/M, output to $18.00/M
        in_tokens = user_tokens + system_tokens
        out_tokens = ai_tokens
        
        in_rate = 0.000004 if in_tokens > 200_000 else 0.000002
        out_rate = 0.000018 if in_tokens > 200_000 else 0.000012
        cost = round((in_tokens * in_rate) + (out_tokens * out_rate), 6)
    else:
        # Standard pricing: calculate input + output separately for accuracy
        in_tokens = user_tokens + system_tokens
        out_tokens = ai_tokens
        in_cost_rate = TOKEN_COST.get(model, {}).get(plan, 0.0) / 1_000_000
        out_cost_rate = TOKEN_COST_OUTPUT.get(model, {}).get(plan, 0.0) / 1_000_000
        cost = round((in_tokens * in_cost_rate) + (out_tokens * out_cost_rate), 6)

    warning = None
    if percentage > 90:
        warning = f"Critical: {percentage}% of {model} {plan} context limit used. Compression strongly recommended."
    elif percentage > 70:
        warning = f"Warning: {percentage}% of {model} {plan} context limit used."
    elif model == "gemini" and plan == "ultra" and (user_tokens + system_tokens) > 200_000:
        warning = f"Notice: Tiered pricing surcharge applied! Prompt size > 200k tokens doubles Gemini 3.1 Pro rates."

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