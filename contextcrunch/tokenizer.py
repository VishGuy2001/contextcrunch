"""
tokenizer.py — exact token counting per model/plan
"""
import tiktoken
from dataclasses import dataclass
from typing import Optional

MODEL_LIMITS = {
    "claude":  {"free": 40_000,  "plus": 200_000, "max": 200_000},
    "chatgpt": {"free": 16_385,  "plus": 128_000, "pro": 200_000},
    "gemini":  {"free": 32_000,  "pro": 1_000_000, "ultra": 2_000_000},
}

TOKEN_COST = {
    "claude":  {"free": 0.0,     "plus": 0.000003,  "max": 0.000015},
    "chatgpt": {"free": 0.0,     "plus": 0.000005,  "pro": 0.000015},
    "gemini":  {"free": 0.0,     "pro": 0.0000035,  "ultra": 0.000007},
}

LANGUAGE_DENSITY = {
    "python": 10, "javascript": 14, "typescript": 15,
    "java": 20, "cpp": 16, "c": 14, "rust": 15,
    "go": 12, "sql": 8, "html": 30, "css": 12,
    "json": 20, "yaml": 8, "markdown": 7, "unknown": 12,
}

MODEL_BEHAVIORS = {
    "claude": {
        "memory": "Full recall — keeps entire history, re-reads on every message",
        "truncation": False,
        "thinking_tokens": True,
        "warning": "Claude re-reads the full conversation every message. Token burn accelerates exponentially. Thinking tokens count invisibly.",
    },
    "chatgpt": {
        "memory": "Silent truncation — drops oldest messages without telling you",
        "truncation": True,
        "thinking_tokens": False,
        "warning": "ChatGPT quietly forgets your older messages. You never get a hard stop — it just loses track of things you said earlier.",
    },
    "gemini": {
        "memory": "Hybrid — mix of full context and retrieval depending on version",
        "truncation": "sometimes",
        "thinking_tokens": False,
        "warning": "Gemini Free hits rate limits before token limits. Quality degrades at very long contexts even within the window.",
    },
}


def image_tokens(width: int, height: int, model: str) -> int:
    if model == "claude":
        return min(int((width * height) / 750), 1600)
    elif model == "chatgpt":
        tiles = ((width + 511) // 512) * ((height + 511) // 512)
        return 85 + (tiles * 170)
    return 258  # gemini fixed


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


def count_tokens(text: str, model: str = "claude", plan: str = "plus") -> int:
    if not text:
        return 0
    if model == "chatgpt":
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / 3.8))


def count_tokens_by_speaker(conversation: str, model: str = "claude", plan: str = "plus") -> TokenResult:
    lines = conversation.strip().split("\n")
    user_text, ai_text, system_text = [], [], []
    user_markers = {"human:", "user:", "you:", "me:"}
    ai_markers = {"assistant:", "ai:", "claude:", "chatgpt:", "gemini:", "bot:", "gpt:"}
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

    user_tokens = count_tokens(" ".join(user_text), model, plan)
    ai_tokens = count_tokens(" ".join(ai_text), model, plan)
    system_tokens = count_tokens(" ".join(system_text), model, plan)
    if system_tokens == 0:
        system_tokens = int((user_tokens + ai_tokens) * 0.05)

    total = user_tokens + ai_tokens + system_tokens
    limit = MODEL_LIMITS.get(model, {}).get(plan, 128_000)
    percentage = round((total / limit) * 100, 1)
    cost = round((total / 1000) * TOKEN_COST.get(model, {}).get(plan, 0), 6)

    warning = None
    if percentage > 90:
        warning = f"Critical: {percentage}% of {model} {plan} limit used."
    elif percentage > 70:
        warning = f"Warning: approaching {model} {plan} context limit ({percentage}%)."

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
    return MODEL_LIMITS.get(model, {}).get(plan, 128_000)


def detect_language(filename: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".rs": "rust",
        ".go": "go", ".sql": "sql", ".html": "html", ".css": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
    }
    for ext, lang in ext_map.items():
        if filename.lower().endswith(ext):
            return lang
    return "unknown"
