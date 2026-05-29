"""
main.py — FastAPI backend for contextcrunch.io
Deployed on Google Cloud Run at api.contextcrunch.io
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
load_dotenv()

from contextcrunch.tokenizer    import count_tokens, count_tokens_by_speaker, get_limit, MODEL_LIMITS, TOKEN_COST, MODEL_BEHAVIORS
from contextcrunch.math_engine  import (
    shannon_entropy, redundancy_score, attention_cost_multiplier,
    theoretical_compression_bound, analyze_text,
    analyze_entropy_deep, analyze_redundancy_deep,
    analyze_attention_deep, analyze_quantization_deep,
    analyze_token_budget,
)
from contextcrunch.compressor   import compress, get_embeddings_for_demo
from contextcrunch.llm_engine   import generate_compression, generate_explanation, improve_prompt
from contextcrunch.file_parser  import parse_file

limiter = Limiter(key_func=get_remote_address)
app     = FastAPI(title="ContextCrunch API", version="0.2.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MAX_TEXT_CHARS   = 100_000
MAX_PROMPT_CHARS =  10_000
MAX_FILE_BYTES   = 20_000_000
MAX_SENTENCES    = 10


# ── REQUEST MODELS ────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    model: str = "claude"
    plan: str  = "sonnet"

class CompressRequest(BaseModel):
    text: str
    model: str     = "claude"
    plan: str      = "sonnet"
    threshold: float = 0.82

class ExplainRequest(BaseModel):
    concept: str
    level: str = "simple"
    user_data: Optional[str] = None

class PromptRequest(BaseModel):
    prompt: str
    model: str = "claude"
    plan: str  = "sonnet"

class EmbedRequest(BaseModel):
    sentences: list[str]

class EntropyRequest(BaseModel):
    text: str

class TokenizeRequest(BaseModel):
    text: str
    model: str = "chatgpt"

class RedundancyRequest(BaseModel):
    text: str

class AttentionRequest(BaseModel):
    token_count: int
    limit: int
    avg_tokens_per_turn: int = 150

class QuantizeRequest(BaseModel):
    embedding_dim: int = 384
    vocab_size: int    = 100277

class TokenBudgetRequest(BaseModel):
    text: str
    model: str = "claude"
    plan: str  = "sonnet"
    limit: int = 200000


# ── CORE ROUTES ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/models")
def get_models():
    return {"limits": MODEL_LIMITS, "costs": TOKEN_COST, "behaviors": MODEL_BEHAVIORS}


@app.post("/analyze")
@limiter.limit("30/minute")
def analyze(req: AnalyzeRequest, request: Request):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(400, "Text too short")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text too long — max {MAX_TEXT_CHARS:,} characters")
    token_result = count_tokens_by_speaker(req.text, req.model, req.plan)
    limit        = get_limit(req.model, req.plan)
    math         = analyze_text(req.text, token_result.total, limit)
    return {
        "tokens": {
            "total":      token_result.total,
            "user":       token_result.user_tokens,
            "ai":         token_result.ai_tokens,
            "system":     token_result.system_tokens,
            "limit":      token_result.limit,
            "percentage": token_result.percentage,
            "cost_usd":   token_result.cost_usd,
        },
        "entropy":           math["entropy"],
        "redundancy":        math["redundancy"],
        "attention":         math["attention"],
        "compression_bound": math["compression_bound"],
        "summary":           math["summary"],
        "warning":           token_result.warning,
        "model":             req.model,
        "plan":              req.plan,
    }


@app.post("/compress")
@limiter.limit("5/minute")
def compress_conversation(req: CompressRequest, request: Request):
    if not req.text or len(req.text.strip()) < 50:
        raise HTTPException(400, "Text too short to compress")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text too long — max {MAX_TEXT_CHARS:,} characters")
    math_result = compress(req.text, req.model, req.plan, req.threshold)
    llm_result  = generate_compression(req.text, math_result["compressed_math"], req.model, req.plan)
    return {
        "original":          req.text,
        "compressed":        llm_result["compressed"],
        "original_tokens":   math_result["original_tokens"],
        "compressed_tokens": math_result["compressed_tokens"],
        "tokens_saved":      math_result["tokens_saved"],
        "compression_ratio": math_result["compression_ratio"],
        "model_used":        llm_result["model_used"],
        "ready_to_paste":    True,
        "instruction":       f"Paste this into a new {req.model} conversation. Same meaning, {math_result['tokens_saved']} fewer tokens.",
    }


@app.post("/improve-prompt")
@limiter.limit("10/minute")
def improve_prompt_route(req: PromptRequest, request: Request):
    if not req.prompt or len(req.prompt.strip()) < 10:
        raise HTTPException(400, "Prompt too short")
    if len(req.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt too long — max {MAX_PROMPT_CHARS:,} characters")
    original_tokens  = count_tokens(req.prompt, req.model, req.plan)
    result           = improve_prompt(req.prompt, req.model, req.plan)
    compressed_tokens = count_tokens(result["compressed"], req.model, req.plan)
    return {
        "original":          req.prompt,
        "compressed":        result["compressed"],
        "original_tokens":   original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved":      max(0, original_tokens - compressed_tokens),
        "changes":           result["changes"],
        "ready_to_paste":    True,
    }


@app.post("/parse-file")
@limiter.limit("20/minute")
async def parse_file_route(request: Request, file: UploadFile = File(...), model: str = "claude", plan: str = "sonnet"):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(400, f"File too large — max {MAX_FILE_BYTES // 1_000_000}MB")
    result = parse_file(file_bytes, file.filename, model, plan)
    return {
        "filename":             result.filename,
        "file_type":            result.file_type,
        "token_estimate":       result.token_estimate,
        "text_preview":         result.text[:500] if result.text else "",
        "breakdown":            result.breakdown,
        "images_found":         result.images_found,
        "image_token_estimate": result.image_token_estimate,
        "language":             result.language,
        "pages":                result.pages,
        "slides":               result.slides,
        "warning":              result.warning,
        "limit":                get_limit(model, plan),
        "percentage":           round((result.token_estimate / get_limit(model, plan)) * 100, 1),
    }


@app.post("/explain")
@limiter.limit("20/minute")
def explain(req: ExplainRequest, request: Request):
    if req.level not in {"simple", "technical"}:
        raise HTTPException(400, "Level must be simple or technical")
    if req.concept and len(req.concept) > 500:
        raise HTTPException(400, "Concept too long — max 500 characters")
    explanation = generate_explanation(req.concept, req.level, req.user_data, "claude", "sonnet")
    return {"concept": req.concept, "level": req.level, "explanation": explanation}


# ── LEARN PAGE — PURE PYTHON ANALYSIS ENDPOINTS ───────────────────────
# No LLM calls. Each endpoint runs real math and returns structured data
# the frontend renders directly. This is what keeps the repo Python-heavy.

@app.post("/learn/entropy")
@limiter.limit("30/minute")
def learn_entropy(req: EntropyRequest, request: Request):
    """
    Full entropy analysis for learn/entropy page.
    Returns char/word/bigram entropy, compression bound, character frequency
    distribution, top words, and compressibility rating.
    """
    if not req.text:
        raise HTTPException(400, "Text required")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text too long — max 10,000 characters")
    return analyze_entropy_deep(req.text)


@app.post("/learn/redundancy")
@limiter.limit("20/minute")
def learn_redundancy(req: RedundancyRequest, request: Request):
    """
    Full redundancy analysis for learn/similarity page.
    Returns per-sentence scoring, filler phrase detection, top redundant pairs,
    and token savings estimate.
    """
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(400, "Text too short — need at least 2 sentences")
    if len(req.text) > 20_000:
        raise HTTPException(400, "Text too long — max 20,000 characters")
    return analyze_redundancy_deep(req.text)


@app.post("/learn/attention")
@limiter.limit("30/minute")
def learn_attention(req: AttentionRequest, request: Request):
    """
    O(n²) attention cost analysis for learn/attention page.
    Returns cost curve, fill-level comparison table, pair counts,
    turns remaining before 80% degradation threshold.
    """
    if req.token_count < 0 or req.limit <= 0:
        raise HTTPException(400, "token_count must be >= 0, limit must be > 0")
    return analyze_attention_deep(req.token_count, req.limit, req.avg_tokens_per_turn)


@app.post("/learn/quantization")
@limiter.limit("30/minute")
def learn_quantization(req: QuantizeRequest, request: Request):
    """
    Quantization format comparison for learn/quantization page.
    Returns float32/float16/int8/int4/binary comparison with memory footprints,
    accuracy loss estimates, and a worked example with real numbers.
    """
    if req.embedding_dim < 1 or req.embedding_dim > 4096:
        raise HTTPException(400, "embedding_dim must be 1–4096")
    return analyze_quantization_deep(req.embedding_dim, req.vocab_size)


@app.post("/learn/token-budget")
@limiter.limit("30/minute")
def learn_token_budget(req: TokenBudgetRequest, request: Request):
    """
    Token budget breakdown for learn/tokens page.
    Returns per-speaker token breakdown, fill percentage, turns remaining,
    and per-query cost estimate.
    """
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(400, "Text too short")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text too long — max {MAX_TEXT_CHARS:,} characters")
    return analyze_token_budget(req.text, req.model, req.plan, req.limit)


# ── EXISTING DEMO ENDPOINTS (kept for embeddings visualization) ───────

@app.post("/demo/embeddings")
@limiter.limit("15/minute")
def demo_embeddings(req: EmbedRequest, request: Request):
    """Embedding similarity matrix for learn/embeddings page."""
    if not req.sentences or len(req.sentences) > MAX_SENTENCES:
        raise HTTPException(400, f"Provide 2–{MAX_SENTENCES} sentences")
    sentences = [s.strip() for s in req.sentences if s.strip()]
    return get_embeddings_for_demo(sentences)


@app.post("/demo/entropy")
@limiter.limit("30/minute")
def demo_entropy(req: EntropyRequest, request: Request):
    """
    Lightweight entropy endpoint for the live calculator on learn/entropy.
    Returns H, compression bound, top character frequencies.
    """
    if not req.text:
        raise HTTPException(400, "Text required")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text too long — max 10,000 characters for entropy demo")
    from collections import Counter as C
    H      = shannon_entropy(req.text)
    bound  = theoretical_compression_bound(req.text)
    freq   = C(req.text)
    total  = len(req.text)
    top    = [
        {"char": k if k != "\n" else "\\n", "count": v, "prob": round(v/total, 4)}
        for k, v in sorted(freq.items(), key=lambda x: -x[1])[:10]
    ]
    interp = (
        "Very low — highly repetitive, easily compressible" if H < 2 else
        "Low — redundant content, compression possible"     if H < 3 else
        "Moderate — typical conversational text"            if H < 3.8 else
        "High — dense information, limited compression"     if H < 4.3 else
        "Very high — information-rich or structured content"
    )
    return {
        "entropy": H, "interpretation": interp,
        "compression_bound": bound, "top_chars": top,
        "char_count": total, "unique_chars": len(freq),
    }


@app.post("/demo/tokenize")
@limiter.limit("30/minute")
def demo_tokenize(req: TokenizeRequest, request: Request):
    """Tokenization breakdown for learn/tokens live demo."""
    import re as _re
    if len(req.text) > 2000:
        raise HTTPException(400, "Text too long — max 2,000 characters")
    parts  = _re.findall(r"\w+|[^\w\s]|\s+", req.text) or []
    tokens = []
    for p in parts:
        if not p.strip():
            if tokens: tokens[-1] += p
            else: tokens.append(p)
        elif len(p) > 6:
            mid = len(p)//2
            tokens.extend([p[:mid], p[mid:]])
        else:
            tokens.append(p)
    tokens = [t for t in tokens if t]
    exact  = count_tokens(req.text, req.model, "plus")
    return {
        "text":               req.text,
        "approximate_tokens": tokens,
        "token_count":        exact,
        "char_count":         len(req.text),
        "chars_per_token":    round(len(req.text)/max(exact,1), 2),
        "model":              req.model,
        "note":               "Exact count via tiktoken for ChatGPT. Claude/Gemini accurate to ±5%.",
    }