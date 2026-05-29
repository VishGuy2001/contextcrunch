# ContextCrunch

**Crunch your AI context. Stop losing work. Stop wasting tokens.**

Live at **[contextcrunch.io](https://contextcrunch.io)** · API at **[api.contextcrunch.io](https://api.contextcrunch.io)**

---

## What it does

**Instant in your browser (free, no server):**
- Live token count as you type — broken down by your messages vs AI responses
- Context window fill % for Claude / ChatGPT / Gemini per plan
- Shannon entropy score — how information-dense is your text?
- Redundancy detection — what percentage says the same thing twice?
- O(n²) latency impact — how much slower are responses getting?

**On compression request (Python backend on Google Cloud Run):**
- Sentence embeddings via `all-MiniLM-L6-v2`
- TurboQuant (Google ICLR 2026) on embeddings
- Cosine similarity to find redundant chunks
- Groq Llama 3.3 70B rewrites for natural flow
- Returns compressed conversation ready to paste back

**Six interactive learn pages** — each with live backend demos, three math levels (simple/technical/academic), and Python code.

---

## Quick start

```bash
git clone https://github.com/VishGuy2001/contextcrunch
cd contextcrunch

# Backend
pip install -r backend/requirements.txt
cp .env.example .env
# Add GROQ_API_KEY and GEMINI_API_KEY to .env

cd backend
uvicorn main:app --reload --port 8000

# Frontend — open frontend/index.html with VS Code Live Server
```

---

## API Keys (both free)

| Key | Get it | Free tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | 6000 req/day |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1500 req/day |

---

## Deploy

**Frontend** → Vercel (root: `frontend/`) → contextcrunch.io
**Backend** → Google Cloud Run → api.contextcrunch.io

```bash
# Deploy backend to Cloud Run
cd backend
gcloud run deploy contextcrunch-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=xxx,GEMINI_API_KEY=xxx \
  --memory 1Gi
```

---

## pip install contextcrunch

```bash
pip install contextcrunch
```

```python
from contextcrunch import count_tokens, compress

tokens = count_tokens("Hello world", model="claude", plan="plus")
result = compress("your long conversation...", model="claude")
print(result["compressed"])
print(result["tokens_saved"])
```

---

## Architecture

```
contextcrunch/
├── contextcrunch/      Python library
│   ├── tokenizer.py    tiktoken — exact per model/plan
│   ├── math_engine.py  Shannon entropy, cosine sim, O(n²)
│   ├── compressor.py   TurboQuant + sentence-transformers
│   ├── llm_engine.py   Groq primary + Gemini fallback
│   └── file_parser.py  PDF, PPTX, DOCX, XLSX, images, code
├── backend/
│   └── main.py         FastAPI — all routes
├── frontend/
│   ├── index.html      Home
│   ├── tool.html       Main analyzer
│   ├── models.html     Model comparison
│   ├── about.html      Story
│   └── learn/          6 interactive learn pages
└── math/               Academic papers
    ├── entropy.md
    ├── vector_quant.md
    ├── turboquant.md
    └── attention.md
```

---

## References

- Zandieh et al. (2025). *TurboQuant*. arXiv:2504.19874. ICLR 2026.
- Cover & Thomas (2006). *Elements of Information Theory*.
- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP.
- Jégou et al. (2011). *Product Quantization*. IEEE TPAMI.

---

Built by [Vishnu Sekar](https://linkedin.com/in/vishnusekar/) · MIT License
