// tool.js — ContextCrunch · verified April 26 2026

const MODELS = {
  claude: {
    label: 'Claude', color: '#c4602a',
    defaultPlan: 'sonnet',
    behavior: 'Full recall — re-reads the entire conversation before every response. You will hit the limit and Claude will stop.',
    warning:  'Claude re-reads everything on every message — longer conversations get progressively slower. Opus 4.7 uses a new tokenizer that can produce up to 35% more tokens for the same text.',
    plans: {
      free:   { label:'Free · Haiku 4.5',    sub:'200k tokens · no API cost',  limit:200000,  cpt:3.5, input:0,    output:0    },
      haiku:  { label:'Pro · Haiku 4.5',     sub:'200k tokens · fast',          limit:200000,  cpt:3.5, input:1.0,  output:5.0  },
      sonnet: { label:'Pro · Sonnet 4.6',    sub:'1M tokens · balanced',        limit:1000000, cpt:3.5, input:3.0,  output:15.0, recommended:true },
      opus:   { label:'Max · Opus 4.7',      sub:'1M tokens · most capable',    limit:1000000, cpt:2.6, input:5.0,  output:25.0 },
    },
    tokenizerNote: {
      free:   'Anthropic Custom BPE · ~3.5 chars/token · Haiku 4.5',
      haiku:  'Anthropic Custom BPE · ~3.5 chars/token · Haiku 4.5',
      sonnet: 'Anthropic Custom BPE · ~3.5 chars/token · Sonnet 4.6',
      opus:   'New tokenizer · ~2.6 chars/token · Opus 4.7 · up to 35% more tokens than prior Claude',
    },
  },
  chatgpt: {
    label: 'ChatGPT', color: '#1a6b4a',
    defaultPlan: 'plus',
    behavior: 'Silent truncation — when the window fills, ChatGPT quietly drops your oldest messages without any warning.',
    warning:  'ChatGPT silently drops earlier messages when the window fills. Context degrades gradually with no indication.',
    plans: {
      free:  { label:'Free · GPT-5.4 Mini',  sub:'32k tokens · no cost',        limit:32000,   cpt:4.0, input:0,    output:0    },
      plus:  { label:'Plus · GPT-5.4',       sub:'272k standard · 1.05M max',   limit:272000,  cpt:4.0, input:2.5,  output:15.0, recommended:true },
      pro:   { label:'Pro · GPT-5.5',        sub:'1.05M tokens · frontier',     limit:1050000, cpt:4.0, input:5.0,  output:30.0 },
    },
    tokenizerNote: {
      free:  'OpenAI cl100k BPE · ~4.0 chars/token · vocab 100,277 · GPT-5.4 Mini',
      plus:  'OpenAI cl100k BPE · ~4.0 chars/token · vocab 100,277 · GPT-5.4 · 2× pricing above 272k',
      pro:   'OpenAI cl100k BPE · ~4.0 chars/token · vocab 100,277 · GPT-5.5',
    },
  },
  gemini: {
    label: 'Gemini', color: '#1e4f8a',
    defaultPlan: 'free',
    behavior: 'Largest context window of the three. Free tier hits rate limits before the token limit. Pro/Ultra charge double above 200k tokens.',
    warning:  'Gemini 2.5 Pro and 3.1 Pro charge 2× input pricing for prompts exceeding 200k tokens — applied to the entire session.',
    plans: {
      free:  { label:'Free · 2.5 Flash',     sub:'1M tokens · rate limited',    limit:1048576, cpt:4.5, input:0,    output:0    },
      pro:   { label:'AI Pro · 2.5 Pro',     sub:'1M tokens · 2× above 200k',   limit:1048576, cpt:4.5, input:1.25, output:10.0, recommended:true },
      ultra: { label:'AI Ultra · 3.1 Pro',   sub:'1M tokens · flagship',        limit:1048576, cpt:4.5, input:2.0,  output:12.0 },
    },
    tokenizerNote: {
      free:  'Google SentencePiece unigram · ~4.5 chars/token · vocab 256,000 · 2.5 Flash',
      pro:   'Google SentencePiece unigram · ~4.5 chars/token · vocab 256,000 · 2.5 Pro · 2× pricing >200k',
      ultra: 'Google SentencePiece unigram · ~4.5 chars/token · vocab 256,000 · 3.1 Pro · 2× pricing >200k',
    },
  },
};

let model = 'claude', plan = 'sonnet', text = '', analysis = null, level = 'simple', compressed = '';

// ── HELPERS ───────────────────────────────────────────────────────────

function P()          { return MODELS[model].plans[plan]; }
function getLimit()   { return P().limit; }
function getCpt()     { return P().cpt; }
function estTokens(t) { return Math.ceil((t||'').length / getCpt()); }
function getPct(tok)  { return Math.min(Math.round(tok / getLimit() * 100), 100); }
function getStatus(p) { return p < 40 ? 'safe' : p < 70 ? 'warning' : 'danger'; }

function fmtCost(tokens) {
  const p = P();
  if(!p.input) return 'Free';
  const c = tokens / 1_000_000 * p.input;
  return c < 0.0001 ? '< $0.0001' : `$${c.toFixed(4)}`;
}

// ── INIT ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  renderModelCards();
  renderPlans();
  updateBehavior();
  setupListeners();
  showEmptyGauges();
  document.getElementById('reset-btn').addEventListener('click', resetTool);
});

// ── RESET ─────────────────────────────────────────────────────────────

function resetTool() {
  text = ''; analysis = null; compressed = '';
  document.getElementById('main-ta').value = '';
  resetFileZone();
  document.getElementById('file-status').textContent  = 'PDF · PPTX · DOCX · plain text';
  document.getElementById('token-live').innerHTML     = '0 tokens';
  document.getElementById('analyze-btn').disabled     = true;
  document.getElementById('compress-btn').disabled    = true;
  const bd = document.getElementById('file-breakdown-card');
  if(bd) bd.remove();
  hideOutput();
  showEmptyGauges();
}

function resetFileZone() {
  const fz = document.getElementById('file-zone');
  fz.classList.remove('has-file');
  fz.innerHTML = 'Drop a file or click &nbsp;·&nbsp; <span style="font-family:var(--mono);font-size:.7rem">PDF &nbsp;·&nbsp; PPTX &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT</span>';
  const fi = document.getElementById('file-input');
  if(fi) fi.value = '';
}

// ── MODEL CARDS ───────────────────────────────────────────────────────

function renderModelCards() {
  document.getElementById('model-cards').querySelectorAll('.model-card').forEach(card => {
    card.addEventListener('click', () => {
      model = card.dataset.model;
      plan  = MODELS[model].defaultPlan;
      document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      renderPlans();
      updateBehavior();
      const bd = document.getElementById('file-breakdown-card');
      if(bd) bd.remove();
      hideOutput();
      text ? updateLiveGauges() : showEmptyGauges();
    });
  });
}

// ── PLAN BUTTONS ──────────────────────────────────────────────────────

function renderPlans() {
  const el = document.getElementById('plan-btns');
  el.innerHTML = Object.entries(MODELS[model].plans).map(([k, v]) => `
    <button class="vbtn ${k === plan ? 'active' : ''} ${v.recommended ? 'recommended' : ''}" data-plan="${k}">
      <span style="display:block;font-size:.72rem">${v.label}</span>
      <span style="display:block;font-size:.6rem;opacity:.65;margin-top:.1rem">${v.sub}</span>
    </button>`).join('');
  el.querySelectorAll('.vbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      plan = btn.dataset.plan;
      el.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateTokenizerNote();
      const bd = document.getElementById('file-breakdown-card');
      if(bd) bd.remove();
      hideOutput();
      text ? updateLiveGauges() : showEmptyGauges();
    });
  });
  updateTokenizerNote();
}

function updateTokenizerNote() {
  document.getElementById('tokenizer-note').textContent = 'ⓘ ' + (MODELS[model].tokenizerNote[plan] || '');
}

function updateBehavior() {
  const m = MODELS[model];
  document.getElementById('beh-tag').textContent           = `How ${m.label} handles context`;
  document.getElementById('beh-text').textContent          = m.behavior;
  document.getElementById('beh-note').style.borderLeftColor = m.color;
}

// ── GAUGES ────────────────────────────────────────────────────────────

function showEmptyGauges() {
  const p = P();
  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card" style="border-left:3px solid var(--accent)">
      <div class="gauge-lbl" style="margin-bottom:.3rem">Ready</div>
      <p style="font-size:.75rem;color:#555;line-height:1.6">Paste a conversation from ${MODELS[model].label}, or upload a PDF, PPTX, DOCX, or text file.</p>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val safe">0%</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">0 / ${p.limit.toLocaleString()} tokens</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.3rem">Cost rate</div>
      <div style="font-family:var(--serif);font-size:1.2rem;color:${p.input ? 'var(--black)' : 'var(--accent)'}">
        ${p.input ? `$${p.input}<span style="font-size:.75rem;font-family:var(--sans);color:var(--muted)">/M tokens</span>` : 'Free'}
      </div>
      <div class="gauge-sub">${p.input ? `$${p.input} input · $${p.output} output per 1M tokens` : 'No cost on free tier'}</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val safe">—</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">Same meaning repeated · sentence embeddings</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val safe">—</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">Filler phrases · repeated tokens</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.2rem">Information density</div>
      <div style="font-family:var(--serif);font-size:1.3rem;color:var(--muted)">—</div>
      <div class="gauge-sub">Shannon entropy · bits/char</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.25rem">Response speed</div>
      <div class="latency-bar-bg"><div class="latency-bar safe" style="width:0%;background:var(--accent)"></div></div>
      <div class="gauge-sub">Slows quadratically as memory fills — O(n²)</div>
    </div>
    <div style="background:var(--accent-light);border:1px solid #a8d4be;border-radius:8px;padding:.8rem;font-size:.75rem;color:#1a4a35;line-height:1.6">
      <strong>${MODELS[model].label}</strong> — ${MODELS[model].behavior}
    </div>`;
}

function updateLiveGauges() {
  const tokens   = estTokens(text);
  const pct      = getPct(tokens);
  const status   = getStatus(pct);
  const H        = TC.entropy(text);
  const semRed   = TC.redundancy(text);
  const tokRed   = TC.tokenRedundancy(text);
  const mult     = TC.attentionMultiplier(tokens, getLimit());
  const p        = P();
  const costStr  = fmtCost(tokens);

  // Per-speaker breakdown
  const lines    = text.split('\n');
  const hCh      = lines.filter(l => l.trim().match(/^(Human|User):/i)).join('').length;
  const aCh      = lines.filter(l => l.trim().match(/^(Assistant|AI):/i)).join('').length;
  const hTok     = Math.ceil(hCh / getCpt());
  const aTok     = Math.ceil(aCh / getCpt());
  const oTok     = Math.max(0, tokens - hTok - aTok);

  const semColor = semRed > 40 ? 'var(--danger)' : semRed > 20 ? 'var(--warn)' : 'var(--accent)';
  const semSt    = semRed > 40 ? 'danger' : semRed > 20 ? 'warning' : 'safe';
  const tokColor = tokRed.score > 20 ? 'var(--warn)' : 'var(--accent)';
  const tokSt    = tokRed.score > 20 ? 'warning' : 'safe';
  const tokMsg   = tokRed.fillers.length
    ? tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ') + (tokRed.fillers.length > 3 ? ` +${tokRed.fillers.length-3}` : '')
    : 'none detected';

  // Special warnings
  const geminiWarn = model === 'gemini' && (plan==='pro'||plan==='ultra') && tokens > 200000
    ? `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">Above 200k tokens — Gemini ${plan==='pro'?'2.5 Pro':'3.1 Pro'} now charging 2× input rate</p></div>` : '';
  const gptWarn = model === 'chatgpt' && plan === 'plus' && tokens > 272000
    ? `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">Above 272k tokens — GPT-5.4 now charging 2× input rate for this session</p></div>` : '';
  const modelWarn = pct > 35 && MODELS[model].warning
    ? `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">${MODELS[model].warning}</p></div>` : '';

  document.getElementById('token-live').innerHTML =
    `<span>${tokens.toLocaleString()} tokens</span>` +
    `<span class="${status}" style="margin-left:5px">· ${pct}%</span>`;

  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${status}">${pct}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${status}" style="width:${pct}%"></div></div>
      <div class="gauge-sub">${tokens.toLocaleString()} / ${getLimit().toLocaleString()} · ${Math.max(0,getLimit()-tokens).toLocaleString()} remaining</div>
      ${geminiWarn}${gptWarn}
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.35rem">Token breakdown</div>
      <table class="bkdn-table">
        <tr><td style="color:var(--muted)">Your messages</td><td>${hTok.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">AI responses</td><td>${aTok.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">Other content</td><td>${oTok.toLocaleString()}</td></tr>
      </table>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Cost estimate</span>
        <span style="font-family:var(--serif);font-size:1.2rem;color:${p.input?'var(--black)':'var(--accent)'}">${costStr}</span>
      </div>
      <div class="gauge-sub">${p.input ? `$${p.input}/M input · $${p.output}/M output` : 'Free tier — no API cost'}</div>
      ${p.input ? `<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-top:.25rem">@ ${tokens.toLocaleString()} tokens input</div>` : ''}
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val ${semSt}" style="color:${semColor}">${semRed}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${semSt}" style="width:${semRed}%;background:${semColor}"></div></div>
      <div class="gauge-sub">${semRed>20?'same meaning repeated — compress recommended':semRed>5?'some overlap detected':'low — unique content'}</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val ${tokSt}" style="color:${tokColor}">${tokRed.score}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${tokSt}" style="width:${Math.min(tokRed.score*2,100)}%;background:${tokColor}"></div></div>
      <div class="gauge-sub">${tokMsg}</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.2rem">Information density</div>
      <div style="font-family:var(--serif);font-size:1.3rem;color:${H<3?'var(--warn)':H<4?'var(--black)':'var(--accent)'}">
        ${H} <span style="font-size:.75rem;font-family:var(--sans);color:var(--muted)">bits/char</span>
      </div>
      <div class="gauge-sub">${H<3?'low — compresses well':H<4?'moderate — typical':'high — information-dense'}</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-lbl" style="margin-bottom:.25rem">Response speed</div>
      <div class="latency-bar-bg"><div class="latency-bar ${status}" style="width:${Math.min(pct*1.1,100)}%;background:${status==='safe'?'var(--accent)':status==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
      <div class="gauge-sub">~${mult}× baseline · ${status==='safe'?'fast':status==='warning'?'slowing':'slow — compress now'}</div>
    </div>
    ${modelWarn}`;
}

// ── FILE HANDLING ─────────────────────────────────────────────────────

async function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if(!['pdf','pptx','ppt','docx','doc','txt','md','csv'].includes(ext)) {
    document.getElementById('file-status').textContent = `Unsupported: ${file.name} — use PDF, PPTX, DOCX, or text`;
    return;
  }
  document.getElementById('file-status').textContent = `${file.name} · reading...`;
  const fz = document.getElementById('file-zone');
  fz.innerHTML = `<span style="font-family:var(--mono);font-size:.7rem;color:var(--muted)">Reading ${file.name}...</span>`;

  try {
    const r = await API.parseFile(file, model, plan);
    const parts = [file.name, `${r.token_estimate.toLocaleString()} tokens`];
    if(r.pages)          parts.push(`${r.pages} pages`);
    if(r.slides?.length) parts.push(`${r.slides.length} slides`);
    if(r.images_found)   parts.push(`${r.images_found} image(s)`);
    document.getElementById('file-status').textContent = parts.join(' · ');
    fz.classList.add('has-file');
    fz.innerHTML = `
      <span style="color:var(--accent);font-family:var(--mono);font-size:.72rem">✓ ${file.name}</span>
      <span style="display:block;color:var(--muted);font-size:.65rem;margin-top:.15rem">${r.token_estimate.toLocaleString()} tokens · ${r.percentage}% of limit · click to change</span>`;
    if(r.text_preview) { document.getElementById('main-ta').value = r.text_preview; text = r.text_preview; }
    showFileBreakdown(r);
    toggleBtns();
    updateLiveGauges();
  } catch(e) {
    fz.classList.remove('has-file');
    fz.innerHTML = `<span style="color:var(--warn);font-family:var(--mono);font-size:.7rem">⚠ Could not parse — try copy-pasting the text</span>`;
    document.getElementById('file-status').textContent = `${file.name} · backend unavailable`;
  }
}

function showFileBreakdown(r) {
  const ex = document.getElementById('file-breakdown-card');
  if(ex) ex.remove();
  const bd = r.breakdown || {};
  const p  = P();
  let rows = '';
  if(r.pages)          rows += `<tr><td style="color:var(--muted)">Pages</td><td>${r.pages}</td></tr>`;
  if(r.slides?.length) rows += `<tr><td style="color:var(--muted)">Slides</td><td>${r.slides.length}</td></tr>`;
  if(bd.text_tokens)   rows += `<tr><td style="color:var(--muted)">Text tokens</td><td>${bd.text_tokens.toLocaleString()}</td></tr>`;
  if(bd.image_tokens)  rows += `<tr><td style="color:var(--muted)">Image tokens</td><td>${bd.image_tokens.toLocaleString()}</td></tr>`;
  if(bd.paragraphs)    rows += `<tr><td style="color:var(--muted)">Paragraphs</td><td>${bd.paragraphs}</td></tr>`;
  rows += `<tr><td style="color:var(--muted)"><strong>Total</strong></td><td><strong>${r.token_estimate.toLocaleString()}</strong></td></tr>`;
  if(p.input) rows += `<tr><td style="color:var(--muted)">Est. cost</td><td>${fmtCost(r.token_estimate)}</td></tr>`;
  rows += `<tr><td style="color:var(--muted)">% of limit</td><td>${r.percentage}%</td></tr>`;
  const card = document.createElement('div');
  card.id = 'file-breakdown-card';
  card.className = 'gauge-card';
  card.style.cssText = 'border-left:3px solid var(--accent)';
  card.innerHTML = `<div class="gauge-lbl" style="margin-bottom:.35rem">📄 ${r.filename}</div><table class="bkdn-table">${rows}</table>${r.warning?`<div style="font-size:.62rem;color:var(--warn);font-family:var(--mono);margin-top:.3rem">⚠ ${r.warning}</div>`:''}`;
  document.getElementById('gauges-pane').insertBefore(card, document.getElementById('gauges-pane').firstChild);
}

// ── LISTENERS ─────────────────────────────────────────────────────────

function setupListeners() {
  const ta = document.getElementById('main-ta');
  ta.addEventListener('input', () => {
    text = ta.value;
    const bd = document.getElementById('file-breakdown-card');
    if(bd) bd.remove();
    toggleBtns();
    text.length > 0 ? updateLiveGauges() : showEmptyGauges();
    hideOutput();
  });

  document.getElementById('file-input').addEventListener('change', e => {
    if(e.target.files[0]) handleFile(e.target.files[0]);
  });
  const fz = document.getElementById('file-zone');
  fz.addEventListener('dragover',  e => { e.preventDefault(); fz.style.borderColor='var(--accent)'; });
  fz.addEventListener('dragleave', () => fz.style.borderColor='var(--border)');
  fz.addEventListener('drop', e => {
    e.preventDefault(); fz.style.borderColor='var(--border)';
    if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  document.getElementById('analyze-btn').addEventListener('click',  runAnalysis);
  document.getElementById('compress-btn').addEventListener('click', runCompress);
  document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(compressed);
    document.getElementById('copy-btn').textContent = '✓ Copied';
    setTimeout(() => document.getElementById('copy-btn').textContent = 'Copy →', 2000);
  });
  document.querySelectorAll('.level-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      level = btn.dataset.level;
      document.querySelectorAll('.level-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if(analysis) renderMath(analysis, level);
    });
  });
}

// ── ANALYSIS & COMPRESSION ───────────────────────────────────────────

async function runAnalysis() {
  if(!text.trim()) return;
  showLoading('Analyzing your conversation');
  try {
    const r    = await API.analyze(text, model, plan);
    analysis   = r;
    const s    = getStatus(r.tokens.percentage);
    const semRed   = r.redundancy.score;
    const semColor = semRed>40?'var(--danger)':semRed>20?'var(--warn)':'var(--accent)';
    const semSt    = semRed>40?'danger':semRed>20?'warning':'safe';
    const tokRed   = TC.tokenRedundancy(text);
    const tokColor = tokRed.score>20?'var(--warn)':'var(--accent)';
    const tokSt    = tokRed.score>20?'warning':'safe';
    const p        = P();
    const costStr  = fmtCost(r.tokens.total);

    document.getElementById('gauges-pane').innerHTML = `
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${s}">${r.tokens.percentage}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${s}" style="width:${r.tokens.percentage}%"></div></div>
        <div class="gauge-sub">${r.tokens.total.toLocaleString()} / ${r.tokens.limit.toLocaleString()} · ${(r.tokens.limit-r.tokens.total).toLocaleString()} remaining</div>
      </div>
      <div class="gauge-card">
        <div class="gauge-lbl" style="margin-bottom:.35rem">Token breakdown</div>
        <table class="bkdn-table">
          <tr><td style="color:var(--muted)">Your messages</td><td>${r.tokens.user.toLocaleString()}</td></tr>
          <tr><td style="color:var(--muted)">AI responses</td><td>${r.tokens.ai.toLocaleString()}</td></tr>
          <tr><td style="color:var(--muted)">System overhead</td><td>${r.tokens.system.toLocaleString()}</td></tr>
        </table>
      </div>
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Cost estimate</span>
          <span style="font-family:var(--serif);font-size:1.2rem;color:${p.input?'var(--black)':'var(--accent)'}">${costStr}</span>
        </div>
        <div class="gauge-sub">${p.input?`$${p.input}/M input · $${p.output}/M output`:'Free tier'}</div>
        ${p.input&&r.tokens.cost_usd>0?`<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-top:.25rem">Exact: $${r.tokens.cost_usd.toFixed(6)}</div>`:''}
      </div>
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val ${semSt}" style="color:${semColor}">${semRed}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${semSt}" style="width:${semRed}%;background:${semColor}"></div></div>
        <div class="gauge-sub">~${r.redundancy.removable.toLocaleString()} tokens · ${r.redundancy.method} detection</div>
      </div>
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val ${tokSt}" style="color:${tokColor}">${tokRed.score}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${tokSt}" style="width:${Math.min(tokRed.score*2,100)}%;background:${tokColor}"></div></div>
        <div class="gauge-sub">${tokRed.fillers.length?tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', '):'none detected'}</div>
      </div>
      <div class="gauge-card">
        <div class="gauge-lbl" style="margin-bottom:.2rem">Information density</div>
        <div style="font-family:var(--serif);font-size:1.3rem;color:${parseFloat(r.entropy)<3?'var(--warn)':parseFloat(r.entropy)<4?'var(--black)':'var(--accent)'}">
          ${r.entropy} <span style="font-size:.75rem;font-family:var(--sans);color:var(--muted)">bits/char</span>
        </div>
        <div class="gauge-sub">${parseFloat(r.entropy)<3?'low — compresses well':parseFloat(r.entropy)<4?'moderate':'high — dense'}</div>
      </div>
      <div class="gauge-card">
        <div class="gauge-lbl" style="margin-bottom:.25rem">Response speed</div>
        <div class="latency-bar-bg"><div class="latency-bar ${r.attention.zone}" style="width:${Math.min(r.attention.percentage*1.1,100)}%;background:${r.attention.zone==='safe'?'var(--accent)':r.attention.zone==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
        <div class="gauge-sub">${r.attention.message}</div>
      </div>
      ${r.warning?`<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">${r.warning}</p></div>`:''}`;
    document.getElementById('math-section').style.display = 'block';
    renderMath(r, level);
  } catch(e) { alert('Analysis failed: ' + e.message); }
  finally { hideLoading(); }
}

async function runCompress() {
  if(!text.trim()) return;
  showLoading('Compressing — removing repeated content and rewriting');
  try {
    const r    = await API.compress(text, model, plan);
    compressed = r.compressed;
    document.getElementById('output-section').style.display = 'block';
    document.getElementById('out-text').textContent    = compressed;
    document.getElementById('saved-badge').textContent = `${r.tokens_saved.toLocaleString()} tokens saved · ${r.compression_ratio}% shorter`;
    document.getElementById('out-note').textContent    = `Paste this into a new ${MODELS[model].label} conversation. Same meaning, ${r.tokens_saved.toLocaleString()} fewer tokens.`;
    document.getElementById('output-section').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(e) { alert('Compression failed: ' + e.message); }
  finally { hideLoading(); }
}

// ── MATH PANEL ────────────────────────────────────────────────────────

function renderMath(r, lv) {
  const el      = document.getElementById('math-area');
  if(!el) return;
  const entropy   = r.entropy ?? '—';
  const semRed    = r.redundancy?.score ?? 0;
  const tokRed    = TC.tokenRedundancy(text);
  const pct       = r.attention?.percentage ?? 0;
  const mult      = r.attention?.multiplier ?? 1;
  const total     = r.tokens?.total?.toLocaleString() ?? '?';
  const removable = r.redundancy?.removable?.toLocaleString() ?? '?';
  const pctD      = r.tokens?.percentage ?? '?';
  const speedup   = semRed > 0 ? Math.round(1/Math.pow(Math.max(1-semRed/100,0.01),2)*10)/10 : 1;
  const p         = P();

  if(lv === 'simple') {
    el.innerHTML = `
      <div class="math-card">
        <h4>What the numbers mean</h4>
        <p>Your conversation is <strong>${total} tokens</strong> — <strong>${pctD}%</strong> of your ${MODELS[model].label} memory limit.
        About <strong>${semRed}%</strong> says things already said earlier in different words — roughly <strong>${removable} tokens</strong> removable without losing meaning.
        ${tokRed.fillers.length ? `There are also <strong>${tokRed.fillers.length} filler phrase(s)</strong> detected: ${tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ')}.` : ''}
        Information density is <strong>${entropy} bits/char</strong> — ${parseFloat(entropy)<3?'low (repetitive, compresses well)':parseFloat(entropy)<4?'moderate (typical conversation)':'high (dense, information-rich)'}.</p>
        ${p.input ? `<p>Estimated input cost at current length: <strong>${fmtCost(r.tokens?.total||0)}</strong> at $${p.input}/M tokens.</p>` : ''}
        <a href="learn/tokens.html" class="learn-lnk">What is a token? →</a>
      </div>
      <div class="math-card">
        <h4>Why responses are getting slower</h4>
        <p>At <strong>${pct}%</strong> memory fill, responses are approximately <strong>${mult}×</strong> slower than the start of this conversation.
        ${MODELS[model].label} re-reads your entire conversation on every single message — and that work grows quadratically, not linearly.
        Compressing now will make it meaningfully faster.</p>
        <a href="learn/attention.html" class="learn-lnk">Why does AI slow down? →</a>
      </div>`;
  } else {
    el.innerHTML = `
      <div class="math-card">
        <h4>Tokenizer — ${MODELS[model].label} · ${plan}</h4>
        <div class="formula">${MODELS[model].tokenizerNote[plan]}
Estimate: tokens ≈ chars / ${getCpt()} → ${total} tokens
${p.input ? `Cost:     $${p.input}/M input · $${p.output}/M output\n          ${fmtCost(r.tokens?.total||0)} for this session` : 'Cost:     free tier'}</div>
        <a href="learn/tokens.html" class="learn-lnk">Tokenization explained →</a>
      </div>
      <div class="math-card">
        <h4>Semantic redundancy · sentence embeddings</h4>
        <div class="formula">Method: ${r.redundancy?.method || 'word_overlap'}
Score:  ${semRed}% of sentences are semantically redundant
Remove: ~${removable} tokens without meaning loss

Pass 1 — Jaccard(A,B) = |A∩B| / |A∪B|        (word overlap)
Pass 2 — TF cosine sim(A,B) = (A·B)/(‖A‖·‖B‖) (semantic)
Stage 1 — sentence-transformers 384-dim cosine > 0.88

Token waste: ${tokRed.score}% filler ratio
Fillers:     ${tokRed.fillers.length ? tokRed.fillers.slice(0,5).join(', ') : 'none'}</div>
        <a href="learn/embeddings.html" class="learn-lnk">How embeddings work →</a>
      </div>
      <div class="math-card">
        <h4>Shannon entropy · H(X) = −Σ p(x) log₂ p(x)</h4>
        <div class="formula">Result:  H = ${entropy} bits/char
Range:   0 (all same char) → 4.7 (uniform ASCII)
English: ~3.5–4.0 · Code: ~4.0–4.5

Lossless compression bound:
  bound = (1 − H / log₂|A|) × 100%
  = ${r.compression_bound?.bound ?? '?'}%</div>
        <a href="learn/entropy.html" class="learn-lnk">Entropy explained →</a>
      </div>
      <div class="math-card">
        <h4>Self-attention O(n²) · latency model</h4>
        <div class="formula">Attention(Q,K,V) = softmax(QKᵀ/√d) · V
Complexity: O(n²·d) per layer

Multiplier = (fill% / 50)²
At ${pct}%:  (${pct}/50)² = ${mult}×

Compression speedup at ${semRed}% reduction:
  1 / (1 − ${(semRed/100).toFixed(2)})² = ${speedup}×</div>
        <a href="learn/attention.html" class="learn-lnk">Attention explained →</a>
      </div>`;
  }
}

// ── UTILITIES ─────────────────────────────────────────────────────────

function toggleBtns() {
  const ok = text.trim().length > 0;
  document.getElementById('analyze-btn').disabled  = !ok;
  document.getElementById('compress-btn').disabled = !ok;
}
function showLoading(msg) {
  document.getElementById('loading-msg').textContent = msg;
  document.getElementById('loading').classList.add('visible');
  document.getElementById('analyze-btn').disabled  = true;
  document.getElementById('compress-btn').disabled = true;
}
function hideLoading() { document.getElementById('loading').classList.remove('visible'); toggleBtns(); }
function hideOutput()  { document.getElementById('output-section').style.display='none'; document.getElementById('math-section').style.display='none'; }