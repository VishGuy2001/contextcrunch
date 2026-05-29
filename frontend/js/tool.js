// tool.js — ContextCrunch · verified May 2026

const MODELS = {
  claude: {
    label: 'Claude', color: '#047857',
    defaultPlan: 'sonnet',
    behavior: 'Full recall — re-reads the entire conversation history before every response.',
    warning: 'Claude re-reads everything on every message — longer conversations get progressively slower. Opus 4.7 uses a dense tokenizer that produces up to 35% more tokens for the same words.',
    plans: {
      haiku:  { label:'Pro · Haiku 4.5',     sub:'200k window · fast',          limit:200000,  cpt:3.5, input:1.0,  output:5.0,  env: { co2: 0.0001, power: 0.00025, water: 0.00008 } },
      sonnet: { label:'Pro · Sonnet 4.6',    sub:'1M window · balanced',        limit:1000000, cpt:3.5, input:3.0,  output:15.0, recommended:true, env: { co2: 0.0004, power: 0.001, water: 0.0003 } },
      opus:   { label:'Max · Opus 4.7',      sub:'1M window · frontier',    limit:1000000, cpt:2.6, input:5.0,  output:25.0, env: { co2: 0.0015, power: 0.0038, water: 0.0012 } },
    },
    tokenizerNote: {
      haiku:  'Anthropic Custom BPE · ~3.5 chars/token · Haiku 4.5',
      sonnet: 'Anthropic Custom BPE · ~3.5 chars/token · Sonnet 4.6',
      opus:   'Dense BPE tokenizer · ~2.6 chars/token · Opus 4.7 · ~35% more tokens',
    },
  },
  chatgpt: {
    label: 'ChatGPT', color: '#10b981',
    defaultPlan: 'plus',
    behavior: 'Silent truncation — drops oldest messages without warning when context fills.',
    warning: 'ChatGPT silently drops oldest messages when context window fills. You never receive an out-of-context error, but details are lost.',
    plans: {
      free:  { label:'Free · GPT-5.4 Mini',  sub:'32k window · standard',       limit:32000,   cpt:4.0, input:0,    output:0,    env: { co2: 0.0001, power: 0.00025, water: 0.00008 } },
      plus:  { label:'Plus · GPT-5.4',       sub:'272k standard · 1.05M max',   limit:272000,  cpt:4.0, input:2.5,  output:15.0, recommended:true, env: { co2: 0.0004, power: 0.001, water: 0.0003 } },
      pro:   { label:'Pro · GPT-5.5',        sub:'1.05M window · frontier',     limit:1050000, cpt:4.0, input:5.0,  output:30.0, env: { co2: 0.0015, power: 0.0038, water: 0.0012 } },
    },
    tokenizerNote: {
      free:  'OpenAI cl100k BPE · ~4.0 chars/token · GPT-5.4 Mini',
      plus:  'OpenAI cl100k BPE · ~4.0 chars/token · GPT-5.4 standard',
      pro:   'OpenAI cl100k BPE · ~4.0 chars/token · GPT-5.5 frontier',
    },
  },
  gemini: {
    label: 'Gemini', color: '#1d4ed8',
    defaultPlan: 'pro',
    behavior: 'Largest context window. Gemini 3.1 Pro supports a massive 2,000,000 token limit.',
    warning: 'Gemini 3.1 Pro charges double (2x input rate) for prompts exceeding 200,000 tokens — applied retroactively to the entire session.',
    plans: {
      free:  { label:'Free · 3.1 Flash-Lite',sub:'1M window · low rates',       limit:1048576, cpt:4.5, input:0.25, output:1.50, env: { co2: 0.0001, power: 0.00025, water: 0.00008 } },
      pro:   { label:'AI Pro · 3.5 Flash',   sub:'1M window · balanced',        limit:1048576, cpt:4.5, input:0.50, output:3.00, recommended:true, env: { co2: 0.0004, power: 0.001, water: 0.0003 } },
      ultra: { label:'AI Ultra · 3.1 Pro',   sub:'2M window · 2x above 200k',   limit:2097152, cpt:4.5, input:2.00, output:12.00, env: { co2: 0.0015, power: 0.0038, water: 0.0012 } },
    },
    tokenizerNote: {
      free:  'Google SentencePiece unigram · ~4.5 chars/token · 3.1 Flash-Lite',
      pro:   'Google SentencePiece unigram · ~4.5 chars/token · 3.5 Flash balanced',
      ultra: 'Google SentencePiece unigram · ~4.5 chars/token · 2M window · Tiered pricing > 200k',
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
  if(model === 'gemini' && plan === 'ultra') {
    const rate = tokens > 200000 ? 4.00 : 2.00;
    const c = tokens / 1_000_000 * rate;
    return c < 0.0001 ? '< $0.0001' : `$${c.toFixed(4)}`;
  }
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
  document.getElementById('token-live').textContent  = '0 tokens';
  document.getElementById('token-live').className    = 'token-live-span safe';
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
  fz.innerHTML = '<span>Drag and drop a file here, or click to browse</span><span style="font-family:var(--mono); font-size:.68rem; color:var(--muted)">Supports PDF, PPTX, DOCX, and TXT</span>';
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
  if(!el) return;
  el.innerHTML = Object.entries(MODELS[model].plans).map(([k, v]) => `
    <button class="vbtn ${k === plan ? 'active' : ''} ${v.recommended ? 'recommended' : ''}" data-plan="${k}">
      <span style="display:block; font-size:.75rem; font-weight:600;">${v.label}</span>
      <span style="display:block; font-size:.6rem; opacity:.7; margin-top:.1rem">${v.sub}</span>
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
  const el = document.getElementById('tokenizer-note');
  if(el) el.textContent = 'Active Tokenizer: ' + (MODELS[model].tokenizerNote[plan] || '');
}

function updateBehavior() {
  const m = MODELS[model];
  document.getElementById('beh-alert-tag').textContent     = `Context Profile: ${m.label}`;
  document.getElementById('beh-alert-text').textContent    = m.behavior;
  document.getElementById('beh-note').textContent         = m.behavior;
  document.getElementById('beh-note').style.borderLeftColor = m.color;
}

// ── INCREMENTAL GAUGES (HIGH-PERFORMANCE) ─────────────────────────────

// ── INCREMENTAL GAUGES (HIGH-PERFORMANCE) ─────────────────────────────

function showEmptyGauges() {
  const p = P();
  
  // Display welcome card, hide breakdown card
  document.getElementById('welcome-card').style.display = 'block';
  document.getElementById('breakdown-card').style.display = 'none';
  
  document.getElementById('welcome-title').textContent = 'Ready to analyze';
  document.getElementById('welcome-desc').textContent = `Paste a conversation or drag a document on the left. The engine will instantly calculate exact token limits, redundancies, and estimated latency impacts.`;

  // Memory gauges
  document.getElementById('memory-val').textContent = '0%';
  document.getElementById('memory-val').className = 'gauge-val safe';
  const memBar = document.getElementById('memory-bar');
  memBar.style.width = '0%';
  memBar.className = 'gauge-fill safe';
  document.getElementById('memory-sub').textContent = `0 / ${p.limit.toLocaleString()} tokens`;

  // Costs
  document.getElementById('cost-val').textContent = p.input ? `$${p.input.toFixed(2)}` : 'Free';
  document.getElementById('cost-sub').textContent = p.input ? `$${p.input}/M input · $${p.output}/M output` : 'No cost on free tier';

  // Semantic redundancy
  document.getElementById('redundancy-val').textContent = '—';
  document.getElementById('redundancy-val').className = 'gauge-val safe';
  const redBar = document.getElementById('redundancy-bar');
  redBar.style.width = '0%';
  redBar.className = 'gauge-fill safe';
  document.getElementById('redundancy-sub').textContent = 'Same meaning repeated · sentence embeddings';

  // Waste
  document.getElementById('waste-val').textContent = '—';
  document.getElementById('waste-val').className = 'gauge-val safe';
  const wasteBar = document.getElementById('waste-bar');
  wasteBar.style.width = '0%';
  wasteBar.className = 'gauge-fill safe';
  document.getElementById('waste-sub').textContent = 'Filler phrases · repeated tokens';

  // Information density
  document.getElementById('density-val').textContent = '—';
  document.getElementById('density-val').style.color = 'var(--muted)';
  document.getElementById('density-sub').textContent = 'Shannon entropy · bits/char';

  // Environmental footprint
  document.getElementById('env-val').textContent = '0.0g';
  document.getElementById('env-val').className = 'gauge-val safe';
  const envBar = document.getElementById('env-bar');
  envBar.style.width = '0%';
  envBar.className = 'gauge-fill safe';
  document.getElementById('env-co2').textContent = '0.0 g CO₂';
  document.getElementById('env-power').textContent = '0.00 Wh';
  document.getElementById('env-water').textContent = '0.0 ml';
  document.getElementById('env-sub').textContent = 'Based on average data center cooling and carbon indices';

  // Latency Speed
  document.getElementById('speed-val').textContent = 'Fast';
  document.getElementById('speed-val').className = 'gauge-val safe';
  const speedBar = document.getElementById('speed-bar');
  speedBar.style.width = '0%';
  speedBar.className = 'gauge-fill safe';
  document.getElementById('speed-sub').textContent = 'Slows quadratically as memory fills — O(n²)';

  // Clean warnings and callouts
  document.getElementById('warnings-container').innerHTML = '';
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

  // Environmental Footprint
  const envCoeffs = p.env || { co2: 0.0004, power: 0.001, water: 0.0003 };
  const envCo2 = tokens * envCoeffs.co2;
  const envPower = tokens * envCoeffs.power;
  const envWater = tokens * envCoeffs.water;
  const envPct = Math.min(Math.round((tokens / 50000) * 100), 100);
  const envStatus = envPct < 40 ? 'safe' : envPct < 70 ? 'warning' : 'danger';

  // Per-speaker breakdown
  const lines    = text.split('\n');
  const hCh      = lines.filter(l => l.trim().match(/^(Human|User|You|Me):/i)).join('').length;
  const aCh      = lines.filter(l => l.trim().match(/^(Assistant|AI|Claude|ChatGPT|Gemini|Bot):/i)).join('').length;
  const hTok     = Math.ceil(hCh / getCpt());
  const aTok     = Math.ceil(aCh / getCpt());
  const oTok     = Math.max(0, tokens - hTok - aTok);

  // Toggle layout sections
  document.getElementById('welcome-card').style.display = 'none';
  document.getElementById('breakdown-card').style.display = 'block';

  // Update token indicators
  const tokenLive = document.getElementById('token-live');
  tokenLive.innerHTML = `${tokens.toLocaleString()} tokens &middot; ${pct}%`;
  tokenLive.className = `token-live-span ${status}`;

  // Update memory gauges
  document.getElementById('memory-val').textContent = `${pct}%`;
  document.getElementById('memory-val').className = `gauge-val ${status}`;
  const memBar = document.getElementById('memory-bar');
  memBar.style.width = `${pct}%`;
  memBar.className = `gauge-fill ${status}`;
  document.getElementById('memory-sub').textContent = `${tokens.toLocaleString()} / ${getLimit().toLocaleString()} tokens · ${Math.max(0, getLimit() - tokens).toLocaleString()} remaining`;

  // Update breakdown columns
  document.getElementById('breakdown-user').textContent = hTok.toLocaleString();
  document.getElementById('breakdown-ai').textContent = aTok.toLocaleString();
  document.getElementById('breakdown-other').textContent = oTok.toLocaleString();

  // Update cost estimation
  document.getElementById('cost-val').textContent = costStr;
  document.getElementById('cost-sub').textContent = p.input ? `$${p.input}/M input · $${p.output}/M output` : 'No cost on free tier';

  // Update semantic redundancy progress
  const semSt = semRed > 40 ? 'danger' : semRed > 20 ? 'warning' : 'safe';
  document.getElementById('redundancy-val').textContent = `${semRed}%`;
  document.getElementById('redundancy-val').className = `gauge-val ${semSt}`;
  const redBar = document.getElementById('redundancy-bar');
  redBar.style.width = `${semRed}%`;
  redBar.className = `gauge-fill ${semSt}`;
  document.getElementById('redundancy-sub').textContent = semRed > 20 ? 'Highly redundant sentences' : semRed > 5 ? 'Some semantic overlap detected' : 'Unique sentences';

  // Update token waste progress
  const tokSt = tokRed.score > 20 ? 'warning' : 'safe';
  const tokMsg = tokRed.fillers.length ? tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ') : 'No filler phrases detected';
  document.getElementById('waste-val').textContent = `${tokRed.score}%`;
  document.getElementById('waste-val').className = `gauge-val ${tokSt}`;
  const wasteBar = document.getElementById('waste-bar');
  wasteBar.style.width = `${Math.min(tokRed.score * 2.5, 100)}%`;
  wasteBar.className = `gauge-fill ${tokSt}`;
  document.getElementById('waste-sub').textContent = tokMsg;

  // Update Shannon entropy
  const densityVal = document.getElementById('density-val');
  densityVal.textContent = `${H} bits/char`;
  densityVal.style.color = H < 3 ? 'var(--warn)' : H < 4 ? 'var(--black)' : 'var(--accent)';
  document.getElementById('density-sub').textContent = H < 3 ? 'Low density (compresses well)' : H < 4 ? 'Moderate density (conversational)' : 'High density (code/data heavy)';

  // Update Environmental footprint Card
  document.getElementById('env-val').textContent = `${envCo2.toFixed(1)}g`;
  document.getElementById('env-val').className = `gauge-val ${envStatus}`;
  const eBar = document.getElementById('env-bar');
  eBar.style.width = `${envPct}%`;
  eBar.className = `gauge-fill ${envStatus}`;
  document.getElementById('env-co2').textContent = `${envCo2.toFixed(2)} g CO₂`;
  document.getElementById('env-power').textContent = `${envPower.toFixed(2)} Wh`;
  document.getElementById('env-water').textContent = `${envWater.toFixed(1)} ml`;
  document.getElementById('env-sub').textContent = `Equivalent to active cloud compute at data center scale`;

  // Update latency response speed progress
  document.getElementById('speed-val').textContent = status === 'safe' ? 'Fast' : status === 'warning' ? 'Slowing' : 'Slow';
  document.getElementById('speed-val').className = `gauge-val ${status}`;
  const speedBar = document.getElementById('speed-bar');
  speedBar.style.width = `${Math.min(pct * 1.1, 100)}%`;
  speedBar.className = `gauge-fill ${status}`;
  document.getElementById('speed-sub').textContent = `~${mult}x baseline · ${status === 'safe' ? 'Optimal execution' : 'Self-attention slowing down'}`;

  // Check warnings
  let warningsHtml = '';
  if (model === 'gemini' && plan === 'ultra' && (hTok + oTok) > 200000) {
    warningsHtml += `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">Above 200k tokens — Gemini 3.1 Pro is charging 2x input rate applied to this session</p></div>`;
  }
  if (pct > 35 && MODELS[model].warning) {
    warningsHtml += `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">${MODELS[model].warning}</p></div>`;
  }
  document.getElementById('warnings-container').innerHTML = warningsHtml;
}

// ── FILE HANDLING ─────────────────────────────────────────────────────

async function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if(!['pdf','pptx','ppt','docx','doc','txt','md','csv'].includes(ext)) {
    return;
  }
  
  const fz = document.getElementById('file-zone');
  fz.innerHTML = `<span style="font-family:var(--mono);font-size:.75rem;color:var(--muted)">Reading ${file.name}...</span>`;

  try {
    const r = await API.parseFile(file, model, plan);
    fz.classList.add('has-file');
    fz.innerHTML = `
      <span style="color:var(--accent);font-family:var(--header);font-weight:600;font-size:.85rem">✓ File parsed successfully</span>
      <span style="display:block;color:var(--muted);font-size:.65rem;margin-top:.15rem">${r.filename} · ${r.token_estimate.toLocaleString()} tokens · ${r.percentage}% of limit</span>`;
    
    if(r.text_preview) { 
      document.getElementById('main-ta').value = r.text_preview; 
      text = r.text_preview; 
    }
    showFileBreakdown(r);
    toggleBtns();
    updateLiveGauges();
  } catch(e) {
    fz.classList.remove('has-file');
    fz.innerHTML = `<span style="color:var(--danger);font-family:var(--header);font-size:.8rem;font-weight:600">Could not parse file. Try copy-pasting the text.</span>`;
  }
}

function showFileBreakdown(r) {
  const ex = document.getElementById('file-breakdown-card');
  if(ex) ex.remove();
  const bd = r.breakdown || {};
  const p  = P();
  let rows = '';
  if(r.pages)          rows += `<tr><td style="color:var(--muted)">Pages</td><td style="text-align:right">${r.pages}</td></tr>`;
  if(r.slides?.length) rows += `<tr><td style="color:var(--muted)">Slides</td><td style="text-align:right">${r.slides.length}</td></tr>`;
  if(bd.text_tokens)   rows += `<tr><td style="color:var(--muted)">Text tokens</td><td style="text-align:right">${bd.text_tokens.toLocaleString()}</td></tr>`;
  if(bd.image_tokens)  rows += `<tr><td style="color:var(--muted)">Image tokens</td><td style="text-align:right">${bd.image_tokens.toLocaleString()}</td></tr>`;
  if(bd.paragraphs)    rows += `<tr><td style="color:var(--muted)">Paragraphs</td><td style="text-align:right">${bd.paragraphs}</td></tr>`;
  rows += `<tr><td style="color:var(--muted)"><strong>Total</strong></td><td style="text-align:right"><strong>${r.token_estimate.toLocaleString()}</strong></td></tr>`;
  if(p.input) rows += `<tr><td style="color:var(--muted)">Est. cost</td><td style="text-align:right">${fmtCost(r.token_estimate)}</td></tr>`;
  
  const card = document.createElement('div');
  card.id = 'file-breakdown-card';
  card.className = 'gauge-card';
  card.style.cssText = 'border-left:3px solid var(--accent-mid)';
  card.innerHTML = `<div class="gauge-lbl" style="margin-bottom:.35rem">File Breakdown: ${r.filename}</div><table class="bkdn-table">${rows}</table>`;
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
  fz.addEventListener('dragover',  e => { e.preventDefault(); fz.classList.add('dragover'); });
  fz.addEventListener('dragleave', () => fz.classList.remove('dragover'));
  fz.addEventListener('drop', e => {
    e.preventDefault();
    fz.classList.remove('dragover');
    if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  document.getElementById('analyze-btn').addEventListener('click',  runAnalysis);
  document.getElementById('compress-btn').addEventListener('click', runCompress);
  
  document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(compressed);
    document.getElementById('copy-btn').textContent = 'Copied';
    setTimeout(() => document.getElementById('copy-btn').textContent = 'Copy to Clipboard', 2000);
  });
  
  document.getElementById('math-toggle-simple').addEventListener('click', () => {
    level = 'simple';
    document.getElementById('math-toggle-simple').classList.add('active');
    document.getElementById('math-toggle-technical').classList.remove('active');
    if (analysis) renderMath(analysis, level);
  });
  
  document.getElementById('math-toggle-technical').addEventListener('click', () => {
    level = 'technical';
    document.getElementById('math-toggle-technical').classList.add('active');
    document.getElementById('math-toggle-simple').classList.remove('active');
    if (analysis) renderMath(analysis, level);
  });

  document.getElementById('math-minimize-btn').addEventListener('click', () => {
    const wrapper = document.getElementById('math-content-wrapper');
    const btn = document.getElementById('math-minimize-btn');
    if (wrapper.style.display === 'none') {
      wrapper.style.display = 'block';
      btn.textContent = 'Minimize';
    } else {
      wrapper.style.display = 'none';
      btn.textContent = 'Expand';
    }
  });
}

// ── ANALYSIS & COMPRESSION ───────────────────────────────────────────

async function runAnalysis() {
  if(!text.trim()) return;
  showLoading('Analyzing conversation structure');
  try {
    const r    = await API.analyze(text, model, plan);
    analysis   = r;
    const s    = getStatus(r.tokens.percentage);
    const semRed   = r.redundancy.score;
    const semSt    = semRed>40?'danger':semRed>20?'warning':'safe';
    const tokRed   = TC.tokenRedundancy(text);
    const tokSt    = tokRed.score>20?'warning':'safe';
    const p        = P();
    const costStr  = r.tokens.cost_usd > 0 ? `$${r.tokens.cost_usd.toFixed(4)}` : 'Free';

    // Environmental Footprint
    const envCoeffs = p.env || { co2: 0.0004, power: 0.001, water: 0.0003 };
    const envCo2 = r.tokens.total * envCoeffs.co2;
    const envPower = r.tokens.total * envCoeffs.power;
    const envWater = r.tokens.total * envCoeffs.water;
    const envPct = Math.min(Math.round((r.tokens.total / 50000) * 100), 100);
    const envStatus = envPct < 40 ? 'safe' : envPct < 70 ? 'warning' : 'danger';

    // Update Indicators
    const tokenLive = document.getElementById('token-live');
    tokenLive.innerHTML = `${r.tokens.total.toLocaleString()} tokens &middot; ${r.tokens.percentage}%`;
    tokenLive.className = `token-live-span ${s}`;

    document.getElementById('welcome-card').style.display = 'none';
    document.getElementById('breakdown-card').style.display = 'block';

    // Memory card
    document.getElementById('memory-val').textContent = `${r.tokens.percentage}%`;
    document.getElementById('memory-val').className = `gauge-val ${s}`;
    const memBar = document.getElementById('memory-bar');
    memBar.style.width = `${r.tokens.percentage}%`;
    memBar.className = `gauge-fill ${s}`;
    document.getElementById('memory-sub').textContent = `${r.tokens.total.toLocaleString()} / ${r.tokens.limit.toLocaleString()} tokens · ${Math.max(0, r.tokens.limit - r.tokens.total).toLocaleString()} remaining`;

    // Costs
    document.getElementById('cost-val').textContent = costStr;
    document.getElementById('cost-sub').textContent = p.input ? `$${p.input}/M input · $${p.output}/M output` : 'No cost on free tier';

    // Breakdown columns
    document.getElementById('breakdown-user').textContent = r.tokens.user.toLocaleString();
    document.getElementById('breakdown-ai').textContent = r.tokens.ai.toLocaleString();
    document.getElementById('breakdown-other').textContent = r.tokens.system.toLocaleString();

    // Semantic redundancy
    document.getElementById('redundancy-val').textContent = `${semRed}%`;
    document.getElementById('redundancy-val').className = `gauge-val ${semSt}`;
    const redBar = document.getElementById('redundancy-bar');
    redBar.style.width = `${semRed}%`;
    redBar.className = `gauge-fill ${semSt}`;
    document.getElementById('redundancy-sub').textContent = `~${r.redundancy.removable.toLocaleString()} tokens removable · sentence-transformers cosine sim`;

    // Token waste
    document.getElementById('waste-val').textContent = `${tokRed.score}%`;
    document.getElementById('waste-val').className = `gauge-val ${tokSt}`;
    const wasteBar = document.getElementById('waste-bar');
    wasteBar.style.width = `${Math.min(tokRed.score * 2.5, 100)}%`;
    wasteBar.className = `gauge-fill ${tokSt}`;
    document.getElementById('waste-sub').textContent = tokRed.fillers.length ? tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ') : 'No filler phrases detected';

    // Shannon entropy
    const densityVal = document.getElementById('density-val');
    densityVal.textContent = `${r.entropy} bits/char`;
    densityVal.style.color = parseFloat(r.entropy) < 3 ? 'var(--warn)' : parseFloat(r.entropy) < 4 ? 'var(--black)' : 'var(--accent)';
    document.getElementById('density-sub').textContent = parseFloat(r.entropy) < 3 ? 'Low density (compresses well)' : parseFloat(r.entropy) < 4 ? 'Moderate density' : 'High density';

    // Update Environmental footprint Card
    document.getElementById('env-val').textContent = `${envCo2.toFixed(1)}g`;
    document.getElementById('env-val').className = `gauge-val ${envStatus}`;
    const eBar = document.getElementById('env-bar');
    eBar.style.width = `${envPct}%`;
    eBar.className = `gauge-fill ${envStatus}`;
    document.getElementById('env-co2').textContent = `${envCo2.toFixed(2)} g CO₂`;
    document.getElementById('env-power').textContent = `${envPower.toFixed(2)} Wh`;
    document.getElementById('env-water').textContent = `${envWater.toFixed(1)} ml`;

    // Response speed
    document.getElementById('speed-val').textContent = r.attention.zone === 'safe' ? 'Fast' : r.attention.zone === 'warning' ? 'Slowing' : 'Slow';
    document.getElementById('speed-val').className = `gauge-val ${r.attention.zone}`;
    const speedBar = document.getElementById('speed-bar');
    speedBar.style.width = `${Math.min(r.attention.percentage * 1.1, 100)}%`;
    speedBar.className = `gauge-fill ${r.attention.zone}`;
    document.getElementById('speed-sub').textContent = r.attention.message;

    // Warnings
    let warningsHtml = '';
    if (r.warning) {
      warningsHtml += `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.68rem;color:#7a3a10;line-height:1.5">${r.warning}</p></div>`;
    }
    document.getElementById('warnings-container').innerHTML = warningsHtml;

    document.getElementById('math-section').style.display = 'block';
    document.getElementById('math-content-wrapper').style.display = 'block';
    document.getElementById('math-minimize-btn').textContent = 'Minimize';
    renderMath(r, level);
  } catch(e) { alert('Analysis failed: ' + e.message); }
  finally { hideLoading(); }
}

async function runCompress() {
  if(!text.trim()) return;
  showLoading('Compressing conversation using similarity pruning and reasoning rewrites');
  try {
    const r    = await API.compress(text, model, plan);
    compressed = r.compressed;
    
    // Environmental savings calculation
    const envCoeffs = P().env || { co2: 0.0004, power: 0.001, water: 0.0003 };
    const savedCo2 = r.tokens_saved * envCoeffs.co2;
    const savedPower = r.tokens_saved * envCoeffs.power;
    const savedWater = r.tokens_saved * envCoeffs.water;

    document.getElementById('output-section').style.display = 'block';
    document.getElementById('out-text').textContent    = compressed;
    document.getElementById('saved-badge').textContent = `Savings: ${r.tokens_saved.toLocaleString()} tokens · ${r.compression_ratio}% reduction`;
    document.getElementById('out-note').textContent    = `Optimized conversation context is ready to copy. By compressing, you prevent ${savedCo2.toFixed(1)}g CO₂ emissions, save ${savedPower.toFixed(2)} Wh of power, and conserve ${savedWater.toFixed(1)} ml of cooling water on your next chat session.`;
    document.getElementById('output-section').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(e) { alert('Compression failed: ' + e.message); }
  finally { hideLoading(); }
}

// ── MATH PANEL (REFINED WITHOUT EMOJIS, SEPARATED LEVELS) ──────────────

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
  const envCoeffs = p.env || { co2: 0.0004, power: 0.001, water: 0.0003 };
  const curCo2 = r.tokens.total * envCoeffs.co2;
  const curPower = r.tokens.total * envCoeffs.power;
  const curWater = r.tokens.total * envCoeffs.water;
  const savedCo2 = (r.redundancy?.removable ?? 0) * envCoeffs.co2;

  if(lv === 'simple') {
    el.innerHTML = `
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">What the numbers mean</h4>
        <p style="font-size:.9rem; color:#475569; margin-bottom:.75rem">Your conversation spans ${total} tokens, utilizing ${pctD}% of your active context window. 
        Approximately ${semRed}% of the chat represents semantic repetition (different words saying the exact same thing), meaning we can safely extract about ${removable} tokens without changing the information.</p>
        ${tokRed.fillers.length ? `<p style="font-size:.9rem; color:#475569; margin-bottom:.75rem">There are also ${tokRed.fillers.length} filler phrases found in the conversation: ${tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ')}.</p>` : ''}
        <p style="font-size:.9rem; color:#475569;">Information density measures ${entropy} bits/character, indicating ${parseFloat(entropy)<3?'extremely redundant text (highly compressible)':parseFloat(entropy)<4?'standard conversational structures':'highly descriptive, information-dense code or prose'}.</p>
      </div>
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Environmental footprint details</h4>
        <p style="font-size:.9rem; color:#475569; margin-bottom:.75rem">Processing this conversation consumed approximately <strong>${curPower.toFixed(2)} Wh</strong> of electricity, evaporated <strong>${curWater.toFixed(1)} ml</strong> of cooling water, and emitted <strong>${curCo2.toFixed(1)}g</strong> of CO₂ equivalent at data center level.</p>
        <p style="font-size:.9rem; color:#475569;">By pruning the ${semRed}% of redundant text, you can prevent <strong>${savedCo2.toFixed(1)}g</strong> of CO₂ emissions on subsequent turns, directly lowering the compute overhead.</p>
      </div>
      <div class="card">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Why response speed degrades</h4>
        <p style="font-size:.9rem; color:#475569; margin-bottom:.75rem">At ${pct}% context fill, queries compute ~${mult}x slower than at the beginning of the chat session. This occurs because the transformer re-reads every message from the top on every single prompt, meaning calculations scale quadratically rather than linearly.</p>
        <p style="font-size:.9rem; color:#475569;">Reducing tokens by ${semRed}% translates to a ${Math.round((1-(1-semRed/100)**2)*100)}% decrease in execution compute, which makes the AI respond significantly faster.</p>
      </div>`;
  } else {
    el.innerHTML = `
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Tokenizer Model - ${MODELS[model].label} · ${plan}</h4>
        <div class="formula">${MODELS[model].tokenizerNote[plan]}
Estimate: tokens = chars / ${getCpt()}
Total:    ${total} tokens
Cost:     $${p.input.toFixed(2)}/M input tokens · $${p.output.toFixed(2)}/M output tokens</div>
      </div>
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Environmental Footprint Metrics</h4>
        <div class="formula">Power Coeff:  ${(envCoeffs.power * 1000).toFixed(2)} Wh / 1k tokens
Water Coeff:  ${(envCoeffs.water * 1000).toFixed(2)} ml / 1k tokens
CO₂ Coeff:     ${(envCoeffs.co2 * 1000).toFixed(2)} g CO₂ / 1k tokens

Calculations:
  CO₂ footprint = ${r.tokens.total} * ${(envCoeffs.co2).toFixed(6)} = ${curCo2.toFixed(3)} g CO₂
  Power usage   = ${r.tokens.total} * ${(envCoeffs.power).toFixed(6)} = ${curPower.toFixed(3)} Wh
  Water usage   = ${r.tokens.total} * ${(envCoeffs.water).toFixed(6)} = ${curWater.toFixed(2)} ml</div>
      </div>
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Semantic Redundancy Matrix & Cosine Similarity</h4>
        <div class="formula">Method: ${r.redundancy?.method || 'sentence_embeddings_cosine'}
Score:  ${semRed}% of sentences flag as redundant (cosine > 0.88)
Remove: ~${removable} tokens without entropy loss

Dot Product Cosine Similarity:
  cosine(A, B) = (A · B) / (||A|| · ||B||)
  threshold = 0.88 (near paraphrase match)</div>
      </div>
      <div class="card" style="margin-bottom: 1rem;">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Shannon Entropy Formula</h4>
        <div class="formula">Entropy H(X) = -Sum( p(x) * log2 p(x) )
Result:  H = ${entropy} bits/char
English bounds: ~3.5 - 4.0 bits/char
Code bounds:    ~4.0 - 4.5 bits/char

Lossless compression theoretical limit:
  limit = (1 - H / log2|Alphabet|) * 100%
  bound = ${r.compression_bound?.bound ?? '?'}%</div>
      </div>
      <div class="card">
        <h4 style="font-family:var(--header); font-weight:600; font-size:1.1rem; margin-bottom:.5rem;">Self-Attention Complexity O(N²)</h4>
        <div class="formula">Attention(Q,K,V) = softmax( Q * K^T / sqrt(d_k) ) * V
Matrix dimensions: Q, K, V in R^(N x D)
Calculations matrix Q*K^T is N x N (quadratic growth)

Empirical degradation multiplier:
  multiplier = ( N_fill / 50 )^2
  At ${pct}% fill: (${pct}/50)^2 = ${mult}x execution cost
  Speedup under ${semRed}% context pruning:
  speedup = 1 / (1 - ${(semRed/100).toFixed(2)})^2 = ${speedup}x</div>
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