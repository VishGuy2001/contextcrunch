// tool.js — main tool page logic

const MODELS = {
  claude: {
    label: 'Claude',
    color: '#c4602a',
    plans: {
      free:   'Free  ·  Haiku 4.5  ·  40k tokens',
      haiku:  'Pro  ·  Haiku 4.5  ·  200k tokens',
      sonnet: 'Pro  ·  Sonnet 4.6  ·  1M tokens',
      opus:   'Max  ·  Opus 4.7  ·  1M tokens',
    },
    defaultPlan: 'sonnet',
    tokenizerNote: {
      free:   'Custom BPE — ~3.5 chars/token',
      haiku:  'Custom BPE — ~3.5 chars/token',
      sonnet: 'Custom BPE — ~3.5 chars/token',
      opus:   'New tokenizer — ~2.6 chars/token (up to 35% more tokens than older Claude)',
    },
    behavior: 'Full recall — keeps your entire conversation and re-reads it on every message. You will hit the limit eventually and Claude will stop.',
    warning: 'Claude re-reads your full conversation on every message. The more you write, the slower and more expensive it gets. Opus 4.7 uses a new tokenizer — up to 35% more tokens for the same text.',
  },
  chatgpt: {
    label: 'ChatGPT',
    color: '#1a6b4a',
    plans: {
      free:  'Free  ·  GPT-4o mini  ·  32k tokens',
      plus:  'Plus  ·  GPT-4o  ·  272k tokens',
      pro:   'Pro  ·  GPT-4o  ·  1.05M tokens',
    },
    defaultPlan: 'plus',
    tokenizerNote: {
      free:  'cl100k BPE — ~4.0 chars/token — vocab 100,277',
      plus:  'cl100k BPE — ~4.0 chars/token — vocab 100,277',
      pro:   'cl100k BPE — ~4.0 chars/token — vocab 100,277',
    },
    behavior: 'Silent forgetting — when your conversation gets too long, ChatGPT quietly drops your oldest messages without telling you.',
    warning: 'ChatGPT is silently dropping your earlier messages when the window fills. Responses gradually lose context without any warning.',
  },
  gemini: {
    label: 'Gemini',
    color: '#1e4f8a',
    plans: {
      free:  'Free  ·  2.5 Flash  ·  1,048,576 tokens',
      pro:   'AI Pro  ·  2.5 Pro  ·  1,048,576 tokens',
      ultra: 'AI Ultra  ·  3.1 Pro  ·  1,048,576 tokens',
    },
    defaultPlan: 'free',
    tokenizerNote: {
      free:  'SentencePiece — ~4.5 chars/token — vocab 256,000',
      pro:   'SentencePiece — ~4.5 chars/token — vocab 256,000',
      ultra: 'SentencePiece — ~4.5 chars/token — vocab 256,000',
    },
    behavior: 'Largest context window of the three. On the free tier, rate limits kick in long before you hit the token limit.',
    warning: 'Gemini Free hits rate limits before the 1M token window. On paid tiers, response quality can drop on very long conversations even within the limit.',
  },
};

// Supported file types — PDF, PPTX, DOCX, plain text
const ACCEPTED_FILES = '.txt,.md,.csv,.pdf,.docx,.doc,.pptx,.ppt';

let model = 'claude', plan = 'sonnet', text = '', analysis = null, level = 'simple', compressed = '';

document.addEventListener('DOMContentLoaded', () => {
  renderPlans();
  updateBehavior();
  setupListeners();
  showEmptyGauges();

  // Reset button
  const rb = document.getElementById('reset-btn');
  if(rb) rb.addEventListener('click', resetTool);
});

function resetTool(){
  text       = '';
  analysis   = null;
  compressed = '';

  const ta = document.getElementById('main-ta');
  if(ta) ta.value = '';

  // Reset file zone to original state
  const fz = document.getElementById('file-zone');
  if(fz){
    fz.classList.remove('has-file');
    fz.innerHTML = 'Drop a file or click &nbsp;·&nbsp; <span style="font-family:var(--mono);font-size:.72rem">PDF &nbsp;·&nbsp; PPTX &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT</span>';
  }

  // Reset file input so same file can be re-uploaded
  const fi = document.getElementById('file-input');
  if(fi) fi.value = '';

  document.getElementById('file-status').textContent     = 'PDF · PPTX · DOCX · plain text';
  document.getElementById('token-live').innerHTML        = '0 tokens';
  document.getElementById('analyze-btn').disabled        = true;
  document.getElementById('compress-btn').disabled       = true;

  // Remove file breakdown card if present
  const bd = document.getElementById('file-breakdown-card');
  if(bd) bd.remove();

  hideOutput();
  showEmptyGauges();
}

// ── GAUGES ────────────────────────────────────────────────────────────

function showEmptyGauges(){
  const limit = TC.getLimit(model, plan);
  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card" style="border-left:3px solid var(--accent)">
      <div style="font-family:var(--mono);font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Ready</div>
      <p style="font-size:.82rem;color:#555;line-height:1.65">
        Paste a conversation from ${MODELS[model].label}, or upload a PDF, PPTX, DOCX, or text file.
        See how full the memory is, what's repeated, and how it's affecting response speed.
      </p>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val safe">0%</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">0 / ${limit.toLocaleString()} tokens · ${TC.formatLimit(limit)} available</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl" style="display:block;margin-bottom:.75rem">Tokenizer</span>
      <div style="font-size:.82rem;color:#555;line-height:1.7">
        <strong>${MODELS[model].label} · ${Object.keys(MODELS[model].plans).find(k=>k===plan)||plan}</strong><br>
        <span style="font-family:var(--mono);font-size:.72rem;color:var(--muted)">${MODELS[model].tokenizerNote[plan]}</span>
      </div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val safe">—</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">Same meaning said twice — detected via sentence embeddings</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val safe">—</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">Filler phrases and repeated tokens</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Information density</span>
      <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:var(--muted)">—</div>
      <div class="gauge-sub">Shannon entropy — bits of unique information per character</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Response speed</span>
      <div class="latency-bar-bg"><div class="latency-bar safe" style="width:0%;background:var(--accent)"></div></div>
      <div class="gauge-sub">Slows down quadratically as memory fills — O(n²)</div>
    </div>
    <div style="background:var(--accent-light);border:1px solid #a8d4be;border-radius:10px;padding:1rem;font-size:.82rem;color:#1a4a35;line-height:1.65">
      <strong>${MODELS[model].label}</strong> — ${MODELS[model].behavior}
    </div>`;
}

function updateLiveGauges(){
  const tokens  = TC.estimate(text, model, plan);
  const pct     = TC.getPct(tokens, model, plan);
  const status  = TC.getStatus(pct);
  const H       = TC.entropy(text);
  const semRed  = TC.redundancy(text);                    // semantic: meaning-level
  const tokRed  = TC.tokenRedundancy(text);               // token: filler phrases
  const limit   = TC.getLimit(model, plan);
  const mult    = TC.attentionMultiplier(tokens, limit);
  const cost    = TC.getCost(tokens, model, plan);

  document.getElementById('token-live').innerHTML =
    `<span>${tokens.toLocaleString()} tokens</span>` +
    `<span class="${status}" style="margin-left:6px">· ${pct}% of ${TC.formatLimit(limit)}</span>`;

  const warn = pct > 30 && MODELS[model].warning
    ? `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.78rem;color:#7a3a10;line-height:1.55">${MODELS[model].warning}</p></div>`
    : '';

  // Semantic redundancy styling
  const semColor  = semRed > 40 ? 'var(--danger)' : semRed > 20 ? 'var(--warn)' : 'var(--accent)';
  const semStatus = semRed > 40 ? 'danger' : semRed > 20 ? 'warning' : 'safe';
  const semMsg    = semRed > 30 ? 'repeated meaning — same idea said multiple times'
                  : semRed > 10 ? 'some semantic overlap detected'
                  : 'low — each sentence carries unique meaning';

  // Token waste styling
  const tokColor  = tokRed.score > 20 ? 'var(--warn)' : 'var(--accent)';
  const tokStatus = tokRed.score > 20 ? 'warning' : 'safe';
  const tokMsg    = tokRed.fillers.length > 0
    ? `filler phrases: ${tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ')}${tokRed.fillers.length > 3 ? ` +${tokRed.fillers.length-3} more` : ''}`
    : 'no significant filler detected';

  // Speaker breakdown
  const lines      = text.split('\n');
  const humanChars = lines.filter(l=>l.trim().startsWith('Human:')||l.trim().startsWith('User:')).join('').length;
  const aiChars    = lines.filter(l=>l.trim().startsWith('Assistant:')||l.trim().startsWith('AI:')).join('').length;
  const cpt        = TC.charsPerToken[model]?.[plan] || 3.8;
  const humanTok   = Math.ceil(humanChars / cpt);
  const aiTok      = Math.ceil(aiChars / cpt);
  const otherTok   = Math.max(0, tokens - humanTok - aiTok);

  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${status}">${pct}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${status}" style="width:${pct}%"></div></div>
      <div class="gauge-sub">${tokens.toLocaleString()} / ${limit.toLocaleString()} tokens · ${TC.formatLimit(limit - tokens)} remaining${parseFloat(cost)>0?' · est. $'+cost:''}</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl" style="display:block;margin-bottom:.5rem">Token breakdown</span>
      <table class="bkdn-table">
        <tr><td style="color:var(--muted)">Your messages</td><td>${humanTok.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">AI responses</td><td>${aiTok.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">Other content</td><td>${otherTok.toLocaleString()}</td></tr>
      </table>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val ${semStatus}" style="color:${semColor}">${semRed}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${semStatus}" style="width:${semRed}%;background:${semColor}"></div></div>
      <div class="gauge-sub">${semMsg}</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val ${tokStatus}" style="color:${tokColor}">${tokRed.score}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${tokStatus}" style="width:${Math.min(tokRed.score*2,100)}%;background:${tokColor}"></div></div>
      <div class="gauge-sub">${tokMsg}</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Information density</span>
      <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:${H<3?'var(--warn)':H<4?'var(--black)':'var(--accent)'}">${H} <span style="font-size:.9rem;font-family:var(--sans);color:var(--muted)">bits/char</span></div>
      <div class="gauge-sub">${H<3?'low — repetitive, compresses well':H<4?'moderate — typical conversation':'high — dense, information-rich'}</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Response speed</span>
      <div class="latency-bar-bg"><div class="latency-bar ${status}" style="width:${Math.min(pct*1.1,100)}%;background:${status==='safe'?'var(--accent)':status==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
      <div class="gauge-sub">~${mult}× baseline · ${status==='safe'?'fast':'status'==='warning'?'slowing — compress soon':'slow — compress now'}</div>
    </div>
    ${warn}`;
}

// ── FILE HANDLING ─────────────────────────────────────────────────────

async function handleFile(file){
  const ext  = file.name.split('.').pop().toLowerCase();
  const name = file.name;

  // Accepted: pdf, pptx, ppt, docx, doc, txt, md, csv
  const supported = ['pdf','pptx','ppt','docx','doc','txt','md','csv'];
  if(!supported.includes(ext)){
    document.getElementById('file-status').textContent = `${name} — unsupported file type. Use PDF, PPTX, DOCX, or text files.`;
    return;
  }

  // Show loading state immediately
  document.getElementById('file-status').textContent = `${name} · reading file...`;
  document.getElementById('file-zone').textContent   = `${name} · processing...`;

  try {
    const r = await API.parseFile(file, model, plan);

    // Build a clean status line
    let statusParts = [name];
    if(r.token_estimate) statusParts.push(`${r.token_estimate.toLocaleString()} tokens`);
    if(r.pages)          statusParts.push(`${r.pages} pages`);
    if(r.slides?.length) statusParts.push(`${r.slides.length} slides`);
    if(r.images_found)   statusParts.push(`${r.images_found} image(s)`);
    if(r.percentage)     statusParts.push(`${r.percentage}% of ${TC.formatLimit(TC.getLimit(model,plan))} limit`);
    if(r.warning)        statusParts.push(`⚠ ${r.warning}`);

    document.getElementById('file-status').textContent = statusParts.join(' · ');
    const fzEl = document.getElementById('file-zone');
    fzEl.classList.add('has-file');
    fzEl.innerHTML = `
      <span style="color:var(--accent);font-family:var(--mono);font-size:.78rem">✓ ${name}</span>
      <span style="color:var(--muted);font-size:.72rem;display:block;margin-top:.2rem">${r.token_estimate.toLocaleString()} tokens · click to upload a different file</span>`;

    // Put extracted text into the textarea so Analyze/Compress work
    if(r.text_preview && r.text_preview.length > 0){
      document.getElementById('main-ta').value = r.text_preview;
      text = r.text_preview;
    } else if(r.token_estimate > 0){
      // File parsed but text not extracted (e.g. image-heavy PDF)
      // Use a placeholder so the gauges show the token count
      text = `[File: ${name}]\nToken estimate: ${r.token_estimate}`;
      document.getElementById('main-ta').value = text;
    }

    toggleBtns();
    updateLiveGauges();

    // Show file-specific breakdown if available
    if(r.breakdown && Object.keys(r.breakdown).length > 0){
      showFileBreakdown(r);
    }

  } catch(e) {
    // Fallback to client-side estimate
    const cpt = TC.charsPerToken[model]?.[plan] || 3.8;
    const est = Math.ceil(file.size / cpt);
    document.getElementById('file-status').textContent = `${name} · ~${est.toLocaleString()} tokens (client estimate — backend unavailable)`;
    document.getElementById('file-zone').innerHTML = `
      <span style="color:var(--warn);font-family:var(--mono);font-size:.78rem">⚠ ${name}</span>
      <span style="color:var(--muted);font-size:.75rem;display:block;margin-top:.25rem">Could not parse — try copy-pasting the text instead</span>`;
    console.error('File parse error:', e);
  }
}

function showFileBreakdown(r){
  // Show a small breakdown card above the gauges for file-specific info
  const existing = document.getElementById('file-breakdown-card');
  if(existing) existing.remove();

  const bd = r.breakdown;
  let rows = '';
  if(r.pages)          rows += `<tr><td style="color:var(--muted)">Pages</td><td>${r.pages}</td></tr>`;
  if(r.slides?.length) rows += `<tr><td style="color:var(--muted)">Slides</td><td>${r.slides.length}</td></tr>`;
  if(bd.text_tokens)   rows += `<tr><td style="color:var(--muted)">Text tokens</td><td>${bd.text_tokens.toLocaleString()}</td></tr>`;
  if(bd.image_tokens)  rows += `<tr><td style="color:var(--muted)">Image tokens</td><td>${bd.image_tokens.toLocaleString()}</td></tr>`;
  if(bd.lines)         rows += `<tr><td style="color:var(--muted)">Lines</td><td>${bd.lines.toLocaleString()}</td></tr>`;
  if(bd.paragraphs)    rows += `<tr><td style="color:var(--muted)">Paragraphs</td><td>${bd.paragraphs.toLocaleString()}</td></tr>`;
  rows += `<tr><td style="color:var(--muted)">Total tokens</td><td><strong>${r.token_estimate.toLocaleString()}</strong></td></tr>`;
  rows += `<tr><td style="color:var(--muted)">% of ${MODELS[model].label} limit</td><td>${r.percentage}%</td></tr>`;

  const card = document.createElement('div');
  card.id = 'file-breakdown-card';
  card.className = 'gauge-card';
  card.style.cssText = 'border-left:3px solid var(--accent);margin-bottom:.75rem';
  card.innerHTML = `
    <span class="gauge-lbl" style="display:block;margin-bottom:.5rem">File: ${r.filename}</span>
    <table class="bkdn-table">${rows}</table>
    ${r.warning?`<div style="font-size:.72rem;color:var(--warn);font-family:var(--mono);margin-top:.5rem">⚠ ${r.warning}</div>`:''}`;

  const pane = document.getElementById('gauges-pane');
  pane.insertBefore(card, pane.firstChild);
}

// ── LISTENERS ─────────────────────────────────────────────────────────

function setupListeners(){
  // Model buttons
  document.querySelectorAll('.mbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      model = btn.dataset.model;
      plan  = MODELS[model].defaultPlan;
      document.querySelectorAll('.mbtn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderPlans();
      updateBehavior();
      // Clear file breakdown and output — token counts change per model
      const bd = document.getElementById('file-breakdown-card');
      if(bd) bd.remove();
      hideOutput();
      text ? updateLiveGauges() : showEmptyGauges();
    });
  });

  // Text input
  const ta = document.getElementById('main-ta');
  ta.addEventListener('input', () => {
    text = ta.value;
    // Clear any file breakdown when user types
    const bd = document.getElementById('file-breakdown-card');
    if(bd) bd.remove();
    toggleBtns();
    text.length > 0 ? updateLiveGauges() : showEmptyGauges();
    hideOutput();
  });

  // File input via click
  document.getElementById('file-input').addEventListener('change', e => {
    if(e.target.files[0]) handleFile(e.target.files[0]);
  });

  // Drag and drop
  const fz = document.getElementById('file-zone');
  fz.addEventListener('dragover', e => {
    e.preventDefault();
    fz.style.borderColor = 'var(--accent)';
    fz.style.color = 'var(--accent)';
  });
  fz.addEventListener('dragleave', () => {
    fz.style.borderColor = 'var(--border)';
    fz.style.color = 'var(--muted)';
  });
  fz.addEventListener('drop', e => {
    e.preventDefault();
    fz.style.borderColor = 'var(--border)';
    fz.style.color = 'var(--muted)';
    if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  // Action buttons
  document.getElementById('analyze-btn').addEventListener('click',  runAnalysis);
  document.getElementById('compress-btn').addEventListener('click', runCompress);
  document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(compressed);
    document.getElementById('copy-btn').textContent = '✓ Copied';
    setTimeout(() => document.getElementById('copy-btn').textContent = 'Copy →', 2000);
  });

  // Level toggle in math panel
  document.querySelectorAll('.level-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      level = btn.dataset.level;
      document.querySelectorAll('.level-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if(analysis) renderMath(analysis, level);
    });
  });
}

// ── PLANS & BEHAVIOR ─────────────────────────────────────────────────

function renderPlans(){
  const el = document.getElementById('plan-btns');
  el.innerHTML = Object.entries(MODELS[model].plans)
    .map(([k,v]) => `<button class="pbtn ${k===plan?'active':''}" data-plan="${k}">${v}</button>`)
    .join('');
  el.querySelectorAll('.pbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      plan = btn.dataset.plan;
      el.querySelectorAll('.pbtn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateTokenizerNote();
      text ? updateLiveGauges() : showEmptyGauges();
    });
  });
  updateTokenizerNote();
}

function updateTokenizerNote(){
  const note = MODELS[model].tokenizerNote?.[plan] || '';
  let el = document.getElementById('tokenizer-note');
  if(!el){
    el = document.createElement('div');
    el.id = 'tokenizer-note';
    el.style.cssText = 'font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-top:.4rem;padding:.3rem .7rem;background:var(--cream);border-radius:5px;display:inline-block';
    const pb = document.getElementById('plan-btns');
    if(pb) pb.insertAdjacentElement('afterend', el);
  }
  el.textContent = 'ⓘ  Tokenizer: ' + note;
}

function updateBehavior(){
  const m = MODELS[model];
  document.getElementById('beh-tag').textContent  = `How ${m.label} handles context`;
  document.getElementById('beh-text').textContent = m.behavior;
  document.getElementById('beh-note').style.borderLeftColor = m.color;
}

// ── ANALYSIS & COMPRESSION ───────────────────────────────────────────

async function runAnalysis(){
  if(!text.trim()) return;
  showLoading('Analyzing your conversation');
  try{
    const r = await API.analyze(text, model, plan);
    analysis = r;

    const s         = TC.getStatus(r.tokens.percentage);
    const semRed    = r.redundancy.score;
    const semColor  = semRed > 40 ? 'var(--danger)' : semRed > 20 ? 'var(--warn)' : 'var(--accent)';
    const semStatus = semRed > 40 ? 'danger' : semRed > 20 ? 'warning' : 'safe';

    // Client-side token waste (filler detection)
    const tokRed    = TC.tokenRedundancy(text);
    const tokColor  = tokRed.score > 20 ? 'var(--warn)' : 'var(--accent)';
    const tokStatus = tokRed.score > 20 ? 'warning' : 'safe';
    const tokMsg    = tokRed.fillers.length > 0
      ? `filler: ${tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ')}${tokRed.fillers.length>3?` +${tokRed.fillers.length-3} more`:''}`
      : 'no significant filler detected';

    document.getElementById('gauges-pane').innerHTML = `
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${s}">${r.tokens.percentage}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${s}" style="width:${r.tokens.percentage}%"></div></div>
        <div class="gauge-sub">${r.tokens.total.toLocaleString()} / ${r.tokens.limit.toLocaleString()} tokens · ${(r.tokens.limit-r.tokens.total).toLocaleString()} remaining${r.tokens.cost_usd>0?' · est. $'+r.tokens.cost_usd.toFixed(4):''}</div>
      </div>
      <div class="gauge-card">
        <span class="gauge-lbl" style="display:block;margin-bottom:.5rem">Token breakdown</span>
        <table class="bkdn-table">
          <tr><td style="color:var(--muted)">Your messages</td><td>${r.tokens.user.toLocaleString()}</td></tr>
          <tr><td style="color:var(--muted)">AI responses</td><td>${r.tokens.ai.toLocaleString()}</td></tr>
          <tr><td style="color:var(--muted)">System overhead</td><td>${r.tokens.system.toLocaleString()}</td></tr>
          ${r.tokens.cost_usd>0?`<tr><td style="color:var(--muted)">Est. cost</td><td>$${r.tokens.cost_usd.toFixed(4)}</td></tr>`:''}
        </table>
      </div>
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Semantic redundancy</span><span class="gauge-val ${semStatus}" style="color:${semColor}">${semRed}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${semStatus}" style="width:${semRed}%;background:${semColor}"></div></div>
        <div class="gauge-sub">~${r.redundancy.removable.toLocaleString()} tokens · same meaning repeated · ${r.redundancy.method} detection</div>
      </div>
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Token waste</span><span class="gauge-val ${tokStatus}" style="color:${tokColor}">${tokRed.score}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${tokStatus}" style="width:${Math.min(tokRed.score*2,100)}%;background:${tokColor}"></div></div>
        <div class="gauge-sub">${tokMsg}</div>
      </div>
      <div class="gauge-card">
        <span class="gauge-lbl">Information density</span>
        <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:${parseFloat(r.entropy)<3?'var(--warn)':parseFloat(r.entropy)<4?'var(--black)':'var(--accent)'}">${r.entropy} <span style="font-size:.9rem;font-family:var(--sans);color:var(--muted)">bits/char</span></div>
        <div class="gauge-sub">${parseFloat(r.entropy)<3?'low — repetitive, compresses well':parseFloat(r.entropy)<4?'moderate — typical conversation':'high — dense, information-rich'}</div>
      </div>
      <div class="gauge-card">
        <span class="gauge-lbl">Response speed</span>
        <div class="latency-bar-bg"><div class="latency-bar ${r.attention.zone}" style="width:${Math.min(r.attention.percentage*1.1,100)}%;background:${r.attention.zone==='safe'?'var(--accent)':r.attention.zone==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
        <div class="gauge-sub">${r.attention.message}</div>
      </div>
      ${r.warning?`<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.78rem;color:#7a3a10;line-height:1.55">${r.warning}</p></div>`:''}`;

    document.getElementById('math-section').style.display = 'block';
    renderMath(r, level);
  } catch(e){ alert('Analysis failed: ' + e.message); }
  finally{ hideLoading(); }
}

async function runCompress(){
  if(!text.trim()) return;
  showLoading('Compressing — removing repeated content and rewriting');
  try{
    const r = await API.compress(text, model, plan);
    compressed = r.compressed;
    document.getElementById('output-section').style.display   = 'block';
    document.getElementById('out-text').textContent           = compressed;
    document.getElementById('saved-badge').textContent        = `${r.tokens_saved.toLocaleString()} tokens saved · ${r.compression_ratio}% shorter`;
    document.getElementById('out-note').textContent           = `Paste this into a new ${MODELS[model].label} conversation. Same meaning, ${r.tokens_saved.toLocaleString()} fewer tokens — more room for what matters.`;
    document.getElementById('output-section').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e){ alert('Compression failed: ' + e.message); }
  finally{ hideLoading(); }
}

// ── MATH PANEL ────────────────────────────────────────────────────────

function renderMath(r, lv){
  const el = document.getElementById('math-area');
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
  const cpt       = TC.charsPerToken[model]?.[plan] || 3.8;
  const tokNote   = MODELS[model].tokenizerNote?.[plan] || '';

  if(lv === 'simple'){
    el.innerHTML = `
      <div class="math-card">
        <h4>What the numbers mean</h4>
        <p>Your conversation is <strong>${total} tokens</strong> — <strong>${pctD}%</strong> of your ${MODELS[model].label} memory limit.
        <strong>${semRed}%</strong> of it says things that were already said earlier in different words —
        roughly <strong>${removable} tokens</strong> that could be removed without losing any meaning.
        ${tokRed.fillers.length > 0 ? `There are also <strong>${tokRed.fillers.length} filler phrase(s)</strong> that add no information: ${tokRed.fillers.slice(0,3).map(f=>`"${f}"`).join(', ')}.` : ''}
        The information density is <strong>${entropy} bits/char</strong> —
        ${parseFloat(entropy)<3?'low, meaning your text is quite repetitive and will compress well':parseFloat(entropy)<4?'moderate, typical for conversational text':'high, meaning the text is information-dense'}.
        </p>
        <p style="font-size:.78rem;font-family:var(--mono);color:var(--muted);margin-top:.5rem">Tokenizer: ${tokNote}</p>
        <a href="learn/tokens.html" class="learn-lnk">What is a token? →</a>
      </div>
      <div class="math-card">
        <h4>Why responses are getting slower</h4>
        <p>At <strong>${pct}%</strong> memory fill, responses are approximately <strong>${mult}×</strong> slower than the start of this conversation.
        This happens because ${MODELS[model].label} re-reads your entire conversation on every single message —
        and that work grows quadratically, not linearly.
        Compressing now will make it meaningfully faster again.</p>
        <a href="learn/attention.html" class="learn-lnk">Why does AI slow down? →</a>
      </div>`;
  } else {
    el.innerHTML = `
      <div class="math-card">
        <h4>Tokenizer — ${MODELS[model].label}</h4>
        <div class="formula">Model: ${MODELS[model].label} · ${plan}
Tokenizer: ${tokNote}
Estimate:  tokens ≈ chars / ${cpt}  →  ${total} tokens

All three models use different tokenizer algorithms:
  Claude Haiku/Sonnet  Custom BPE       ~3.5 chars/token
  Claude Opus 4.7      New BPE (+35%)   ~2.6 chars/token
  ChatGPT              cl100k BPE       ~4.0 chars/token  vocab 100,277
  Gemini               SentencePiece    ~4.5 chars/token  vocab 256,000

Same text → different token counts across models.</div>
        <a href="learn/tokens.html" class="learn-lnk">Tokenization explained →</a>
      </div>
      <div class="math-card">
        <h4>Semantic redundancy — sentence embeddings</h4>
        <div class="formula">Method: ${r.redundancy?.method || 'word_overlap'}
Score: ${semRed}% of sentences are semantically redundant
Removable: ~${removable} tokens

Detection pipeline:
  Pass 1 — Jaccard(A,B) = |A∩B| / |A∪B|  (word overlap)
  Pass 2 — TF cosine sim(A,B) = (A·B)/(‖A‖·‖B‖)  (semantic)
  Threshold: Jaccard ≥ 0.4  OR  cosine ≥ 0.65

Stage 1 compression (sentence-transformers, 384-dim):
  embed each sentence → find cosine sim > 0.88
  drop near-duplicate sentences before LLM rewrite</div>
        <a href="learn/embeddings.html" class="learn-lnk">How embeddings work →</a>
      </div>
      <div class="math-card">
        <h4>Token waste — filler detection</h4>
        <div class="formula">Score: ${tokRed.score}% filler token ratio
Filler phrases found: ${tokRed.fillers.length > 0 ? tokRed.fillers.slice(0,5).join(', ') : 'none'}
Filler tokens: ~${tokRed.fillerTokens || 0}

These patterns add zero information:
  Openers:      "I'd be happy to", "Of course!", "Certainly"
  Closers:      "I hope this helps!", "Let me know if..."
  Restatements: "As I mentioned", "To reiterate"
  Hedges:       "it's worth noting that", "generally speaking"

LLMLingua-2 removes these at token level before LLM rewrite.</div>
      </div>
      <div class="math-card">
        <h4>Shannon entropy &amp; compression bound</h4>
        <div class="formula">H(X) = −Σ p(x) · log₂ p(x)
  p(x) = probability of character x
  result: H = ${entropy} bits/char
  0 = completely repetitive  ·  4.7 = max density

Lossless compression bound:
  bound = (1 − H/log₂|A|) × 100%
  ${parseFloat(entropy)>0?`= (1 − ${entropy}/${Math.log2(Math.max(new Set(text).size,2)).toFixed(2)}) × 100% = ${r.compression_bound?.bound ?? '?'}%`:'—'}</div>
        <a href="learn/entropy.html" class="learn-lnk">Entropy explained →</a>
      </div>
      <div class="math-card">
        <h4>Self-attention O(n²) — latency model</h4>
        <div class="formula">Attention(Q,K,V) = softmax(QKᵀ/√d) · V
  Complexity: O(n²·d) per layer

Latency multiplier = (context% / 50)²
  at ${pct}%: (${pct}/50)² = ${mult}×

Compression speedup at ${semRed}% reduction:
  speedup = 1/(1−${(semRed/100).toFixed(2)})² = ${speedup}×</div>
        <a href="learn/attention.html" class="learn-lnk">Attention explained →</a>
      </div>`;
  }
}

// ── UTILITIES ─────────────────────────────────────────────────────────

function toggleBtns(){
  const ok = text.trim().length > 0;
  document.getElementById('analyze-btn').disabled  = !ok;
  document.getElementById('compress-btn').disabled = !ok;
}

function showLoading(msg){
  document.getElementById('loading-msg').textContent = msg;
  document.getElementById('loading').classList.add('visible');
  document.getElementById('analyze-btn').disabled  = true;
  document.getElementById('compress-btn').disabled = true;
}

function hideLoading(){
  document.getElementById('loading').classList.remove('visible');
  toggleBtns();
}

function hideOutput(){
  document.getElementById('output-section').style.display = 'none';
  document.getElementById('math-section').style.display   = 'none';
}