// tool.js — main tool page logic

const MODELS = {
  claude: {
    label: 'Claude',
    color: '#c4602a',
    plans: {
      haiku:  'Haiku 4.5  ·  200k tokens  ·  Fast',
      sonnet: 'Sonnet 4.6  ·  1M tokens  ·  Balanced',
      opus:   'Opus 4.7  ·  1M tokens  ·  Most capable',
    },
    defaultPlan: 'sonnet',
    tokenizerNote: {
      haiku:  'Custom BPE — ~3.5 chars/token',
      sonnet: 'Custom BPE — ~3.5 chars/token',
      opus:   'New tokenizer in Opus 4.7 — ~2.6 chars/token (up to 35% more tokens than older Claude)',
    },
    behavior: 'Full recall — keeps your entire conversation and re-reads it on every message. You will hit the limit eventually and Claude will stop.',
    warning: 'Claude re-reads your full conversation on every message. The more you write, the slower and more expensive it gets. Opus 4.7 also uses a new tokenizer — up to 35% more tokens for the same text.',
  },
  chatgpt: {
    label: 'ChatGPT',
    color: '#1a6b4a',
    plans: {
      free:  'Free  ·  GPT-5.4 Mini  ·  ~32k tokens',
      plus:  'Plus  ·  GPT-5.4  ·  272k tokens',
      pro:   'Pro  ·  GPT-5.4 Pro  ·  1.05M tokens',
    },
    defaultPlan: 'plus',
    tokenizerNote: {
      free:  'cl100k BPE — ~4.0 chars/token — vocab 100,277',
      plus:  'cl100k BPE — ~4.0 chars/token — vocab 100,277',
      pro:   'cl100k BPE — ~4.0 chars/token — vocab 100,277',
    },
    behavior: 'Silent forgetting — when your conversation gets too long, ChatGPT quietly drops your oldest messages without telling you.',
    warning: 'ChatGPT is quietly forgetting your earlier messages. It never tells you — responses just gradually lose track of things you said earlier in the conversation.',
  },
  gemini: {
    label: 'Gemini',
    color: '#1e4f8a',
    plans: {
      free:  'Free  ·  2.5 Flash  ·  1,048,576 tokens',
      pro:   'AI Pro  ·  2.5 Pro  ·  1,048,576 tokens',
      ultra: 'AI Ultra  ·  3.1 Pro Preview  ·  1,048,576 tokens',
    },
    defaultPlan: 'free',
    tokenizerNote: {
      free:  'SentencePiece unigram — ~4.5 chars/token — vocab 256,000',
      pro:   'SentencePiece unigram — ~4.5 chars/token — vocab 256,000',
      ultra: 'SentencePiece unigram — ~4.5 chars/token — vocab 256,000',
    },
    behavior: 'Largest context window of the three. On the free tier, rate limits kick in long before you hit the token limit.',
    warning: 'Gemini Free hits rate limits long before the 1M token window. On paid tiers, response quality can drop on very long conversations even within the limit.',
  },
};

const ACCEPTED_FILES = '.txt,.md,.csv,.pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.png,.jpg,.jpeg,.webp,.gif,.py,.js,.ts,.jsx,.tsx,.java,.cpp,.c,.h,.cs,.rs,.go,.sql,.html,.css,.json,.yaml,.yml,.rb,.php,.swift,.kt,.r,.sh,.bash,.ps1,.ipynb';

let model='claude', plan='sonnet', text='', analysis=null, level='simple', compressed='';

document.addEventListener('DOMContentLoaded', () => {
  renderPlans();
  updateBehavior();
  setupListeners();
  const fi = document.getElementById('file-input');
  if(fi) fi.setAttribute('accept', ACCEPTED_FILES);
  const fz = document.getElementById('file-zone');
  if(fz) fz.textContent = 'Drop any file or click · PDF · PPTX · DOCX · XLSX · Images · Code · CSV · Text';
  showEmptyGauges();
});

function showEmptyGauges(){
  const limit = TC.getLimit(model, plan);
  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card" style="border-left:3px solid var(--accent)">
      <div style="font-family:var(--mono);font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Ready</div>
      <p style="font-size:.82rem;color:#555;line-height:1.65">Paste any conversation from ${MODELS[model].label} and see instantly how full the memory is, what's repeated, and how it's affecting response speed.</p>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val safe">0%</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">0 / ${limit.toLocaleString()} tokens · ${TC.formatLimit(limit)} available</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl" style="display:block;margin-bottom:.75rem">Tokenizer</span>
      <div style="font-size:.82rem;color:#555;line-height:1.7">
        <strong>${MODELS[model].label}</strong><br>
        <span style="font-family:var(--mono);font-size:.72rem;color:var(--muted)">${MODELS[model].tokenizerNote[plan]}</span>
      </div>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Repeated content</span><span class="gauge-val safe">—</span></div>
      <div class="gauge-track"><div class="gauge-fill safe" style="width:0%"></div></div>
      <div class="gauge-sub">Paste text to detect repeated content</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Information density</span>
      <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:var(--muted)">—</div>
      <div class="gauge-sub">How much unique information is in your text</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Response speed</span>
      <div class="latency-bar-bg"><div class="latency-bar safe" style="width:0%;background:var(--accent)"></div></div>
      <div class="gauge-sub">Gets slower as memory fills up</div>
    </div>
    <div style="background:var(--accent-light);border:1px solid #a8d4be;border-radius:10px;padding:1rem;font-size:.82rem;color:#1a4a35;line-height:1.65">
      <strong>${MODELS[model].label}</strong> — ${MODELS[model].behavior}
    </div>`;
}

function setupListeners(){
  document.querySelectorAll('.mbtn').forEach(btn => {
    btn.addEventListener('click', () => {
      model = btn.dataset.model;
      plan  = MODELS[model].defaultPlan;
      document.querySelectorAll('.mbtn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderPlans();
      updateBehavior();
      text ? updateLive() : showEmptyGauges();
    });
  });

  const ta = document.getElementById('main-ta');
  ta.addEventListener('input', () => {
    text = ta.value;
    toggleBtns();
    text.length > 0 ? updateLive() : showEmptyGauges();
    hideOutput();
  });

  document.getElementById('file-input').addEventListener('change', e => {
    if(e.target.files[0]) handleFile(e.target.files[0]);
  });
  const fz = document.getElementById('file-zone');
  fz.addEventListener('dragover',  e => { e.preventDefault(); fz.style.borderColor='var(--accent)'; });
  fz.addEventListener('dragleave', () => fz.style.borderColor='var(--border)');
  fz.addEventListener('drop', e => {
    e.preventDefault();
    fz.style.borderColor='var(--border)';
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
      text ? updateLive() : showEmptyGauges();
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

function updateLive(){
  const tokens = TC.estimate(text, model, plan);
  const pct    = TC.getPct(tokens, model, plan);
  const status = TC.getStatus(pct);
  const H      = TC.entropy(text);
  const red    = TC.redundancy(text, model, plan);
  const bd     = TC.breakdown(text, model, plan);
  const limit  = TC.getLimit(model, plan);
  const mult   = TC.attnMult(pct);

  document.getElementById('token-live').innerHTML =
    `<span>${tokens.toLocaleString()} tokens</span>` +
    `<span class="${status}" style="margin-left:6px">· ${pct}% of ${TC.formatLimit(limit)}</span>`;

  const warn = pct > 30 && MODELS[model].warning
    ? `<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.78rem;color:#7a3a10;line-height:1.55">${MODELS[model].warning}</p></div>`
    : '';

  const redColor  = red > 50 ? 'var(--danger)' : red > 25 ? 'var(--warn)' : 'var(--accent)';
  const redStatus = red > 50 ? 'danger' : red > 25 ? 'warning' : 'safe';
  const redMsg    = red > 30 ? 'a lot of repetition — compress recommended' : red > 10 ? 'some repeated content detected' : 'low repetition';

  document.getElementById('gauges-pane').innerHTML = `
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${status}">${pct}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${status}" style="width:${pct}%"></div></div>
      <div class="gauge-sub">${tokens.toLocaleString()} / ${limit.toLocaleString()} tokens · ${TC.formatLimit(limit - tokens)} remaining</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl" style="display:block;margin-bottom:.5rem">Token breakdown</span>
      <table class="bkdn-table">
        <tr><td style="color:var(--muted)">Your messages</td><td>${bd.user.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">AI responses</td><td>${bd.ai.toLocaleString()}</td></tr>
        <tr><td style="color:var(--muted)">System overhead</td><td>${bd.system.toLocaleString()}</td></tr>
      </table>
    </div>
    <div class="gauge-card">
      <div class="gauge-hdr"><span class="gauge-lbl">Repeated content</span><span class="gauge-val ${redStatus}" style="color:${redColor}">${red}%</span></div>
      <div class="gauge-track"><div class="gauge-fill ${redStatus}" style="width:${red}%;background:${redColor}"></div></div>
      <div class="gauge-sub">~${Math.round(tokens*red/100).toLocaleString()} tokens wasted · ${redMsg}</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Information density</span>
      <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:${H < 3 ? 'var(--warn)' : H < 4 ? 'var(--black)' : 'var(--accent)'}">${H} <span style="font-size:.9rem;font-family:var(--sans);color:var(--muted)">bits/char</span></div>
      <div class="gauge-sub">${H < 3 ? 'low — lots of repetition, compresses well' : H < 4 ? 'moderate — typical conversation' : 'high — dense, information-rich text'}</div>
    </div>
    <div class="gauge-card">
      <span class="gauge-lbl">Response speed</span>
      <div class="latency-bar-bg"><div class="latency-bar ${status}" style="width:${Math.min(pct*1.1,100)}%;background:${status==='safe'?'var(--accent)':status==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
      <div class="gauge-sub">~${mult}× slower than the start · ${status === 'safe' ? 'still fast' : status === 'warning' ? 'noticeably slower — compress soon' : 'significantly slow — compress now'}</div>
    </div>
    ${warn}`;
}

async function handleFile(file){
  const ext = file.name.split('.').pop().toLowerCase();
  const clientEst = TC.estimateFile(ext, file.size, model, plan);
  document.getElementById('file-status').textContent = `${file.name} · ~${clientEst.toLocaleString()} tokens (estimating...)`;
  try{
    const r = await API.parseFile(file, model, plan);
    let s = `${file.name} · ${r.token_estimate.toLocaleString()} tokens`;
    if(r.pages)            s += ` · ${r.pages} pages`;
    if(r.slides?.length)   s += ` · ${r.slides.length} slides`;
    if(r.images_found > 0) s += ` · ${r.images_found} image(s)`;
    if(r.language)         s += ` · ${r.language}`;
    s += ` · ${r.percentage}% of ${TC.formatLimit(TC.getLimit(model,plan))} limit`;
    document.getElementById('file-status').textContent = s;
    if(r.text_preview){ document.getElementById('main-ta').value = r.text_preview; text = r.text_preview; }
    toggleBtns(); updateLive();
  } catch(e){
    document.getElementById('file-status').textContent = `${file.name} · ~${clientEst.toLocaleString()} tokens (client estimate)`;
  }
}

async function runAnalysis(){
  if(!text.trim()) return;
  showLoading('Analyzing your conversation');
  try{
    const r = await API.analyze(text, model, plan);
    analysis = r;
    const s = TC.getStatus(r.tokens.percentage);
    const redColor  = r.redundancy.score > 50 ? 'var(--danger)' : r.redundancy.score > 25 ? 'var(--warn)' : 'var(--accent)';
    const redStatus = r.redundancy.score > 50 ? 'danger' : r.redundancy.score > 25 ? 'warning' : 'safe';
    document.getElementById('gauges-pane').innerHTML = `
      <div class="gauge-card">
        <div class="gauge-hdr"><span class="gauge-lbl">Memory used</span><span class="gauge-val ${s}">${r.tokens.percentage}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${s}" style="width:${r.tokens.percentage}%"></div></div>
        <div class="gauge-sub">${r.tokens.total.toLocaleString()} / ${r.tokens.limit.toLocaleString()} tokens · ${(r.tokens.limit - r.tokens.total).toLocaleString()} remaining</div>
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
        <div class="gauge-hdr"><span class="gauge-lbl">Repeated content</span><span class="gauge-val ${redStatus}" style="color:${redColor}">${r.redundancy.score}%</span></div>
        <div class="gauge-track"><div class="gauge-fill ${redStatus}" style="width:${r.redundancy.score}%;background:${redColor}"></div></div>
        <div class="gauge-sub">~${r.redundancy.removable.toLocaleString()} tokens wasted · ${r.redundancy.score > 30 ? 'compress recommended' : 'low repetition'}</div>
      </div>
      <div class="gauge-card">
        <span class="gauge-lbl">Information density</span>
        <div style="font-family:var(--serif);font-size:1.6rem;display:block;margin:.2rem 0;color:${parseFloat(r.entropy)<3?'var(--warn)':parseFloat(r.entropy)<4?'var(--black)':'var(--accent)'}">${r.entropy} <span style="font-size:.9rem;font-family:var(--sans);color:var(--muted)">bits/char</span></div>
        <div class="gauge-sub">${parseFloat(r.entropy)<3?'low — lots of repetition':'moderate — typical conversation'}</div>
      </div>
      <div class="gauge-card">
        <span class="gauge-lbl">Response speed</span>
        <div class="latency-bar-bg"><div class="latency-bar ${r.attention.zone}" style="width:${Math.min(r.attention.percentage*1.1,100)}%;background:${r.attention.zone==='safe'?'var(--accent)':r.attention.zone==='warning'?'var(--warn)':'var(--danger)'}"></div></div>
        <div class="gauge-sub">${r.attention.message}</div>
      </div>
      ${r.warning?`<div class="model-warn"><div class="warn-icon">!</div><p style="font-size:.78rem;color:#7a3a10;line-height:1.55">${r.warning}</p></div>`:''}`;
    document.getElementById('math-section').style.display = 'block';
    renderMath(r, level);
  } catch(e){ alert('Analysis failed: '+e.message); }
  finally{ hideLoading(); }
}

async function runCompress(){
  if(!text.trim()) return;
  showLoading('Compressing — finding repeated content and rewriting');
  try{
    const r = await API.compress(text, model, plan);
    compressed = r.compressed;
    document.getElementById('output-section').style.display = 'block';
    document.getElementById('out-text').textContent    = compressed;
    document.getElementById('saved-badge').textContent = `${r.tokens_saved.toLocaleString()} tokens saved · ${r.compression_ratio}% shorter`;
    document.getElementById('out-note').textContent    = `Paste this into a new ${MODELS[model].label} conversation. Same meaning, ${r.tokens_saved.toLocaleString()} fewer tokens — more room for what matters.`;
    document.getElementById('output-section').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(e){ alert('Compression failed: '+e.message); }
  finally{ hideLoading(); }
}

function renderMath(r, lv){
  const el = document.getElementById('math-area');
  if(!el) return;
  const entropy   = r.entropy || '—';
  const redScore  = r.redundancy?.score || 0;
  const pct       = r.attention?.percentage || 0;
  const mult      = r.attention?.multiplier || 1;
  const total     = r.tokens?.total?.toLocaleString() || '?';
  const removable = r.redundancy?.removable?.toLocaleString() || '?';
  const speedup   = redScore > 0 ? Math.round(1/Math.pow(Math.max(1-redScore/100,0.01),2)*10)/10 : 1;
  const cpt       = TC.charsPerToken[model]?.[plan] || 3.8;
  const tokNote   = MODELS[model].tokenizerNote?.[plan] || '';

  if(lv === 'simple'){
    el.innerHTML = `
      <div class="math-card">
        <h4>What the numbers mean</h4>
        <p>Your conversation is <strong>${total} tokens</strong> — ${r.tokens?.percentage||'?'}% of your ${MODELS[model].label} memory limit.
        About <strong>${redScore}%</strong> of it says things that were already said earlier —
        roughly <strong>${removable} tokens</strong> that could be removed without losing any meaning.
        The information density score is <strong>${entropy} bits/char</strong> —
        ${parseFloat(entropy)<3?'low, meaning your text is quite repetitive and will compress well':parseFloat(entropy)<4?'moderate, which is typical for conversational text':'high, meaning your text is information-dense and harder to compress'}.</p>
        <p style="font-size:.78rem;font-family:var(--mono);color:var(--muted);margin-top:.5rem">Tokenizer: ${tokNote}</p>
        <a href="learn/tokens.html" class="learn-lnk">What is a token? →</a>
      </div>
      <div class="math-card">
        <h4>Why responses are getting slower</h4>
        <p>At <strong>${pct}%</strong> memory fill, responses are approximately <strong>${mult}×</strong> slower than the start of this conversation.
        This happens because your AI re-reads your entire conversation history on every single message —
        and that work grows much faster than the conversation itself. Compressing now will make it faster again.</p>
        <a href="learn/attention.html" class="learn-lnk">Why does AI slow down? →</a>
      </div>`;
  } else {
    el.innerHTML = `
      <div class="math-card">
        <h4>Tokenizer — each model counts differently</h4>
        <div class="formula">Current: ${MODELS[model].label} · ${MODELS[model].plans[plan]}
Tokenizer: ${tokNote}
Estimate:  tokens ≈ chars / ${cpt}  →  ${total} tokens

Each model uses a different tokenizer algorithm and vocabulary:
  Claude Haiku/Sonnet  Custom BPE      ~3.5 chars/token
  Claude Opus 4.7      New BPE (+35%)  ~2.6 chars/token
  ChatGPT              cl100k BPE      ~4.0 chars/token  vocab 100,277
  Gemini               SentencePiece   ~4.5 chars/token  vocab 256,000

The same text produces different token counts across models.</div>
        <a href="learn/tokens.html" class="learn-lnk">Tokenization explained →</a>
      </div>
      <div class="math-card">
        <h4>Shannon entropy &amp; redundancy detection</h4>
        <div class="formula">H(X) = −Σ p(x) · log₂ p(x)
  p(x) = probability of character x in the text
  result: H = ${entropy} bits/char
  0 = completely repetitive  →  4.7 = maximum density

Redundancy: Jaccard similarity on word sets + TF cosine similarity
  Jaccard(A,B) = |A∩B| / |A∪B|  — catches exact/near-exact repeats
  sim(A,B) = (A·B) / (‖A‖·‖B‖)  — catches semantic overlap
  result: ${redScore}% of sentence pairs are redundant
          ~${removable} tokens removable</div>
        <a href="learn/entropy.html" class="learn-lnk">Entropy →</a>&nbsp;
        <a href="learn/embeddings.html" class="learn-lnk">Similarity →</a>
      </div>
      <div class="math-card">
        <h4>Self-attention complexity — why AI slows down</h4>
        <div class="formula">Attention(Q,K,V) = softmax(QKᵀ / √d) · V
  Q, K ∈ ℝⁿˣᵈ  ·  n = tokens  ·  d = dimensions
  complexity: O(n²·d) per layer — every token attends to every other

Latency multiplier = (context% / 50)²
  at ${pct}%:  (${pct}/50)² = ${mult}×

If you compress ${redScore}% of tokens:
  speedup ≈ 1 / (1 − ${(redScore/100).toFixed(2)})² = ${speedup}×</div>
        <a href="learn/attention.html" class="learn-lnk">Attention explained →</a>
      </div>
      <div class="math-card">
        <h4>How compression works</h4>
        <div class="formula">Step 1 — Semantic similarity (sentence-transformers, 384-dim):
  embed each sentence → 384-dimensional vector
  find pairs with cosine similarity above threshold
  flag semantically redundant content

Step 2 — LLM rewriting (Groq Llama 3.3 70B):
  send flagged redundancies + original text to Llama
  rewrite: same meaning, remove repeated content
  return shorter version ready to paste

Result: same conversation, fewer tokens, more room.</div>
        <a href="learn/embeddings.html" class="learn-lnk">How embeddings work →</a>
      </div>`;
  }
}

function toggleBtns(){ const ok=text.trim().length>0; document.getElementById('analyze-btn').disabled=!ok; document.getElementById('compress-btn').disabled=!ok; }
function showLoading(msg){ document.getElementById('loading-msg').textContent=msg; document.getElementById('loading').classList.add('visible'); document.getElementById('analyze-btn').disabled=true; document.getElementById('compress-btn').disabled=true; }
function hideLoading(){ document.getElementById('loading').classList.remove('visible'); toggleBtns(); }
function hideOutput(){ document.getElementById('output-section').style.display='none'; document.getElementById('math-section').style.display='none'; }