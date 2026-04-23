// api.js — complete file
const BACKEND_URL = window.BACKEND_URL || 'https://api.contextcrunch.io';

const API = {
  async analyze(text, model='claude', plan='sonnet'){
    const r = await fetch(`${BACKEND_URL}/analyze`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,model,plan})});
    if(!r.ok) throw new Error(`Analyze failed: ${r.status}`);
    return r.json();
  },
  async compress(text, model='claude', plan='sonnet', threshold=0.82){
    const r = await fetch(`${BACKEND_URL}/compress`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,model,plan,threshold})});
    if(!r.ok) throw new Error(`Compress failed: ${r.status}`);
    return r.json();
  },
  async improvePrompt(prompt, model='claude', plan='sonnet'){
    const r = await fetch(`${BACKEND_URL}/improve-prompt`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model,plan})});
    if(!r.ok) throw new Error(`Improve failed: ${r.status}`);
    return r.json();
  },
  async parseFile(file, model='claude', plan='sonnet'){
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(`${BACKEND_URL}/parse-file?model=${model}&plan=${plan}`,{method:'POST',body:fd});
    if(!r.ok) throw new Error(`Parse failed: ${r.status}`);
    return r.json();
  },
  async explain(concept, level='simple', userData=null){
    const r = await fetch(`${BACKEND_URL}/explain`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({concept,level,user_data:userData})});
    if(!r.ok) throw new Error(`Explain failed: ${r.status}`);
    return r.json();
  },
  async demo(concept, userInput, demoType='tokenize'){
    const r = await fetch(`${BACKEND_URL}/demo`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({concept,user_input:userInput,demo_type:demoType})});
    if(!r.ok) throw new Error(`Demo failed: ${r.status}`);
    return r.json();
  },
  async demoEmbeddings(sentences){
    const r = await fetch(`${BACKEND_URL}/demo/embeddings`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentences})});
    if(!r.ok) throw new Error(`Embeddings demo failed: ${r.status}`);
    return r.json();
  },
  async demoEntropy(text){
    const r = await fetch(`${BACKEND_URL}/demo/entropy`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    if(!r.ok) throw new Error(`Entropy demo failed: ${r.status}`);
    return r.json();
  },
  async demoTokenize(text, model='chatgpt'){
    const r = await fetch(`${BACKEND_URL}/demo/tokenize`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,model})});
    if(!r.ok) throw new Error(`Tokenize demo failed: ${r.status}`);
    return r.json();
  },
  async getModels(){
    const r = await fetch(`${BACKEND_URL}/models`);
    if(!r.ok) throw new Error('Models failed');
    return r.json();
  },
};

const TC = {
  limits: {
    claude:  { haiku: 200000, sonnet: 1000000, opus: 1000000 },
    chatgpt: { free: 32000, plus: 272000, pro: 1050000 },
    gemini:  { free: 1048576, pro: 1048576, ultra: 1048576 },
  },

  charsPerToken: {
    claude:  { haiku: 3.5, sonnet: 3.5, opus: 2.6 },
    chatgpt: { free: 4.0, plus: 4.0, pro: 4.0 },
    gemini:  { free: 4.5, pro: 4.5, ultra: 4.5 },
  },

  estimate(text, model='claude', plan='sonnet'){
    if(!text) return 0;
    const cpt = this.charsPerToken[model]?.[plan] || 3.8;
    return Math.ceil(text.length / cpt);
  },

  estimateFile(ext, sizeBytes, model='claude', plan='sonnet'){
    const code = ['py','js','ts','jsx','tsx','java','cpp','c','h','cs','rs','go','sql','html','css','json','yaml','yml','rb','php','swift','kt','r','sh','bash','ps1','ipynb'];
    if(['png','jpg','jpeg','webp','gif'].includes(ext)) return this.imageTokens(512, 512, model);
    if(code.includes(ext)) return Math.ceil(sizeBytes / 2.5);
    const cpt = this.charsPerToken[model]?.[plan] || 3.8;
    return Math.ceil(sizeBytes / cpt);
  },

  getLimit(model, plan){ return this.limits[model]?.[plan] || 200000; },
  getPct(tokens, model, plan){ return Math.min(Math.round((tokens / this.getLimit(model, plan)) * 100), 100); },
  getStatus(pct){ return pct < 40 ? 'safe' : pct < 70 ? 'warning' : 'danger'; },

  formatLimit(n){
    if(n >= 1000000) return (n / 1000000).toFixed(n % 1000000 === 0 ? 0 : 2) + 'M';
    if(n >= 1000) return Math.round(n / 1000) + 'k';
    return n.toString();
  },

  entropy(text){
    if(!text || text.length < 2) return 0;
    const freq = {};
    for(const ch of text) freq[ch] = (freq[ch] || 0) + 1;
    const n = text.length;
    let H = 0;
    for(const count of Object.values(freq)){
      const p = count / n;
      H -= p * Math.log2(p);
    }
    return Math.round(H * 1000) / 1000;
  },

  // Redundancy detection using a two-pass approach:
  //
  // Pass 1 — Exact/near-exact match (Jaccard similarity)
  //   Jaccard(A,B) = |A ∩ B| / |A ∪ B|
  //   Best for: repeated phrases, near-duplicate sentences, typo variants
  //   "How are you" vs "How are you" → Jaccard = 1.0 (exact duplicate)
  //   "How is it going" vs "How is it going?" → Jaccard = 0.8 (near-duplicate)
  //
  // Pass 2 — Semantic overlap (TF cosine similarity)
  //   sim(A,B) = (A·B) / (‖A‖·‖B‖)
  //   Best for: paraphrases with different wording but same meaning
  //   Only applied to longer sentences where stopword removal leaves content
  //
  // Why NOT cosine-only on short text:
  //   Short sentences like "How are you" consist entirely of stopwords.
  //   After stopword removal the TF vector is empty → cosine = 0 always.
  //   Jaccard operates on the raw word set before any filtering, catching
  //   exactly what cosine misses on short conversational text.
  redundancy(text, model='claude', plan='sonnet'){
    if(!text || text.length < 2) return 0;

    // Split on sentence boundaries and newlines
    const raw = text
      .split(/(?<=[.!?,;])\s+|\n+/)
      .map(s => s.trim().toLowerCase())
      .filter(s => s.length > 0);

    if(raw.length < 2) return 0;

    // Normalise: remove punctuation, collapse whitespace
    const norm = raw.map(s => s.replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim());

    // Jaccard similarity: |intersection| / |union| on word sets
    // Catches exact duplicates AND near-duplicates regardless of word count
    function jaccard(a, b){
      const wa = new Set(a.split(' ').filter(w => w));
      const wb = new Set(b.split(' ').filter(w => w));
      if(wa.size === 0 && wb.size === 0) return 1;
      if(wa.size === 0 || wb.size === 0) return 0;
      let inter = 0;
      for(const w of wa) if(wb.has(w)) inter++;
      return inter / (wa.size + wb.size - inter);
    }

    // TF cosine similarity for longer sentences (semantic overlap)
    // Only used when sentences have enough content words after stopword removal
    const stop = new Set(['the','a','an','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','i','you','he','she','it','we','they','this','that','and','or','but','in','on','at','to','for','of','with','by','from','as','not']);

    function tfCosine(a, b){
      const wa = a.split(' ').filter(w => w.length > 1 && !stop.has(w));
      const wb = b.split(' ').filter(w => w.length > 1 && !stop.has(w));
      if(wa.length < 2 || wb.length < 2) return 0; // not enough content for cosine
      const tf = (words) => {
        const v = {}, n = words.length;
        for(const w of words) v[w] = (v[w] || 0) + 1/n;
        return v;
      };
      const va = tf(wa), vb = tf(wb);
      let dot = 0, nA = 0, nB = 0;
      for(const [w, v] of Object.entries(va)){ dot += v*(vb[w]||0); nA += v*v; }
      for(const v of Object.values(vb)) nB += v*v;
      const d = Math.sqrt(nA)*Math.sqrt(nB);
      return d > 0 ? dot/d : 0;
    }

    // Count redundant sentence pairs using both methods
    // A pair is redundant if EITHER Jaccard >= 0.6 OR cosine >= 0.75
    // Jaccard 0.6 = 60% word overlap — clearly the same idea
    // Cosine 0.75 = strong semantic similarity on content words
    let redundantPairs = 0;
    const totalPairs = (raw.length * (raw.length - 1)) / 2;

    for(let i = 0; i < norm.length; i++){
      for(let j = i+1; j < norm.length; j++){
        const j_sim = jaccard(norm[i], norm[j]);
        const c_sim = j_sim < 0.6 ? tfCosine(norm[i], norm[j]) : 0; // skip cosine if jaccard already caught it
        if(j_sim >= 0.6 || c_sim >= 0.75) redundantPairs++;
      }
    }

    // Express as % of all sentence pairs that are redundant
    const rawPct = Math.round((redundantPairs / Math.max(totalPairs, 1)) * 100);

    // Cap at 90% — some repetition is intentional (emphasis, structure)
    return Math.min(rawPct, 90);
  },

  attnMult(pct){ return Math.round((pct / 50) ** 2 * 100) / 100; },

  breakdown(text, model='claude', plan='sonnet'){
    const lines = text.split('\n');
    let u=[], a=[], speaker='user';
    const um=['human:','user:','you:','me:'];
    const am=['assistant:','ai:','claude:','chatgpt:','gemini:','bot:'];
    for(const line of lines){
      const lo=line.toLowerCase().trim();
      if(um.some(m=>lo.startsWith(m)))      { speaker='user'; u.push(line); }
      else if(am.some(m=>lo.startsWith(m))) { speaker='ai';   a.push(line); }
      else (speaker==='ai'?a:u).push(line);
    }
    const ut=this.estimate(u.join(' '),model,plan), at=this.estimate(a.join(' '),model,plan);
    return { user:ut, ai:at, system:Math.round((ut+at)*0.05) };
  },

  imageTokens(w, h, model){
    if(model==='claude')  return Math.min(Math.round((w*h)/750), 1600);
    if(model==='chatgpt') return 85+(Math.ceil(w/512)*Math.ceil(h/512))*170;
    return 258;
  },
};

window.API = API;
window.TC  = TC;