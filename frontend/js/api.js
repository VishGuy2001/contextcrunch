// api.js — all calls to api.contextcrunch.io
// Model data verified April 2026 from official documentation

const BACKEND_URL = window.BACKEND_URL || 'https://contextcrunch-api-753105082654.us-central1.run.app';

const API = {
  BASE: BACKEND_URL,

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
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${BACKEND_URL}/parse-file?model=${model}&plan=${plan}`,{method:'POST',body:fd});
    if(!r.ok) throw new Error(`Parse failed: ${r.status}`);
    return r.json();
  },
  async explain(concept, level='simple', userData=null){
    const r = await fetch(`${BACKEND_URL}/explain`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({concept,level,user_data:userData})});
    if(!r.ok) throw new Error(`Explain failed: ${r.status}`);
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
  async learnRedundancy(text){
    const r = await fetch(`${BACKEND_URL}/learn/redundancy`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    if(!r.ok) throw new Error(`Redundancy analysis failed: ${r.status}`);
    return r.json();
  },
  async learnAttention(tokenCount, limit, avgPerTurn=150){
    const r = await fetch(`${BACKEND_URL}/learn/attention`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token_count:tokenCount,limit,avg_tokens_per_turn:avgPerTurn})});
    if(!r.ok) throw new Error(`Attention analysis failed: ${r.status}`);
    return r.json();
  },
  async learnQuantization(embeddingDim=384, vocabSize=100277){
    const r = await fetch(`${BACKEND_URL}/learn/quantization`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({embedding_dim:embeddingDim,vocab_size:vocabSize})});
    if(!r.ok) throw new Error(`Quantization analysis failed: ${r.status}`);
    return r.json();
  },
  async content(page){
    const r = await fetch(`${BACKEND_URL}/content/${page}`);
    if(!r.ok) throw new Error(`Content fetch failed: ${r.status}`);
    return r.json();
  },
  async getModels(){
    const r = await fetch(`${BACKEND_URL}/models`);
    if(!r.ok) throw new Error('Models failed');
    return r.json();
  },
};

// ── CLIENT-SIDE MATH ──────────────────────────────────────────────────

const TC = {
  // Context window limits — verified April 2026
  limits: {
    claude:  { free: 40000, haiku: 200000, sonnet: 1000000, opus: 1000000 },
    chatgpt: { free: 32000, plus: 272000, pro: 1050000 },
    gemini:  { free: 1048576, pro: 1048576, ultra: 1048576 },
  },

  // Chars per token — each model uses a different tokenizer algorithm
  // Claude Haiku/Sonnet: Custom BPE ~3.5 chars/token
  // Claude Opus 4.7:     New BPE — up to 35% more tokens — ~2.6 chars/token
  // ChatGPT:             cl100k BPE ~4.0 chars/token, vocab 100,277
  // Gemini:              SentencePiece unigram ~4.5 chars/token, vocab 256,000
  charsPerToken: {
    claude:  { free: 3.5, haiku: 3.5, sonnet: 3.5, opus: 2.6 },
    chatgpt: { free: 4.0, plus: 4.0, pro: 4.0 },
    gemini:  { free: 4.5, pro: 4.5, ultra: 4.5 },
  },

  // Cost per 1M tokens (input) — USD, verified April 2026
  costs: {
    claude:  { free: 0, haiku: 0.80, sonnet: 3.0, opus: 15.0 },
    chatgpt: { free: 0, plus: 2.5,   pro: 2.5 },
    gemini:  { free: 0, pro: 1.25,   ultra: 2.50 },
  },

  // File token multipliers by type
  // Images: billed per tile (Claude/GPT) or flat rate (Gemini)
  // PDFs:   each page ~1,500-2,500 tokens depending on density
  // Code:   ~2.5 chars/token (more efficient than prose)
  fileTokens(ext, sizeBytes, pages, model, plan){
    const codeExts = ['py','js','ts','jsx','tsx','java','cpp','c','h','cs','rs','go',
                      'sql','html','css','json','yaml','yml','rb','php','swift','kt','r',
                      'sh','bash','ps1','ipynb'];
    const imageExts = ['png','jpg','jpeg','webp','gif','bmp'];

    if(imageExts.includes(ext)){
      // Claude/ChatGPT: tile-based. Gemini: flat ~258 tokens per image
      if(model === 'gemini') return 258;
      return this.imageTokens(1024, 1024, model); // assume medium res
    }
    if(ext === 'pdf' && pages){
      // ~1,800 tokens/page average (text-heavy), varies by layout
      return pages * 1800;
    }
    if(['pptx','ppt'].includes(ext) && pages){
      return pages * 400; // slides are sparser
    }
    if(['docx','doc'].includes(ext)){
      return Math.ceil(sizeBytes / 4.0); // ~4 chars/token for prose
    }
    if(['xlsx','xls','csv'].includes(ext)){
      return Math.ceil(sizeBytes / 2.0); // structured data is token-dense
    }
    if(codeExts.includes(ext)){
      return Math.ceil(sizeBytes / 2.5);
    }
    const cpt = this.charsPerToken[model]?.[plan] || 3.8;
    return Math.ceil(sizeBytes / cpt);
  },

  // Image tile calculation (Claude + ChatGPT vision)
  // Both use 512px tiles: each tile = 170 tokens (Claude) or 85 tokens (GPT)
  imageTokens(width, height, model){
    const tiles = Math.ceil(width/512) * Math.ceil(height/512);
    const base  = model === 'chatgpt' ? 85 : 170;
    return tiles * base + (model === 'claude' ? 300 : 85); // base cost
  },

  estimate(text, model='claude', plan='sonnet'){
    if(!text) return 0;
    const cpt = this.charsPerToken[model]?.[plan] || 3.8;
    return Math.ceil(text.length / cpt);
  },

  getLimit(model, plan){ return this.limits[model]?.[plan] || 200000; },
  getPct(tokens, model, plan){ return Math.min(Math.round((tokens / this.getLimit(model, plan)) * 100), 100); },
  getStatus(pct){ return pct < 40 ? 'safe' : pct < 70 ? 'warning' : 'danger'; },
  getCost(tokens, model, plan){
    const rate = this.costs[model]?.[plan] || 0;
    return (tokens / 1_000_000 * rate).toFixed(6);
  },

  formatLimit(n){
    if(n >= 1_000_000) return (n/1_000_000).toFixed(n%1_000_000===0?0:1)+'M';
    if(n >= 1000) return Math.round(n/1000)+'k';
    return n.toString();
  },

  // H(X) = -Σ p(x) · log₂ p(x) over character distribution
  // Range: 0 (all same char) to ~4.7 (uniform ASCII)
  // Low H = repetitive/compressible. High H = information-dense.
  entropy(text){
    if(!text || text.length < 2) return 0;
    const freq = {};
    for(const ch of text) freq[ch] = (freq[ch]||0)+1;
    const n = text.length;
    let H = 0;
    for(const c of Object.values(freq)){ const p=c/n; H -= p*Math.log2(p); }
    return Math.round(H*1000)/1000;
  },

  // Semantic redundancy (Jaccard + TF-cosine on sentence pairs)
  // Returns score 0-90 as a percentage of redundant sentences
  redundancy(text){
    if(!text || text.length < 2) return 0;
    const sentences = text.split(/(?<=[.!?])\s+|\n+/).map(s=>s.trim().toLowerCase()).filter(s=>s.length>0);
    if(sentences.length < 2) return 0;
    const norm = sentences.map(s=>s.replace(/[^a-z0-9\s]/g,'').replace(/\s+/g,' ').trim());
    const stop = new Set(['the','a','an','is','are','was','were','be','been','have','has','had',
      'do','does','did','will','would','could','should','i','you','he','she','it','we','they',
      'this','that','and','or','but','in','on','at','to','for','of','with','by','from','as',
      'not','just','so','what','how','when','where','who','which','if','then','than','too',
      'very','also','can','get','me','my','your','our','their','its','ok','yes','no','hi']);

    function jaccard(a,b){
      const wa=new Set(a.split(' ').filter(w=>w&&!stop.has(w)));
      const wb=new Set(b.split(' ').filter(w=>w&&!stop.has(w)));
      if(!wa.size&&!wb.size){ const ra=new Set(a.split(' ').filter(w=>w)); const rb=new Set(b.split(' ').filter(w=>w)); if(!ra.size||!rb.size) return 0; let i=0; for(const w of ra) if(rb.has(w)) i++; return i/(ra.size+rb.size-i); }
      if(!wa.size||!wb.size) return 0;
      let i=0; for(const w of wa) if(wb.has(w)) i++;
      return i/(wa.size+wb.size-i);
    }
    function tfCosine(a,b){
      const wa=a.split(' ').filter(w=>w.length>1&&!stop.has(w));
      const wb=b.split(' ').filter(w=>w.length>1&&!stop.has(w));
      if(wa.length<2||wb.length<2) return 0;
      const tf=words=>{const v={},n=words.length;for(const w of words)v[w]=(v[w]||0)+1/n;return v;};
      const va=tf(wa),vb=tf(wb);
      let dot=0,nA=0,nB=0;
      for(const[w,v]of Object.entries(va)){dot+=v*(vb[w]||0);nA+=v*v;}
      for(const v of Object.values(vb))nB+=v*v;
      const d=Math.sqrt(nA)*Math.sqrt(nB);
      return d>0?dot/d:0;
    }
    const redundantSet=new Set();
    for(let i=0;i<norm.length;i++) for(let j=i+1;j<norm.length;j++) if(!redundantSet.has(j)){ const js=jaccard(norm[i],norm[j]); const cs=js<0.4?tfCosine(norm[i],norm[j]):0; if(js>=0.4||cs>=0.65) redundantSet.add(j); }
    return Math.min(Math.round((redundantSet.size/Math.max(sentences.length,1))*100),90);
  },

  // Token-level redundancy — filler phrases and repeated tokens
  // Returns count of wasteful token patterns found
  tokenRedundancy(text){
    if(!text) return {score:0, fillers:[], repeatedPhrases:[]};
    const fillers = [
      "i'd be happy to","of course","great question","certainly","absolutely",
      "as i mentioned","as i said","to reiterate","let me explain","in other words",
      "it's worth noting","needless to say","as previously","i hope this helps",
      "please let me know","happy to help","feel free to","does that make sense",
      "i want to make sure","sounds good","that's a great point","no problem",
      "you're welcome","thanks for asking","i understand","i see","got it",
    ];
    const lower = text.toLowerCase();
    const found = fillers.filter(f => lower.includes(f));
    const fillerTokens = found.reduce((sum,f) => sum + Math.ceil(f.split(' ').length * 1.1), 0);

    // Repeated n-gram detection (3-gram+)
    const words = text.toLowerCase().split(/\s+/).filter(w=>w.length>2);
    const ngrams = {};
    for(let i=0;i<words.length-2;i++){
      const g=words.slice(i,i+3).join(' ');
      ngrams[g]=(ngrams[g]||0)+1;
    }
    const repeatedPhrases = Object.entries(ngrams)
      .filter(([,c])=>c>1)
      .sort((a,b)=>b[1]-a[1])
      .slice(0,5)
      .map(([phrase,count])=>({phrase,count}));

    const totalTokens = this.estimate(text);
    const score = Math.min(Math.round((fillerTokens/Math.max(totalTokens,1))*100*3),50);

    return {score, fillers:found.slice(0,8), repeatedPhrases, fillerTokens};
  },

  // O(n²) attention cost — T(n) ∝ n²
  // multiplier = (fill_pct/50)² — calibrated at 50% = 1× baseline
  attentionMultiplier(tokens, limit){
    if(!limit) return 1;
    const pct=(tokens/limit)*100;
    return Math.round((pct/50)**2*100)/100;
  },
};