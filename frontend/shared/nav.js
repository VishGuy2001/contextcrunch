// nav.js — shared navigation + backend wake-up
(function(){
  const path = window.location.pathname;
  const isLearn = path.includes('/learn/');
  const depth = isLearn ? '../' : './';

  const html = `
<nav class="nav" role="navigation">
  <a href="${depth}index.html" class="nav-logo">
    <span class="logo-dot"></span>ContextCrunch
  </a>
  <ul class="nav-links" id="nav-links">
    <li><a href="${depth}tool.html">Tool</a></li>
    <li><a href="${depth}models.html">Models</a></li>
    <li class="nav-dropdown">
      <button class="nav-drop-btn">Learn ▾</button>
      <ul class="dropdown-menu">
        <li><a href="${depth}learn/index.html">Overview</a></li>
        <li><a href="${depth}learn/tokens.html">Tokens & context</a></li>
        <li><a href="${depth}learn/embeddings.html">Embeddings</a></li>
        <li><a href="${depth}learn/entropy.html">Entropy</a></li>
        <li><a href="${depth}learn/quantization.html">Quantization</a></li>
        <li><a href="${depth}learn/attention.html">Attention & latency</a></li>
        <li><a href="${depth}learn/prompts.html">Prompt efficiency</a></li>
      </ul>
    </li>
    <li><a href="https://github.com/VishGuy2001/contextcrunch" target="_blank" rel="noreferrer">GitHub</a></li>
    <li><a href="${depth}tool.html" class="btn btn-primary nav-cta">Try free →</a></li>
  </ul>
  <button class="nav-hamburger" id="nav-ham">☰</button>
</nav>
<style>
.nav{position:sticky;top:0;z-index:100;background:rgba(250,250,248,.93);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:56px}
.nav-logo{font-family:var(--serif);font-size:1.2rem;color:var(--black);display:flex;align-items:center;gap:8px}
.logo-dot{width:9px;height:9px;border-radius:50%;background:var(--accent);display:inline-block}
.nav-links{display:flex;align-items:center;gap:1.75rem;list-style:none}
.nav-links a{font-size:.875rem;color:var(--muted);transition:color .2s}
.nav-links a:hover{color:var(--black)}
.nav-cta{color:var(--white)!important;padding:.45rem 1.1rem!important;font-size:.85rem!important}
.nav-cta:hover{color:var(--white)!important}
.nav-dropdown{position:relative}
.nav-drop-btn{background:none;border:none;font-size:.875rem;color:var(--muted);cursor:pointer;font-family:var(--sans);padding:0}
.nav-drop-btn:hover{color:var(--black)}
.dropdown-menu{display:none;position:absolute;top:calc(100% + 8px);left:50%;transform:translateX(-50%);background:var(--white);border:1px solid var(--border);border-radius:10px;padding:.5rem;min-width:200px;box-shadow:0 4px 20px rgba(0,0,0,.08);list-style:none;z-index:200}
.nav-dropdown:hover .dropdown-menu{display:block}
.dropdown-menu li a{display:block;padding:.5rem .75rem;border-radius:6px;font-size:.85rem;color:var(--black)}
.dropdown-menu li a:hover{background:var(--cream)}
.nav-hamburger{display:none;background:none;border:none;font-size:1.3rem;cursor:pointer;color:var(--black)}
@media(max-width:768px){
  .nav-hamburger{display:block}
  .nav-links{display:none;position:absolute;top:56px;left:0;right:0;background:var(--white);border-bottom:1px solid var(--border);flex-direction:column;padding:1.5rem 2rem;gap:1rem;align-items:flex-start}
  .nav-links.open{display:flex}
  .dropdown-menu{position:static;transform:none;box-shadow:none;border:none;padding-left:1rem;display:block}
}
</style>`;

  const root = document.getElementById('nav-root');
  if(root){ root.innerHTML = html;
    document.getElementById('nav-ham').addEventListener('click', ()=> document.getElementById('nav-links').classList.toggle('open'));
  }

  // Pre-warm backend silently on every page load
  const BACKEND = window.BACKEND_URL || 'https://api.contextcrunch.io';
  window.BACKEND_URL = BACKEND;
  fetch(BACKEND+'/health').catch(()=>{});
})();
