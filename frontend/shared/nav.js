// nav.js — shared navigation with fixed dropdown
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
    <li class="nav-dropdown" id="nav-dropdown">
      <button class="nav-drop-btn" id="nav-drop-btn" aria-expanded="false">Learn &#9662;</button>
      <ul class="dropdown-menu" id="dropdown-menu" role="menu">
        <li><a href="${depth}learn/index.html">Overview</a></li>
        <li><a href="${depth}learn/tokens.html">Tokens &amp; context</a></li>
        <li><a href="${depth}learn/embeddings.html">Embeddings</a></li>
        <li><a href="${depth}learn/entropy.html">Entropy</a></li>
        <li><a href="${depth}learn/quantization.html">Quantization</a></li>
        <li><a href="${depth}learn/attention.html">Attention &amp; latency</a></li>
        <li><a href="${depth}learn/prompts.html">Prompt efficiency</a></li>
      </ul>
    </li>
    <li><a href="https://github.com/VishGuy2001/contextcrunch" target="_blank" rel="noreferrer">GitHub</a></li>
    <li><a href="${depth}tool.html" class="btn btn-primary nav-cta">Try free &rarr;</a></li>
  </ul>
  <button class="nav-hamburger" id="nav-ham" aria-label="Menu">&#9776;</button>
</nav>
<style>
.nav{position:sticky;top:0;z-index:100;background:rgba(250,250,248,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:56px}
.nav-logo{font-family:var(--serif);font-size:1.2rem;color:var(--black);display:flex;align-items:center;gap:8px;text-decoration:none}
.logo-dot{width:9px;height:9px;border-radius:50%;background:var(--accent);display:inline-block;flex-shrink:0}
.nav-links{display:flex;align-items:center;gap:1.75rem;list-style:none;margin:0;padding:0}
.nav-links a{font-size:.875rem;color:var(--muted);transition:color .2s;text-decoration:none}
.nav-links a:hover{color:var(--black)}
.nav-cta{color:var(--white)!important;padding:.45rem 1.1rem!important;font-size:.85rem!important}
.nav-cta:hover{color:var(--white)!important}
.nav-dropdown{position:relative}
.nav-drop-btn{background:none;border:none;font-size:.875rem;color:var(--muted);cursor:pointer;font-family:inherit;padding:0;line-height:1}
.nav-drop-btn:hover,.nav-drop-btn.open{color:var(--black)}
.dropdown-menu{
  display:none;
  position:absolute;
  top:calc(100% + 12px);
  left:50%;
  transform:translateX(-50%);
  background:var(--white);
  border:1px solid var(--border);
  border-radius:10px;
  padding:.5rem;
  min-width:210px;
  box-shadow:0 8px 24px rgba(0,0,0,.1);
  list-style:none;
  margin:0;
  z-index:300
}
.dropdown-menu.open{display:block}
.dropdown-menu li a{display:block;padding:.55rem .85rem;border-radius:6px;font-size:.85rem;color:var(--black);text-decoration:none}
.dropdown-menu li a:hover{background:var(--cream)}
.nav-hamburger{display:none;background:none;border:none;font-size:1.3rem;cursor:pointer;color:var(--black);padding:0}
@media(max-width:768px){
  .nav-hamburger{display:block}
  .nav-links{
    display:none;
    position:fixed;
    top:56px;left:0;right:0;
    background:var(--white);
    border-bottom:1px solid var(--border);
    flex-direction:column;
    padding:1.25rem 2rem 1.5rem;
    gap:.85rem;
    align-items:flex-start;
    box-shadow:0 8px 24px rgba(0,0,0,.08);
    z-index:99
  }
  .nav-links.open{display:flex}
  .dropdown-menu{
    position:static;
    transform:none;
    box-shadow:none;
    border:none;
    padding-left:1rem;
    display:block;
    min-width:auto
  }
  .nav-drop-btn{display:none}
}
</style>`;

  const root = document.getElementById('nav-root');
  if(!root) return;
  root.innerHTML = html;

  // Hamburger toggle
  const ham = document.getElementById('nav-ham');
  const links = document.getElementById('nav-links');
  if(ham && links){
    ham.addEventListener('click', (e) => {
      e.stopPropagation();
      links.classList.toggle('open');
    });
  }

  // Dropdown — click to open/close, close on outside click
  const dropBtn  = document.getElementById('nav-drop-btn');
  const dropMenu = document.getElementById('dropdown-menu');
  if(dropBtn && dropMenu){
    dropBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = dropMenu.classList.contains('open');
      dropMenu.classList.toggle('open', !isOpen);
      dropBtn.classList.toggle('open', !isOpen);
      dropBtn.setAttribute('aria-expanded', !isOpen);
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
      if(!dropBtn.contains(e.target) && !dropMenu.contains(e.target)){
        dropMenu.classList.remove('open');
        dropBtn.classList.remove('open');
        dropBtn.setAttribute('aria-expanded', 'false');
      }
    });

    // Close when a link inside is clicked
    dropMenu.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        dropMenu.classList.remove('open');
        dropBtn.classList.remove('open');
      });
    });
  }

  // Pre-warm backend silently
  const BACKEND = window.BACKEND_URL || 'https://contextcrunch-api-753105082654.us-central1.run.app';
  window.BACKEND_URL = BACKEND;
  fetch(BACKEND + '/health').catch(() => {});
})();