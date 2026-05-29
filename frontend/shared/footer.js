// footer.js — shared footer
(function(){
  const path = window.location.pathname;
  const depth = path.includes('/learn/') ? '../' : './';
  const html = `
<footer style="background:var(--black);color:var(--white);padding:4rem 2rem 2rem">
  <div style="max-width:1060px;margin:0 auto">
    <div style="display:grid;grid-template-columns:280px 1fr;gap:4rem;margin-bottom:3rem">
      <div>
        <a href="${depth}index.html" style="font-family:var(--serif);font-size:1.2rem;color:var(--white);display:flex;align-items:center;gap:8px;margin-bottom:.75rem;text-decoration:none">
          <span style="width:8px;height:8px;border-radius:50%;background:var(--accent-mid);display:inline-block"></span>ContextCrunch
        </a>
        <p style="font-size:.88rem;color:#888;line-height:1.65;margin-bottom:.75rem">Get more from every AI conversation.<br>See what's eating your context. Fix it instantly.</p>
        <p style="font-size:.8rem;color:#666;font-family:var(--mono)">Built by <a href="https://github.com/VishGuy2001" target="_blank" style="color:var(--accent-mid)">Vishnu Sekar</a></p>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2rem">
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Product</span>
          <a href="${depth}tool.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Try the tool</a>
          <a href="${depth}models.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Model comparison</a>
          <a href="${depth}about.html" style="font-size:.85rem;color:#aaa;text-decoration:none">About</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Learn</span>
          <a href="${depth}learn/tokens.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Tokens</a>
          <a href="${depth}learn/embeddings.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Embeddings</a>
          <a href="${depth}learn/entropy.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Entropy</a>
          <a href="${depth}learn/quantization.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Quantization</a>
          <a href="${depth}learn/attention.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Attention</a>
          <a href="${depth}learn/prompts.html" style="font-size:.85rem;color:#aaa;text-decoration:none">Prompts</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Developer</span>
          <a href="https://github.com/VishGuy2001/contextcrunch" target="_blank" style="font-size:.85rem;color:#aaa;text-decoration:none">GitHub</a>
          <a href="https://researchnu.com" target="_blank" style="font-size:.85rem;color:#aaa;text-decoration:none">ResearchNU</a>
          <a href="https://bracketgenius.vercel.app/" target="_blank" style="font-size:.85rem;color:#aaa;text-decoration:none">BracketGenius</a>
        </div>
      </div>
    </div>
    <div style="border-top:1px solid #222;padding-top:1.5rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.75rem">
      <span style="font-size:.72rem;color:#555;font-family:var(--mono)">Free &middot; Open source &middot; No data stored &middot; Works with Claude, ChatGPT &amp; Gemini</span>
      <span style="font-size:.72rem;color:#444;font-family:var(--mono)">&copy; ${new Date().getFullYear()} Vishnu Sekar &middot; MIT License</span>
    </div>
  </div>
</footer>`;
  const root = document.getElementById('footer-root');
  if(root) root.innerHTML = html;
})();