(function(){
  const path = window.location.pathname;
  const depth = path.includes('/learn/') ? '../' : './';
  const html = `
<footer style="background:var(--black);color:var(--white);padding:4rem 2rem 2rem">
  <div style="max-width:1060px;margin:0 auto">
    <div style="display:grid;grid-template-columns:280px 1fr;gap:4rem;margin-bottom:3rem">
      <div>
        <a href="${depth}index.html" style="font-family:var(--serif);font-size:1.2rem;color:var(--white);display:flex;align-items:center;gap:8px;margin-bottom:.75rem">
          <span style="width:8px;height:8px;border-radius:50%;background:var(--accent-mid);display:inline-block"></span>ContextCrunch
        </a>
        <p style="font-size:.88rem;color:#888;line-height:1.65;margin-bottom:.75rem">Get more from every AI conversation.<br>See what's eating your context. Fix it instantly.</p>
        <p style="font-size:.8rem;color:#666;font-family:var(--mono)">Built by <a href="https://linkedin.com/in/vishnusekar/" target="_blank" style="color:var(--accent-mid)">Vishnu Sekar</a></p>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:2rem">
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Product</span>
          <a href="${depth}tool.html" style="font-size:.85rem;color:#aaa">Try the tool</a>
          <a href="${depth}models.html" style="font-size:.85rem;color:#aaa">Model comparison</a>
          <a href="${depth}about.html" style="font-size:.85rem;color:#aaa">About</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Learn</span>
          <a href="${depth}learn/tokens.html" style="font-size:.85rem;color:#aaa">Tokens</a>
          <a href="${depth}learn/embeddings.html" style="font-size:.85rem;color:#aaa">Embeddings</a>
          <a href="${depth}learn/entropy.html" style="font-size:.85rem;color:#aaa">Entropy</a>
          <a href="${depth}learn/quantization.html" style="font-size:.85rem;color:#aaa">Quantization</a>
          <a href="${depth}learn/attention.html" style="font-size:.85rem;color:#aaa">Attention</a>
          <a href="${depth}learn/prompts.html" style="font-size:.85rem;color:#aaa">Prompts</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Developer</span>
          <a href="https://github.com/VishGuy2001/contextcrunch" target="_blank" style="font-size:.85rem;color:#aaa">GitHub</a>
          <a href="https://pypi.org/project/contextcrunch-io" target="_blank" style="font-size:.85rem;color:#aaa">PyPI</a>
          <a href="${depth}math/README.md" style="font-size:.85rem;color:#aaa">Research notes</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:.6rem">
          <span style="font-family:var(--mono);font-size:.68rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Also by Vishnu</span>
          <a href="https://researchnu.com" target="_blank" style="font-size:.85rem;color:#aaa">ResearchNU</a>
          <a href="https://bracketgenius.vercel.app/" target="_blank" style="font-size:.85rem;color:#aaa">BracketGenius</a>
          <a href="https://linkedin.com/in/vishnusekar/" target="_blank" style="font-size:.85rem;color:#aaa">LinkedIn</a>
        </div>
      </div>
    </div>
    <div style="border-top:1px solid #222;padding-top:1.5rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.75rem">
      <span style="font-size:.72rem;color:#555;font-family:var(--mono)">Free · Open source · No data stored · Works with Claude, ChatGPT &amp; Gemini</span>
      <span style="font-size:.72rem;color:#444;font-family:var(--mono)">© ${new Date().getFullYear()} Vishnu Sekar · MIT License</span>
    </div>
  </div>
</footer>`;
  const root = document.getElementById('footer-root');
  if(root) root.innerHTML = html;
})();