// learn-base.js — shared utilities for all learn pages

// Render math level content
function setLevel(level, btn, levels) {
  document.querySelectorAll('.level-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const el = document.getElementById('level-content');
  if (el) el.innerHTML = levels[level] || '';
  // Re-highlight code blocks if any
  document.querySelectorAll('.code-block').forEach(b => {
    b.style.opacity = '0';
    setTimeout(() => { b.style.transition = 'opacity .3s'; b.style.opacity = '1'; }, 10);
  });
}

// Show loading state in demo result area
function demoLoading(resultId, msg = 'Calling backend') {
  const el = document.getElementById(resultId);
  if (el) el.innerHTML = `<div style="font-family:var(--mono);font-size:.82rem;color:var(--muted);padding:1rem;text-align:center"><span class="spin">◌</span> ${msg}<span class="loading-dots"></span></div>`;
}

// Show error in demo result area
function demoError(resultId, msg) {
  const el = document.getElementById(resultId);
  if (el) el.innerHTML = `<div style="background:var(--danger-light);border:1px solid #f0a0a0;border-radius:8px;padding:1rem;font-size:.85rem;color:var(--danger)">${msg}</div>`;
}

// Format API response text nicely
function formatDemoResult(text) {
  if (!text) return '';
  // Convert **bold** markers
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Convert backtick code
  text = text.replace(/`([^`]+)`/g, '<code style="font-family:var(--mono);background:var(--cream);padding:.1rem .3rem;border-radius:3px;font-size:.85em">$1</code>');
  // Paragraphs
  return text.split('\n\n').map(p => `<p style="margin-bottom:.75rem;font-size:.9rem;line-height:1.7;color:#333">${p.replace(/\n/g,'<br>')}</p>`).join('');
}

window.setLevel = setLevel;
window.demoLoading = demoLoading;
window.demoError = demoError;
window.formatDemoResult = formatDemoResult;
