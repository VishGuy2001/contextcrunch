/**
 * content-loader.js
 * Fetches educational content from the Python /content/<page> endpoint
 * and renders it into #level-content.
 *
 * All text, formulas, and code examples live in Python content_engine.py.
 * This file is the thin bridge between the HTML shell and the Python backend.
 */

let _pageContent = null;
let _currentLevel = 'simple';

/**
 * Fetch content for a page from Python, cache it, render the given level.
 * Called once on page load.
 */
async function loadPageContent(page, initialLevel = 'simple') {
  _currentLevel = initialLevel;
  try {
    const res = await fetch(`${API.BASE}/content/${page}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _pageContent = await res.json();
    renderLevel(_currentLevel);
  } catch (e) {
    console.warn('Content fetch failed, hiding level section:', e);
    document.getElementById('level-content').style.display = 'none';
  }
}

/**
 * Switch the displayed level. Called by level toggle buttons.
 * Overrides the setLevel() function from learn-base.js for pages
 * using the Python content system.
 */
function setLevel(level, btn) {
  _currentLevel = level;
  document.querySelectorAll('.level-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (_pageContent) renderLevel(level);
}

/**
 * Render a level's content into #level-content.
 * Handles sections with optional formula and code blocks.
 */
function renderLevel(level) {
  const container = document.getElementById('level-content');
  if (!_pageContent || !_pageContent.levels || !_pageContent.levels[level]) {
    container.innerHTML = '';
    return;
  }

  const data = _pageContent.levels[level];
  let html = '';

  // Waste patterns section (prompts page only)
  if (_pageContent.waste_patterns && level === 'simple') {
    html += renderWastePatterns(_pageContent.waste_patterns);
  }

  if (data.sections) {
    data.sections.forEach(section => {
      html += `<div class="cs">`;
      if (section.heading) {
        html += `<h2>${escHtml(section.heading)}</h2>`;
      }
      if (section.body) {
        html += `<p>${escHtml(section.body)}</p>`;
      }
      if (section.formula) {
        html += `<div class="formula">${escHtml(section.formula)}</div>`;
      }
      if (section.code) {
        html += `<div class="code-block">${escHtml(section.code)}</div>`;
      }
      html += `</div>`;
    });
  }

  container.innerHTML = html;
}

function renderWastePatterns(patterns) {
  let html = `<div class="cs"><h2>Common token waste patterns</h2><div class="waste-grid">`;
  patterns.forEach(p => {
    html += `<div class="waste-card">
      <span class="waste-tag">${escHtml(p.tag)}</span>
      <div style="font-size:.85rem;color:#333;line-height:1.5">${escHtml(p.example)}</div>
      <span class="waste-saving">→ ${escHtml(p.fix)}</span>
    </div>`;
  });
  html += `</div></div>`;
  return html;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}