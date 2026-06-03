"""FastAPI web server: ticker input → live-streamed pipeline progress → embedded report.

Run with:  uv run serve
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .cache import Cache
from .config import load_settings
from .pipeline import PipelineError, run_pipeline
from .report.render import write_report

app = FastAPI(title="stock_v3")


def main() -> None:
    import uvicorn
    uvicorn.run("stock_v3.server:app", host="127.0.0.1", port=8080, reload=False)
_settings = load_settings()
_cache = Cache(_settings.cache_dir)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_reports_dir = Path("reports")

# Serve generated reports + the AXIS stylesheet as static files.
_reports_dir.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(_reports_dir)), name="reports")
_assets_dir = Path(__file__).parent / "report" / "assets"
app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_INDEX_HTML)


@app.get("/analyze/stream")
async def analyze_stream(ticker: str, narrative: bool = False):
    """SSE endpoint. Streams progress events, then a final 'done' with the report URL."""
    ticker = re.sub(r"[^A-Za-z0-9.\-]", "", ticker).upper()[:10]
    if not ticker:
        return StreamingResponse(_error_stream("Invalid ticker"), media_type="text/event-stream")
    return StreamingResponse(
        _run_and_stream(ticker, narrative),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/report/{filename}")
async def serve_report(filename: str):
    path = _reports_dir / filename
    if not path.exists() or not path.suffix == ".html":
        return HTMLResponse("Report not found", status_code=404)
    return FileResponse(path)


@app.get("/compare", response_class=HTMLResponse)
async def compare(tickers: str, narrative: bool = False):
    """Run the pipeline for several tickers and return a side-by-side comparison page."""
    from .report.compare import render_comparison

    symbols = _parse_tickers(tickers)
    if len(symbols) < 2:
        return HTMLResponse(
            "<p style='font-family:sans-serif;padding:40px'>Enter at least two tickers "
            "to compare, e.g. <code>/compare?tickers=NVDA,AMD,AAPL</code>.</p>",
            status_code=400,
        )

    loop = asyncio.get_event_loop()

    def run_one(sym: str):
        try:
            return run_pipeline(sym, _settings, _cache, use_llm=narrative)
        except PipelineError:
            return None

    results = await asyncio.gather(
        *[loop.run_in_executor(_executor, run_one, s) for s in symbols]
    )
    reports = [r for r in results if r is not None]
    if len(reports) < 2:
        bad = [s for s, r in zip(symbols, results) if r is None]
        return HTMLResponse(
            f"<p style='font-family:sans-serif;padding:40px'>Couldn't analyze enough valid "
            f"tickers. Check: {', '.join(bad) or 'symbols'}.</p>",
            status_code=400,
        )
    return HTMLResponse(render_comparison(reports))


def _parse_tickers(raw: str) -> list[str]:
    seen: list[str] = []
    for part in re.split(r"[,\s]+", raw):
        sym = re.sub(r"[^A-Za-z0-9.\-]", "", part).upper()[:10]
        if sym and sym not in seen:
            seen.append(sym)
    return seen[:6]  # cap at 6 for layout sanity


# --------------------------------------------------------------------------- #
async def _run_and_stream(ticker: str, use_llm: bool):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def progress(msg: str):
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    def worker():
        try:
            progress(f"Fetching quote for {ticker}…")
            from .sources import prices as prices_src
            quote_r = prices_src.fetch_quote(ticker, _settings, _cache)
            if not quote_r.usable:
                raise PipelineError(f"No price data for {ticker}. Check the symbol.")

            progress(f"Found {quote_r.value.company} @ {quote_r.value.currency} "
                     f"{quote_r.value.price:,.2f}")
            progress("Pulling fundamentals, history, holders, options, short interest…")

            from .sources import edgar, finra, finnhub_src, fred
            from .sources import prices as prices_src
            history_r  = prices_src.fetch_history(ticker, _settings, _cache)
            fund_r     = prices_src.fetch_fundamentals(ticker, _settings, _cache)
            holders_r  = prices_src.fetch_institutional_holders(ticker, _settings, _cache)
            options_r  = prices_src.fetch_options_proxy(ticker, quote_r.value.price, _settings, _cache)
            short_r    = finra.fetch_short_interest(ticker, _settings, _cache)
            insider_r  = edgar.fetch_insider_activity(ticker, _settings, _cache)

            progress("Fetching macro data…" if _settings.has_fred else
                     "Macro layer skipped (no FRED key — add to .env for full coverage)")
            macro_r = fred.fetch_macro(_settings, _cache)

            progress("Fetching analyst consensus…" if _settings.has_finnhub else
                     "Analyst consensus skipped (no Finnhub key)")
            consensus_r = finnhub_src.fetch_consensus(ticker, _settings, _cache)
            earnings_r  = finnhub_src.fetch_earnings_catalysts(ticker, _settings, _cache)

            progress("Running analysis…")
            report = run_pipeline(ticker, _settings, _cache, use_llm=use_llm)

            progress("Rendering report…")
            path = write_report(report, _reports_dir)

            v = report.verdict
            progress(
                f"✓ {ticker} — {v.label} {v.conviction_score:.0f}/100 · "
                f"Confidence {v.confidence:.0f}/100 ({v.confidence_band})"
            )
            loop.call_soon_threadsafe(queue.put_nowait,
                                      f"__done__:/reports/{path.name}")
        except PipelineError as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"__error__:{exc}")
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, f"__error__:Unexpected error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(_executor, worker)

    while True:
        msg = await queue.get()
        if msg is None:
            break
        yield f"data: {json.dumps(msg)}\n\n"
        await asyncio.sleep(0)  # flush


async def _error_stream(msg: str):
    yield f"data: {json.dumps(f'__error__:{msg}')}\n\n"


# --------------------------------------------------------------------------- #
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>stock_v3</title>
<link rel="stylesheet" href="/assets/axis.css">
<style>
  html, body { height: 100%; }
  body { margin: 0; overflow: hidden; }

  .shell { display: grid; grid-template-columns: 248px 1fr; height: 100vh; }

  /* ---------- left rail ---------- */
  .rail { display: flex; flex-direction: column; padding: 20px 16px; gap: 4px; min-height: 0; }
  .rail .brand { display: flex; align-items: center; gap: 11px; padding: 4px 8px 22px; }
  .rail .brand .mark { width: 36px; height: 36px; border-radius: 10px; background: var(--accent);
    display: grid; place-items: center; color: var(--accent-text); font-weight: 700;
    font-size: 19px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.3); }
  .rail .brand .name { font-size: var(--fs-h3); font-weight: 700; letter-spacing: -0.01em; }
  .rail .eyebrow { padding: 4px 10px; margin-top: 4px; }

  .nav-item { display: flex; align-items: center; gap: 11px; height: 42px; padding: 0 12px;
    border-radius: var(--radius-md); color: var(--text-secondary); cursor: pointer;
    font-size: var(--fs-body); font-weight: 500; border: none; background: transparent;
    width: 100%; text-align: left; font-family: var(--font-sans);
    transition: background var(--dur-fast), color var(--dur-fast); }
  .nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
  .nav-item.active { color: var(--text-primary); font-weight: 600; }
  .nav-item .ic { width: 18px; height: 18px; flex-shrink: 0; opacity: 0.9; }

  .rail .spacer { flex: 1; min-height: 16px; }
  .rail .rail-foot { border-top: 1px solid var(--border-subtle); padding-top: 12px; }

  /* theme toggle row */
  .theme-row { display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; }
  .theme-row .lbl { font-size: var(--fs-sm); color: var(--text-secondary); }
  .theme-toggle { display: inline-flex; align-items: center; background: transparent;
    border: none; cursor: pointer; padding: 2px; }
  .theme-toggle .track { width: 44px; height: 25px; border-radius: var(--radius-pill);
    background: var(--bg-inset); border: 1px solid var(--border-default); position: relative;
    transition: background var(--dur) var(--ease-out), border-color var(--dur); }
  .theme-toggle .thumb { position: absolute; top: 50%; left: 2px; transform: translateY(-50%);
    width: 19px; height: 19px; border-radius: 50%; background: var(--text-tertiary);
    display: grid; place-items: center; font-size: 10px; color: var(--bg-surface);
    transition: left var(--dur) var(--ease-out), background var(--dur); }
  html[data-theme="light"] .theme-toggle .track { background: var(--accent); border-color: transparent; }
  html[data-theme="light"] .theme-toggle .thumb { left: 22px; background: var(--accent-text); color: var(--accent); }
  .theme-toggle:focus-visible { outline: none; }
  .theme-toggle:focus-visible .track { box-shadow: 0 0 0 3px var(--accent-soft); }

  /* ---------- main: topbar + workspace ---------- */
  .main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px;
    height: 64px; padding: 0 clamp(16px, 3vw, 32px); flex-shrink: 0; }
  .topbar .title h1 { font-size: var(--fs-h2); font-weight: 700; letter-spacing: -0.015em;
    font-family: var(--font-sans); line-height: 1.1; }
  .topbar .title .sub { font-size: var(--fs-sm); color: var(--text-tertiary); margin-top: 2px; }
  .topbar .actions { display: flex; align-items: center; gap: 10px; }

  .workspace { flex: 1; overflow-y: auto; padding: clamp(16px, 3vw, 32px); min-height: 0; }
  .panel { max-width: 720px; }

  .field { display: flex; gap: 10px; align-items: stretch; }
  .field .input { flex: 1; height: 52px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; }
  .field .input::placeholder { text-transform: none; font-weight: 400; letter-spacing: 0; }
  .hint { color: var(--text-tertiary); font-size: var(--fs-xs); margin-top: 10px; }
  .lede { color: var(--text-secondary); font-size: var(--fs-body); margin-bottom: 22px; line-height: 1.5; max-width: 560px; }

  .examples { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
  .chip { height: 34px; padding: 0 14px; border-radius: var(--radius-pill);
    background: var(--bg-inset); border: 1px solid var(--border-default);
    color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--fs-xs);
    cursor: pointer; display: inline-flex; align-items: center;
    transition: all var(--dur-fast); }
  .chip:hover { border-color: var(--border-strong); color: var(--text-primary); }

  .opt { margin-top: 18px; display: flex; align-items: center; gap: 9px; font-size: var(--fs-sm);
    color: var(--text-secondary); }
  .opt input { accent-color: var(--accent); width: 18px; height: 18px; cursor: pointer; }
  .opt label { cursor: pointer; } .opt code { font-family: var(--font-mono);
    font-size: var(--fs-xs); color: var(--text-tertiary); }

  #log { margin-top: 24px; display: none; max-width: 560px; }
  #log-list { list-style: none; padding: 0; margin: 0; }
  #log-list li { padding: 9px 0; border-bottom: 1px solid var(--border-subtle);
    font-size: var(--fs-sm); color: var(--text-secondary); display: flex; gap: 12px;
    align-items: baseline; }
  #log-list li:last-child { border: none; }
  #log-list li.final { color: var(--positive); font-weight: 600; }
  #log-list li.err { color: var(--critical); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted);
    flex-shrink: 0; }
  .dot.spin { background: var(--accent); animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .25; } }

  /* results: iframe fills the workspace */
  #result { display: none; flex-direction: column; height: 100%; }
  #result.show { display: flex; }
  #panel-view.hide { display: none; }
  iframe { width: 100%; flex: 1; border: 1px solid var(--border-default);
    border-radius: var(--radius-lg); background: var(--bg-base); min-height: 70vh; }

  /* ---------- mobile: rail collapses to a top strip ---------- */
  @media (max-width: 760px) {
    body { overflow: auto; }
    .shell { grid-template-columns: 1fr; height: auto; min-height: 100vh; }
    .rail { flex-direction: row; align-items: center; flex-wrap: wrap; gap: 8px;
      padding: 12px 16px; position: sticky; top: 0; z-index: 10; }
    .rail .brand { padding: 0 8px 0 0; }
    .rail .eyebrow, .rail .spacer { display: none; }
    .nav-item { width: auto; height: 38px; }
    .rail .rail-foot { border-top: none; padding-top: 0; margin-left: auto; }
    .workspace { padding: 20px 16px 64px; height: auto; overflow: visible; }
    iframe { min-height: 80vh; }
  }
</style>
</head>
<body class="glass">
<div class="shell">

  <!-- ---------- RAIL ---------- -->
  <aside class="rail">
    <div class="brand">
      <div class="mark">A</div>
      <div class="name">stock_v3</div>
    </div>
    <div class="eyebrow">Workspace</div>
    <button class="nav-item active" id="nav-one" onclick="setMode('one')">
      <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
      Single report
    </button>
    <button class="nav-item" id="nav-cmp" onclick="setMode('cmp')">
      <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="7" height="16" rx="1"/><rect x="14" y="4" width="7" height="16" rx="1"/></svg>
      Compare
    </button>
    <div class="spacer"></div>
    <div class="rail-foot">
      <div class="theme-row">
        <span class="lbl">Theme</span>
        <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle dark / light">
          <span class="track"><span class="thumb" id="theme-glyph">&#9790;</span></span>
        </button>
      </div>
    </div>
  </aside>

  <!-- ---------- MAIN ---------- -->
  <div class="main">
    <header class="topbar">
      <div class="title">
        <h1 id="tb-title">Analyze a ticker</h1>
        <div class="sub" id="tb-sub">Institutional research from live free-data</div>
      </div>
      <div class="actions" id="tb-actions">
        <a id="frame-link" class="btn btn-ghost" href="#" target="_blank" style="display:none">Open in new tab &#8599;</a>
        <button id="back-btn" class="btn btn-ghost" onclick="goBack()" style="display:none">&#8592; Back</button>
      </div>
    </header>

    <div class="workspace">
      <!-- INPUT PANEL -->
      <div id="panel-view">
        <section class="axis-card panel">
          <p class="lede" id="lede">Generate an institutional research report from live free-data —
            verdict, scenarios, an interactive probability cone and a full trade plan.</p>

          <div class="field">
            <input id="ticker" class="input" type="text" placeholder="e.g. NVDA"
                   maxlength="60" autocomplete="off" autofocus>
            <button id="btn" class="btn btn-primary" onclick="run()">Analyze</button>
          </div>
          <div class="hint" id="hint">Enter one symbol, then press Enter.</div>

          <div class="examples" id="examples"></div>

          <div class="opt">
            <input type="checkbox" id="narrative">
            <label for="narrative">Enrich prose with Claude <code>(ANTHROPIC_API_KEY)</code></label>
          </div>

          <div id="log"><ul id="log-list"></ul></div>
        </section>
      </div>

      <!-- RESULT PANEL -->
      <div id="result">
        <iframe id="frame" src="about:blank" title="Result"></iframe>
      </div>
    </div>
  </div>
</div>

<script>
let mode = 'one';
const SINGLE = ['NVDA','AAPL','MSFT','PLTR'];
const COMPARE = ['NVDA, AMD, AVGO', 'AAPL, MSFT, GOOGL', 'PLTR, SNOW, NET'];

// ---- theme (persisted; defaults to dark) ----
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const g = document.getElementById('theme-glyph');
  if (g) g.innerHTML = theme === 'light' ? '&#9728;' : '&#9790;';
}
function toggleTheme() {
  const next = (document.documentElement.getAttribute('data-theme') === 'light') ? 'dark' : 'light';
  localStorage.setItem('stockv3-theme', next);
  applyTheme(next); themeFrame();
}
function themeFrame() {
  const theme = document.documentElement.getAttribute('data-theme');
  const frame = document.getElementById('frame');
  try { const d = frame && frame.contentDocument;
    if (d && d.documentElement) d.documentElement.setAttribute('data-theme', theme); } catch (e) {}
}
applyTheme(localStorage.getItem('stockv3-theme') || 'dark');

// ---- mode (rail nav) ----
function setMode(m) {
  mode = m;
  document.getElementById('nav-one').classList.toggle('active', m === 'one');
  document.getElementById('nav-cmp').classList.toggle('active', m === 'cmp');
  const tt = document.getElementById('tb-title'), ts = document.getElementById('tb-sub');
  const inp = document.getElementById('ticker'), btn = document.getElementById('btn');
  const hint = document.getElementById('hint'), lede = document.getElementById('lede');
  if (m === 'one') {
    tt.textContent = 'Analyze a ticker';
    ts.textContent = 'Institutional research from live free-data';
    lede.textContent = 'Generate an institutional research report from live free-data — verdict, scenarios, an interactive probability cone and a full trade plan.';
    inp.placeholder = 'e.g. NVDA'; btn.textContent = 'Analyze';
    hint.textContent = 'Enter one symbol, then press Enter.';
  } else {
    tt.textContent = 'Compare tickers';
    ts.textContent = 'Side-by-side metric matrix';
    lede.textContent = 'Run several names side by side — verdict, conviction, risk/reward and key metrics, with the leader in each row marked.';
    inp.placeholder = 'e.g. NVDA, AMD, AVGO'; btn.textContent = 'Compare';
    hint.textContent = 'Enter 2–6 symbols separated by commas.';
  }
  renderExamples();
  goBack();
}
function renderExamples() {
  const wrap = document.getElementById('examples'); wrap.innerHTML = '';
  (mode === 'one' ? SINGLE : COMPARE).forEach(ex => {
    const c = document.createElement('button'); c.className = 'chip'; c.textContent = ex;
    c.onclick = () => { document.getElementById('ticker').value = ex; document.getElementById('ticker').focus(); };
    wrap.appendChild(c);
  });
}

// ---- result view switching ----
function showResult(url) {
  document.getElementById('panel-view').classList.add('hide');
  document.getElementById('result').classList.add('show');
  document.getElementById('back-btn').style.display = '';
  const link = document.getElementById('frame-link');
  link.href = url; link.style.display = '';
}
function goBack() {
  document.getElementById('result').classList.remove('show');
  document.getElementById('panel-view').classList.remove('hide');
  document.getElementById('back-btn').style.display = 'none';
  document.getElementById('frame-link').style.display = 'none';
  const f = document.getElementById('frame'); f.src = 'about:blank';
  document.getElementById('log').style.display = 'none';
}

function run() {
  const raw = document.getElementById('ticker').value.trim();
  if (!raw) { document.getElementById('ticker').focus(); return; }
  const narrative = document.getElementById('narrative').checked;
  return mode === 'one' ? runSingle(raw, narrative) : runCompare(raw, narrative);
}

function runCompare(raw, narrative) {
  const btn = document.getElementById('btn'), frame = document.getElementById('frame');
  btn.disabled = true; btn.textContent = 'Running…';
  const url = '/compare?tickers=' + encodeURIComponent(raw) + '&narrative=' + narrative;
  frame.onload = () => { btn.disabled = false; btn.textContent = 'Compare'; themeFrame(); };
  frame.src = url; showResult(url);
}

function runSingle(ticker, narrative) {
  ticker = ticker.split(/[,\\s]+/)[0].toUpperCase();
  const btn = document.getElementById('btn');
  const log = document.getElementById('log'), list = document.getElementById('log-list');
  btn.disabled = true; log.style.display = 'block'; list.innerHTML = ''; let lastLi = null;

  function addLine(text, cls) {
    const li = document.createElement('li'); if (cls) li.className = cls;
    li.innerHTML = '<span class="dot' + (cls ? '' : ' spin') + '"></span><span>' + text + '</span>';
    list.appendChild(li);
    if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
    lastLi = li; li.scrollIntoView({behavior:'smooth', block:'nearest'}); return li;
  }

  const url = '/analyze/stream?ticker=' + encodeURIComponent(ticker) + '&narrative=' + narrative;
  const es = new EventSource(url);
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.startsWith('__done__:')) {
      es.close(); const path = msg.slice(9);
      if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
      addLine('Report ready.', 'final'); btn.disabled = false;
      const f = document.getElementById('frame'); f.onload = themeFrame; f.src = path;
      showResult(path);
    } else if (msg.startsWith('__error__:')) {
      es.close(); if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
      addLine(msg.slice(10), 'err'); btn.disabled = false;
    } else { addLine(msg); }
  };
  es.onerror = () => { es.close();
    if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
    addLine('Connection error — check the server.', 'err'); btn.disabled = false; };
}

document.getElementById('ticker').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
renderExamples();
</script>
</body>
</html>
"""
