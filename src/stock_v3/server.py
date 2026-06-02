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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>stock_v3 — Equity Research</title>
<link rel="stylesheet" href="/assets/axis.css">
<style>
  body { min-height: 100vh; display: flex; flex-direction: column; align-items: center;
    padding: clamp(24px, 8vh, 80px) 20px 64px; }
  .shell { width: 100%; max-width: 600px; }
  .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 28px; }
  .brand .mark { width: 38px; height: 38px; border-radius: 10px; background: var(--accent);
    display: grid; place-items: center; color: var(--accent-text); font-weight: 700;
    font-size: 20px; font-family: var(--font-sans); }
  .brand .name { font-size: var(--fs-h3); font-weight: 700; letter-spacing: -0.01em; }
  .brand .name span { color: var(--text-tertiary); font-weight: 400; }

  .card { background: var(--bg-surface); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl); padding: clamp(24px, 5vw, 36px); box-shadow: var(--shadow-md); }
  h1 { font-size: clamp(22px, 5vw, 27px); font-weight: 700; letter-spacing: -0.02em;
    font-family: var(--font-sans); }
  .lede { color: var(--text-secondary); font-size: var(--fs-body); margin: 8px 0 24px; line-height: 1.5; }

  .seg { display: inline-flex; background: var(--bg-inset); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill); padding: 3px; margin-bottom: 20px; }
  .seg button { height: 36px; padding: 0 20px; border: none; background: transparent;
    color: var(--text-secondary); font-family: var(--font-sans); font-size: var(--fs-sm);
    font-weight: 600; border-radius: var(--radius-pill); cursor: pointer;
    transition: all var(--dur) var(--ease-out); }
  .seg button.active { background: var(--accent); color: var(--accent-text); }

  .field { display: flex; gap: 10px; align-items: stretch; }
  .field .input { flex: 1; height: 52px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; }
  .field .input::placeholder { text-transform: none; font-weight: 400; letter-spacing: 0; }
  .hint { color: var(--text-tertiary); font-size: var(--fs-xs); margin-top: 10px; }

  .opt { margin-top: 16px; display: flex; align-items: center; gap: 9px; font-size: var(--fs-sm);
    color: var(--text-secondary); }
  .opt input { accent-color: var(--accent); width: 18px; height: 18px; cursor: pointer; }
  .opt label { cursor: pointer; } .opt code { font-family: var(--font-mono);
    font-size: var(--fs-xs); color: var(--text-tertiary); }

  .examples { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
  .chip { height: 34px; padding: 0 14px; border-radius: var(--radius-pill);
    background: var(--bg-inset); border: 1px solid var(--border-default);
    color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--fs-xs);
    cursor: pointer; display: inline-flex; align-items: center;
    transition: all var(--dur) var(--ease-out); }
  .chip:hover { border-color: var(--accent-line); color: var(--text-primary); }

  #log { margin-top: 24px; display: none; }
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

  #frame-wrap { margin-top: 36px; width: 100%; max-width: 1100px; display: none; }
  .frame-bar { display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; }
  .frame-bar a { color: var(--accent-hover); text-decoration: none; font-size: var(--fs-sm); }
  .frame-bar a:hover { text-decoration: underline; }
  iframe { width: 100%; height: 86vh; border: 1px solid var(--border-default);
    border-radius: var(--radius-lg); background: var(--bg-base); }
</style>
</head>
<body>
<div class="shell">
  <div class="brand">
    <div class="mark">A</div>
    <div class="name">stock<span>_v3</span> · Equity Research</div>
  </div>

  <div class="card">
    <h1 id="title">Analyze a ticker</h1>
    <p class="lede" id="lede">Generate an institutional research report from live free-data —
      verdict, scenarios, probability cone and a full trade plan.</p>

    <div class="seg" role="tablist">
      <button id="seg-one" class="active" onclick="setMode('one')">Single report</button>
      <button id="seg-cmp" onclick="setMode('cmp')">Compare</button>
    </div>

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
  </div>
</div>

<div id="frame-wrap">
  <div class="frame-bar">
    <span class="eyebrow">Result</span>
    <a id="frame-link" href="#" target="_blank">Open in new tab &#8599;</a>
  </div>
  <iframe id="frame" src="about:blank" title="Result"></iframe>
</div>

<script>
let mode = 'one';
const SINGLE = ['NVDA','AAPL','MSFT','PLTR'];
const COMPARE = ['NVDA, AMD, AVGO', 'AAPL, MSFT, GOOGL', 'PLTR, SNOW, NET'];

function setMode(m) {
  mode = m;
  document.getElementById('seg-one').classList.toggle('active', m === 'one');
  document.getElementById('seg-cmp').classList.toggle('active', m === 'cmp');
  const t = document.getElementById('title'), l = document.getElementById('lede');
  const inp = document.getElementById('ticker'), btn = document.getElementById('btn');
  const hint = document.getElementById('hint');
  if (m === 'one') {
    t.textContent = 'Analyze a ticker';
    l.textContent = 'Generate an institutional research report from live free-data — verdict, scenarios, probability cone and a full trade plan.';
    inp.placeholder = 'e.g. NVDA'; btn.textContent = 'Analyze';
    hint.textContent = 'Enter one symbol, then press Enter.';
  } else {
    t.textContent = 'Compare tickers';
    l.textContent = 'Run several names side by side — verdict, conviction, risk/reward and key metrics, with the leader in each row marked.';
    inp.placeholder = 'e.g. NVDA, AMD, AVGO'; btn.textContent = 'Compare';
    hint.textContent = 'Enter 2–6 symbols separated by commas.';
  }
  renderExamples();
}

function renderExamples() {
  const wrap = document.getElementById('examples');
  wrap.innerHTML = '';
  (mode === 'one' ? SINGLE : COMPARE).forEach(ex => {
    const c = document.createElement('button');
    c.className = 'chip'; c.textContent = ex;
    c.onclick = () => { document.getElementById('ticker').value = ex;
      document.getElementById('ticker').focus(); };
    wrap.appendChild(c);
  });
}

function run() {
  const raw = document.getElementById('ticker').value.trim();
  if (!raw) { document.getElementById('ticker').focus(); return; }
  const narrative = document.getElementById('narrative').checked;
  return mode === 'one' ? runSingle(raw, narrative) : runCompare(raw, narrative);
}

function runCompare(raw, narrative) {
  const btn = document.getElementById('btn');
  const wrap = document.getElementById('frame-wrap'), frame = document.getElementById('frame');
  btn.disabled = true; btn.textContent = 'Running…';
  const url = '/compare?tickers=' + encodeURIComponent(raw) + '&narrative=' + narrative;
  frame.src = url; wrap.style.display = 'block';
  document.getElementById('frame-link').href = url;
  frame.onload = () => { btn.disabled = false; btn.textContent = 'Compare';
    wrap.scrollIntoView({behavior:'smooth'}); };
}

function runSingle(ticker, narrative) {
  ticker = ticker.split(/[,\\s]+/)[0].toUpperCase();
  const btn = document.getElementById('btn');
  const log = document.getElementById('log'), list = document.getElementById('log-list');
  const wrap = document.getElementById('frame-wrap');
  btn.disabled = true; log.style.display = 'block'; wrap.style.display = 'none';
  list.innerHTML = ''; let lastLi = null;

  function addLine(text, cls) {
    const li = document.createElement('li');
    if (cls) li.className = cls;
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
      wrap.style.display = 'block';
      document.getElementById('frame-link').href = path;
      document.getElementById('frame').src = path;
      wrap.scrollIntoView({behavior:'smooth'});
    } else if (msg.startsWith('__error__:')) {
      es.close(); if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
      addLine(msg.slice(10), 'err'); btn.disabled = false;
    } else { addLine(msg); }
  };
  es.onerror = () => { es.close();
    if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
    addLine('Connection error — check the server.', 'err'); btn.disabled = false; };
}

document.getElementById('ticker').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') run();
});
renderExamples();
</script>
</body>
</html>
"""
