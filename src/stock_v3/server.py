"""FastAPI web server: ticker input → live-streamed pipeline progress → embedded report.

Run with:  uv run serve
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .cache import Cache
from .config import load_settings
from .pipeline import PipelineError, run_pipeline
from .report.render import write_report

app = FastAPI(title="stock_v3")


def main() -> None:
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("stock_v3.server:app", host=host, port=port, reload=False)
_settings = load_settings()
_cache = Cache(_settings.cache_dir)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_reports_dir = Path("reports")

# Serve generated reports + the AXIS stylesheet as static files.
_reports_dir.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(_reports_dir)), name="reports")
_assets_dir = Path(__file__).parent / "report" / "assets"
app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

# Cache-bust the stylesheet link so theme/material edits show up without a manual hard-reload.
try:
    _CSS_VER = str(int((_assets_dir / "axis.css").stat().st_mtime))
except OSError:
    _CSS_VER = "0"


# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_INDEX_HTML.replace("__CSSVER__", _CSS_VER))


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


@app.post("/portfolio")
async def portfolio(request: Request):
    """Aggregate a portfolio: live prices, per-ticker P&L, totals, and each name's verdict.

    Body: {"lots": [{"ticker": "NVDA", "shares": 10, "buy_price": 180.5}, ...],
           "verdicts": true}
    Returns JSON the frontend renders. Lots for the same ticker are merged (avg cost)."""
    from .sources import prices as prices_src

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {"error": "Invalid JSON body"}

    lots = body.get("lots") or []
    want_verdicts = bool(body.get("verdicts", True))

    # Group lots by ticker → total shares + total cost.
    by_ticker: dict[str, dict] = {}
    for lot in lots:
        sym = re.sub(r"[^A-Za-z0-9.\-]", "", str(lot.get("ticker", ""))).upper()[:10]
        try:
            shares = float(lot.get("shares"))
            buy = float(lot.get("buy_price"))
        except (TypeError, ValueError):
            continue
        if not sym or shares <= 0 or buy < 0:
            continue
        agg = by_ticker.setdefault(sym, {"shares": 0.0, "cost": 0.0, "lots": 0})
        agg["shares"] += shares
        agg["cost"] += shares * buy
        agg["lots"] += 1

    if not by_ticker:
        return {"error": "No valid holdings. Each lot needs ticker, shares and buy price."}

    symbols = list(by_ticker)[:20]
    loop = asyncio.get_event_loop()

    def price_one(sym: str):
        try:
            q = prices_src.fetch_quote(sym, _settings, _cache)
            return (sym, q.value) if q.usable else (sym, None)
        except Exception:  # noqa: BLE001
            return (sym, None)

    def verdict_one(sym: str):
        try:
            rep = run_pipeline(sym, _settings, _cache, use_llm=False)
            return (sym, rep.verdict.label, rep.verdict.conviction_score)
        except Exception:  # noqa: BLE001
            return (sym, None, None)

    quote_results = dict(
        await asyncio.gather(*[loop.run_in_executor(_executor, price_one, s) for s in symbols])
    )
    verdict_results: dict[str, tuple] = {}
    if want_verdicts:
        vlist = await asyncio.gather(
            *[loop.run_in_executor(_executor, verdict_one, s) for s in symbols]
        )
        verdict_results = {v[0]: (v[1], v[2]) for v in vlist}

    positions = []
    total_cost = total_value = 0.0
    for sym in symbols:
        agg = by_ticker[sym]
        quote = quote_results.get(sym)
        shares, cost = agg["shares"], agg["cost"]
        avg_cost = cost / shares if shares else 0.0
        price = quote.price if quote else None
        value = price * shares if price is not None else None
        pnl = (value - cost) if value is not None else None
        pnl_pct = (pnl / cost * 100) if (pnl is not None and cost) else None
        total_cost += cost
        if value is not None:
            total_value += value
        label, conviction = verdict_results.get(sym, (None, None))
        positions.append({
            "ticker": sym,
            "company": quote.company if quote else sym,
            "shares": round(shares, 4),
            "lots": agg["lots"],
            "avg_cost": round(avg_cost, 2),
            "price": round(price, 2) if price is not None else None,
            "cost_basis": round(cost, 2),
            "market_value": round(value, 2) if value is not None else None,
            "pnl": round(pnl, 2) if pnl is not None else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "verdict": label,
            "conviction": conviction,
            "priced": price is not None,
        })

    # sort by market value descending (biggest positions first)
    positions.sort(key=lambda p: p["market_value"] or 0, reverse=True)
    total_pnl = total_value - total_cost
    return {
        "positions": positions,
        "totals": {
            "cost_basis": round(total_cost, 2),
            "market_value": round(total_value, 2),
            "pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl / total_cost * 100, 2) if total_cost else 0.0,
        },
    }


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
<link rel="stylesheet" href="/assets/axis.css?v=__CSSVER__">
<style>
  html, body { height: 100%; }
  body { margin: 0; }
  .app { display: flex; flex-direction: column; height: 100vh; height: 100dvh; }

  /* ---- scroll-reactive frosted top bar ---- */
  .topbar { position: sticky; top: 0; z-index: 30; display: flex; align-items: center;
    justify-content: space-between; gap: 12px; padding: 12px clamp(14px, 4vw, 24px);
    padding-top: max(12px, env(safe-area-inset-top)); flex-shrink: 0; }
  .topbar .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .topbar .brand .mark { width: 30px; height: 30px; border-radius: 8px; background: var(--accent);
    display: grid; place-items: center; color: var(--accent-text); font-weight: 700; font-size: 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3); flex-shrink: 0; }
  .topbar .title { min-width: 0; }
  .topbar .title h1 { font-size: var(--fs-h3); font-weight: 700; letter-spacing: -0.01em;
    line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .topbar .title .sub { font-size: var(--fs-micro); color: var(--text-tertiary); margin-top: 1px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .topbar .actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .icon-btn { width: 38px; height: 38px; border-radius: var(--radius-pill); border: 1px solid var(--border-default);
    background: var(--glass-inset, var(--bg-inset)); color: var(--text-secondary); cursor: pointer;
    display: grid; place-items: center; font-size: 15px; -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
    transition: all var(--dur-fast); }
  .icon-btn:hover { color: var(--text-primary); border-color: var(--border-strong); }
  .back-btn { width: auto; padding: 0 14px; gap: 6px; font-family: var(--font-sans);
    font-size: var(--fs-sm); font-weight: 600; }

  /* ---- scrollable workspace ---- */
  .scroll { flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch; min-height: 0;
    padding: 8px clamp(14px, 4vw, 24px) 96px; }
  .view { max-width: 760px; margin: 0 auto; display: none; }
  .view.active { display: block; }

  .panel { padding: clamp(18px, 5vw, 26px); }
  .lede { color: var(--text-secondary); font-size: var(--fs-sm); margin-bottom: 18px;
    line-height: 1.5; }
  .field { display: flex; gap: 8px; align-items: stretch; }
  .field .input { flex: 1; height: 48px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
  .field .input::placeholder { text-transform: none; font-weight: 400; letter-spacing: 0; }
  .field .btn { height: 48px; }
  .hint { color: var(--text-tertiary); font-size: var(--fs-xs); margin-top: 9px; }
  .examples { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 16px; }
  .chip { height: 32px; padding: 0 13px; border-radius: var(--radius-pill);
    background: var(--bg-inset); border: 1px solid var(--border-default); color: var(--text-secondary);
    font-family: var(--font-mono); font-size: var(--fs-xs); cursor: pointer; display: inline-flex;
    align-items: center; -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); transition: all var(--dur-fast); }
  .chip:hover { border-color: var(--border-strong); color: var(--text-primary); }
  .opt { margin-top: 16px; display: flex; align-items: center; gap: 9px; font-size: var(--fs-sm);
    color: var(--text-secondary); }
  .opt input { accent-color: var(--accent); width: 17px; height: 17px; cursor: pointer; }
  .opt code { font-family: var(--font-mono); font-size: var(--fs-xs); color: var(--text-tertiary); }

  #log { margin-top: 20px; display: none; }
  #log-list { list-style: none; padding: 0; margin: 0; }
  #log-list li { padding: 8px 0; border-bottom: 1px solid var(--border-subtle); font-size: var(--fs-sm);
    color: var(--text-secondary); display: flex; gap: 11px; align-items: baseline; }
  #log-list li:last-child { border: none; }
  #log-list li.final { color: var(--positive); font-weight: 600; }
  #log-list li.err { color: var(--critical); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
  .dot.spin { background: var(--accent); animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .25; } }

  /* result iframe (single & comparison) */
  #result { display: none; }
  #result.show { display: block; }
  iframe { width: 100%; border: 1px solid var(--border-default); border-radius: var(--radius-lg);
    background: var(--bg-base); height: calc(100dvh - 150px); min-height: 480px; }

  /* ---- portfolio ---- */
  .lot-form { display: grid; grid-template-columns: 1.4fr 1fr 1fr auto; gap: 8px; align-items: stretch; }
  .lot-form .input { height: 44px; font-size: var(--fs-sm); }
  .lot-form .input.tk { text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em; }
  .lot-form .add { height: 44px; width: 44px; padding: 0; font-size: 22px; }
  .lot-list { margin-top: 14px; display: flex; flex-direction: column; gap: 7px; }
  .lot { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: var(--radius-md);
    background: var(--bg-inset); border: 1px solid var(--border-subtle); font-size: var(--fs-sm); }
  .lot .lt-tk { font-weight: 700; font-family: var(--font-mono); min-width: 56px; }
  .lot .lt-detail { color: var(--text-secondary); font-family: var(--font-mono); flex: 1; }
  .lot .lt-x { margin-left: auto; cursor: pointer; color: var(--text-tertiary); width: 26px; height: 26px;
    border-radius: 50%; display: grid; place-items: center; border: none; background: transparent; font-size: 16px; }
  .lot .lt-x:hover { background: var(--critical-soft); color: var(--critical); }
  .empty-note { color: var(--text-tertiary); font-size: var(--fs-sm); padding: 14px 2px; }

  .pf-total { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border-subtle);
    border-radius: var(--radius-md); overflow: hidden; margin-bottom: 12px; }
  .pf-total .cell { background: var(--bg-elevated); padding: 12px 14px; min-width: 0; }
  .pf-total .cell.wide { grid-column: 1 / -1; }
  .pf-total .k { color: var(--text-tertiary); font-size: var(--fs-micro); text-transform: uppercase; letter-spacing: 0.04em; }
  .pf-total .v { font-size: var(--fs-body); font-weight: 700; font-family: var(--font-mono);
    font-variant-numeric: tabular-nums; margin-top: 3px; white-space: nowrap; }
  .pos { display: flex; flex-direction: column; gap: 8px; }
  .pos-card { padding: 14px; }
  .pos-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
  .pos-head .tk { font-weight: 700; font-size: var(--fs-h3); }
  .pos-head .co { color: var(--text-tertiary); font-size: var(--fs-xs); }
  .pos-pnl { font-family: var(--font-mono); font-weight: 700; font-size: var(--fs-body); text-align: right; }
  .pos-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 12px; margin-top: 11px; }
  .pos-grid .k { color: var(--text-tertiary); font-size: var(--fs-micro); text-transform: uppercase; letter-spacing: 0.03em; }
  .pos-grid .v { font-family: var(--font-mono); font-size: var(--fs-sm); font-weight: 600; margin-top: 1px; }
  .pos-foot { margin-top: 11px; padding-top: 10px; border-top: 1px solid var(--border-subtle);
    display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .pos-foot .verdict { font-size: var(--fs-xs); }
  .gain { color: var(--positive); } .loss { color: var(--critical); }

  /* ---- frosted bottom tab bar ---- */
  .tabbar { position: fixed; left: 0; right: 0; bottom: 0; z-index: 40; display: flex;
    padding: 6px 8px; padding-bottom: max(6px, env(safe-area-inset-bottom)); }
  .tab { flex: 1; background: transparent; border: none; cursor: pointer; padding: 7px 4px;
    display: flex; flex-direction: column; align-items: center; gap: 3px; color: var(--text-tertiary);
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 600; border-radius: var(--radius-md);
    transition: color var(--dur-fast); -webkit-tap-highlight-color: transparent; }
  .tab .ic { width: 22px; height: 22px; }
  .tab.active { color: var(--accent-hover); }
  .tab:active { transform: scale(0.94); }

  /* desktop: center content, tabs become a slimmer floating bar */
  @media (min-width: 820px) {
    .scroll { padding-bottom: 100px; }
    .tabbar { left: 50%; transform: translateX(-50%); bottom: 18px; max-width: 460px;
      border-radius: var(--radius-pill); padding: 6px; }
    .tab { border-radius: var(--radius-pill); }
  }
</style>
</head>
<body class="glass">
<div class="app">

  <!-- TOP BAR -->
  <header class="topbar" id="topbar">
    <div class="brand">
      <div class="mark">A</div>
      <div class="title">
        <h1 id="tb-title">Single Report</h1>
        <div class="sub" id="tb-sub">Institutional research, one ticker</div>
      </div>
    </div>
    <div class="actions">
      <a id="frame-link" class="icon-btn" href="#" target="_blank" title="Open in new tab" style="display:none">&#8599;</a>
      <button id="back-btn" class="icon-btn back-btn" onclick="goBack()" style="display:none">&#8592; Back</button>
      <button class="icon-btn" onclick="toggleTheme()" title="Toggle theme" id="theme-btn">&#9790;</button>
    </div>
  </header>

  <!-- SCROLLABLE WORKSPACE -->
  <main class="scroll" id="scroll">

    <!-- SINGLE / COMPARISON share the input + result machinery -->
    <section class="view active" id="view-report">
      <div id="panel-view">
        <div class="axis-card panel">
          <p class="lede" id="lede"></p>
          <div class="field">
            <input id="ticker" class="input" type="text" placeholder="e.g. NVDA" maxlength="60" autocomplete="off">
            <button id="btn" class="btn btn-primary" onclick="run()">Analyze</button>
          </div>
          <div class="hint" id="hint"></div>
          <div class="examples" id="examples"></div>
          <div class="opt">
            <input type="checkbox" id="narrative">
            <label for="narrative">Enrich prose with Claude <code>(ANTHROPIC_API_KEY)</code></label>
          </div>
          <div id="log"><ul id="log-list"></ul></div>
        </div>
      </div>
      <div id="result"><iframe id="frame" src="about:blank" title="Result"></iframe></div>
    </section>

    <!-- PORTFOLIO CONSOLIDATION -->
    <section class="view" id="view-portfolio">
      <div class="axis-card panel">
        <p class="lede">Add your holdings — multiple lots per ticker are merged. We pull live prices and
          aggregate cost basis, market value and unrealised P&amp;L, with each name's current verdict.</p>
        <div class="lot-form">
          <input id="pf-tk" class="input tk" placeholder="Ticker" maxlength="10" autocomplete="off">
          <input id="pf-sh" class="input" type="number" placeholder="Shares" min="0" step="any" inputmode="decimal">
          <input id="pf-bp" class="input" type="number" placeholder="Buy $" min="0" step="any" inputmode="decimal">
          <button class="btn btn-primary add" onclick="addLot()" title="Add lot">+</button>
        </div>
        <div class="lot-list" id="lot-list"></div>
        <button id="pf-run" class="btn btn-primary" style="margin-top:16px;width:100%" onclick="runPortfolio()">Consolidate portfolio</button>
        <div id="pf-status" class="hint" style="text-align:center"></div>
      </div>
      <div id="pf-result" style="margin-top:12px"></div>
    </section>

  </main>

  <!-- BOTTOM TAB BAR -->
  <nav class="tabbar">
    <button class="tab active" id="tab-one" onclick="setMode('one')">
      <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
      Single
    </button>
    <button class="tab" id="tab-cmp" onclick="setMode('cmp')">
      <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="7" height="16" rx="1"/><rect x="14" y="4" width="7" height="16" rx="1"/></svg>
      Comparison
    </button>
    <button class="tab" id="tab-pf" onclick="setMode('pf')">
      <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-6 9 6v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z"/></svg>
      Portfolio
    </button>
  </nav>
</div>

<script>
let mode = 'one';
const SINGLE = ['NVDA','AAPL','MSFT','PLTR'];
const COMPARE = ['NVDA, AMD, AVGO', 'AAPL, MSFT, GOOGL', 'PLTR, SNOW, NET'];

// ---- theme ----
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  var b = document.getElementById('theme-btn'); if (b) b.innerHTML = t === 'light' ? '&#9728;' : '&#9790;';
  themeFrame();
}
function toggleTheme() {
  var n = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  localStorage.setItem('stockv3-theme', n); applyTheme(n);
}
function themeFrame() {
  var t = document.documentElement.getAttribute('data-theme'), f = document.getElementById('frame');
  try { if (f && f.contentDocument) f.contentDocument.documentElement.setAttribute('data-theme', t); } catch(e){}
}
applyTheme(localStorage.getItem('stockv3-theme') || 'dark');

// ---- scroll-reactive top bar ----
document.getElementById('scroll').addEventListener('scroll', function (e) {
  document.getElementById('topbar').classList.toggle('scrolled', e.target.scrollTop > 6);
}, { passive: true });

// ---- mode switching (bottom tabs) ----
function setMode(m) {
  mode = m;
  document.getElementById('tab-one').classList.toggle('active', m === 'one');
  document.getElementById('tab-cmp').classList.toggle('active', m === 'cmp');
  document.getElementById('tab-pf').classList.toggle('active', m === 'pf');
  document.getElementById('view-report').classList.toggle('active', m === 'one' || m === 'cmp');
  document.getElementById('view-portfolio').classList.toggle('active', m === 'pf');
  var tt = document.getElementById('tb-title'), ts = document.getElementById('tb-sub');
  if (m === 'pf') {
    tt.textContent = 'Portfolio Consolidation'; ts.textContent = 'Holdings, P&L and verdicts';
  } else if (m === 'cmp') {
    tt.textContent = 'Comparison Report'; ts.textContent = 'Side-by-side, leader marked';
    document.getElementById('ticker').placeholder = 'e.g. NVDA, AMD, AVGO';
    document.getElementById('btn').textContent = 'Compare';
    document.getElementById('lede').textContent = 'Run several names side by side — verdict, conviction, risk/reward and key metrics, with the leader in each row marked.';
    document.getElementById('hint').textContent = 'Enter 2–6 symbols separated by commas.';
    renderExamples(); goBack();
  } else {
    tt.textContent = 'Single Report'; ts.textContent = 'Institutional research, one ticker';
    document.getElementById('ticker').placeholder = 'e.g. NVDA';
    document.getElementById('btn').textContent = 'Analyze';
    document.getElementById('lede').textContent = 'Generate an institutional research report from live free-data — verdict, scenario cone, probability calculator, scorecard and trade plan up front; the full data a tap away.';
    document.getElementById('hint').textContent = 'Enter one symbol, then press Enter.';
    renderExamples(); goBack();
  }
  document.getElementById('scroll').scrollTop = 0;
}
function renderExamples() {
  var wrap = document.getElementById('examples'); wrap.innerHTML = '';
  (mode === 'cmp' ? COMPARE : SINGLE).forEach(function (ex) {
    var c = document.createElement('button'); c.className = 'chip'; c.textContent = ex;
    c.onclick = function () { document.getElementById('ticker').value = ex; document.getElementById('ticker').focus(); };
    wrap.appendChild(c);
  });
}

// ---- single / comparison result ----
function showResult(url) {
  document.getElementById('panel-view').style.display = 'none';
  document.getElementById('result').classList.add('show');
  document.getElementById('back-btn').style.display = '';
  var link = document.getElementById('frame-link'); link.href = url; link.style.display = '';
}
function goBack() {
  document.getElementById('result').classList.remove('show');
  document.getElementById('panel-view').style.display = '';
  document.getElementById('back-btn').style.display = 'none';
  document.getElementById('frame-link').style.display = 'none';
  document.getElementById('frame').src = 'about:blank';
  document.getElementById('log').style.display = 'none';
}
function run() {
  var raw = document.getElementById('ticker').value.trim();
  if (!raw) { document.getElementById('ticker').focus(); return; }
  var narrative = document.getElementById('narrative').checked;
  return mode === 'cmp' ? runCompare(raw, narrative) : runSingle(raw, narrative);
}
function runCompare(raw, narrative) {
  var btn = document.getElementById('btn'), frame = document.getElementById('frame');
  btn.disabled = true; btn.textContent = 'Running…';
  var url = '/compare?tickers=' + encodeURIComponent(raw) + '&narrative=' + narrative;
  frame.onload = function () { btn.disabled = false; btn.textContent = 'Compare'; themeFrame(); };
  frame.src = url; showResult(url);
}
function runSingle(ticker, narrative) {
  ticker = ticker.split(/[,\\s]+/)[0].toUpperCase();
  var btn = document.getElementById('btn'), log = document.getElementById('log'), list = document.getElementById('log-list');
  btn.disabled = true; log.style.display = 'block'; list.innerHTML = ''; var lastLi = null;
  function addLine(text, cls) {
    var li = document.createElement('li'); if (cls) li.className = cls;
    li.innerHTML = '<span class="dot' + (cls ? '' : ' spin') + '"></span><span>' + text + '</span>';
    list.appendChild(li);
    if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
    lastLi = li; li.scrollIntoView({behavior:'smooth', block:'nearest'}); return li;
  }
  var es = new EventSource('/analyze/stream?ticker=' + encodeURIComponent(ticker) + '&narrative=' + narrative);
  es.onmessage = function (e) {
    var msg = JSON.parse(e.data);
    if (msg.startsWith('__done__:')) {
      es.close(); var path = msg.slice(9);
      if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
      addLine('Report ready.', 'final'); btn.disabled = false;
      var f = document.getElementById('frame'); f.onload = themeFrame; f.src = path; showResult(path);
    } else if (msg.startsWith('__error__:')) {
      es.close(); if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
      addLine(msg.slice(10), 'err'); btn.disabled = false;
    } else { addLine(msg); }
  };
  es.onerror = function () { es.close();
    if (lastLi) lastLi.querySelector('.dot').classList.remove('spin');
    addLine('Connection error — check the server.', 'err'); btn.disabled = false; };
}

// ---- portfolio ----
var lots = JSON.parse(localStorage.getItem('stockv3-lots') || '[]');
function saveLots() { localStorage.setItem('stockv3-lots', JSON.stringify(lots)); }
function addLot() {
  var tk = document.getElementById('pf-tk').value.trim().toUpperCase().replace(/[^A-Z0-9.\\-]/g,'');
  var sh = parseFloat(document.getElementById('pf-sh').value);
  var bp = parseFloat(document.getElementById('pf-bp').value);
  if (!tk || !(sh > 0) || !(bp >= 0)) { document.getElementById('pf-status').textContent = 'Enter ticker, shares and buy price.'; return; }
  lots.push({ ticker: tk, shares: sh, buy_price: bp }); saveLots(); renderLots();
  document.getElementById('pf-tk').value=''; document.getElementById('pf-sh').value=''; document.getElementById('pf-bp').value='';
  document.getElementById('pf-tk').focus(); document.getElementById('pf-status').textContent='';
}
function removeLot(i) { lots.splice(i,1); saveLots(); renderLots(); }
function renderLots() {
  var el = document.getElementById('lot-list');
  if (!lots.length) { el.innerHTML = '<div class="empty-note">No holdings yet. Add a lot above.</div>'; return; }
  el.innerHTML = '';
  lots.forEach(function (l, i) {
    var d = document.createElement('div'); d.className = 'lot';
    d.innerHTML = '<span class="lt-tk">' + l.ticker + '</span>' +
      '<span class="lt-detail">' + l.shares + ' sh @ $' + l.buy_price + '</span>' +
      '<button class="lt-x" title="Remove">&times;</button>';
    d.querySelector('.lt-x').onclick = function () { removeLot(i); };
    el.appendChild(d);
  });
}
function money(v) { return v == null ? '—' : '$' + Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
function vClass(label) {
  if (!label) return 'v-hold';
  return 'v-' + label.toLowerCase().replace(/[^a-z]/g,'');
}
function runPortfolio() {
  if (!lots.length) { document.getElementById('pf-status').textContent = 'Add at least one holding.'; return; }
  var btn = document.getElementById('pf-run'); btn.disabled = true; btn.textContent = 'Consolidating…';
  document.getElementById('pf-status').textContent = 'Pulling live prices and verdicts…';
  fetch('/portfolio', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ lots: lots, verdicts: true }) })
  .then(function (r) { return r.json(); })
  .then(function (d) {
    btn.disabled = false; btn.textContent = 'Consolidate portfolio'; document.getElementById('pf-status').textContent = '';
    if (d.error) { document.getElementById('pf-status').textContent = d.error; return; }
    renderPortfolio(d);
  })
  .catch(function () { btn.disabled = false; btn.textContent = 'Consolidate portfolio';
    document.getElementById('pf-status').textContent = 'Request failed.'; });
}
function renderPortfolio(d) {
  var t = d.totals, out = '';
  var tcls = t.pnl >= 0 ? 'gain' : 'loss', sign = t.pnl >= 0 ? '+' : '';
  out += '<div class="axis-card panel" style="padding:16px">' +
    '<div class="pf-total">' +
      '<div class="cell"><div class="k">Cost basis</div><div class="v">' + money(t.cost_basis) + '</div></div>' +
      '<div class="cell"><div class="k">Market value</div><div class="v">' + money(t.market_value) + '</div></div>' +
      '<div class="cell wide"><div class="k">Unrealised P&L</div><div class="v ' + tcls + '">' +
        sign + money(t.pnl) + '  ·  ' + sign + t.pnl_pct.toFixed(2) + '%</div></div>' +
    '</div>' +
    '</div>';
  out += '<div class="pos" style="margin-top:10px">';
  d.positions.forEach(function (p) {
    var cls = (p.pnl != null && p.pnl >= 0) ? 'gain' : 'loss';
    var sg = (p.pnl != null && p.pnl >= 0) ? '+' : '';
    out += '<div class="axis-card pos-card card-hover">' +
      '<div class="pos-head"><div><span class="tk">' + p.ticker + '</span> <span class="co">' + (p.company||'') + '</span></div>' +
        '<div class="pos-pnl ' + cls + '">' + (p.pnl!=null ? sg+money(p.pnl)+' ('+sg+p.pnl_pct.toFixed(1)+'%)' : '—') + '</div></div>' +
      '<div class="pos-grid">' +
        '<div><div class="k">Shares</div><div class="v">' + p.shares + (p.lots>1?' · '+p.lots+' lots':'') + '</div></div>' +
        '<div><div class="k">Avg cost</div><div class="v">' + money(p.avg_cost) + '</div></div>' +
        '<div><div class="k">Price</div><div class="v">' + money(p.price) + '</div></div>' +
        '<div><div class="k">Cost basis</div><div class="v">' + money(p.cost_basis) + '</div></div>' +
        '<div><div class="k">Mkt value</div><div class="v">' + money(p.market_value) + '</div></div>' +
        '<div><div class="k">Weight</div><div class="v">' + (p.market_value && t.market_value ? (p.market_value/t.market_value*100).toFixed(1)+'%' : '—') + '</div></div>' +
      '</div>' +
      (p.verdict ? '<div class="pos-foot"><span class="verdict ' + vClass(p.verdict) + '">' + p.verdict + '</span>' +
        '<span style="font-family:var(--font-mono);font-size:var(--fs-xs);color:var(--text-tertiary)">Conviction ' + p.conviction + '/100</span></div>' : '') +
      '</div>';
  });
  out += '</div>';
  document.getElementById('pf-result').innerHTML = out;
}

document.getElementById('ticker').addEventListener('keydown', function (e) { if (e.key === 'Enter') run(); });
document.getElementById('pf-bp').addEventListener('keydown', function (e) { if (e.key === 'Enter') addLot(); });
setMode('one'); renderLots();

// liquid-glass ambient drift on visible panels
(function () {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  document.body.classList.add('anim');
  document.querySelectorAll('.view.active .panel.axis-card').forEach(function (panel) {
    if (!panel.querySelector(':scope > .liquid')) {
      var l = document.createElement('i'); l.className = 'liquid'; panel.insertBefore(l, panel.firstChild);
    }
  });
})();
</script>
</body>
</html>
"""
