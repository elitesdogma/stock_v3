# stock_v3

Institutional-grade equity research, on the command line or in the browser. `analyze NVDA` pulls live free-data, scores the name across seven categories, and writes a self-contained HTML report — verdict, scenarios, an interactive probability cone, trade plan, and a coverage panel that states exactly what the free data could and couldn't deliver. `uv run serve` gives the same thing a web UI, plus side-by-side multi-ticker comparison.

The engine is **deterministic-first**: every score, probability, ratio, and the final Buy/Hold/Avoid call comes from auditable rules and reproduces run-to-run. An optional `--narrative` flag layers Claude-written prose on top of those numbers without ever changing them.

The interface uses the **AXIS** design system — navy + muted-emerald, Hanken Grotesk with IBM Plex Mono for every figure, dark-first with a light theme. Reports are responsive down to iPhone width and print cleanly to PDF.

```
analyze NVDA
  → reports/NVDA-20260603-0034.report.html
  Verdict:    Accumulate (78/100)
  Confidence: 72/100 — Moderate Conviction
  Risk/Reward: 3.0:1
```

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (uv fetches it automatically).

```bash
uv sync
cp .env.example .env   # then add your free keys (see below)
uv run analyze NVDA
```

## API keys

The tool runs with **zero keys** — sources without a key degrade to `unavailable` and their scorecard category is neutralized, not faked. Add keys to unlock full coverage. All three are free.

| Key | Unlocks | Get it |
|---|---|---|
| `FRED_API_KEY` | Macro layer (yields, curve, CPI/PPI, VIX, dollar) | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) — instant |
| `FINNHUB_API_KEY` | Analyst consensus, price targets, earnings calendar | [finnhub.io](https://finnhub.io/) — free tier, 60 calls/min |
| `SEC_USER_AGENT` | Insider (Form 4) activity via EDGAR | No signup — just set a descriptive `"name email"` string |
| `ANTHROPIC_API_KEY` | `--narrative` prose enrichment (optional) | [console.anthropic.com](https://console.anthropic.com/) |

Price, fundamentals, OHLCV/technicals, institutional holders, the options gamma proxy, and short interest all come from Yahoo and need no key.

## Usage

### CLI

```bash
uv run analyze NVDA                 # full report, opens in browser
uv run analyze AAPL --no-open       # write only
uv run analyze PLTR --narrative     # add Claude prose (falls back to templates w/o key)
uv run analyze F --out ~/research   # custom output dir
uv run analyze TSLA --no-cache      # bypass the on-disk cache, hit live APIs
```

### Web UI

```bash
uv run serve            # → http://127.0.0.1:8080
```

Type a symbol for a single report (progress streams live, then the report loads inline), or switch to **Compare** mode and enter 2–6 tickers for a side-by-side matrix. Direct routes:

- `GET /` — landing page (single report or compare)
- `GET /compare?tickers=NVDA,AMD,AAPL` — comparison matrix, leader in each metric marked

Open any report's HTML and print-to-PDF for a shareable deliverable. Reports are self-contained — inline SVG charts and one small inline script for the probability slider, no CDN — so they render offline (web fonts degrade to system fonts).

## What it produces

The 15-section institutional structure: executive summary, macro, positioning, fundamentals, valuation, technicals, catalysts, key risks, scenarios, portfolio impact, position sizing, the seven-category scorecard, and the final verdict — plus a trade plan with entry zone, ATR-anchored stop, laddered targets, reward-to-risk, and Kelly-capped allocation.

**Probability cone + calculator.** The scenario section renders a volatility cone (the thinkorswim/Schwab standard: ±1σ ≈ 68%, ±2σ ≈ 95%) projecting the price distribution over 2 / 4 / 8 / 13 weeks, with the bull/base/bear targets overlaid. An interactive calculator lets you pick a horizon and drag a target price to read `P(at or above)` / `P(at or below)` live — computed from a zero-drift lognormal model using implied volatility (ATM options) where available, realized 1-year volatility otherwise. It maps dispersion, not direction.

**Multi-ticker comparison.** A succinct matrix distilling each full report to its decision-grade row — verdict, conviction, risk/reward, growth, margins, valuation, scenario skew — with the best cell in each metric highlighted (directional: higher conviction wins, lower P/E wins).

## Data limitations (read this)

This is a free-data tool. It is honest about the gaps rather than papering over them — the report's coverage panel timestamps every source and flags every N/A.

- **Prices and quotes are delayed.** Yahoo is ~15–20 min behind; treat the report as end-of-day, not real-time.
- **13F ownership lags ~45 days** by SEC regulation. Always labeled with its vintage.
- **Short interest is the bi-monthly FINRA settlement**, up to two weeks stale. **Borrow rates are not available on free data** — marked N/A.
- **Gamma exposure is an estimate, not authoritative GEX.** It's derived from the option chain under the standard "dealers short puts / long calls" assumption and is labeled as a proxy throughout. Don't trade off it as if it were a paid dealer-positioning feed.
- **Market breadth** (% of index above its 200-day) has no clean free source and is left out of the macro layer rather than approximated badly.
- **yfinance is scraping-based and rate-limited.** The tool caches aggressively and retries, but a Yahoo-side block will surface as `unavailable` for the affected source. Re-run; the cache fills in.

Upgrading any source to a paid provider (Polygon, FMP, a real GEX feed) is a drop-in change behind the `SourceResult` contract in `src/stock_v3/sources/`.

## Not financial advice

Structured research output. Not advice, not a recommendation, not a solicitation. Verify every figure independently before acting.

## Development

```bash
uv run pytest          # 28 tests: scorecard bands, probability sums, R:R math,
                       # neutralization, indicator math, offline render
```

Architecture: `sources/` (each free API → a `SourceResult` carrying ok/stale/unavailable + as-of) → `analysis/` (pure deterministic classifiers and the scorecard/verdict math) → `report/` (inline-SVG charts + Jinja2 → one HTML file). `pipeline.py` wires it; `cli.py` is the thin entry point. The narrative LLM layer in `narrative/llm.py` is strictly additive.
