"""Render a ResearchReport into a single self-contained HTML file.

All formatting helpers live here (the template stays logic-light). Charts are generated
as inline SVG and passed through; nothing external is referenced, so the file is portable.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ResearchReport
from . import charts

_TEMPLATES = Path(__file__).parent / "templates"
_AXIS_CSS = (Path(__file__).parent / "assets" / "axis.css").read_text(encoding="utf-8")

_VERDICT_CLASS = {
    "Strong Buy": "v-strongbuy",
    "Buy": "v-buy",
    "Accumulate": "v-accumulate",
    "Watchlist": "v-watchlist",
    "Hold": "v-hold",
    "Avoid": "v-avoid",
}


def render_report(r: ResearchReport) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    context = {
        "r": r,
        "axis_css": _AXIS_CSS,
        "verdict_class": _VERDICT_CLASS.get(r.verdict.label, "v-hold"),
        "narrative_source": r.narrative.generated_by,
        # charts
        "gauge_svg": charts.gauge(r.verdict.conviction_score),
        "price_chart_svg": _price_chart(r),
        "scorecard_svg": charts.scorecard_bars(
            [(c.name, c.score, c.neutralized) for c in r.scorecard]
        ),
        "scenario_svg": charts.scenario_fan(
            r.quote.price, [(s.name, s.price_target, s.probability) for s in r.scenarios]
        ),
        "cone_svg": _cone_svg(r),
        "cone_widget": _cone_widget(r),
        "has_cone": r.probability_cone is not None,
        # derived display strings
        "headline_stats": _headline_stats(r),
        "trade_plan_block": _trade_plan_block(r),
        # rationales pulled from scorecard rows so prose and scores stay consistent
        "macro_rationale": _rationale(r, "Macro Environment"),
        "fundamentals_rationale": _rationale(r, "Fundamentals"),
        "valuation_rationale": _rationale(r, "Valuation"),
        "technical_rationale": _rationale(r, "Technicals"),
        "positioning_stance": _positioning_stance(r),
        "insider_signal": _insider_signal(r),
        "technical_trend": _technical_trend(r),
        # formatters
        "fmt_money": _fmt_money,
        "fmt_pct": _fmt_pct,
        "fmt_pct_raw": _fmt_pct_raw,
        "fmt_n": _fmt_n,
        "fmt_big": _fmt_big,
        "fmt_big_signed": _fmt_big_signed,
    }
    return template.render(**context)


def write_report(r: ResearchReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = r.generated_at.strftime("%Y%m%d-%H%M")
    path = out_dir / f"{r.quote.ticker}-{stamp}.report.html"
    path.write_text(render_report(r), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
def _cone_svg(r: ResearchReport) -> str:
    cone = r.probability_cone
    if cone is None:
        return ""
    return charts.probability_cone(
        cone.spot,
        cone.bands,
        [(s.name, s.price_target, s.probability) for s in r.scenarios],
    )


def _cone_widget(r: ResearchReport) -> str:
    """Interactive probability calculator: pick a horizon, drag a price, see P(≥) live.

    Self-contained — the lognormal math is reimplemented in a small inline script seeded with
    spot + annual vol, so the report needs no network and works offline. This is the only JS
    in the report and degrades gracefully (the static cone chart already conveys the ranges)."""
    cone = r.probability_cone
    if cone is None:
        return ""

    horizons = [(b.weeks, b.days) for b in cone.bands]
    # Slider price range: ±2σ of the longest horizon, generous enough to explore.
    last = cone.bands[-1]
    half = (last.high_1sd - last.low_1sd) / 2
    lo = round(max(0.01, last.p50 - 2.4 * half), 2)
    hi = round(last.p50 + 2.4 * half, 2)
    step = round((hi - lo) / 200, 2) or 0.01
    default_price = round(cone.spot * 1.05, 2)

    params = json.dumps({
        "spot": cone.spot,
        "vol": cone.annual_vol,
        "volSource": cone.vol_source,
        "horizons": horizons,
        "lo": lo, "hi": hi, "step": step, "default": default_price,
    })

    return f"""
<div class="cone-calc" data-cone='{escape(params, {"'": "&#39;"})}'>
  <div class="cone-calc-head">
    <div class="eyebrow">Probability calculator</div>
    <div class="cone-vol mono"></div>
  </div>
  <div class="cone-horizons" role="tablist"></div>
  <div class="cone-readout">
    <div class="cone-target">
      <span class="cone-label">Target price</span>
      <span class="cone-price mono"></span>
    </div>
    <input type="range" class="axis-range cone-slider" min="{lo}" max="{hi}" step="{step}"
           value="{default_price}" aria-label="Target price">
    <div class="cone-probs">
      <div class="cone-prob up">
        <div class="cone-prob-val mono"></div>
        <div class="cone-prob-lbl">P(at or above)</div>
      </div>
      <div class="cone-prob down">
        <div class="cone-prob-val mono"></div>
        <div class="cone-prob-lbl">P(at or below)</div>
      </div>
      <div class="cone-prob move">
        <div class="cone-prob-val mono"></div>
        <div class="cone-prob-lbl">Move from spot</div>
      </div>
    </div>
  </div>
</div>
<script>
(function() {{
  const root = document.currentScript.previousElementSibling;
  const cfg = JSON.parse(root.getAttribute('data-cone'));
  let horizonDays = cfg.horizons[1] ? cfg.horizons[1][1] : cfg.horizons[0][1];

  function normCdf(x) {{
    // Abramowitz-Stegun erf approximation.
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }}
  function probAbove(target) {{
    const t = horizonDays / 365;
    const sig = cfg.vol * Math.sqrt(t);
    const d = (Math.log(cfg.spot / target) - 0.5 * sig * sig) / sig;
    return normCdf(d);
  }}

  const slider = root.querySelector('.cone-slider');
  const priceEl = root.querySelector('.cone-price');
  const upEl = root.querySelector('.cone-prob.up .cone-prob-val');
  const downEl = root.querySelector('.cone-prob.down .cone-prob-val');
  const moveEl = root.querySelector('.cone-prob.move .cone-prob-val');
  const volEl = root.querySelector('.cone-vol');
  volEl.textContent = (cfg.vol * 100).toFixed(0) + '% vol · ' + cfg.volSource;

  const hwrap = root.querySelector('.cone-horizons');
  cfg.horizons.forEach((h, i) => {{
    const b = document.createElement('button');
    b.className = 'cone-htab' + (h[1] === horizonDays ? ' active' : '');
    b.textContent = h[0] + ' wk';
    b.onclick = () => {{
      horizonDays = h[1];
      hwrap.querySelectorAll('.cone-htab').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      update();
    }};
    hwrap.appendChild(b);
  }});

  function fmt(v) {{ return v.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}); }}
  function update() {{
    const target = parseFloat(slider.value);
    const pUp = probAbove(target);
    priceEl.textContent = '$' + fmt(target);
    upEl.textContent = (pUp * 100).toFixed(0) + '%';
    downEl.textContent = ((1 - pUp) * 100).toFixed(0) + '%';
    const mv = (target / cfg.spot - 1) * 100;
    moveEl.textContent = (mv >= 0 ? '+' : '') + mv.toFixed(1) + '%';
    moveEl.className = 'cone-prob-val mono ' + (mv >= 0 ? 'delta-up' : 'delta-down');
  }}
  slider.addEventListener('input', update);
  update();
}})();
</script>"""


def _price_chart(r: ResearchReport) -> str:
    if r.technicals is None or not r.technicals.price_history:
        return ""
    return charts.price_chart(
        r.technicals.price_history,
        sma_50=r.technicals.sma_50,
        sma_200=r.technicals.sma_200,
    )


def _headline_stats(r: ResearchReport) -> list[tuple[str, str]]:
    stats: list[tuple[str, str]] = [
        ("Conviction", f"{r.verdict.conviction_score:.0f}/100"),
        ("Verdict", r.verdict.label),
    ]
    if r.trade_plan.risk_reward is not None:
        stats.append(("Risk/Reward", f"{r.trade_plan.risk_reward:.1f}:1"))
    if r.trade_plan.stop_loss is not None:
        stats.append(("Stop", _fmt_money(r.trade_plan.stop_loss)))
    if r.consensus and r.consensus.target_mean:
        upside = (r.consensus.target_mean / r.quote.price - 1) * 100
        stats.append(("Consensus", f"{_fmt_money(r.consensus.target_mean)} ({upside:+.0f}%)"))
    if r.fundamentals and r.fundamentals.forward_pe:
        stats.append(("Fwd P/E", f"{r.fundamentals.forward_pe:.1f}"))
    return stats


def _trade_plan_block(r: ResearchReport) -> str:
    """The monospace trade plan, matching the framework's template. HTML-escaped values
    with light span coloring; returned as safe markup."""
    p = r.trade_plan
    q = r.quote

    def row(key: str, value: str, accent: bool = False) -> str:
        cls = "pv accent" if accent else "pv"
        # padded key for the desktop `pre` alignment; the wrapper div lets mobile flex it.
        return (f'<div class="plan-row"><span class="pk">{escape(key):<22}</span>'
                f'<span class="{cls}">{escape(value)}</span></div>')

    entry = f"{p.entry_zone[0]} – {p.entry_zone[1]}" if p.entry_zone else "—"
    targets = " / ".join(_fmt_money(t) for t in p.targets) if p.targets else "—"
    alloc = f"{p.portfolio_allocation_pct:.1f}%" if p.portfolio_allocation_pct else "—"
    size = f"{p.position_size_shares:.0f} sh (per $100k, 1% risk)" if p.position_size_shares else "—"
    rr = f"{p.risk_reward:.1f} : 1" if p.risk_reward else "—"
    catalyst = r.catalysts[0].label if r.catalysts else "none in window"

    lines = [
        row("Ticker", f"{q.ticker} — {q.company}", accent=True),
        row("Current price", _fmt_money(q.price)),
        row("Sector", f"{q.sector or '—'} / {q.industry or '—'}"),
        '<div class="plan-gap"></div>',
        row("Entry zone", entry),
        row("Ideal buy", _fmt_money(p.ideal_buy)),
        row("Stop loss", _fmt_money(p.stop_loss)),
        row("Targets (T1/T2/T3)", targets),
        '<div class="plan-gap"></div>',
        row("Risk / Reward", rr),
        row("Holding period", p.holding_period),
        row("Allocation", alloc),
        row("Position size", size),
        '<div class="plan-gap"></div>',
        row("Next catalyst", catalyst),
        row("Confidence", f"{r.verdict.confidence:.0f}/100 ({r.verdict.confidence_band})"),
        row("Verdict", r.verdict.label, accent=True),
    ]
    return "".join(lines)


def _rationale(r: ResearchReport, category: str) -> str:
    for c in r.scorecard:
        if c.name == category:
            return c.rationale
    return ""


def _positioning_stance(r: ResearchReport) -> str:
    if r.positioning is None:
        return "—"
    change = r.positioning.institutional_total_change
    if change is None:
        return "Neutral (no change data)"
    if change > 0.02:
        return "Accumulating"
    if change < -0.02:
        return "Distributing"
    return "Neutral"


def _insider_signal(r: ResearchReport) -> str:
    if r.positioning is None or not r.positioning.insider_events:
        return "no recent activity"
    net = r.positioning.insider_net_value_90d
    if net is None:
        return "activity reported"
    return "net buying" if net > 0 else "net selling" if net < 0 else "mixed"


def _technical_trend(r: ResearchReport) -> str:
    t = r.technicals
    if t is None or t.sma_50 is None or t.sma_200 is None:
        return "—"
    px = r.quote.price
    if px > t.sma_50 and px > t.sma_200:
        return "Uptrend"
    if px < t.sma_50 and px < t.sma_200:
        return "Downtrend"
    return "Consolidation"


# --------------------------- formatters ------------------------------------ #
def _fmt_money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:,.2f}"


def _fmt_pct(v: float | None) -> str:
    """Fraction → percent (0.18 → 18.0%)."""
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_pct_raw(v: float | None) -> str:
    """Already-in-percent value (yields) → append nothing extra."""
    if v is None:
        return "—"
    return f"{v:.2f}"


def _fmt_n(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}"


def _fmt_big(v: float | None) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:.2f}T"
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _fmt_big_signed(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "−" if v < 0 else ""
    return sign + _fmt_big(abs(v))
