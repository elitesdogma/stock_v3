"""The report must render from a fully-mocked ResearchReport with no network access."""

from __future__ import annotations

import datetime as dt

from stock_v3.analysis.probability import build_probability_cone
from stock_v3.models import (
    AnalystConsensus,
    CategoryScore,
    CatalystEvent,
    CoverageEntry,
    Fundamentals,
    MacroSnapshot,
    NarrativeSections,
    OptionsSnapshot,
    Positioning,
    Quote,
    Regime,
    ResearchReport,
    Scenario,
    Technicals,
    TradePlan,
    Verdict,
)
from stock_v3.report.render import render_report


def _mock_report() -> ResearchReport:
    return ResearchReport(
        generated_at=dt.datetime(2026, 6, 3, 9, 30),
        quote=Quote("NVDA", "NVIDIA Corp", 1180.50, "USD", 2.9e12, "Technology",
                    "Semiconductors", "NMS", 2.46e9, dt.datetime(2026, 6, 3, 9, 30)),
        fundamentals=Fundamentals(
            revenue_ttm=1.3e11, revenue_growth_yoy=0.62, revenue_growth_qoq=0.12,
            gross_margin=0.75, operating_margin=0.55, ebitda_margin=0.60, net_margin=0.50,
            fcf_margin=0.45, fcf_ttm=5.8e10, cash=4e10, total_debt=1e10, net_debt=-3e10,
            current_ratio=3.2, pe=55, forward_pe=38, ev_ebitda=42, price_to_sales=22,
            peg=1.1, price_to_fcf=50,
        ),
        macro=MacroSnapshot(4.2, 4.4, 0.2, 0.026, 0.028, 0.022, 16.5, 102.0, 0.01, None,
                            dt.date(2026, 6, 2)),
        macro_regime=Regime.NEUTRAL,
        positioning=Positioning([], 0.03, dt.date(2026, 3, 31), [], 1500.0, 0.012, 1.8,
                                dt.date(2026, 5, 15), None),
        options=OptionsSnapshot(0.55, 1200.0, 1250.0, 1150.0, True),
        technicals=Technicals(
            ema_8=1170, ema_21=1140, sma_50=1100, sma_200=950, rsi_14=63.5, macd=12.0,
            macd_signal=9.0, stoch_rsi=72.0, atr_14=38.0, bb_upper=1220, bb_lower=1080,
            bb_width=0.12, volume=4.2e7, avg_volume_20d=3.8e7, weekly_sma_50=1000,
            weekly_close=1180.50,
            price_history=[(dt.date(2026, 5, 1) + dt.timedelta(days=i), 1000 + i * 6)
                           for i in range(30)],
        ),
        consensus=AnalystConsensus("Buy", 1.8, 1350.0, 1600.0, 1050.0, 52, 6, 1),
        catalysts=[
            CatalystEvent(dt.date(2026, 6, 20), "Q2 earnings", "earnings", "high"),
            CatalystEvent(dt.date(2026, 6, 15), "CPI release (approx)", "macro", "medium"),
        ],
        scenarios=[
            Scenario("Bull", 1600.0, 0.30, "AI capex acceleration, share gains."),
            Scenario("Base", 1350.0, 0.50, "In-line growth, multiple holds."),
            Scenario("Bear", 980.0, 0.20, "Capex digestion, multiple compression."),
        ],
        trade_plan=TradePlan((1160.0, 1190.0), 1100.0, 1104.0, [1218.0, 1390.0, 1560.0],
                             3.2, "6-18 months", 18.5, 13.0, 76.5),
        scorecard=[
            CategoryScore("Macro Environment", 5.5, "Neutral regime, VIX 16.5."),
            CategoryScore("Fundamentals", 9.0, "Hyper growth, exceptional margins."),
            CategoryScore("Valuation", 6.0, "Rich but PEG ~1.1 defensible."),
            CategoryScore("Technicals", 8.0, "Uptrend, RSI 63, MACD bullish."),
            CategoryScore("Institutional Positioning", 7.5, "Accumulating; insiders net buyers."),
            CategoryScore("Catalysts", 7.0, "Earnings in ~17 days."),
            CategoryScore("Risk/Reward", 7.5, "3.2:1 to primary target."),
        ],
        verdict=Verdict(78.0, "Accumulate", 82.0, "High Conviction"),
        coverage=[
            CoverageEntry("Macro", "FRED", "ok", "2026-06-02", None),
            CoverageEntry("Positioning", "SEC EDGAR", "stale", "2026-05-15", "13F lags 45d"),
            CoverageEntry("Options", "Yahoo", "ok", "2026-06-03", "Gamma is an estimate"),
        ],
        narrative=NarrativeSections(
            executive_summary="NVIDIA screens as a high-quality compounder at a full but "
            "PEG-defensible multiple; the setup favors accumulation into earnings strength.",
            thesis="Durable AI-infrastructure demand underwrites a 78/100 conviction.",
            bull="", base="", bear="",
            risks="Primary risk is a capex-digestion air-pocket compressing the multiple.",
            portfolio_impact="Adds concentrated semiconductor/large-cap-growth factor exposure.",
            generated_by="deterministic-template",
        ),
        key_risks=[
            "Customer capex digestion could stall revenue growth.",
            "Rich multiple amplifies downside on any guidance miss.",
            "Export-control and geopolitical overhang on China revenue.",
        ],
        probability_cone=build_probability_cone(1180.50, implied_vol=0.42, daily_history=None),
    )


def test_render_produces_html():
    html = render_report(_mock_report())
    assert html.startswith("<!DOCTYPE html>")
    assert "NVDA" in html
    assert "NVIDIA Corp" in html
    assert "Accumulate" in html
    # coverage panel surfaces the stale 13F
    assert "13F lags 45d" in html
    # deterministic-template indicator is shown
    assert "deterministic template" in html
    # trade plan rendered
    assert "Risk / Reward" in html
    # charts embedded as inline SVG, not external refs
    assert "<svg" in html
    assert "http://www.w3.org/2000/svg" in html
    # AXIS theming applied (navy+emerald tokens inlined)
    assert "--emerald-500" in html
    assert "Hanken Grotesk" in html
    # probability cone + interactive calculator present
    assert "Probability Cone" in html or "probability cone" in html.lower()
    assert "cone-calc" in html
    assert "probAbove" in html  # the inline lognormal slider math
    # the only script is the self-contained cone calculator (no remote JS src)
    assert "<script src" not in html.lower()


def test_render_handles_missing_optional_sections():
    report = _mock_report()
    report.fundamentals = None
    report.macro = None
    report.options = None
    report.consensus = None
    html = render_report(report)
    assert "unavailable" in html.lower()
    assert html.startswith("<!DOCTYPE html>")
