"""Indicator math sanity + classifier boundaries for macro/fundamentals/technicals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_v3.analysis.fundamentals import assess_fundamentals
from stock_v3.analysis.macro import assess_macro
from stock_v3.analysis.technical_assessment import assess_technicals
from stock_v3.analysis.technicals import compute_technicals
from stock_v3.models import Fundamentals, MacroSnapshot, Regime, Technicals


def _trending_frame(days: int = 260, slope: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    base = 100 + slope * np.arange(days)
    noise = np.sin(np.arange(days) / 5) * 1.5
    close = base + noise
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(days, 1_000_000.0),
        },
        index=idx,
    )


def test_compute_technicals_on_uptrend():
    t = compute_technicals(_trending_frame())
    assert t.rsi_14 is not None and 0 <= t.rsi_14 <= 100
    assert t.sma_50 is not None and t.sma_200 is not None
    # In a steady uptrend, fast EMA sits above slow SMA and RSI is elevated.
    assert t.ema_8 > t.sma_200
    assert t.rsi_14 > 50


def test_assess_technicals_calls_uptrend():
    frame = _trending_frame()
    t = compute_technicals(frame)
    price = float(frame["Close"].iloc[-1])
    a = assess_technicals(t, price)
    assert a.trend == "Uptrend"
    assert a.score >= 6.0


def test_macro_riskoff_on_high_vix():
    macro = MacroSnapshot(
        ust_2y=4.5, ust_10y=4.2, yield_curve_10y_2y=-0.3, cpi_yoy=0.05,
        core_cpi_yoy=0.045, ppi_yoy=0.04, vix=35.0, dxy=105.0, dxy_trend_3m=0.03,
        breadth_pct_above_200d=None, as_of=None,
    )
    a = assess_macro(macro)
    assert a.regime == Regime.RISK_OFF
    assert a.vix_regime == "Risk-Off"
    assert a.score < 4.0


def test_macro_riskon_on_low_vix_normal_curve():
    macro = MacroSnapshot(
        ust_2y=3.5, ust_10y=4.2, yield_curve_10y_2y=0.7, cpi_yoy=0.022,
        core_cpi_yoy=0.023, ppi_yoy=0.02, vix=13.0, dxy=100.0, dxy_trend_3m=0.0,
        breadth_pct_above_200d=None, as_of=None,
    )
    a = assess_macro(macro)
    assert a.regime == Regime.RISK_ON
    assert a.score >= 6.5


def test_fundamentals_hypergrowth_high_score():
    f = Fundamentals(
        revenue_ttm=1e10, revenue_growth_yoy=0.55, revenue_growth_qoq=0.12,
        gross_margin=0.7, operating_margin=0.30, ebitda_margin=0.35, net_margin=0.25,
        fcf_margin=0.25, fcf_ttm=2.5e9, cash=5e9, total_debt=1e9, net_debt=-4e9,
        current_ratio=2.5, pe=40, forward_pe=28, ev_ebitda=22, price_to_sales=12,
        peg=0.9, price_to_fcf=30,
    )
    a = assess_fundamentals(f)
    assert a.growth_class == "Hyper Growth"
    assert a.fundamentals_score >= 7.5
    assert a.valuation_class in {"Undervalued", "Fairly Valued"}  # PEG < 1 rescues high P/E


def test_fundamentals_contracting_unprofitable_low_score():
    f = Fundamentals(
        revenue_ttm=1e8, revenue_growth_yoy=-0.15, revenue_growth_qoq=-0.05,
        gross_margin=0.2, operating_margin=-0.10, ebitda_margin=-0.05, net_margin=-0.20,
        fcf_margin=-0.15, fcf_ttm=-2e7, cash=1e7, total_debt=5e7, net_debt=4e7,
        current_ratio=0.8, pe=None, forward_pe=None, ev_ebitda=None, price_to_sales=8,
        peg=None, price_to_fcf=None,
    )
    a = assess_fundamentals(f)
    assert a.growth_class == "Contracting"
    assert a.fundamentals_score <= 4.0
