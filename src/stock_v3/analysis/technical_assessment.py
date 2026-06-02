"""Layer 4 scoring — turns raw Technicals into trend/momentum/volatility classification
and a 1-10 technical score. Kept separate from indicator computation (technicals.py) so
the math and the judgment are independently testable."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Technicals


@dataclass
class TechnicalAssessment:
    trend: str  # Uptrend / Downtrend / Consolidation
    momentum: str  # Oversold / Neutral / Overbought
    volatility: str  # Compression / Normal / Expansion
    volume_signal: str
    score: float  # 1..10
    rationale: str


def assess_technicals(t: Technicals, price: float) -> TechnicalAssessment:
    trend = _trend(t, price)
    momentum = _momentum(t.rsi_14, t.stoch_rsi)
    volatility = _volatility(t.bb_width)
    volume_signal = _volume(t.volume, t.avg_volume_20d)

    score = _score(t, price, trend, momentum)
    rationale = (
        f"{trend} (px {_n(price)} vs 50d {_n(t.sma_50)}, 200d {_n(t.sma_200)}); "
        f"RSI {_n(t.rsi_14)} ({momentum}); MACD {_macd_state(t)}; "
        f"BBands {volatility}; volume {volume_signal}."
    )
    return TechnicalAssessment(
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume_signal=volume_signal,
        score=score,
        rationale=rationale,
    )


def _trend(t: Technicals, price: float) -> str:
    above_50 = t.sma_50 is not None and price > t.sma_50
    above_200 = t.sma_200 is not None and price > t.sma_200
    stacked_up = (
        t.ema_8 is not None and t.ema_21 is not None and t.ema_8 > t.ema_21
    )
    if above_50 and above_200 and stacked_up:
        return "Uptrend"
    if (not above_50) and (not above_200):
        return "Downtrend"
    return "Consolidation"


def _momentum(rsi: float | None, stoch: float | None) -> str:
    if rsi is None:
        return "Unknown"
    if rsi >= 70:
        return "Overbought"
    if rsi <= 30:
        return "Oversold"
    return "Neutral"


def _volatility(bb_width: float | None) -> str:
    if bb_width is None:
        return "Unknown"
    if bb_width < 0.08:
        return "Compression (breakout watch)"
    if bb_width > 0.20:
        return "Expansion"
    return "Normal"


def _volume(volume: float | None, avg20: float | None) -> str:
    if volume is None or avg20 is None or avg20 == 0:
        return "Unknown"
    ratio = volume / avg20
    if ratio >= 1.5:
        return f"elevated ({ratio:.1f}x 20d avg)"
    if ratio <= 0.6:
        return f"light ({ratio:.1f}x 20d avg)"
    return f"average ({ratio:.1f}x 20d avg)"


def _macd_state(t: Technicals) -> str:
    if t.macd is None or t.macd_signal is None:
        return "n/a"
    return "bullish" if t.macd > t.macd_signal else "bearish"


def _score(t: Technicals, price: float, trend: str, momentum: str) -> float:
    score = 5.0
    score += {"Uptrend": 2.0, "Consolidation": 0.0, "Downtrend": -2.0}[trend]

    if t.macd is not None and t.macd_signal is not None:
        score += 0.75 if t.macd > t.macd_signal else -0.75

    # Momentum extremes are mean-reversion nudges, not trend votes.
    if momentum == "Oversold":
        score += 0.5
    elif momentum == "Overbought":
        score -= 0.5

    # Weekly confirmation: price above the long weekly average reinforces the daily read.
    if t.weekly_close is not None and t.weekly_sma_50 is not None:
        score += 0.5 if t.weekly_close > t.weekly_sma_50 else -0.5

    return round(max(1.0, min(10.0, score)), 1)


def _n(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "n/a"
