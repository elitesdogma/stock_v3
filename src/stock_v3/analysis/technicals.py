"""Layer 4 — technical indicators, computed locally from the OHLCV frame.

Indicators are hand-rolled in pandas (RSI/MACD/StochRSI/ATR/Bollinger) so the output is
deterministic and dependency-light. This module is pure: a DataFrame in, a Technicals out.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..models import Technicals


def compute_technicals(daily: pd.DataFrame) -> Technicals:
    close = daily["Close"]
    high, low, volume = daily["High"], daily["Low"], daily["Volume"]

    macd_line, signal_line = _macd(close)
    bb_upper, bb_lower, bb_width = _bollinger(close)
    weekly = _weekly(close)

    return Technicals(
        ema_8=_last(_ema(close, 8)),
        ema_21=_last(_ema(close, 21)),
        sma_50=_last(close.rolling(50).mean()),
        sma_200=_last(close.rolling(200).mean()),
        rsi_14=_last(_rsi(close, 14)),
        macd=_last(macd_line),
        macd_signal=_last(signal_line),
        stoch_rsi=_last(_stoch_rsi(close, 14)),
        atr_14=_last(_atr(high, low, close, 14)),
        bb_upper=_last(bb_upper),
        bb_lower=_last(bb_lower),
        bb_width=_last(bb_width),
        volume=_last(volume),
        avg_volume_20d=_last(volume.rolling(20).mean()),
        weekly_sma_50=weekly["sma_50"],
        weekly_close=weekly["close"],
        price_history=_history_points(close),
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # A window with zero losses gives rs=inf → RSI=100 (correct), but 0/0 windows give
    # NaN; only the genuinely undefined early-warmup rows should stay NaN.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    macd_line = _ema(series, 12) - _ema(series, 26)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def _stoch_rsi(series: pd.Series, period: int) -> pd.Series:
    rsi = _rsi(series, period)
    lowest = rsi.rolling(period).min()
    highest = rsi.rolling(period).max()
    rng = (highest - lowest).replace(0, np.nan)
    return ((rsi - lowest) / rng) * 100


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, lower, width


def _weekly(close: pd.Series) -> dict[str, float | None]:
    weekly_close = close.resample("W").last().dropna()
    if weekly_close.empty:
        return {"close": None, "sma_50": None}
    sma_50 = weekly_close.rolling(50).mean()
    return {"close": _last(weekly_close), "sma_50": _last(sma_50)}


def _history_points(close: pd.Series, max_points: int = 180) -> list[tuple[dt.date, float]]:
    tail = close.dropna().tail(max_points)
    return [(idx.date(), round(float(val), 2)) for idx, val in tail.items()]


def _last(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    value = float(clean.iloc[-1])
    return round(value, 4) if np.isfinite(value) else None
