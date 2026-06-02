"""Probability cone — the industry-standard volatility projection.

Mirrors the thinkorswim / Schwab "probability cone": project the price distribution forward
using annualized volatility (implied from ATM options where available, else realized from
1-year returns). Prices are lognormal, so percentile bands and P(S_T ≥ K) come from the
normal CDF on log-returns.

Defaults follow the benchmark: the ±1σ band is the 68.27% range, ±2σ ≈ 95%. Horizons of
2 / 4 / 8 / 13 weeks match how desks quote near-term expected moves.

Pure functions: volatility + spot in, a ProbabilityCone out.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..models import ConeBand, ProbabilityCone

_HORIZONS_WEEKS = (2, 4, 8, 13)
_TRADING_DAYS = 252
_CAL_DAYS = 365.0


def build_probability_cone(
    spot: float,
    *,
    implied_vol: float | None,
    daily_history: pd.DataFrame | None,
    horizons_weeks: tuple[int, ...] = _HORIZONS_WEEKS,
) -> ProbabilityCone | None:
    vol, source = _resolve_vol(implied_vol, daily_history)
    if vol is None or spot <= 0:
        return None

    bands = [_band(spot, vol, w) for w in horizons_weeks]
    return ProbabilityCone(
        spot=round(spot, 2),
        annual_vol=round(vol, 4),
        vol_source=source,
        bands=bands,
        max_weeks=max(horizons_weeks),
    )


def prob_above(spot: float, target: float, annual_vol: float, days: int) -> float:
    """P(S_T ≥ target) under the lognormal (zero-drift, risk-neutral-ish) model.

    Using d2 with μ=0: P = N( ln(spot/target) / (σ√t) − ½σ√t ). Zero drift keeps the cone
    centered on spot, which is the convention for a short-horizon probability cone (we are
    not forecasting direction, only dispersion)."""
    if spot <= 0 or target <= 0 or annual_vol <= 0 or days <= 0:
        return float("nan")
    t = days / _CAL_DAYS
    sig = annual_vol * math.sqrt(t)
    d = (math.log(spot / target) - 0.5 * sig * sig) / sig
    return _norm_cdf(d)


def prob_below(spot: float, target: float, annual_vol: float, days: int) -> float:
    p = prob_above(spot, target, annual_vol, days)
    return float("nan") if math.isnan(p) else 1.0 - p


def expected_move(spot: float, annual_vol: float, days: int) -> float:
    """1σ expected move in price terms at the horizon."""
    t = days / _CAL_DAYS
    return spot * annual_vol * math.sqrt(t)


# --------------------------------------------------------------------------- #
def _band(spot: float, vol: float, weeks: int) -> ConeBand:
    days = weeks * 7
    t = days / _CAL_DAYS
    sig = vol * math.sqrt(t)  # std dev of log-return at horizon

    def price_at(z: float) -> float:
        # lognormal quantile with zero drift: S0 * exp(σ√t · z − ½σ²t)
        return round(spot * math.exp(sig * z - 0.5 * sig * sig), 2)

    return ConeBand(
        weeks=weeks,
        days=days,
        expected_move_pct=round(vol * math.sqrt(t), 4),
        p10=price_at(-1.2816),
        p25=price_at(-0.6745),
        p50=price_at(0.0),
        p75=price_at(0.6745),
        p90=price_at(1.2816),
        low_1sd=price_at(-1.0),
        high_1sd=price_at(1.0),
    )


def _resolve_vol(
    implied_vol: float | None, daily_history: pd.DataFrame | None
) -> tuple[float | None, str]:
    if implied_vol is not None and 0.05 <= implied_vol <= 3.0:
        return implied_vol, "implied (ATM options)"
    realized = _realized_vol(daily_history)
    if realized is not None:
        return realized, "realized (1y returns)"
    return None, "unavailable"


def _realized_vol(daily: pd.DataFrame | None) -> float | None:
    """Annualized close-to-close volatility over the trailing ~1 year."""
    if daily is None or "Close" not in daily or len(daily) < 30:
        return None
    closes = daily["Close"].dropna().tail(_TRADING_DAYS)
    if len(closes) < 30:
        return None
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if log_returns.empty:
        return None
    daily_sigma = float(log_returns.std())
    annual = daily_sigma * math.sqrt(_TRADING_DAYS)
    if annual <= 0 or annual > 3.0:
        return None
    return round(annual, 4)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
