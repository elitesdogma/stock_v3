"""Short interest source.

FINRA is the regulatory origin of US short-interest data, published bi-monthly. FINRA's
own consolidated-by-ticker feed requires OAuth credentials, but the same settled figures
are surfaced (with the FINRA settlement date) through Yahoo's fundamentals — short % of
float, days-to-cover (short ratio), and the prior-period count for trend. We read those,
label the settlement vintage, and mark borrow rate N/A (never free).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..cache import Cache
from ..config import Settings
from .base import SourceResult

_SOURCE = "FINRA short interest (via Yahoo)"
_TTL = 12 * 3600


class ShortInterest:
    __slots__ = (
        "short_pct_float",
        "days_to_cover",
        "shares_short",
        "shares_short_prior",
        "settlement",
        "borrow_rate",
    )

    def __init__(
        self,
        short_pct_float: float | None,
        days_to_cover: float | None,
        shares_short: float | None,
        shares_short_prior: float | None,
        settlement: dt.date | None,
        borrow_rate: float | None,
    ) -> None:
        self.short_pct_float = short_pct_float
        self.days_to_cover = days_to_cover
        self.shares_short = shares_short
        self.shares_short_prior = shares_short_prior
        self.settlement = settlement
        self.borrow_rate = borrow_rate


def fetch_short_interest(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[ShortInterest]:
    cache_key = f"{ticker}:shortinterest"
    try:
        cached = cache.get(_SOURCE, cache_key, _TTL)
        if cached is not None:
            return _wrap(_from_dict(cached))

        import yfinance as yf

        info = dict(yf.Ticker(ticker).info or {})
        short_pct = _num(info.get("shortPercentOfFloat"))
        days_to_cover = _num(info.get("shortRatio"))
        shares_short = _num(info.get("sharesShort"))
        if short_pct is None and days_to_cover is None and shares_short is None:
            return SourceResult.unavailable(_SOURCE, "No short-interest data reported")

        si = ShortInterest(
            short_pct_float=short_pct,
            days_to_cover=days_to_cover,
            shares_short=shares_short,
            shares_short_prior=_num(info.get("sharesShortPriorMonth")),
            settlement=_epoch_to_date(info.get("dateShortInterest")),
            borrow_rate=None,  # never available on free data
        )
        cache.set(_SOURCE, cache_key, _to_dict(si))
        return _wrap(si)
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Short-interest error: {exc}")


def _wrap(si: ShortInterest) -> SourceResult[ShortInterest]:
    return SourceResult.of(
        si, _SOURCE, as_of=si.settlement, stale=True,
        note="Bi-monthly FINRA settlement; borrow rate N/A on free data",
        field_gaps=("borrow_rate",),
    )


def _epoch_to_date(raw) -> dt.date | None:
    if raw is None:
        return None
    try:
        return dt.datetime.utcfromtimestamp(int(raw)).date()
    except (TypeError, ValueError, OSError):
        return None


def _num(raw) -> float | None:
    if raw is None or raw == "" or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _to_dict(si: ShortInterest) -> dict:
    return {
        "short_pct_float": si.short_pct_float,
        "days_to_cover": si.days_to_cover,
        "shares_short": si.shares_short,
        "shares_short_prior": si.shares_short_prior,
        "settlement": si.settlement.isoformat() if si.settlement else None,
        "borrow_rate": si.borrow_rate,
    }


def _from_dict(d: dict) -> ShortInterest:
    return ShortInterest(
        short_pct_float=d.get("short_pct_float"),
        days_to_cover=d.get("days_to_cover"),
        shares_short=d.get("shares_short"),
        shares_short_prior=d.get("shares_short_prior"),
        settlement=dt.date.fromisoformat(d["settlement"]) if d.get("settlement") else None,
        borrow_rate=d.get("borrow_rate"),
    )
