"""FRED macro source. Stable, free (key required), 120 req/min.

Covers Layer 1 fully. CPI/PPI arrive as index levels, so year-over-year inflation is
computed here rather than pulled directly. Treasury yields, the 10y-2y spread, VIX and
the broad dollar index come ready-to-use.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..cache import Cache
from ..config import Settings
from ..models import MacroSnapshot
from .base import SourceResult

_SOURCE = "FRED"
_TTL = 6 * 3600  # macro series update daily at most; 6h is plenty and spares the API

# series IDs
_UST_2Y = "DGS2"
_UST_10Y = "DGS10"
_CURVE = "T10Y2Y"
_CPI = "CPIAUCSL"
_CORE_CPI = "CPILFESL"
_PPI = "PPIACO"
_VIX = "VIXCLS"
_DXY = "DTWEXBGS"  # nominal broad USD index


def fetch_macro(settings: Settings, cache: Cache) -> SourceResult[MacroSnapshot]:
    if not settings.has_fred:
        return SourceResult.unavailable(
            _SOURCE, "FRED_API_KEY not set — macro layer skipped"
        )

    try:
        from fredapi import Fred

        fred = Fred(api_key=settings.fred_api_key)

        def latest(series_id: str) -> float | None:
            cached = cache.get(_SOURCE, series_id, _TTL)
            if cached is not None:
                series = _to_series(cached)
            else:
                series = fred.get_series(series_id, observation_start=_lookback())
                cache.set(_SOURCE, series_id, _series_payload(series))
            return _last_valid(series)

        def yoy(series_id: str) -> float | None:
            cached = cache.get(_SOURCE, series_id, _TTL)
            if cached is not None:
                series = _to_series(cached)
            else:
                series = fred.get_series(series_id, observation_start=_lookback(months=15))
                cache.set(_SOURCE, series_id, _series_payload(series))
            return _yoy_change(series)

        ust_2y = latest(_UST_2Y)
        ust_10y = latest(_UST_10Y)
        curve = latest(_CURVE)
        if curve is None and ust_2y is not None and ust_10y is not None:
            curve = round(ust_10y - ust_2y, 2)

        snapshot = MacroSnapshot(
            ust_2y=ust_2y,
            ust_10y=ust_10y,
            yield_curve_10y_2y=curve,
            cpi_yoy=yoy(_CPI),
            core_cpi_yoy=yoy(_CORE_CPI),
            ppi_yoy=yoy(_PPI),
            vix=latest(_VIX),
            dxy=latest(_DXY),
            dxy_trend_3m=_trend(fred, cache, _DXY),
            breadth_pct_above_200d=None,  # no clean free API; computed elsewhere or left N/A
            as_of=dt.date.today(),
        )
        gaps = tuple(
            name
            for name, val in {
                "cpi_yoy": snapshot.cpi_yoy,
                "ppi_yoy": snapshot.ppi_yoy,
                "vix": snapshot.vix,
                "dxy": snapshot.dxy,
            }.items()
            if val is None
        )
        return SourceResult.of(
            snapshot, _SOURCE, as_of=dt.date.today(), field_gaps=gaps,
            note="Breadth (% above 200d) not sourced from FRED",
        )
    except Exception as exc:  # noqa: BLE001 — any FRED failure degrades, never crashes the run
        return SourceResult.unavailable(_SOURCE, f"FRED error: {exc}")


def _lookback(months: int = 2) -> str:
    start = dt.date.today() - dt.timedelta(days=31 * months)
    return start.isoformat()


def _last_valid(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.iloc[-1]), 4)


def _yoy_change(series: pd.Series | None) -> float | None:
    """Year-over-year fractional change from a monthly index level series."""
    if series is None:
        return None
    clean = series.dropna()
    if len(clean) < 13:
        return None
    latest_val = float(clean.iloc[-1])
    year_ago = float(clean.iloc[-13])
    if year_ago == 0:
        return None
    return round((latest_val - year_ago) / year_ago, 4)


def _trend(fred, cache: Cache, series_id: str) -> float | None:
    """Fractional change over the trailing ~3 months."""
    cached = cache.get(_SOURCE, f"{series_id}__trend", _TTL)
    if cached is not None:
        series = _to_series(cached)
    else:
        series = fred.get_series(series_id, observation_start=_lookback(months=4))
        cache.set(_SOURCE, f"{series_id}__trend", _series_payload(series))
    if series is None:
        return None
    clean = series.dropna()
    if len(clean) < 20:
        return None
    start, end = float(clean.iloc[0]), float(clean.iloc[-1])
    if start == 0:
        return None
    return round((end - start) / start, 4)


def _series_payload(series: pd.Series | None) -> dict | None:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    return {
        "index": [d.isoformat() for d in clean.index],
        "values": [float(v) for v in clean.values],
    }


def _to_series(payload: dict) -> pd.Series:
    idx = pd.to_datetime(payload["index"])
    return pd.Series(payload["values"], index=idx)
