"""SEC EDGAR source via edgartools (MIT, no API key — just a descriptive User-Agent).

Covers Layer 2 insider activity from Form 4. 13F institutional holdings are exposed
through the *issuer's* filings only indirectly, so for a personal free tool we read
the aggregate institutional picture from yfinance's holders data instead and reserve
EDGAR for the high-signal insider transactions and (optionally) latest financials.

Form 4 parsing fetches XML per filing, so we cap the scan to the most recent filings
inside a 90-day window — enough to gauge insider buy/sell pressure without a slow crawl.
"""

from __future__ import annotations

import datetime as dt

from ..cache import Cache
from ..config import Settings
from ..models import InsiderEvent
from .base import SourceResult

_SOURCE = "SEC EDGAR (Form 4)"
_TTL = 12 * 3600
_MAX_FILINGS = 25
_WINDOW_DAYS = 90


def fetch_insider_activity(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[list[InsiderEvent]]:
    cache_key = f"{ticker}:insider"
    cached = cache.get(_SOURCE, cache_key, _TTL)
    if cached is not None:
        events = [_event_from_dict(d) for d in cached]
        return _wrap(events, stale=False)

    try:
        import edgar

        edgar.set_identity(settings.sec_user_agent)
        company = edgar.Company(ticker)
        filings = company.get_filings(form="4")
        if not filings or len(filings) == 0:
            return SourceResult.of([], _SOURCE, as_of=dt.date.today(),
                                   note="No Form 4 filings found")

        cutoff = dt.date.today() - dt.timedelta(days=_WINDOW_DAYS)
        events: list[InsiderEvent] = []
        for filing in filings[: _MAX_FILINGS * 2]:  # scan a bit extra; some fall outside window
            fdate = _filing_date(filing)
            if fdate is not None and fdate < cutoff:
                break  # filings are newest-first; once past the window we're done
            event = _parse_form4(filing, fdate)
            if event is not None:
                events.append(event)
            if len(events) >= _MAX_FILINGS:
                break

        cache.set(_SOURCE, cache_key, [_event_to_dict(e) for e in events])
        return _wrap(events, stale=False)
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"EDGAR error: {exc}")


def _wrap(events: list[InsiderEvent], *, stale: bool) -> SourceResult[list[InsiderEvent]]:
    return SourceResult.of(events, _SOURCE, as_of=dt.date.today(), stale=stale)


def _filing_date(filing) -> dt.date | None:
    raw = getattr(filing, "filing_date", None)
    return _coerce_date(raw)


def _parse_form4(filing, fdate: dt.date | None) -> InsiderEvent | None:
    try:
        obj = filing.obj()
    except Exception:  # noqa: BLE001 — a single unparseable filing is skipped, not fatal
        return None

    insider = _str(getattr(obj, "insider_name", None)) or "Unknown"
    role = _str(getattr(obj, "position", None)) or "Insider"

    buys = _safe_float(getattr(obj, "common_stock_purchases", None))
    sales = _safe_float(getattr(obj, "common_stock_sales", None))

    # Prefer explicit purchase/sale share counts; fall back to net shares_traded.
    if buys and buys > 0:
        txn, shares = "buy", buys
    elif sales and sales > 0:
        txn, shares = "sell", sales
    else:
        net = _safe_float(getattr(obj, "shares_traded", None))
        if net is None or net == 0:
            return None
        txn = "buy" if net > 0 else "sell"
        shares = abs(net)

    return InsiderEvent(
        date=fdate or dt.date.today(),
        insider=insider,
        role=role,
        transaction=txn,
        shares=shares,
        value=None,  # Form 4 price-per-share parsing is inconsistent; left N/A rather than guessed
    )


# --------------------------------------------------------------------------- #
# coercion helpers — Form 4 fields vary in type across filings
# --------------------------------------------------------------------------- #
def _coerce_date(raw) -> dt.date | None:
    if raw is None:
        return None
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _safe_float(raw) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _str(raw) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _event_to_dict(e: InsiderEvent) -> dict:
    return {
        "date": e.date.isoformat(),
        "insider": e.insider,
        "role": e.role,
        "transaction": e.transaction,
        "shares": e.shares,
        "value": e.value,
    }


def _event_from_dict(d: dict) -> InsiderEvent:
    return InsiderEvent(
        date=dt.date.fromisoformat(d["date"]),
        insider=d["insider"],
        role=d["role"],
        transaction=d["transaction"],
        shares=d["shares"],
        value=d.get("value"),
    )
