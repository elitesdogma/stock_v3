"""Finnhub source: analyst consensus, price targets, upcoming earnings.

Free tier is 60 calls/min and 20-min delayed. Some endpoints (notably price_target)
are premium-gated and return 403; each call is isolated so a gated endpoint degrades
that one field rather than the whole source.
"""

from __future__ import annotations

import datetime as dt

from ..cache import Cache
from ..config import Settings
from ..models import AnalystConsensus, CatalystEvent
from .base import SourceResult

_SOURCE = "Finnhub"
_TTL = 3 * 3600

# recommendation_trends returns counts; map to a 1(best)..5(worst) normalized score.
_REC_WEIGHTS = {
    "strongBuy": 1.0,
    "buy": 2.0,
    "hold": 3.0,
    "sell": 4.0,
    "strongSell": 5.0,
}


def fetch_consensus(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[AnalystConsensus]:
    if not settings.has_finnhub:
        return SourceResult.unavailable(
            _SOURCE, "FINNHUB_API_KEY not set — analyst consensus skipped"
        )

    cache_key = f"{ticker}:consensus"
    cached = cache.get(_SOURCE, cache_key, _TTL)
    if cached is not None:
        return SourceResult.of(_consensus_from_dict(cached), _SOURCE, as_of=dt.date.today())

    try:
        import finnhub

        client = finnhub.Client(api_key=settings.finnhub_api_key)
        trends = _safe_call(lambda: client.recommendation_trends(ticker)) or []
        target = _safe_call(lambda: client.price_target(ticker)) or {}

        rating, rating_score = _summarize_trends(trends)
        upgrades, downgrades = _momentum(trends)

        consensus = AnalystConsensus(
            rating=rating,
            rating_score=rating_score,
            target_mean=_num(target.get("targetMean")),
            target_high=_num(target.get("targetHigh")),
            target_low=_num(target.get("targetLow")),
            num_analysts=_analyst_count(trends),
            recent_upgrades=upgrades,
            recent_downgrades=downgrades,
        )
        gaps = tuple(
            n for n, v in {"target_mean": consensus.target_mean,
                           "rating": consensus.rating}.items() if v is None
        )
        note = None
        if consensus.target_mean is None:
            note = "Price targets unavailable on free tier"
        cache.set(_SOURCE, cache_key, _consensus_to_dict(consensus))
        return SourceResult.of(consensus, _SOURCE, as_of=dt.date.today(),
                               field_gaps=gaps, note=note)
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Finnhub error: {exc}")


def fetch_earnings_catalysts(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[list[CatalystEvent]]:
    if not settings.has_finnhub:
        return SourceResult.unavailable(_SOURCE, "FINNHUB_API_KEY not set")

    cache_key = f"{ticker}:earnings"
    cached = cache.get(_SOURCE, cache_key, _TTL)
    if cached is not None:
        return SourceResult.of([_event_from_dict(d) for d in cached], _SOURCE)

    try:
        import finnhub

        client = finnhub.Client(api_key=settings.finnhub_api_key)
        today = dt.date.today()
        horizon = today + dt.timedelta(days=90)
        data = _safe_call(
            lambda: client.earnings_calendar(
                _from=today.isoformat(), to=horizon.isoformat(), symbol=ticker
            )
        ) or {}
        events: list[CatalystEvent] = []
        for row in data.get("earningsCalendar", []) or []:
            edate = _coerce_date(row.get("date"))
            if edate is None:
                continue
            events.append(
                CatalystEvent(
                    date=edate,
                    label=f"Q{row.get('quarter', '?')} earnings",
                    kind="earnings",
                    importance="high",
                )
            )
        cache.set(_SOURCE, cache_key, [_event_to_dict(e) for e in events])
        return SourceResult.of(events, _SOURCE, as_of=dt.date.today())
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Finnhub earnings error: {exc}")


# --------------------------------------------------------------------------- #
def _safe_call(fn):
    """Isolate a single Finnhub endpoint so a premium-gated 403 degrades one field."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _summarize_trends(trends: list[dict]) -> tuple[str | None, float | None]:
    if not trends:
        return None, None
    latest = trends[0]  # newest period first
    weighted = 0.0
    total = 0
    for key, weight in _REC_WEIGHTS.items():
        count = int(latest.get(key, 0) or 0)
        weighted += weight * count
        total += count
    if total == 0:
        return None, None
    score = weighted / total
    return _label_for_score(score), round(score, 2)


def _label_for_score(score: float) -> str:
    if score <= 1.5:
        return "Strong Buy"
    if score <= 2.5:
        return "Buy"
    if score <= 3.5:
        return "Hold"
    if score <= 4.5:
        return "Sell"
    return "Strong Sell"


def _analyst_count(trends: list[dict]) -> int | None:
    if not trends:
        return None
    latest = trends[0]
    total = sum(int(latest.get(k, 0) or 0) for k in _REC_WEIGHTS)
    return total or None


def _momentum(trends: list[dict]) -> tuple[int | None, int | None]:
    """Compare buy-side conviction between the two most recent periods."""
    if len(trends) < 2:
        return None, None
    cur, prev = trends[0], trends[1]

    def bullish(t: dict) -> int:
        return int(t.get("strongBuy", 0) or 0) + int(t.get("buy", 0) or 0)

    delta = bullish(cur) - bullish(prev)
    if delta > 0:
        return delta, 0
    if delta < 0:
        return 0, abs(delta)
    return 0, 0


def _num(raw) -> float | None:
    if raw in (None, 0, "0", ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _coerce_date(raw) -> dt.date | None:
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _consensus_to_dict(c: AnalystConsensus) -> dict:
    return c.__dict__.copy()


def _consensus_from_dict(d: dict) -> AnalystConsensus:
    return AnalystConsensus(**d)


def _event_to_dict(e: CatalystEvent) -> dict:
    return {"date": e.date.isoformat() if e.date else None, "label": e.label,
            "kind": e.kind, "importance": e.importance}


def _event_from_dict(d: dict) -> CatalystEvent:
    return CatalystEvent(
        date=dt.date.fromisoformat(d["date"]) if d.get("date") else None,
        label=d["label"], kind=d["kind"], importance=d["importance"],
    )
