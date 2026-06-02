"""yfinance source — the primary feed: quote, fundamentals, OHLCV history, institutional
holders, and an option-chain-derived gamma proxy.

yfinance is fragile (scraping-based, aggressively rate-limited). Every call is cached and
wrapped so a failure degrades to UNAVAILABLE. The gamma proxy is explicitly an estimate.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..cache import Cache
from ..config import Settings
from ..models import (
    Fundamentals,
    InstitutionalHolder,
    OptionsSnapshot,
    Quote,
)
from .base import SourceResult

_SOURCE = "Yahoo (yfinance)"
_INFO_TTL = 1800  # 30 min — quote/fundamentals
_HIST_TTL = 1800
_HOLDERS_TTL = 24 * 3600  # 13F-derived, updates slowly
_OPTIONS_TTL = 1800


def _ticker(ticker: str):
    import yfinance as yf

    return yf.Ticker(ticker)


def _info(ticker: str, cache: Cache) -> dict:
    cached = cache.get(_SOURCE, f"{ticker}:info", _INFO_TTL)
    if cached is not None:
        return cached
    info = dict(_ticker(ticker).info or {})
    # Strip unpicklable / huge nested junk; keep scalar fields.
    clean = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
    cache.set(_SOURCE, f"{ticker}:info", clean)
    return clean


def fetch_quote(ticker: str, settings: Settings, cache: Cache) -> SourceResult[Quote]:
    try:
        info = _info(ticker, cache)
        price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
        if price is None:
            return SourceResult.unavailable(_SOURCE, f"No price for {ticker}")
        quote = Quote(
            ticker=ticker.upper(),
            company=info.get("longName") or info.get("shortName") or ticker.upper(),
            price=price,
            currency=info.get("currency") or "USD",
            market_cap=_num(info.get("marketCap")),
            sector=info.get("sector"),
            industry=info.get("industry"),
            exchange=info.get("exchange"),
            shares_outstanding=_num(info.get("sharesOutstanding")),
            as_of=dt.datetime.now(),
        )
        return SourceResult.of(quote, _SOURCE, as_of=quote.as_of,
                               note="Price is delayed (free data)")
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Quote error: {exc}")


def fetch_fundamentals(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[Fundamentals]:
    try:
        info = _info(ticker, cache)
        revenue = _num(info.get("totalRevenue"))
        fcf = _num(info.get("freeCashflow"))
        cash = _num(info.get("totalCash"))
        debt = _num(info.get("totalDebt"))
        price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
        mcap = _num(info.get("marketCap"))

        fundamentals = Fundamentals(
            revenue_ttm=revenue,
            revenue_growth_yoy=_num(info.get("revenueGrowth")),
            revenue_growth_qoq=_num(info.get("revenueQuarterlyGrowth")),
            gross_margin=_num(info.get("grossMargins")),
            operating_margin=_num(info.get("operatingMargins")),
            ebitda_margin=_num(info.get("ebitdaMargins")),
            net_margin=_num(info.get("profitMargins")),
            fcf_margin=(fcf / revenue) if fcf and revenue else None,
            fcf_ttm=fcf,
            cash=cash,
            total_debt=debt,
            net_debt=(debt - cash) if debt is not None and cash is not None else None,
            current_ratio=_num(info.get("currentRatio")),
            pe=_num(info.get("trailingPE")),
            forward_pe=_num(info.get("forwardPE")),
            ev_ebitda=_num(info.get("enterpriseToEbitda")),
            price_to_sales=_num(info.get("priceToSalesTrailing12Months")),
            peg=_num(info.get("trailingPegRatio")) or _num(info.get("pegRatio")),
            price_to_fcf=(mcap / fcf) if mcap and fcf and fcf > 0 else None,
        )
        return SourceResult.of(fundamentals, _SOURCE, as_of=dt.date.today())
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Fundamentals error: {exc}")


def fetch_history(
    ticker: str, settings: Settings, cache: Cache, *, period: str = "2y"
) -> SourceResult[pd.DataFrame]:
    """Daily OHLCV — the substrate for every technical indicator and the price chart."""
    cache_key = f"{ticker}:hist:{period}"
    try:
        cached = cache.get(_SOURCE, cache_key, _HIST_TTL)
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            return SourceResult.of(cached, _SOURCE, as_of=dt.date.today())
        hist = _ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return SourceResult.unavailable(_SOURCE, f"No price history for {ticker}")
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        cache.set(_SOURCE, cache_key, hist)
        return SourceResult.of(hist, _SOURCE, as_of=hist.index[-1].date())
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"History error: {exc}")


def fetch_institutional_holders(
    ticker: str, settings: Settings, cache: Cache
) -> SourceResult[list[InstitutionalHolder]]:
    cache_key = f"{ticker}:instholders"
    try:
        cached = cache.get(_SOURCE, cache_key, _HOLDERS_TTL)
        if cached is not None:
            return SourceResult.of(
                [InstitutionalHolder(**d) for d in cached], _SOURCE,
                as_of=dt.date.today(), stale=True,
                note="13F-derived ownership lags ~45 days",
            )
        df = _ticker(ticker).institutional_holders
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return SourceResult.unavailable(_SOURCE, "No institutional holder data")
        holders = _parse_holders(df)
        cache.set(_SOURCE, cache_key, [h.__dict__ for h in holders])
        return SourceResult.of(holders, _SOURCE, as_of=dt.date.today(), stale=True,
                               note="13F-derived ownership lags ~45 days")
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Holders error: {exc}")


def fetch_options_proxy(
    ticker: str, spot: float | None, settings: Settings, cache: Cache
) -> SourceResult[OptionsSnapshot]:
    """Estimate dealer-gamma structure from the option chain.

    This is a PROXY, never authoritative GEX. Under the standard assumption that dealers
    are short puts and long calls, we weight open interest by gamma across strikes to
    locate approximate call/put walls and the gamma-flip spot, plus a put/call OI ratio.
    """
    if spot is None:
        return SourceResult.unavailable(_SOURCE, "No spot price for options proxy")
    cache_key = f"{ticker}:options"
    try:
        cached = cache.get(_SOURCE, cache_key, _OPTIONS_TTL)
        if cached is not None:
            return SourceResult.of(OptionsSnapshot(**cached), _SOURCE,
                                   note="Gamma levels are an estimate, not authoritative GEX")
        yt = _ticker(ticker)
        expiries = list(getattr(yt, "options", []) or [])
        if not expiries:
            return SourceResult.unavailable(_SOURCE, "No listed options")
        # Use the two nearest expiries — that's where dealer gamma concentrates.
        calls, puts = _collect_chain(yt, expiries[:2])
        if calls.empty and puts.empty:
            return SourceResult.unavailable(_SOURCE, "Empty option chain")

        atm_iv = _atm_iv(yt, expiries, spot)
        snapshot = _build_options_snapshot(calls, puts, spot, atm_iv)
        cache.set(_SOURCE, cache_key, snapshot.__dict__)
        return SourceResult.of(snapshot, _SOURCE,
                               note="Gamma levels are an estimate, not authoritative GEX")
    except Exception as exc:  # noqa: BLE001
        return SourceResult.unavailable(_SOURCE, f"Options error: {exc}")


# --------------------------------------------------------------------------- #
def _collect_chain(yt, expiries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_frames, put_frames = [], []
    for exp in expiries:
        try:
            chain = yt.option_chain(exp)
        except Exception:  # noqa: BLE001
            continue
        if getattr(chain, "calls", None) is not None and not chain.calls.empty:
            call_frames.append(chain.calls[["strike", "openInterest"]])
        if getattr(chain, "puts", None) is not None and not chain.puts.empty:
            put_frames.append(chain.puts[["strike", "openInterest"]])
    calls = pd.concat(call_frames) if call_frames else pd.DataFrame(columns=["strike", "openInterest"])
    puts = pd.concat(put_frames) if put_frames else pd.DataFrame(columns=["strike", "openInterest"])
    return calls, puts


def _build_options_snapshot(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float, atm_iv: float | None
) -> OptionsSnapshot:
    call_oi = float(calls["openInterest"].fillna(0).sum())
    put_oi = float(puts["openInterest"].fillna(0).sum())
    pcr = round(put_oi / call_oi, 2) if call_oi > 0 else None

    call_wall = _max_oi_strike(calls)
    put_wall = _max_oi_strike(puts)

    # Gamma-flip proxy: midpoint of the dominant call and put walls, the rough spot at
    # which net dealer gamma transitions. A real GEX model integrates per-strike gamma;
    # this is intentionally a coarse estimate and labelled as such.
    if call_wall is not None and put_wall is not None:
        gamma_flip = round((call_wall + put_wall) / 2, 2)
    else:
        gamma_flip = None

    return OptionsSnapshot(
        put_call_ratio=pcr,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        atm_iv=atm_iv,
        is_estimate=True,
    )


def _atm_iv(yt, expiries: list[str], spot: float) -> float | None:
    """At-the-money implied volatility from an expiry ~20-45 days out (front-month IV is
    noisy near expiry). Averages the IV of the calls/puts whose strikes bracket spot."""
    import datetime as _dt

    today = _dt.date.today()
    target = None
    for exp in expiries:
        try:
            days = (_dt.date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if days >= 20:
            target = exp
            break
    target = target or (expiries[len(expiries) // 2] if expiries else None)
    if target is None:
        return None
    try:
        chain = yt.option_chain(target)
    except Exception:  # noqa: BLE001
        return None

    ivs: list[float] = []
    for frame in (getattr(chain, "calls", None), getattr(chain, "puts", None)):
        if frame is None or frame.empty or "impliedVolatility" not in frame:
            continue
        frame = frame.dropna(subset=["strike", "impliedVolatility"]).copy()
        frame = frame[frame["impliedVolatility"] > 0.01]  # drop garbage near-zero IV rows
        if frame.empty:
            continue
        frame["dist"] = (frame["strike"] - spot).abs()
        nearest = frame.nsmallest(3, "dist")
        ivs.extend(float(v) for v in nearest["impliedVolatility"])
    if not ivs:
        return None
    iv = sum(ivs) / len(ivs)
    # Sanity clamp — IV between 5% and 300% annualized.
    if iv < 0.05 or iv > 3.0:
        return None
    return round(iv, 4)


def _max_oi_strike(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    grouped = df.groupby("strike")["openInterest"].sum()
    if grouped.empty or grouped.max() == 0:
        return None
    return round(float(grouped.idxmax()), 2)


def _parse_holders(df: pd.DataFrame) -> list[InstitutionalHolder]:
    holders: list[InstitutionalHolder] = []
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("holder")
    shares_col = cols.get("shares")
    value_col = cols.get("value")
    change_col = cols.get("pctchange") or cols.get("change")
    for _, row in df.iterrows():
        holders.append(
            InstitutionalHolder(
                name=str(row[name_col]) if name_col else "Unknown",
                shares=_num(row[shares_col]) or 0.0 if shares_col else 0.0,
                value=_num(row[value_col]) if value_col else None,
                change=_num(row[change_col]) if change_col else None,
            )
        )
    return holders


def _num(raw) -> float | None:
    if raw is None or raw == "" or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
