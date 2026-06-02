"""Layer 6 — quantitative risk framework: stop placement, reward-to-risk, position sizing,
and Kelly-capped allocation.

Stops are ATR-anchored (volatility-aware) with a structural fallback to the 50-day MA.
Position size follows the 1%-portfolio-risk rule; allocation is bounded by 25% fractional
Kelly and the conviction-tier ceilings from the framework.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TradePlan


@dataclass
class RiskInputs:
    price: float
    atr: float | None
    sma_50: float | None
    sma_200: float | None
    target_mean: float | None  # analyst consensus target, if available
    conviction_score: float  # 0..100, from the scorecard
    win_probability: float  # base-case + bull-case probability mass


def build_trade_plan(
    r: RiskInputs, *, portfolio_value: float = 100_000.0, risk_pct: float = 0.01
) -> TradePlan:
    stop = _stop(r)
    risk_per_share = max(r.price - stop, 0.01) if stop is not None else None

    targets = _targets(r, risk_per_share)
    rr = _reward_to_risk(r.price, stop, targets)

    allocation_pct = _allocation(r, rr)
    position_shares = _position_size(portfolio_value, risk_pct, risk_per_share)
    entry_zone = _entry_zone(r)

    return TradePlan(
        entry_zone=entry_zone,
        ideal_buy=round(r.sma_50, 2) if r.sma_50 and r.sma_50 < r.price else round(r.price, 2),
        stop_loss=round(stop, 2) if stop is not None else None,
        targets=[round(t, 2) for t in targets],
        risk_reward=round(rr, 2) if rr is not None else None,
        holding_period=_holding_period(r.conviction_score),
        portfolio_allocation_pct=round(allocation_pct, 2) if allocation_pct else None,
        position_size_shares=round(position_shares, 2) if position_shares else None,
        risk_per_share=round(risk_per_share, 2) if risk_per_share else None,
    )


def _stop(r: RiskInputs) -> float | None:
    """ATR-based stop (2x ATR below price), floored to not sit above a structural support
    that's already closer. Falls back to a fixed 8% stop when ATR is unavailable."""
    candidates = []
    if r.atr is not None and r.atr > 0:
        candidates.append(r.price - 2.0 * r.atr)
    if r.sma_50 is not None and r.sma_50 < r.price:
        candidates.append(r.sma_50 * 0.985)  # just below the 50d
    if not candidates:
        return r.price * 0.92  # 8% structural fallback
    # Choose the tightest sensible stop that still gives the trade room (the highest stop
    # below price), so risk-per-share isn't needlessly wide.
    valid = [c for c in candidates if c < r.price]
    return max(valid) if valid else r.price * 0.92


def _targets(r: RiskInputs, risk_per_share: float | None) -> list[float]:
    """Three laddered targets. Prefer R-multiples off the stop (1.5R/3R/5R); blend in the
    analyst mean target when present so the ladder is anchored to real expectations."""
    if risk_per_share is None or risk_per_share <= 0:
        base = r.price
        return [base * 1.08, base * 1.18, base * 1.30]
    ladder = [r.price + m * risk_per_share for m in (1.5, 3.0, 5.0)]
    if r.target_mean is not None and r.target_mean > r.price:
        # Slot the consensus target into the ladder rather than ignoring it.
        ladder[1] = (ladder[1] + r.target_mean) / 2
    return ladder


def _reward_to_risk(price: float, stop: float | None, targets: list[float]) -> float | None:
    if stop is None or not targets:
        return None
    risk = price - stop
    if risk <= 0:
        return None
    primary_target = targets[1] if len(targets) > 1 else targets[0]
    reward = primary_target - price
    if reward <= 0:
        return None
    return reward / risk


def _allocation(r: RiskInputs, rr: float | None) -> float | None:
    """Kelly-capped allocation. Fractional Kelly (25%) sets the ceiling; conviction tier
    caps it further per the framework (10% speculative / 20% standard / 30% exceptional)."""
    if rr is None or rr <= 0:
        return None
    p = max(0.0, min(1.0, r.win_probability))
    b = rr
    kelly = (p * (b + 1) - 1) / b  # full Kelly fraction
    fractional = max(0.0, kelly * 0.25)

    tier_cap = _conviction_cap(r.conviction_score)
    return min(fractional, tier_cap) * 100.0


def _conviction_cap(conviction: float) -> float:
    if conviction >= 90:
        return 0.30
    if conviction >= 70:
        return 0.20
    return 0.10


def _position_size(
    portfolio_value: float, risk_pct: float, risk_per_share: float | None
) -> float | None:
    if risk_per_share is None or risk_per_share <= 0:
        return None
    dollar_risk = portfolio_value * risk_pct
    return dollar_risk / risk_per_share


def _entry_zone(r: RiskInputs) -> tuple[float, float] | None:
    if r.atr is None or r.atr <= 0:
        return (round(r.price * 0.98, 2), round(r.price * 1.01, 2))
    low = r.price - 0.5 * r.atr
    high = r.price + 0.25 * r.atr
    return (round(low, 2), round(high, 2))


def _holding_period(conviction: float) -> str:
    if conviction >= 80:
        return "6-18 months (position trade)"
    if conviction >= 65:
        return "2-6 months (swing)"
    return "weeks (tactical, tight management)"
