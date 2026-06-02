"""Scenario analysis — bull / base / bear price targets and probabilities.

Targets are anchored to volatility (ATR) and, where available, the analyst target band.
Probabilities are tilted by the blended technical+fundamental score and ALWAYS renormalize
to sum to exactly 1.0 — the framework requires probabilities to total 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Scenario


@dataclass
class ScenarioInputs:
    price: float
    atr: float | None
    target_high: float | None
    target_low: float | None
    tilt_score: float  # 0..10 blended technical+fundamental lean (5 = neutral)


def build_scenarios(s: ScenarioInputs) -> list[Scenario]:
    bull_px, base_px, bear_px = _price_targets(s)
    p_bull, p_base, p_bear = _probabilities(s.tilt_score)

    return [
        Scenario(
            name="Bull",
            price_target=round(bull_px, 2),
            probability=p_bull,
            drivers="Trend continuation, multiple expansion, positive catalyst surprise.",
        ),
        Scenario(
            name="Base",
            price_target=round(base_px, 2),
            probability=p_base,
            drivers="In-line fundamentals, range-respecting price action, no regime shock.",
        ),
        Scenario(
            name="Bear",
            price_target=round(bear_px, 2),
            probability=p_bear,
            drivers="Guidance cut, multiple compression, macro/liquidity deterioration.",
        ),
    ]


def _price_targets(s: ScenarioInputs) -> tuple[float, float, float]:
    atr = s.atr if s.atr and s.atr > 0 else s.price * 0.03
    bull = s.price + 8 * atr
    bear = s.price - 6 * atr
    base = s.price + 1 * atr

    # Anchor extremes to the analyst band when present so they're not pure volatility math.
    if s.target_high is not None and s.target_high > s.price:
        bull = (bull + s.target_high) / 2
    if s.target_low is not None and s.target_low < s.price:
        bear = (bear + s.target_low) / 2

    bear = min(bear, s.price * 0.97)  # bear must sit below spot
    bull = max(bull, s.price * 1.03)  # bull must sit above spot
    return bull, base, bear


def _probabilities(tilt: float) -> tuple[float, float, float]:
    """Base case always carries the most mass; bull/bear shift with the tilt. Renormalized
    to sum to exactly 1.0."""
    lean = (tilt - 5.0) / 5.0  # -1 (max bear) .. +1 (max bull)
    p_base = 0.50
    p_bull = 0.25 + 0.15 * lean
    p_bear = 0.25 - 0.15 * lean

    p_bull = max(0.05, p_bull)
    p_bear = max(0.05, p_bear)
    total = p_bull + p_base + p_bear
    return (
        round(p_bull / total, 2),
        round(p_base / total, 2),
        round(p_bear / total, 2),
    )


def normalize_probabilities(scenarios: list[Scenario]) -> list[Scenario]:
    """Final guard: force probabilities to sum to exactly 1.00 after any rounding drift."""
    total = sum(s.probability for s in scenarios)
    if total == 0:
        return scenarios
    adjusted = [
        Scenario(s.name, s.price_target, s.probability / total, s.drivers) for s in scenarios
    ]
    # Push residual rounding error onto the base case so the visible numbers sum to 1.00.
    drift = round(1.0 - sum(round(s.probability, 2) for s in adjusted), 2)
    for s in adjusted:
        if s.name == "Base":
            s.probability = round(s.probability + drift, 2)
        else:
            s.probability = round(s.probability, 2)
    return adjusted
