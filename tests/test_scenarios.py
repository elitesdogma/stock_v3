"""Scenario probabilities must sum to exactly 1.00 and targets must straddle spot."""

from __future__ import annotations

import pytest

from stock_v3.analysis.scenarios import (
    ScenarioInputs,
    build_scenarios,
    normalize_probabilities,
)


@pytest.mark.parametrize("tilt", [0.0, 2.5, 5.0, 7.5, 10.0])
def test_probabilities_sum_to_one(tilt):
    scenarios = normalize_probabilities(
        build_scenarios(ScenarioInputs(price=100.0, atr=3.0, target_high=140.0,
                                       target_low=80.0, tilt_score=tilt))
    )
    total = sum(s.probability for s in scenarios)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_bull_above_bear_below_spot():
    scenarios = build_scenarios(
        ScenarioInputs(price=100.0, atr=2.0, target_high=130.0, target_low=85.0,
                       tilt_score=6.0)
    )
    by_name = {s.name: s for s in scenarios}
    assert by_name["Bull"].price_target > 100.0
    assert by_name["Bear"].price_target < 100.0
    assert by_name["Base"].price_target > 0


def test_bullish_tilt_raises_bull_probability():
    bullish = build_scenarios(
        ScenarioInputs(price=100, atr=2, target_high=None, target_low=None, tilt_score=9.0)
    )
    bearish = build_scenarios(
        ScenarioInputs(price=100, atr=2, target_high=None, target_low=None, tilt_score=1.0)
    )
    p_bull = next(s.probability for s in bullish if s.name == "Bull")
    p_bull_bear_tilt = next(s.probability for s in bearish if s.name == "Bull")
    assert p_bull > p_bull_bear_tilt


def test_no_atr_falls_back_gracefully():
    scenarios = build_scenarios(
        ScenarioInputs(price=50.0, atr=None, target_high=None, target_low=None, tilt_score=5.0)
    )
    assert all(s.price_target is not None for s in scenarios)
