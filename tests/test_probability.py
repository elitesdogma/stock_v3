"""Probability-cone math: lognormal P(above), cone monotonicity, vol-source fallback."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_v3.analysis.probability import (
    build_probability_cone,
    expected_move,
    prob_above,
    prob_below,
)


def test_prob_above_at_spot_is_about_half():
    # At the money, zero-drift lognormal sits just under 50% (the −½σ²t drag).
    p = prob_above(100.0, 100.0, annual_vol=0.30, days=30)
    assert 0.45 < p < 0.50


def test_prob_above_monotonic_in_target():
    spot, vol, days = 100.0, 0.35, 60
    higher = prob_above(spot, 120.0, vol, days)
    lower = prob_above(spot, 90.0, vol, days)
    assert higher < lower  # harder to exceed a higher target
    assert 0.0 < higher < lower < 1.0


def test_prob_above_and_below_sum_to_one():
    p_up = prob_above(250.0, 270.0, 0.4, 45)
    p_dn = prob_below(250.0, 270.0, 0.4, 45)
    assert p_up + p_dn == pytest.approx(1.0, abs=1e-9)


def test_prob_above_more_certain_with_longer_horizon():
    # A far-OTM target becomes more reachable as horizon (and dispersion) grows.
    near = prob_above(100.0, 130.0, 0.5, 7)
    far = prob_above(100.0, 130.0, 0.5, 90)
    assert far > near


def test_expected_move_scales_with_sqrt_time():
    m1 = expected_move(100.0, 0.32, 30)
    m4 = expected_move(100.0, 0.32, 120)
    # 4x the time → 2x the move (sqrt scaling).
    assert m4 / m1 == pytest.approx(2.0, abs=0.05)


def test_cone_uses_implied_vol_when_present():
    cone = build_probability_cone(100.0, implied_vol=0.40, daily_history=None)
    assert cone is not None
    assert cone.annual_vol == 0.40
    assert "implied" in cone.vol_source
    assert cone.bands[0].weeks == 2
    # bands widen with horizon
    assert cone.bands[-1].expected_move_pct > cone.bands[0].expected_move_pct


def test_cone_falls_back_to_realized_vol():
    # Synthetic random-walk history → realized vol fallback when no IV.
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(rets))
    hist = pd.DataFrame({"Close": prices},
                        index=pd.date_range("2025-01-01", periods=252, freq="B"))
    cone = build_probability_cone(float(prices[-1]), implied_vol=None, daily_history=hist)
    assert cone is not None
    assert "realized" in cone.vol_source
    assert cone.annual_vol > 0


def test_cone_bands_ordered():
    cone = build_probability_cone(100.0, implied_vol=0.30, daily_history=None)
    for band in cone.bands:
        assert band.p10 < band.p25 < band.p50 < band.p75 < band.p90
        assert band.low_1sd < band.p50 < band.high_1sd


def test_cone_none_without_any_vol():
    assert build_probability_cone(100.0, implied_vol=None, daily_history=None) is None
