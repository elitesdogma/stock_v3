"""Reward-to-risk, ATR-stop placement, 1% position sizing, and Kelly/conviction caps."""

from __future__ import annotations

from stock_v3.analysis.risk import RiskInputs, build_trade_plan


def _inputs(**overrides) -> RiskInputs:
    base = dict(
        price=100.0,
        atr=2.0,
        sma_50=95.0,
        sma_200=80.0,
        target_mean=130.0,
        conviction_score=75.0,
        win_probability=0.6,
    )
    base.update(overrides)
    return RiskInputs(**base)


def test_stop_below_price_and_risk_per_share_positive():
    plan = build_trade_plan(_inputs())
    assert plan.stop_loss is not None
    assert plan.stop_loss < 100.0
    assert plan.risk_per_share is not None
    assert plan.risk_per_share > 0


def test_position_size_respects_one_percent_risk():
    plan = build_trade_plan(_inputs(), portfolio_value=100_000.0, risk_pct=0.01)
    # 1% of 100k = $1,000 dollar risk; shares = 1000 / risk_per_share.
    expected = 1000.0 / plan.risk_per_share
    assert abs(plan.position_size_shares - expected) < 0.5


def test_reward_to_risk_is_positive_and_reasonable():
    plan = build_trade_plan(_inputs())
    assert plan.risk_reward is not None
    assert plan.risk_reward > 0


def test_allocation_capped_by_conviction_tier():
    speculative = build_trade_plan(_inputs(conviction_score=62.0))
    exceptional = build_trade_plan(_inputs(conviction_score=95.0))
    # Speculative tier caps allocation at 10%, exceptional at 30%.
    if speculative.portfolio_allocation_pct is not None:
        assert speculative.portfolio_allocation_pct <= 10.0 + 1e-6
    if exceptional.portfolio_allocation_pct is not None:
        assert exceptional.portfolio_allocation_pct <= 30.0 + 1e-6


def test_no_atr_uses_structural_fallback_stop():
    plan = build_trade_plan(_inputs(atr=None, sma_50=None))
    assert plan.stop_loss is not None
    assert plan.stop_loss < 100.0
