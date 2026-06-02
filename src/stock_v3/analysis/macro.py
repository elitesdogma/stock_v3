"""Layer 1 — macro environment classification and scoring.

Pure functions over a MacroSnapshot. Produces the overall regime (Risk-On / Neutral /
Risk-Off) and a 1-10 macro score with a human-readable rationale. When the snapshot is
missing entirely (no FRED key), callers neutralize the category rather than calling here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import MacroSnapshot, Regime


@dataclass
class MacroAssessment:
    regime: Regime
    vix_regime: str
    rate_environment: str
    inflation_trend: str
    dollar_bias: str
    score: float  # 1..10
    rationale: str


def assess_macro(macro: MacroSnapshot) -> MacroAssessment:
    vix_regime = _vix_regime(macro.vix)
    rate_env = _rate_environment(macro.yield_curve_10y_2y, macro.ust_10y)
    inflation = _inflation_trend(macro.cpi_yoy, macro.core_cpi_yoy)
    dollar = _dollar_bias(macro.dxy_trend_3m)

    score, regime = _score(macro, vix_regime, inflation)
    rationale = (
        f"VIX {_fmt(macro.vix)} ({vix_regime}); "
        f"10y {_fmt(macro.ust_10y)}%, curve {_fmt(macro.yield_curve_10y_2y)}pp ({rate_env}); "
        f"CPI {_pct(macro.cpi_yoy)} ({inflation}); USD {dollar}."
    )
    return MacroAssessment(
        regime=regime,
        vix_regime=vix_regime,
        rate_environment=rate_env,
        inflation_trend=inflation,
        dollar_bias=dollar,
        score=score,
        rationale=rationale,
    )


def _vix_regime(vix: float | None) -> str:
    if vix is None:
        return "Unknown"
    if vix < 15:
        return "Low"
    if vix < 20:
        return "Normal"
    if vix < 30:
        return "Elevated"
    return "Risk-Off"


def _rate_environment(curve: float | None, ten_year: float | None) -> str:
    if curve is None:
        return "Unknown"
    if curve < 0:
        return "Inverted curve (late-cycle / recession signal)"
    if curve < 0.5:
        return "Flat curve"
    return "Normal positive curve"


def _inflation_trend(cpi: float | None, core: float | None) -> str:
    ref = core if core is not None else cpi
    if ref is None:
        return "Unknown"
    if ref > 0.04:
        return "Elevated / sticky"
    if ref > 0.025:
        return "Moderating but above target"
    return "Near target"


def _dollar_bias(trend: float | None) -> str:
    if trend is None:
        return "neutral"
    if trend > 0.02:
        return "strengthening (headwind for commodities/EM/exporters)"
    if trend < -0.02:
        return "weakening (tailwind for commodities/EM/exporters)"
    return "range-bound"


def _score(
    macro: MacroSnapshot, vix_regime: str, inflation: str
) -> tuple[float, Regime]:
    """Blend volatility, rates and inflation into a 1-10 macro tailwind score."""
    score = 5.0

    score += {"Low": 2.0, "Normal": 1.0, "Elevated": -1.5, "Risk-Off": -3.0,
              "Unknown": 0.0}[vix_regime]

    curve = macro.yield_curve_10y_2y
    if curve is not None:
        if curve < 0:
            score -= 1.5  # inversion is a risk-off tilt for equities broadly
        elif curve > 0.5:
            score += 0.5

    if inflation == "Near target":
        score += 1.0
    elif inflation == "Elevated / sticky":
        score -= 1.0

    score = max(1.0, min(10.0, score))

    if score >= 6.5:
        regime = Regime.RISK_ON
    elif score >= 4.0:
        regime = Regime.NEUTRAL
    else:
        regime = Regime.RISK_OFF
    return round(score, 1), regime


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"
