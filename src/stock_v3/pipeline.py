"""Orchestration: fetch every source, run the deterministic analysis, assemble a
ResearchReport. The CLI stays thin; all the wiring (including coverage-panel construction
and scorecard neutralization of unavailable categories) lives here.
"""

from __future__ import annotations

import datetime as dt

from .analysis import catalysts as catalysts_mod
from .analysis import fundamentals as fundamentals_mod
from .analysis import macro as macro_mod
from .analysis import positioning as positioning_mod
from .analysis import probability as probability_mod
from .analysis import risk as risk_mod
from .analysis import scenarios as scenarios_mod
from .analysis import technical_assessment as tech_assess_mod
from .analysis import technicals as technicals_mod
from .analysis.scorecard import compute_verdict, neutralized_score
from .cache import Cache
from .config import Settings
from .models import (
    CategoryScore,
    CoverageEntry,
    NarrativeSections,
    Regime,
    ResearchReport,
    Verdict,
)
from .narrative.llm import build_narrative
from .sources import edgar, finnhub_src, finra, fred, prices
from .sources.base import SourceResult, Status, freshness_label


def run_pipeline(
    ticker: str, settings: Settings, cache: Cache, *, use_llm: bool = False
) -> ResearchReport:
    ticker = ticker.upper().strip()

    # ---- fetch everything (each returns a SourceResult; failures degrade) ----
    quote_r = prices.fetch_quote(ticker, settings, cache)
    if not quote_r.usable:
        raise PipelineError(
            f"Cannot analyze {ticker}: no price data ({quote_r.note}). "
            "Check the symbol and your network."
        )
    quote = quote_r.value

    fundamentals_r = prices.fetch_fundamentals(ticker, settings, cache)
    history_r = prices.fetch_history(ticker, settings, cache)
    holders_r = prices.fetch_institutional_holders(ticker, settings, cache)
    options_r = prices.fetch_options_proxy(ticker, quote.price, settings, cache)
    short_r = finra.fetch_short_interest(ticker, settings, cache)
    insider_r = edgar.fetch_insider_activity(ticker, settings, cache)
    macro_r = fred.fetch_macro(settings, cache)
    consensus_r = finnhub_src.fetch_consensus(ticker, settings, cache)
    earnings_r = finnhub_src.fetch_earnings_catalysts(ticker, settings, cache)

    # ---- analysis ----
    technicals = technicals_mod.compute_technicals(history_r.value) if history_r.usable else None
    tech_assessment = (
        tech_assess_mod.assess_technicals(technicals, quote.price) if technicals else None
    )
    fund_assessment = (
        fundamentals_mod.assess_fundamentals(fundamentals_r.value)
        if fundamentals_r.usable else None
    )
    macro_assessment = macro_mod.assess_macro(macro_r.value) if macro_r.usable else None
    pos_assessment = positioning_mod.assess_positioning(
        holders=holders_r.value if holders_r.usable else None,
        holders_vintage=holders_r.as_of if holders_r.usable else None,
        insiders=insider_r.value if insider_r.usable else None,
        short_pct_float=short_r.value.short_pct_float if short_r.usable else None,
        days_to_cover=short_r.value.days_to_cover if short_r.usable else None,
        short_settlement=short_r.value.settlement if short_r.usable else None,
        borrow_rate=None,
    )
    cat_assessment = catalysts_mod.assess_catalysts(
        earnings_r.value if earnings_r.usable else []
    )

    # ---- scorecard (neutralize categories whose inputs are missing) ----
    scorecard = _build_scorecard(
        macro_assessment, fund_assessment, tech_assessment, pos_assessment, cat_assessment
    )

    # Risk/Reward needs the conviction-from-everything-else, so compute a provisional verdict
    # first, build the trade plan, then fold Risk/Reward back in for the final verdict.
    provisional = compute_verdict(scorecard)
    tilt = _blend_tilt(tech_assessment, fund_assessment)
    scenarios = scenarios_mod.normalize_probabilities(
        scenarios_mod.build_scenarios(
            scenarios_mod.ScenarioInputs(
                price=quote.price,
                atr=technicals.atr_14 if technicals else None,
                target_high=consensus_r.value.target_high if consensus_r.usable else None,
                target_low=consensus_r.value.target_low if consensus_r.usable else None,
                tilt_score=tilt,
            )
        )
    )
    win_prob = _win_probability(scenarios)
    trade_plan = risk_mod.build_trade_plan(
        risk_mod.RiskInputs(
            price=quote.price,
            atr=technicals.atr_14 if technicals else None,
            sma_50=technicals.sma_50 if technicals else None,
            sma_200=technicals.sma_200 if technicals else None,
            target_mean=consensus_r.value.target_mean if consensus_r.usable else None,
            conviction_score=provisional.conviction_score,
            win_probability=win_prob,
        )
    )
    _apply_risk_reward_score(scorecard, trade_plan.risk_reward)
    verdict = compute_verdict(scorecard)

    # ---- probability cone (implied vol, realized-vol fallback) ----
    cone = probability_mod.build_probability_cone(
        quote.price,
        implied_vol=options_r.value.atm_iv if options_r.usable else None,
        daily_history=history_r.value if history_r.usable else None,
    )

    # ---- coverage panel + assembly ----
    coverage = _build_coverage(
        macro_r, holders_r, insider_r, short_r, options_r, consensus_r, history_r, fundamentals_r
    )
    macro_regime = macro_assessment.regime if macro_assessment else Regime.NEUTRAL
    key_risks = _key_risks(
        quote, fund_assessment, tech_assessment, pos_assessment, macro_assessment, verdict
    )

    narrative = build_narrative(
        ticker=ticker,
        quote=quote,
        verdict=verdict,
        scorecard=scorecard,
        scenarios=scenarios,
        macro_regime=macro_regime,
        key_risks=key_risks,
        settings=settings,
        use_llm=use_llm,
    )

    return ResearchReport(
        generated_at=dt.datetime.now(),
        quote=quote,
        fundamentals=fundamentals_r.value if fundamentals_r.usable else None,
        macro=macro_r.value if macro_r.usable else None,
        macro_regime=macro_regime,
        positioning=pos_assessment.positioning,
        options=options_r.value if options_r.usable else None,
        technicals=technicals,
        consensus=consensus_r.value if consensus_r.usable else None,
        catalysts=cat_assessment.events,
        scenarios=scenarios,
        trade_plan=trade_plan,
        scorecard=scorecard,
        verdict=verdict,
        coverage=coverage,
        narrative=narrative,
        key_risks=key_risks,
        probability_cone=cone,
    )


class PipelineError(RuntimeError):
    """Raised when analysis cannot proceed (e.g. no price for the symbol)."""


# --------------------------------------------------------------------------- #
def _build_scorecard(macro, fund, tech, pos, cat) -> list[CategoryScore]:
    rows: list[CategoryScore] = []

    rows.append(
        CategoryScore("Macro Environment", macro.score, macro.rationale)
        if macro else neutralized_score("Macro Environment", "no FRED key / macro data")
    )
    rows.append(
        CategoryScore("Fundamentals", fund.fundamentals_score, fund.fundamentals_rationale)
        if fund else neutralized_score("Fundamentals", "fundamentals unavailable")
    )
    rows.append(
        CategoryScore("Valuation", fund.valuation_score, fund.valuation_rationale)
        if fund else neutralized_score("Valuation", "valuation inputs unavailable")
    )
    rows.append(
        CategoryScore("Technicals", tech.score, tech.rationale)
        if tech else neutralized_score("Technicals", "no price history")
    )
    # Positioning always produces an assessment, but if it has no real signal we still show it.
    rows.append(CategoryScore("Institutional Positioning", pos.score, pos.rationale))
    rows.append(CategoryScore("Catalysts", cat.score, cat.rationale))
    # Risk/Reward placeholder; filled by _apply_risk_reward_score once the plan exists.
    rows.append(neutralized_score("Risk/Reward", "pending trade-plan computation"))
    return rows


def _apply_risk_reward_score(scorecard: list[CategoryScore], rr: float | None) -> None:
    for i, row in enumerate(scorecard):
        if row.name != "Risk/Reward":
            continue
        if rr is None:
            return  # leave neutralized
        if rr >= 3.0:
            score, note = 8.5, f"{rr:.1f}:1 — exceeds 1:3 target"
        elif rr >= 2.0:
            score, note = 7.0, f"{rr:.1f}:1 — meets 1:2 minimum"
        elif rr >= 1.0:
            score, note = 4.5, f"{rr:.1f}:1 — below 1:2, flagged"
        else:
            score, note = 3.0, f"{rr:.1f}:1 — unfavorable"
        scorecard[i] = CategoryScore("Risk/Reward", score, note)
        return


def _blend_tilt(tech, fund) -> float:
    """Blend technical + fundamental scores into a 0-10 directional lean for scenarios."""
    parts = []
    if tech is not None:
        parts.append(tech.score)
    if fund is not None:
        parts.append((fund.fundamentals_score + fund.valuation_score) / 2)
    if not parts:
        return 5.0
    return sum(parts) / len(parts)


def _win_probability(scenarios) -> float:
    """Base + bull probability mass — the chance the trade works, for Kelly sizing."""
    prob = 0.0
    for s in scenarios:
        if s.name in ("Base", "Bull"):
            prob += s.probability
    return max(0.1, min(0.9, prob))


def _build_coverage(
    macro_r, holders_r, insider_r, short_r, options_r, consensus_r, history_r, fundamentals_r
) -> list[CoverageEntry]:
    def entry(layer: str, r: SourceResult) -> CoverageEntry:
        return CoverageEntry(
            layer=layer,
            source=r.source,
            status=r.status.value,
            as_of=freshness_label(r),
            note=r.note,
        )

    return [
        entry("Price / Fundamentals", fundamentals_r),
        entry("Technicals (OHLCV)", history_r),
        entry("Macro", macro_r),
        entry("Institutional 13F", holders_r),
        entry("Insider (Form 4)", insider_r),
        entry("Short interest", short_r),
        entry("Options (gamma proxy)", options_r),
        entry("Analyst consensus", consensus_r),
    ]


def _key_risks(quote, fund, tech, pos, macro, verdict) -> list[str]:
    risks: list[str] = []
    if fund is not None:
        if fund.valuation_class == "Overvalued":
            risks.append("Elevated valuation amplifies downside on any guidance miss or "
                         "multiple compression.")
        if fund.balance_sheet and ("tight" in fund.balance_sheet.lower()
                                    or "leveraged" in fund.balance_sheet.lower()):
            risks.append(f"Balance-sheet risk: {fund.balance_sheet.lower()}.")
        if fund.fcf_class == "Cash-burning":
            risks.append("Negative free cash flow raises dilution/refinancing risk.")
    if tech is not None and tech.trend == "Downtrend":
        risks.append("Price is in a confirmed downtrend; trend is not yet a tailwind.")
    if macro is not None and macro.regime == Regime.RISK_OFF:
        risks.append(f"Risk-off macro regime ({macro.vix_regime} volatility) is a headwind "
                     "for beta and multiples.")
    if pos is not None and "selling" in pos.insider_signal:
        risks.append("Insiders are net sellers over the trailing window.")
    if pos is not None and "elevated squeeze" in pos.squeeze_read:
        risks.append("Crowded short interest can drive violent two-sided volatility.")
    if verdict.confidence < 60:
        risks.append("Confidence is below the institutional edge threshold — data gaps or "
                     "mixed signals; size conservatively.")
    if not risks:
        risks.append("No single dominant risk flagged; standard market, liquidity, and "
                     "single-name concentration risks apply.")
    return risks
