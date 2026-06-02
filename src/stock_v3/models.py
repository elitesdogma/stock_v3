"""Domain models flowing through the pipeline: sources populate them, analysis reads
them and emits scores, the renderer consumes the assembled ResearchReport.

Raw source payloads are wrapped in SourceResult (see sources/base.py); the analysis
outputs here are plain dataclasses because by that point a value has either been
derived or explicitly neutralized.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Raw / semi-raw payloads produced by sources
# --------------------------------------------------------------------------- #
@dataclass
class Quote:
    ticker: str
    company: str
    price: float
    currency: str
    market_cap: float | None
    sector: str | None
    industry: str | None
    exchange: str | None
    shares_outstanding: float | None
    as_of: dt.datetime


@dataclass
class Fundamentals:
    revenue_ttm: float | None
    revenue_growth_yoy: float | None  # fraction, e.g. 0.18 == +18%
    revenue_growth_qoq: float | None
    gross_margin: float | None
    operating_margin: float | None
    ebitda_margin: float | None
    net_margin: float | None
    fcf_margin: float | None
    fcf_ttm: float | None
    cash: float | None
    total_debt: float | None
    net_debt: float | None
    current_ratio: float | None
    pe: float | None
    forward_pe: float | None
    ev_ebitda: float | None
    price_to_sales: float | None
    peg: float | None
    price_to_fcf: float | None


@dataclass
class MacroSnapshot:
    ust_2y: float | None
    ust_10y: float | None
    yield_curve_10y_2y: float | None  # 10y minus 2y, in pct points
    cpi_yoy: float | None
    core_cpi_yoy: float | None
    ppi_yoy: float | None
    vix: float | None
    dxy: float | None
    dxy_trend_3m: float | None  # fractional change over ~3 months
    breadth_pct_above_200d: float | None  # share of an index above its 200d, if computed
    as_of: dt.date | None


@dataclass
class InsiderEvent:
    date: dt.date
    insider: str
    role: str
    transaction: str  # "buy" / "sell"
    shares: float
    value: float | None


@dataclass
class InstitutionalHolder:
    name: str
    shares: float
    value: float | None
    change: float | None  # share delta vs prior filing if known


@dataclass
class Positioning:
    institutional_holders: list[InstitutionalHolder]
    institutional_total_change: float | None  # net share delta across reported holders
    holders_vintage: dt.date | None  # 13F is ~45 days stale by regulation
    insider_events: list[InsiderEvent]
    insider_net_value_90d: float | None  # buys minus sells, last ~90 days
    short_interest_pct_float: float | None
    days_to_cover: float | None
    short_interest_settlement: dt.date | None
    borrow_rate: float | None  # almost always None on free data


@dataclass
class OptionsSnapshot:
    put_call_ratio: float | None
    gamma_flip: float | None  # estimated spot where net dealer gamma crosses zero
    call_wall: float | None
    put_wall: float | None
    atm_iv: float | None = None  # annualized at-the-money implied volatility (fraction)
    is_estimate: bool = True  # free-data GEX is always a proxy, never authoritative


@dataclass
class ConeBand:
    """One horizon's worth of the probability cone."""

    weeks: int
    days: int
    expected_move_pct: float  # 1σ move as a fraction of spot
    p10: float  # 10th-percentile price (~-1.28σ)
    p25: float
    p50: float  # median (lognormal)
    p75: float
    p90: float  # 90th-percentile price (~+1.28σ)
    low_1sd: float  # -1σ price band
    high_1sd: float  # +1σ price band


@dataclass
class ProbabilityCone:
    """Industry-standard volatility cone: projects the price distribution forward using
    implied (or realized-fallback) volatility. Powers the scenario graph + interactive slider."""

    spot: float
    annual_vol: float  # the σ used (annualized)
    vol_source: str  # "implied (ATM options)" or "realized (1y returns)"
    bands: list[ConeBand]
    max_weeks: int


@dataclass
class Technicals:
    ema_8: float | None
    ema_21: float | None
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    stoch_rsi: float | None
    atr_14: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_width: float | None  # (upper-lower)/mid — compression vs expansion
    volume: float | None
    avg_volume_20d: float | None
    # weekly-timeframe trend confirmation
    weekly_sma_50: float | None
    weekly_close: float | None
    price_history: list[tuple[dt.date, float]] = field(default_factory=list)  # for charts


@dataclass
class CatalystEvent:
    date: dt.date | None
    label: str
    kind: str  # "earnings" / "macro" / "corporate"
    importance: str  # "high" / "medium" / "low"


@dataclass
class AnalystConsensus:
    rating: str | None  # e.g. "Buy"
    rating_score: float | None  # 1 (strong buy) .. 5 (strong sell) style, normalized below
    target_mean: float | None
    target_high: float | None
    target_low: float | None
    num_analysts: int | None
    recent_upgrades: int | None
    recent_downgrades: int | None


# --------------------------------------------------------------------------- #
# Analysis outputs
# --------------------------------------------------------------------------- #
class Regime(str, Enum):
    RISK_ON = "Risk-On"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk-Off"


@dataclass
class CategoryScore:
    """One row of the Institutional Scorecard."""

    name: str
    score: float  # 1..10
    rationale: str
    neutralized: bool = False  # True when inputs were unavailable and we defaulted to 5


@dataclass
class Scenario:
    name: str  # Bull / Base / Bear
    price_target: float | None
    probability: float  # 0..1
    drivers: str


@dataclass
class TradePlan:
    entry_zone: tuple[float, float] | None
    ideal_buy: float | None
    stop_loss: float | None
    targets: list[float]
    risk_reward: float | None
    holding_period: str
    portfolio_allocation_pct: float | None  # suggested, capped by Kelly + conviction
    position_size_shares: float | None
    risk_per_share: float | None


@dataclass
class Verdict:
    conviction_score: float  # 0..100
    label: str  # Strong Buy .. Avoid
    confidence: float  # 0..100, blends signal strength + data completeness
    confidence_band: str  # Exceptional Conviction .. Insufficient Edge


# --------------------------------------------------------------------------- #
# Coverage panel + the fully assembled report
# --------------------------------------------------------------------------- #
@dataclass
class CoverageEntry:
    layer: str
    source: str
    status: str  # ok / stale / unavailable
    as_of: str
    note: str | None


@dataclass
class NarrativeSections:
    """Prose sections — either LLM-written or deterministic templates. Keyed by section."""

    executive_summary: str
    thesis: str
    bull: str
    base: str
    bear: str
    risks: str
    portfolio_impact: str
    generated_by: str  # "claude" or "deterministic-template"


@dataclass
class ResearchReport:
    generated_at: dt.datetime
    quote: Quote
    fundamentals: Fundamentals | None
    macro: MacroSnapshot | None
    macro_regime: Regime
    positioning: Positioning | None
    options: OptionsSnapshot | None
    technicals: Technicals | None
    consensus: AnalystConsensus | None
    catalysts: list[CatalystEvent]
    scenarios: list[Scenario]
    trade_plan: TradePlan
    scorecard: list[CategoryScore]
    verdict: Verdict
    coverage: list[CoverageEntry]
    narrative: NarrativeSections
    key_risks: list[str]
    probability_cone: ProbabilityCone | None = None
