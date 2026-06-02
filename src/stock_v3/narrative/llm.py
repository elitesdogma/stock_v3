"""Narrative generation — deterministic templates by default, optional Claude enrichment.

The narrative is STRICTLY additive: it receives already-computed numbers (verdict, scores,
scenarios) and writes prose around them. It never changes a score, probability, or rating.

Without an Anthropic key (or without --narrative), deterministic templates fill every prose
section so the report's shape is identical — only the wording differs. With the key, Claude
rewrites the same sections in richer analyst prose, fed the deterministic facts as ground truth.
"""

from __future__ import annotations

from ..config import Settings
from ..models import (
    CategoryScore,
    NarrativeSections,
    Quote,
    Regime,
    Scenario,
    Verdict,
)

_MODEL = "claude-opus-4-8"


def build_narrative(
    *,
    ticker: str,
    quote: Quote,
    verdict: Verdict,
    scorecard: list[CategoryScore],
    scenarios: list[Scenario],
    macro_regime: Regime,
    key_risks: list[str],
    settings: Settings,
    use_llm: bool,
) -> NarrativeSections:
    facts = _facts(ticker, quote, verdict, scorecard, scenarios, macro_regime, key_risks)

    if use_llm and settings.has_anthropic:
        llm_sections = _try_llm(facts, settings)
        if llm_sections is not None:
            return llm_sections

    return _template_sections(facts)


# --------------------------------------------------------------------------- #
def _facts(ticker, quote, verdict, scorecard, scenarios, macro_regime, key_risks) -> dict:
    by_name = {c.name: c for c in scorecard}
    scen = {s.name: s for s in scenarios}
    return {
        "ticker": ticker,
        "company": quote.company,
        "price": quote.price,
        "sector": quote.sector or "its sector",
        "verdict": verdict.label,
        "conviction": verdict.conviction_score,
        "confidence": verdict.confidence,
        "confidence_band": verdict.confidence_band,
        "macro_regime": macro_regime.value,
        "scores": {n: c.score for n, c in by_name.items()},
        "neutralized": [n for n, c in by_name.items() if c.neutralized],
        "rationales": {n: c.rationale for n, c in by_name.items()},
        "bull": scen.get("Bull"),
        "base": scen.get("Base"),
        "bear": scen.get("Bear"),
        "key_risks": key_risks,
    }


def _template_sections(f: dict) -> NarrativeSections:
    strongest = _extreme(f["scores"], highest=True)
    weakest = _extreme(f["scores"], highest=False)

    exec_summary = (
        f"{f['company']} ({f['ticker']}) scores {f['conviction']:.0f}/100, a "
        f"“{f['verdict']}” in the current {f['macro_regime'].lower()} macro regime. "
        f"The thesis leans on {strongest.lower()} while {weakest.lower()} is the principal "
        f"offset. Confidence is {f['confidence']:.0f}/100 ({f['confidence_band'].lower()})."
    )

    thesis = (
        f"Aggregating macro, fundamentals, valuation, technicals, positioning, catalysts and "
        f"risk/reward, {f['ticker']} resolves to a {f['conviction']:.0f}/100 conviction and a "
        f"“{f['verdict']}” stance. "
        + (f"Note: {', '.join(f['neutralized'])} could not be assessed on free data and were "
           f"neutralized, which is reflected in the {f['confidence_band'].lower()} confidence. "
           if f["neutralized"] else "")
        + "This is a structured, evidence-weighted read — not a directive to trade."
    )

    bull = _scen_line("Upside", f["bull"], f["price"])
    base = _scen_line("Base", f["base"], f["price"])
    bear = _scen_line("Downside", f["bear"], f["price"])

    risks = (
        "Principal risks: " + " ".join(f["key_risks"][:3])
        if f["key_risks"] else "No dominant idiosyncratic risk flagged."
    )

    portfolio = (
        f"Sizing should respect single-name concentration and the factor exposure {f['ticker']} "
        f"adds (sector: {f['sector']}). The trade plan caps allocation by fractional-Kelly and "
        f"the conviction tier; treat the suggested size as an upper bound, not a target."
    )

    return NarrativeSections(
        executive_summary=exec_summary,
        thesis=thesis,
        bull=bull,
        base=base,
        bear=bear,
        risks=risks,
        portfolio_impact=portfolio,
        generated_by="deterministic-template",
    )


def _scen_line(label: str, scenario: Scenario | None, price: float) -> str:
    if scenario is None or scenario.price_target is None:
        return f"{label}: not modeled."
    move = (scenario.price_target / price - 1) * 100
    return (
        f"{label} case ({scenario.probability * 100:.0f}% probability): "
        f"${scenario.price_target:,.2f} ({move:+.0f}%). {scenario.drivers}"
    )


def _extreme(scores: dict[str, float], *, highest: bool) -> str:
    live = {n: s for n, s in scores.items()}
    if not live:
        return "the available signals"
    name = max(live, key=live.get) if highest else min(live, key=live.get)
    return name


def _try_llm(facts: dict, settings: Settings):
    """Best-effort Claude enrichment with prompt caching. Returns None on any failure so the
    caller falls back to templates — the LLM path must never break a report."""
    try:
        import json

        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        system = (
            "You are an institutional equity research analyst. You will be given a set of "
            "ALREADY-COMPUTED quantitative facts about a stock (scores, verdict, scenarios, "
            "risks). Write concise, professional research prose for each requested section. "
            "Critical rules: (1) never contradict or alter the provided numbers — they are "
            "ground truth; (2) no hype, no price predictions beyond the given scenario targets; "
            "(3) institutional tone, not retail. Return strict JSON with keys: "
            "executive_summary, thesis, bull, base, bear, risks, portfolio_impact."
        )
        # The system block is cached: it's static across every ticker in a session, so repeated
        # runs reuse it and only pay for the small per-ticker user message.
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1500,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": _llm_prompt(facts)}],
        )
        text = response.content[0].text.strip()
        data = json.loads(_extract_json(text))
        return NarrativeSections(
            executive_summary=data["executive_summary"],
            thesis=data["thesis"],
            bull=data["bull"],
            base=data["base"],
            bear=data["bear"],
            risks=data["risks"],
            portfolio_impact=data["portfolio_impact"],
            generated_by="claude",
        )
    except Exception:  # noqa: BLE001 — degrade to templates on any LLM/parse/network failure
        return None


def _llm_prompt(facts: dict) -> str:
    import json

    payload = {k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in facts.items()}
    return (
        "Write the research prose sections for this stock. Facts (ground truth, do not alter):\n\n"
        + json.dumps(payload, default=str, indent=2)
        + "\n\nReturn ONLY the JSON object."
    )


def _extract_json(text: str) -> str:
    """Strip markdown fences if the model wrapped the JSON."""
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
    return text
