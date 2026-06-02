"""`analyze TICKER` — run the full pipeline and write/open an HTML research report."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .cache import Cache
from .config import load_settings
from .pipeline import PipelineError, run_pipeline
from .report.render import write_report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(cache_enabled=not args.no_cache)
    cache = Cache(settings.cache_dir, enabled=not args.no_cache)

    _print_preflight(settings)

    try:
        report = run_pipeline(args.ticker, settings, cache, use_llm=args.narrative)
    except PipelineError as exc:
        print(f"\n✗ {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out).expanduser()
    path = write_report(report, out_dir)

    _print_summary(report, path)

    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze",
        description="Institutional equity research report for a ticker (free data).",
    )
    parser.add_argument("ticker", help="Stock symbol, e.g. NVDA")
    parser.add_argument("--narrative", action="store_true",
                        help="Enrich prose via Claude (needs ANTHROPIC_API_KEY); "
                             "falls back to deterministic templates otherwise.")
    parser.add_argument("--out", default="reports", help="Output directory (default: reports/)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open the report in a browser.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the on-disk cache (always hit live APIs).")
    return parser.parse_args(argv)


def _print_preflight(settings) -> None:
    keys = []
    keys.append(("FRED", settings.has_fred))
    keys.append(("Finnhub", settings.has_finnhub))
    keys.append(("Anthropic", settings.has_anthropic))
    missing = [name for name, present in keys if not present]
    if missing:
        print(f"  ℹ  Running without: {', '.join(missing)} "
              f"(those layers degrade gracefully). See .env.example.")


def _print_summary(report, path: Path) -> None:
    v = report.verdict
    print(f"\n  {report.quote.ticker} — {report.quote.company}")
    print(f"  Verdict:    {v.label}  ({v.conviction_score:.0f}/100)")
    print(f"  Confidence: {v.confidence:.0f}/100 — {v.confidence_band}")
    if report.trade_plan.risk_reward:
        print(f"  Risk/Reward: {report.trade_plan.risk_reward:.1f}:1")
    neutralized = [c.name for c in report.scorecard if c.neutralized]
    if neutralized:
        print(f"  Data gaps:  {', '.join(neutralized)} (neutralized)")
    print(f"\n  ✓ Report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
