"""Runtime configuration: environment-derived settings and cache location."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_SEC_UA = "stock_v3 research-tool contact@example.com"


@dataclass(frozen=True)
class Settings:
    """Immutable view of everything the pipeline needs from the environment."""

    fred_api_key: str | None
    finnhub_api_key: str | None
    sec_user_agent: str
    anthropic_api_key: str | None
    cache_dir: Path
    cache_enabled: bool = True

    @property
    def has_fred(self) -> bool:
        return bool(self.fred_api_key)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings(cache_dir: Path | None = None, *, cache_enabled: bool = True) -> Settings:
    """Read settings from the environment. Missing keys are tolerated, not fatal —
    each source degrades to `unavailable` rather than crashing the run."""
    resolved_cache = cache_dir or Path(os.getenv("STOCK_V3_CACHE", ".cache")).expanduser()
    return Settings(
        fred_api_key=_clean(os.getenv("FRED_API_KEY")),
        finnhub_api_key=_clean(os.getenv("FINNHUB_API_KEY")),
        sec_user_agent=_clean(os.getenv("SEC_USER_AGENT")) or _DEFAULT_SEC_UA,
        anthropic_api_key=_clean(os.getenv("ANTHROPIC_API_KEY")),
        cache_dir=resolved_cache,
        cache_enabled=cache_enabled,
    )


def _clean(value: str | None) -> str | None:
    """Treat blank / placeholder env values as absent."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "your_key_here", "changeme"}:
        return None
    return stripped
