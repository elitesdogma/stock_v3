"""On-disk TTL cache.

yfinance *will* rate-limit during a multi-call run, so caching is mandatory, not a
nicety. Keys are namespaced per source; values are JSON. Pandas frames are stored via
their split-orient JSON so a DataFrame round-trips without a parquet engine dependency.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_FRAME_SENTINEL = "__stock_v3_dataframe__"


class Cache:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()[:20]
        return self.root / f"{namespace}__{digest}.json"

    def get(self, namespace: str, key: str, ttl_seconds: float) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            return None
        if ttl_seconds >= 0 and (time.time() - path.stat().st_mtime) > ttl_seconds:
            return None
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return _decode(raw)

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        try:
            path.write_text(json.dumps(_encode(value)))
        except (OSError, TypeError):
            # A cache write failure must never break a run.
            pass

    def get_or_fetch(
        self, namespace: str, key: str, ttl_seconds: float, fetch: Callable[[], Any]
    ) -> Any:
        cached = self.get(namespace, key, ttl_seconds)
        if cached is not None:
            return cached
        value = fetch()
        if value is not None:
            self.set(namespace, key, value)
        return value


def _encode(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {_FRAME_SENTINEL: value.to_json(orient="split", date_format="iso")}
    return value


def _decode(raw: Any) -> Any:
    if isinstance(raw, dict) and _FRAME_SENTINEL in raw:
        from io import StringIO

        return pd.read_json(StringIO(raw[_FRAME_SENTINEL]), orient="split")
    return raw
