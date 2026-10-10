"""Cached, rate-limited, loud HTTP client (DECISIONS.md D26; credentials per R5).

* Every successful response body is cached on disk under ``data/cache/<source>/``. The cache key
  excludes credentials. A cached request is never re-sent, so runs are resumable.
* Rate limiting: a minimum interval between live requests per source, plus exponential backoff
  that honours ``Retry-After`` on HTTP 429/503.
* Failures are loud: non-retryable HTTP errors, exhausted retries and non-JSON bodies raise
  :class:`FetchError`. Messages contain a redacted URL, never the credential.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from alphavol.config import REPO_ROOT
from alphavol.logging_utils import get_logger

log = get_logger(__name__)
CACHE_ROOT = REPO_ROOT / "data" / "cache"
RETRYABLE = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """A fetch failed; the message never contains credentials."""


class MissingCredentialError(FetchError):
    pass


def read_key(env_var: str) -> str:
    """Read an API key from the environment only (never from files; R5)."""
    val = os.environ.get(env_var, "").strip()
    if not val:
        raise MissingCredentialError(f"environment variable {env_var} is not set")
    return val


@dataclass
class CachedClient:
    source: str
    base_url: str
    min_interval: float = 1.0
    key_env: str | None = None
    key_param: str | None = None
    max_retries: int = 6
    timeout: float = 60.0
    expect: str = "json"                     # "json" or "text"
    cache_root: Path = CACHE_ROOT
    session: requests.Session = field(default_factory=requests.Session)
    _last: float = 0.0

    def _cache_path(self, path: str, params: dict[str, Any], tag: str | None = None) -> Path:
        clean = {k: params[k] for k in sorted(params)}
        key = hashlib.sha256(json.dumps([self.base_url, path, clean, tag], sort_keys=True, default=str).encode()).hexdigest()
        return self.cache_root / self.source / key[:2] / f"{key}.json"

    def _redacted(self, path: str, params: dict[str, Any]) -> str:
        query = urlencode(params)
        if self.key_env:
            query = f"{query}&{self.key_param or 'apikey'}=***" if query else f"{self.key_param or 'apikey'}=***"
        return f"{self.base_url}{path}?{query}"

    def _scrub(self, text: str) -> str:
        """Remove the credential from any text that might be echoed back by the server."""
        if self.key_env and os.environ.get(self.key_env):
            text = text.replace(os.environ[self.key_env], "***")
        return text

    def get(self, path: str = "", params: dict[str, Any] | None = None, *, cache_tag: str | None = None) -> Any:
        """GET with cache. Returns parsed JSON (or text if ``expect='text'``).

        ``cache_tag`` is part of the cache key but never sent; use it to force a fresh fetch of an
        endpoint whose parameters do not change (e.g. a full-history download at a later date).
        """
        params = dict(params or {})
        cpath = self._cache_path(path, params, cache_tag)
        if cpath.exists():
            return json.loads(cpath.read_text())["body"]
        live = dict(params)
        if self.key_env:
            live[self.key_param or "apikey"] = read_key(self.key_env)
        shown = self._redacted(path, params)
        delay = self.min_interval
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(self.base_url + path, params=live, timeout=self.timeout)
            except requests.RequestException as exc:   # network error: retry, never echo the URL
                log.warning("%s: network error (%s) on attempt %d for %s", self.source, type(exc).__name__, attempt, shown)
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code in RETRYABLE:
                ra = resp.headers.get("Retry-After")
                sleep_for = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                log.warning("%s: HTTP %d, retry %d/%d in %.0fs for %s", self.source, resp.status_code, attempt,
                            self.max_retries, sleep_for, shown)
                time.sleep(sleep_for)
                delay *= 2
                continue
            if resp.status_code != 200:
                raise FetchError(f"{self.source}: HTTP {resp.status_code} for {shown}: {self._scrub(resp.text[:300])!r}")
            if self.expect == "json":
                try:
                    body = resp.json()
                except ValueError:
                    raise FetchError(f"{self.source}: non-JSON body for {shown}: {self._scrub(resp.text[:300])!r}") from None
            else:
                body = resp.text
            cpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": shown, "fetched_at": time.time(), "body": body}))
            tmp.replace(cpath)                      # atomic: no half-written cache entries
            return body
        raise FetchError(f"{self.source}: exhausted {self.max_retries} retries for {shown}")
