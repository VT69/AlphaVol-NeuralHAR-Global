"""HTTP client: caching, retries, loud failures, credential redaction (no network)."""
from __future__ import annotations

import json

import pytest
import requests

from alphavol.fetch import http as H


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(tmp_path, responses, **kw):
    return H.CachedClient(source="t", base_url="https://x.test", min_interval=0, cache_root=tmp_path,
                          session=FakeSession(responses), **kw)


def test_cache_makes_second_call_free(tmp_path):
    c = _client(tmp_path, [FakeResp(200, {"a": 1})])
    assert c.get("/p", {"q": 1}) == {"a": 1}
    assert c.get("/p", {"q": 1}) == {"a": 1}
    assert len(c.session.calls) == 1


def test_retry_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    c = _client(tmp_path, [FakeResp(429, "slow down", {"Retry-After": "1"}),
                           requests.ConnectionError("boom"), FakeResp(200, {"ok": True})])
    assert c.get("/p") == {"ok": True} and len(c.session.calls) == 3


def test_non_retryable_and_non_json_are_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    with pytest.raises(H.FetchError):
        _client(tmp_path, [FakeResp(403, "forbidden")]).get("/p")
    with pytest.raises(H.FetchError):
        _client(tmp_path, [FakeResp(200, "<html>error</html>")]).get("/p")
    with pytest.raises(H.FetchError):
        _client(tmp_path, [FakeResp(503, "x")] * 2, max_retries=2).get("/p")


def test_credentials_never_in_errors_or_cache(tmp_path, monkeypatch):
    secret = "SUPERSECRETKEY123"
    monkeypatch.setenv("FAKE_KEY", secret)
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    c = _client(tmp_path, [FakeResp(401, f"invalid key {secret}")], key_env="FAKE_KEY", key_param="api-key")  # server echoes key
    with pytest.raises(H.FetchError) as ei:
        c.get("/p", {"q": "x"})
    assert secret not in str(ei.value) and "***" in str(ei.value)
    c2 = _client(tmp_path, [FakeResp(200, {"ok": 1})], key_env="FAKE_KEY", key_param="api-key")
    c2.get("/p", {"q": "y"})
    assert c2.session.calls[0][1]["api-key"] == secret          # key is sent ...
    for f in tmp_path.rglob("*.json"):
        assert secret not in f.read_text()                       # ... but never stored


def test_missing_key_is_loud(tmp_path, monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    with pytest.raises(H.MissingCredentialError):
        _client(tmp_path, [], key_env="ABSENT_KEY", key_param="k").get("/p")
