"""Client for AIsa, the radar's one paid dependency: its tool router (X posts, Google
Trends and whois data through DataForSEO) and its OpenAI-compatible chat endpoint (the
name judge). One key: AISA_API_KEY, or the one the AIsa CLI stores when you log in.

The tool router runs a batch's calls one by one, about 10 s each, so batches are kept to
BATCH calls and the timeout is long. Every call is billed: a failed batch is never
retried here, so one bad day costs one batch at most.

Prices seen in October 2026: an account's last 20 posts $0.003, one Trends task $0.0012,
one whois overview query about $0.13.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import UA

ROUTER = os.environ.get("AISA_ROUTER_BASE_URL", "https://tools.aisa.one") + "/v1/tool-router/aisa-batch-use"


def _chat_url() -> str:
    """Chat endpoint. RADAR_LLM_BASE_URL points the LLM calls (judge, asks, pains,
    usecases, tutorials) at any OpenAI-compatible provider; falls back to AIsa.
    Accepts a full `.../chat/completions` URL, a `.../v1` base, or a bare host."""
    base = (os.environ.get("RADAR_LLM_BASE_URL") or "").strip()
    base = base or os.environ.get("AISA_API_BASE_URL", "").strip()
    base = base.rstrip("/")
    if base and "://" not in base:
        base = "https://" + base  # bare host — urlparse needs a scheme to see it
    if not base:
        return "https://api.aisa.one/v1/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    try:
        host = urllib.parse.urlparse(base).hostname or ""
    except ValueError:
        host = ""  # malformed base — surfaces as a per-call AisaError later
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        # OpenRouter serves ONLY /api/v1 — normalize every input shape
        # (bare host, /v1, /api, /api/v1) to it before appending
        for suffix in ("/api/v1", "/api", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base + "/api/v1/chat/completions"
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


CHAT = _chat_url()


def _aisa_host(url: str) -> bool:
    """Resolved URL host is aisa.one or a subdomain — never a substring match
    (aisa.one.evil.tld must not receive AIsa-only credentials/headers)."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    return host == "aisa.one" or host.endswith(".aisa.one")


def _custom_endpoint() -> bool:
    """Resolved chat endpoint is not an aisa.one host. Keyed on the resolved
    URL (CHAT), not env vars: both RADAR_LLM_BASE_URL and AISA_API_BASE_URL
    can reroute chat."""
    return not _aisa_host(CHAT)
MAX_CALLS = 20      # the router's own limit per batch
BATCH = 5           # what call_all sends at once, about 50 s a batch
TIMEOUT = 240
CHAT_TIMEOUT = 90


class AisaError(RuntimeError):
    def __init__(self, message: str = "", *, transient: bool = False,
                 status: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.status = status  # HTTP status when server answered, else None
                              # (None = network-layer failure: the request may
                              # have completed server-side — a blind retry can
                              # double-bill on the paid aisa endpoint)


def key() -> str:
    """AISA_API_KEY, else the key the AIsa CLI stored at login (~/.aisa).
    Router calls only — chat auth goes through llm_key()."""
    k = os.environ.get("AISA_API_KEY", "").strip()
    if k:
        return k
    home = Path.home() / ".aisa"
    try:
        if (home / "key").exists():
            return (home / "key").read_text().strip()
        if (home / "tokens.json").exists():
            return json.loads((home / "tokens.json").read_text()).get("accessToken", "") or _no_key()
    except (OSError, ValueError):
        pass
    return _no_key()


def llm_key() -> str:
    """Key for chat calls. Credential policy binds to the RESOLVED host, not
    to which env var pointed at it: the AIsa credential only ever goes to
    aisa.one, and every non-aisa endpoint — set via RADAR_LLM_BASE_URL or
    upstream's AISA_API_BASE_URL alike — requires RADAR_LLM_API_KEY. Proxied
    AIsa deployments must therefore set RADAR_LLM_API_KEY explicitly (fail
    closed beats a silent Bearer leak). On aisa.one, RADAR_LLM_API_KEY must
    NOT be used — it is scoped to the custom endpoint."""
    if _custom_endpoint():
        k = os.environ.get("RADAR_LLM_API_KEY", "").strip()
        if not k:
            raise AisaError("custom chat endpoint: set RADAR_LLM_API_KEY")
        return k
    return key()


def _no_key() -> str:
    raise AisaError("no AIsa key: set AISA_API_KEY, or log in with the AIsa CLI (`aisa login`)")


def _loopback(host: str | None) -> bool:
    """Real loopback only — `127.evil.tld` must not sneak past the cleartext
    guard on a string prefix."""
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped and mapped.is_loopback)


def _post(url: str, body: dict, timeout: int, api_key: str | None = None,
          auth: str | None = None) -> dict:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        raise AisaError(f"malformed endpoint URL: {url[:80]}") from exc
    if parsed.scheme == "http" and not _loopback(parsed.hostname):
        # a Bearer over cleartext is one typo'd env var away from a public leak
        raise AisaError(f"cleartext http:// endpoint refused: {parsed.hostname}")
    if auth is None:
        if api_key is None:
            # the AIsa credential only defaults in for aisa.one hosts — a
            # rerouted router (AISA_ROUTER_BASE_URL=third-party) must not
            # silently receive it as Bearer
            if not _aisa_host(url):
                raise AisaError(f"non-AIsa endpoint requires an explicit api_key: {parsed.hostname or url[:80]}")
            api_key = key()
        auth = f"Bearer {api_key}"
    headers = {"Content-Type": "application/json", "Authorization": auth,
               "User-Agent": UA}
    if _aisa_host(url):
        headers["x-aisa-source"] = "new-term-radar"
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        transient = exc.code == 429 or 500 <= exc.code < 600
        try:
            detail = ' '.join(exc.read().decode('utf-8', 'replace').split())[:200]
        except (OSError, http.client.HTTPException):
            detail = exc.reason or ""  # a truncated error body must not escape raw
        raise AisaError(f"HTTP {exc.code}: {detail}", transient=transient,
                        status=exc.code) from exc
    except (OSError, ValueError) as exc:
        # OSError-family (URLError, TimeoutError, connection resets) is
        # transient; ValueError (malformed JSON body) is deterministic
        raise AisaError(f"{type(exc).__name__}: {exc}",
                        transient=not isinstance(exc, ValueError)) from exc
    except http.client.HTTPException as exc:
        # IncompleteRead / BadStatusLine / LineTooLong — NOT OSError-derived,
        # would escape raw past the AisaError degradation contract
        raise AisaError(f"{type(exc).__name__}: {exc}", transient=True) from exc


def _have_router_creds() -> bool:
    """AIsa router credentials available, without key()'s raise."""
    if os.environ.get("AISA_API_KEY", "").strip():
        return True
    home = Path.home() / ".aisa"
    return (home / "key").exists() or (home / "tokens.json").exists()


def call(calls: list[dict]) -> tuple[dict[str, dict], int]:
    """Run one batch of tool calls. Returns ({call_id: data}, cost in micro-USD). Calls
    that failed are left out of the dict; only a failed request raises."""
    if not calls:
        return {}, 0
    if len(calls) > MAX_CALLS:
        raise AisaError(f"{len(calls)} calls in one batch, the cap is {MAX_CALLS}")
    # Pluggable data providers: RADAR_DATA_PROVIDER forces one; without router
    # creds the provider layer auto-selects (see radar/datasrc.py)
    provider = os.environ.get("RADAR_DATA_PROVIDER", "").strip()
    if provider or not _have_router_creds():
        from . import datasrc
        return datasrc.call(calls, provider or None)
    body = _post(ROUTER, {"calls": calls}, TIMEOUT)
    if "error" in body:
        raise AisaError(str(body["error"].get("message") or body["error"])[:200])
    out, cost = {}, 0
    for r in body.get("results", []):
        cost += int(r.get("customer_cost_micros_usd") or 0)
        if r.get("successful"):
            out[r["call_id"]] = r.get("data") or {}
    return out, cost


def call_all(calls: list[dict]) -> tuple[dict[str, dict], int]:
    """call() in batches of BATCH. A failed batch raises and stops the rest."""
    out, cost = {}, 0
    for i in range(0, len(calls), BATCH):
        data, c = call(calls[i:i + BATCH])
        out.update(data)
        cost += c
    return out, cost


def _env_int(name: str) -> int | None:
    """Positive int env override; a malformed value fails loudly as AisaError
    (callers degrade per-chunk) instead of a bare ValueError."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        v = -1
    if v <= 0:
        raise AisaError(f"bad {name}: {raw!r} (positive int required)")
    return v


def chat(model: str, system: str, user: str, max_tokens: int | None = None) -> str:
    """OpenAI-compatible chat. Retries transient failures twice (free-tier
    providers rate-limit often); router calls stay unretried — they bill per
    call. Budget policy: on the aisa endpoint an explicit max_tokens arg is
    verbatim cost control (None → 1500 default); on a custom endpoint every
    budget — explicit or not — is floored at 4000, because free reasoning
    models burn the budget on hidden reasoning before content.
    RADAR_LLM_MAX_TOKENS only ever raises, never lowers.
    RADAR_LLM_TIMEOUT likewise floors the 240/90 default."""
    env_mt = _env_int("RADAR_LLM_MAX_TOKENS")
    if _custom_endpoint():
        # custom endpoints often run reasoning models that burn the budget on
        # hidden reasoning before content — floor EVERY budget at 4000,
        # including call-site args sized for aisa's non-reasoning model
        # (600–2500 deterministically returns an empty answer)
        max_tokens = max(max_tokens or 0, 4000, env_mt or 0)
    else:
        # aisa path: an explicit per-call budget is verbatim cost control —
        # None picks the platform default, env can only raise
        max_tokens = max(max_tokens or 1500, env_mt or 0)
    env_to = _env_int("RADAR_LLM_TIMEOUT")
    timeout = max(240 if _custom_endpoint() else CHAT_TIMEOUT, env_to or 0)
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
    last: Exception | None = None
    # every attempt keeps the full floored timeout — halving it would make
    # retries useless for exactly the case they exist for (a slow free-tier
    # reasoning model timing out at the full budget). Instead a deadline caps
    # total wall time at ~2 attempts' worth: fast failures (429s) still get
    # all 3 tries, full-length timeouts get at most ~2.
    deadline = time.monotonic() + 2 * timeout
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            raise last or AisaError("chat retries exhausted (deadline)")
        try:
            resp = _post(CHAT, body, min(timeout, int(remaining)), api_key=llm_key())
            break
        except AisaError as exc:
            last = exc
            if not exc.transient or attempt == 2:
                raise
            if not _custom_endpoint() and exc.status != 429:
                # paid endpoint: retry ONLY a definite 429 rejection — a
                # network failure or 5xx may have completed (and billed)
                # server-side, so a blind retry can double-bill
                raise
            time.sleep(5 * (attempt + 1))
    else:
        raise last or AisaError("chat failed")
    try:
        msg = resp["choices"][0]["message"]
        if not isinstance(msg, dict):
            raise TypeError("message is not an object")
        content = msg.get("content")
        if isinstance(content, list):  # content-parts format (e.g. Gemini via OpenRouter)
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content:
            ch = resp["choices"][0]
            if ch.get("finish_reason") == "length" or _custom_endpoint():
                raise AisaError(
                    f"empty chat answer: finish_reason={ch.get('finish_reason')} "
                    f"usage={resp.get('usage')} — reasoning models burn max_tokens "
                    f"on hidden reasoning; raise the budget or pick another model")
            return ""  # aisa hiccup — upstream tolerance ("" degrades quietly)
        if not isinstance(content, str):  # truthy non-str quirk — degrade, don't leak
            raise TypeError(f"content is {type(content).__name__}, not str")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise AisaError(f"unexpected chat answer: {str(resp)[:160]}") from exc
