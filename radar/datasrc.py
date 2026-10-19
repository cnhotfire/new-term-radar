"""Pluggable data providers for router-style tool calls.

The AIsa tool router is a billing proxy in front of third-party data APIs —
the tool names ARE the proxied API's identity (`post_dataforseo_*`), and its
`arguments.body` is already the upstream request body. When no AISA_API_KEY
is configured, `aisa.call()` delegates here: a provider translates each tool
call onto its own API and shapes the answer back into the DataForSEO
`{call_id: {"tasks": [{"result": ...}]}}` contract the upstream parsing
expects, so usecases/competition/review code never learns which provider ran.

Selection: `RADAR_DATA_PROVIDER=webcafe|dataforseo` forces one provider;
otherwise the first provider whose credentials are present wins, in
_PROVIDERS order. Providers serve ONLY the tools they map — anything else
raises AisaError and the caller's existing degradation path handles it
(e.g. evidence() records a per-name error and moves on).
"""
from __future__ import annotations

import base64
import os
import time
import urllib.parse
from datetime import date, datetime, timezone

from . import aisa
from .aisa import AisaError

# The router tool names = the upstream DataForSEO endpoints they proxy.
DFS_VOLUME = "post_dataforseo_keywords_gads_search_volume_live"
DFS_TRENDS = "post_dataforseo_keywords_trends_explore_live"
DFS_SERP = "post_dataforseo_serp_google_organic_live"
DFS_KD = "post_dataforseo_labs_google_keyword_overview_live"

_TIMEOUT = 90  # per request — DFS trends live averages ~30 s, webcafe similar


def _dig(obj, key: str):
    """First list-of-dicts containing `key`, searched recursively. Provider
    envelopes differ; the payload rows inside are what callers want. Returns
    None on a miss so dict siblings keep being searched — an early [] would
    short-circuit the loop at the first scalar/dict value."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and key in obj[0]:
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _dig(v, key)
            if found:
                return found
    return None


def _num(row: dict, *keys) -> int | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def _gl(location_code) -> str:
    """DataForSEO numeric location → web.cafe's `gl` country code."""
    if not location_code:
        return "us"
    gl = {2840: "us", 2826: "gb", 2124: "ca", 2036: "au", 2276: "de",
          2392: "jp", 2702: "sg"}.get(location_code)
    if gl is None:
        # a real-but-unmapped code silently pulling US data is worse than a
        # loud error — extend the map when adding geos
        raise AisaError(f"unmapped location_code {location_code!r} (extend _gl)")
    return gl


def _range_of(body: dict) -> str:
    """DFS date_from/date_to window → web.cafe's range enum. Rounds UP —
    truncating to the nearest shorter window would silently cut series a
    caller (e.g. review) asked for."""
    try:
        days = (date.fromisoformat(str(body["date_to"]))
                - date.fromisoformat(str(body["date_from"]))).days
    except (KeyError, ValueError):
        days = 90
    if days <= 0:   # inverted/empty window — default, not the smallest enum
        days = 90
    for label, span in (("7d", 7), ("30d", 30), ("90d", 90),
                        ("12m", 365), ("5y", 1826)):
        if span >= days:
            return label
    return "5y"


def _domain(url: str) -> str:
    host = urllib.parse.urlparse(url if "//" in url else "https://" + url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def _point_day(p: dict) -> str | None:
    """One trend point's date — ISO string or epoch seconds/millis. Anything
    else returns None so the point is skipped: a poison point must not crash
    the call, and a garbage date must not leak downstream where the contract
    no longer catches it."""
    d = p.get("date", p.get("time", p.get("timestamp", "")))
    if isinstance(d, bool):
        return None
    if isinstance(d, (int, float)):
        if d >= 1e12:   # epoch millis — seconds can't reach this until year 33k
            d /= 1000
        if not 1_000_000_000 <= d <= 253_402_300_799:  # <2001 is garbage for
            return None                                # trend data (starts '04)
        return datetime.fromtimestamp(d, tz=timezone.utc).date().isoformat()
    day = str(d or "")[:10]
    try:
        return date.fromisoformat(day).isoformat()
    except ValueError:
        return None


class _Provider:
    name = ""
    tools: tuple = ()
    min_interval = 0.0   # seconds between provider requests (rate limits)

    def __init__(self):
        self._last = 0.0
        self._spent = 0   # micro-USD billed provider-side; survives a mid-batch
                          # raise so the next successful call reports real spend

    def available(self) -> bool:
        raise NotImplementedError

    def _gate(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def call(self, calls: list[dict]) -> tuple[dict, int]:
        """Same contract as aisa.call(): {call_id: {"tasks": [...]}}, cost micros.
        A provider-side request failure raises AisaError — for a direct provider
        each call IS the request, and callers degrade per failed request."""
        out = {}
        for c in calls:
            tool = c.get("tool") or ""
            if tool not in self.tools:
                raise AisaError(f"{self.name}: no route for tool {tool!r}")
            cid = c.get("call_id")
            if not cid:
                raise AisaError(f"{self.name}: call without call_id")
            task, _micros = self._handle(tool, c.get("arguments") or {})
            if task is not None:
                out[cid] = {"tasks": [task]}
        # report everything billed since the last SUCCESSFUL call — a raise
        # mid-batch leaves _spent intact so the next call still accounts it
        spent, self._spent = self._spent, 0
        return out, spent

    def _handle(self, tool: str, arguments: dict) -> tuple[dict | None, int]:
        raise NotImplementedError


class WebCafe(_Provider):
    """seo.web.cafe 开放 API — one token covers volume, trends, SERP and KD.
    Prices (member credits, ~¥0.025/分): volume 2–5/call (≤50 kw),
    trends 2–4/call, serp 2/call, kd 2/call. Tool endpoints share a
    10-requests/minute cap, hence the 6.1 s gate."""

    name = "webcafe"
    tools = (DFS_VOLUME, DFS_TRENDS, DFS_SERP, DFS_KD)
    min_interval = 6.1
    # credits → micro-USD for the spend line: ¥0.025/分 ÷ ¥7.1/$ ≈ $0.0035
    CREDIT_MICROS = 3520

    def _base(self) -> str:
        return (os.environ.get("WEBCAFE_BASE_URL", "https://seo.web.cafe")
                .rstrip("/") + "/api/v1/")

    def available(self) -> bool:
        return bool(os.environ.get("WEBCAFE_TOKEN", "").strip())

    def _post(self, endpoint: str, payload: dict) -> tuple[dict, int]:
        self._gate()
        # Bearer token auth — aisa._post's cleartext/malformed-URL guards apply
        r = aisa._post(self._base() + endpoint, payload, _TIMEOUT,
                       api_key=os.environ.get("WEBCAFE_TOKEN", "").strip())
        if r.get("ok") is False:   # webcafe envelopes its errors in-band too
            raise AisaError(f"webcafe: {str(r.get('error'))[:160]}"
                            f" (code={r.get('code')})")
        credits = r.get("credits")
        charged = credits.get("charged") if isinstance(credits, dict) else 0
        try:                       # same coercion as the DFS path — a weird
            micros = int(float(charged or 0) * self.CREDIT_MICROS)  # billing
        except (TypeError, ValueError):      # field must not kill a call whose
            micros = 0                       # data already landed
        self._spent += micros
        return r, micros

    def _handle(self, tool: str, arguments: dict):
        # every shaping surprise must surface as AisaError — upstream catches
        # exactly that and degrades per call; a bare KeyError/TypeError would
        # instead take out the whole usecases block for the day
        try:
            body = (arguments.get("body") or [{}])[0]
            if tool == DFS_VOLUME:
                return self._volume(body)
            if tool == DFS_TRENDS:
                return self._trends(body)
            if tool == DFS_SERP:
                return self._serp(body)
            return self._kd(body)
        except AisaError:
            raise
        except Exception as exc:
            raise AisaError(f"webcafe {tool}: {type(exc).__name__}: {exc}") from exc

    def _volume(self, body: dict):
        # data.keywords maps each keyword → {avg, months:[{ym,v}], bids…} — or
        # null when Ads has no data; skipping it reads as volume 0 upstream,
        # same as DFS omitting the row. keyword_volume caps at 50 keywords
        # per request — chunk and merge rather than silently truncate
        kws = body.get("keywords") or []
        result, micros = [], 0
        for i in range(0, len(kws), 50):
            r, m = self._post("keyword_volume", {
                "keywords": kws[i:i + 50],
                "gl": _gl(body.get("location_code")),
                "hl": body.get("language_code") or "en"})
            micros += m
            kwmap = (r.get("data") or {}).get("keywords") or {}
            for kw, row in kwmap.items():
                if not isinstance(row, dict):
                    continue
                ms = [{"year": int(str(m2.get("ym") or "")[:4] or 0),
                       "month": int(str(m2.get("ym") or "")[5:7] or 0),
                       "search_volume": _num(m2, "v", "search_volume") or 0}
                      for m2 in (row.get("months") or []) if isinstance(m2, dict)]
                # DFS monthly_searches is newest-first and upstream labels the
                # estimate with ms[0]; webcafe lists oldest→newest
                ms.reverse()
                result.append({"keyword": kw,
                               "search_volume": _num(row, "avg", "search_volume", "volume"),
                               "monthly_searches": ms})
        return {"status_code": 20000, "result": result}, micros

    def _trends(self, body: dict):
        # data.points[] = {time: "YYYY-MM-DD", timestamp, value, values:[per kw]}
        kws = body.get("keywords") or []
        if not kws:   # empty like _volume/_kd — no request, no error
            return {"status_code": 20000,
                    "result": [{"items": [{"data": []}]}]}, 0
        r, micros = self._post("google_trends", {
            "keyword": kws[0], "compare": kws[1:5],
            "range": _range_of(body)})
        points = ((r.get("data") or {}).get("points")
                  or _dig(r, "timestamp") or _dig(r, "time") or [])
        data = []
        for p in points:
            if not isinstance(p, dict):
                continue
            day = _point_day(p)
            if day is None:
                continue
            vals = p.get("values")
            if not (isinstance(vals, list) and vals
                    and all(isinstance(v, (int, float)) for v in vals)):
                v = p.get("value")
                vals = [v if isinstance(v, (int, float)) else 0]
            data.append({"date_from": day, "values": vals})
        # upstream readers take points[-1] as "now" — never trust the API's
        # ordering; ISO date strings sort chronologically
        data.sort(key=lambda p: p["date_from"])
        return {"status_code": 20000,
                "result": [{"items": [{"data": data}]}]}, micros

    def _serp(self, body: dict):
        # data.organic[] = {title, link, snippet, position}; no ads key —
        # upstream's paid-ads probe just reports false here
        r, micros = self._post("serp", {
            "q": body.get("keyword") or "",
            "gl": _gl(body.get("location_code")),
            "hl": body.get("language_code") or "en"})
        rows = (r.get("data") or {}).get("organic")
        if rows is None:
            # _dig(r, "link") would happily return related/answer_box lists and
            # stamp them "organic" — a missing block is a shaping surprise, so
            # it must raise like the rest of this module's contract says
            raise AisaError("serp response missing data.organic")
        if not isinstance(rows, list):
            raise AisaError(f"serp data.organic is not a list: {type(rows).__name__}")
        items = []
        for i, row in enumerate(rows):
            if len(items) >= 10:
                break
            if not isinstance(row, dict):
                continue
            url = row.get("url") or row.get("link") or ""
            if not url:
                continue
            items.append({"type": "organic",
                          "domain": row.get("domain") or _domain(url),
                          "url": url, "title": row.get("title") or "",
                          "rank_group": _num(row, "rank_group", "position", "rank") or i + 1})
        return {"status_code": 20000, "result": [{"items": items}]}, micros

    def _kd(self, body: dict):
        # data.score is the 0–100 哥飞版 difficulty (data.level/details are the
        # human report). One keyword per request — batch inside the tool call
        # and reassemble the DFS items[] contract
        items, micros = [], 0
        for kw in (body.get("keywords") or []):
            r, m = self._post("keyword_difficulty", {
                "keyword": kw,
                "gl": _gl(body.get("location_code")),
                "hl": body.get("language_code") or "en"})
            micros += m
            kd = _num(r.get("data") or {}, "score", "difficulty", "kd")
            items.append({"keyword": kw,
                          "keyword_properties": {"keyword_difficulty": kd}})
        return {"status_code": 20000, "result": [{"items": items}]}, micros


class DataForSEO(_Provider):
    """api.dataforseo.org direct — tool names map to endpoints verbatim and
    the request body passes through unmodified; the response is already in
    the contract shape. Basic-auth login:password from the DFS dashboard.
    Google Ads live endpoints cap at 12 req/min, hence the 5.1 s gate."""

    name = "dataforseo"
    min_interval = 5.1
    tools = (DFS_VOLUME, DFS_TRENDS, DFS_SERP, DFS_KD)
    PATHS = {
        DFS_VOLUME: "keywords_data/google_ads/search_volume/live",
        DFS_TRENDS: "keywords_data/google_trends/explore/live",
        DFS_SERP: "serp/google/organic/live",
        DFS_KD: "dataforseo_labs/google/keyword_overview/live",
    }

    def available(self) -> bool:
        return bool(os.environ.get("DATAFORSEO_LOGIN", "").strip()
                    and os.environ.get("DATAFORSEO_PASSWORD", "").strip())

    def _auth(self) -> str:
        pair = (os.environ.get("DATAFORSEO_LOGIN", "").strip() + ":"
                + os.environ.get("DATAFORSEO_PASSWORD", "").strip())
        return "Basic " + base64.b64encode(pair.encode()).decode()

    def _handle(self, tool: str, arguments: dict):
        # same shaping contract as WebCafe: provider-side surprises surface as
        # AisaError so upstream degrades per call, not per whole block
        try:
            self._gate()
            url = "https://api.dataforseo.org/v3/" + self.PATHS[tool]
            r = aisa._post(url, arguments.get("body") or [], _TIMEOUT,
                           auth=self._auth())
            try:                                  # DFS bills in USD; a weird
                micros = int(float(r.get("cost") or 0) * 1_000_000)   # cost
            except (TypeError, ValueError):       # must not kill the call —
                micros = 0                        # the data already landed
            self._spent += micros          # billed spend survives even a
            if r.get("status_code") not in (None, 20000):     # failed task
                raise AisaError(f"dataforseo: {r.get('status_code')} "
                                f"{str(r.get('status_message'))[:120]}")
            task = (r.get("tasks") or [{}])[0]
            # DFS nests per-task failures (40xxx, result:null) below the
            # top-level status — without this check a billed task error
            # silently degrades to empty data
            if task.get("status_code") not in (None, 20000):
                raise AisaError(f"dataforseo {tool}: task "
                                f"{task.get('status_code')} "
                                f"{str(task.get('status_message'))[:120]}")
            return {"status_code": 20000,
                    "result": r.get("result") or task.get("result")}, micros
        except AisaError:
            raise
        except Exception as exc:
            raise AisaError(f"dataforseo {tool}: {type(exc).__name__}: {exc}") from exc


_PROVIDERS = {"webcafe": WebCafe, "dataforseo": DataForSEO}

# instances persist per name — aisa.call_all() batches calls through
# datasrc.call(), and a fresh instance each batch would reset the rate-limit
# gate, blowing past the provider's req/min cap mid-run
_INSTANCES: dict[str, _Provider] = {}


def call(calls: list[dict], provider: str | None = None) -> tuple[dict, int]:
    prov = _select(provider)
    if prov is None:
        raise AisaError("no data provider: set AISA_API_KEY, or "
                        "RADAR_DATA_PROVIDER with WEBCAFE_TOKEN / "
                        "DATAFORSEO_LOGIN+PASSWORD")
    return prov.call(calls)


def _select(name: str | None) -> _Provider | None:
    if name:
        cls = _PROVIDERS.get(name)
        if cls is None:
            raise AisaError(f"unknown RADAR_DATA_PROVIDER: {name!r} "
                            f"(have: {', '.join(_PROVIDERS)})")
        p = _INSTANCES.get(name)
        if p is None:
            p = _INSTANCES[name] = cls()
        if not p.available():
            raise AisaError(f"RADAR_DATA_PROVIDER={name} but its credentials "
                            f"are not set")
        return p
    for n, cls in _PROVIDERS.items():
        p = _INSTANCES.get(n)
        if p is None:
            p = _INSTANCES[n] = cls()
        if p.available():
            return p
    return None
