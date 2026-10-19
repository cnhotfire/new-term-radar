"""The radar's pure logic: name extraction, signal scoring, the store, the judge's parsing,
the review's verdict, the reports. Network calls are not exercised here."""
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from radar import __main__ as cli
from radar import breakout, config, gsc, i18n, judge, new_sites, notify, report, review, sitemaps, store
from radar.sources import base_name, novel_names
from radar.terms import names

NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


# --- names(): what a post can name -------------------------------------------------

def test_names_keeps_versioned_model_names_whole():
    found = names("Introducing Claude Opus 5.5, our best model.")
    assert "claude opus 5.5" in found and "opus 5.5" in found


def test_names_drops_phrases_made_only_of_stop_words():
    assert "claude code" not in names("Try it in Claude Code today")


def test_names_takes_a_name_that_starts_the_sentence():
    assert "strata" in names("Strata now runs Qwen3.8-Flash-Next on a 12GB RTX 5070. Big day.")


def test_names_stops_at_the_end_of_a_sentence():
    assert not any("big" in n and "5070" in n for n in names("on a RTX 5070. Big day."))


def test_names_ignores_urls_and_handles():
    assert names("see https://Example.com/Foo by @SomeOne") == set()


def test_names_breaks_on_chinese_text_between_words():
    found = names("用 Opus 5.5 配音（Gemini 3.8 Flash TTS，我提供了 API Key）")
    assert "gemini 3.8 flash" in found and not any("tts" in n and "api" in n for n in found)


# --- ① X: only names the accounts have not used before ------------------------------

def _post(account, text, likes=0, hours_ago=2):
    return {"account": account, "text": text, "likes": likes, "url": "", "created": NOW - timedelta(hours=hours_ago)}


def test_x_reports_a_new_name_two_accounts_used():
    rows = novel_names([_post("a", "Strata is fast"), _post("b", "Tried Strata today")], {}, NOW)
    assert [r["term"] for r in rows] == ["strata"] and rows[0]["sources"] == ["a", "b"]
    assert rows[0]["note"] == "X: 2 accounts, 0 likes"


def test_x_skips_names_already_in_the_vocabulary():
    rows = novel_names([_post("a", "Strata is fast"), _post("b", "Tried Strata today")], {"strata": "2026-09-01"}, NOW)
    assert rows == []


def test_x_one_account_needs_likes_and_an_unmistakable_name():
    assert novel_names([_post("a", "Qwen3.9 is fast", likes=20)], {}, NOW) == []
    assert [r["term"] for r in novel_names([_post("a", "Qwen3.9 is fast", likes=500)], {}, NOW)] == ["qwen3.9"]
    # a plain capitalised word may only be the start of a sentence: one account is not enough
    assert novel_names([_post("a", "Strata is fast", likes=500)], {}, NOW) == []


def test_x_one_post_yields_at_most_three_names():
    post = _post("a", "New: Alpha1 Beta2 Gamma3 Delta4 Eps5 Zeta6 models", likes=900)
    assert len(novel_names([post], {}, NOW)) <= 3


def test_x_ignores_old_posts_and_retweets():
    old = [_post("a", "Strata is fast", hours_ago=48), _post("b", "Strata rocks", hours_ago=48)]
    rts = [_post("a", "RT @x: Strata is fast"), _post("b", "RT @x: Strata rocks")]
    assert novel_names(old, {}, NOW) == [] and novel_names(rts, {}, NOW) == []


# --- ① HF: copies of one model group under its base name -----------------------------

def test_hf_base_name_strips_quantisation_and_size():
    assert base_name("abenzerps/Qwen-Image-2.1-Uncensored-GGUF") == "qwen image 2.1"
    assert base_name("Qwen/Qwen3.8-Flash-Next-A6B-Instruct") == "qwen3.8 flash next"
    assert base_name("Lightricks/LTX-2.5") == "ltx 2.5"


def test_hf_reads_trending_models_and_spaces_and_merges_a_model_with_its_demo(tmp_path, monkeypatch):
    import radar.sources as src
    lists = {"models": [{"id": "old/Old-Model"}], "spaces": [{"id": "old/old-demo"}]}
    monkeypatch.setattr(src, "_trending", lambda what: lists[what])
    first = src.hf_new({}, tmp_path)
    assert first["first_run"] and first["terms"] == [] and first["models"] == 1 and first["spaces"] == 1
    lists["models"] = [{"id": "krea/Krea-2", "pipeline_tag": "text-to-image", "likes": 900}, {"id": "old/Old-Model"}]
    lists["spaces"] = [{"id": "krea/Krea-2-demo", "sdk": "gradio", "likes": 300}, {"id": "lighton/LightOnOCR-3-webgpu", "sdk": "static"}]
    later = src.hf_new({}, tmp_path)
    terms = {r["term"]: r for r in later["terms"]}
    assert set(terms) == {"krea 2", "lightonocr 3"} and not later["first_run"]
    assert terms["krea 2"]["examples"] == ["krea/Krea-2 (text-to-image, ♥900)", "krea/Krea-2-demo (gradio, ♥300)"]
    assert terms["lightonocr 3"]["note"] == "Hugging Face Spaces: new at #2"


# --- ② breakout: one repo taking off --------------------------------------------------

def _repo(name, stars, days_old):
    created = (NOW - timedelta(days=days_old)).isoformat().replace("+00:00", "Z")
    return {"full_name": name, "stargazers_count": stars, "created_at": created, "description": ""}


def test_breakout_flags_a_fast_repo_and_skips_a_slow_one():
    rows = breakout.score([_repo("Niko1221/Strata", 7734, 9), _repo("a/slow", 900, 12)], None, NOW)
    assert [r["repo"] for r in rows] == ["Niko1221/Strata"]


def test_breakout_flags_a_jump_since_yesterday_even_when_the_average_is_low():
    rows = breakout.score([_repo("a/late", 1500, 13)], {"a/late": 900}, NOW)
    assert rows and rows[0]["delta"] == 600


def test_breakout_variants_include_handle_and_display_name():
    row = {"owner": "Niko1221", "name": "Strata"}
    assert breakout.variants(row, "Niko") == ["strata", "niko1221 strata", "niko strata"]
    assert breakout.variants(row, "") == ["strata", "niko1221 strata"]


# --- ③ sitemaps: one entry in 17 languages counts once ---------------------------------

def test_locale_prefix_is_stripped():
    urls = ["https://lobehub.com/skills/x", "https://lobehub.com/zh/skills/x", "https://lobehub.com/zh-TW/skills/x"]
    assert {re.sub(sitemaps._LOCALE, r"\1", u) for u in urls} == {"https://lobehub.com/skills/x"}


def test_paths_are_not_mistaken_for_locales():
    assert re.sub(sitemaps._LOCALE, r"\1", "https://lobehub.com/mcp/foo") == "https://lobehub.com/mcp/foo"


def test_sitemap_diff_reports_a_word_new_urls_share(cfg, tmp_path, monkeypatch):
    cfg["sitemaps"] = {"site": {"index": "https://example.com/sitemap.xml", "pause": 0}}
    old = {f"https://example.com/tools/thing-{i}" for i in range(40)}
    new = old | {f"https://example.com/tools/hyperframes-{w}" for w in ("starter", "captions", "studio")}
    answers = iter([old, new])
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: next(answers))
    first = sitemaps.diff(cfg, tmp_path)
    assert first["sites"]["site"]["first_run"] and first["terms"] == []
    second = sitemaps.diff(cfg, tmp_path)
    assert second["sites"]["site"]["new"] == 3 and second["terms"][0]["term"] == "hyperframes"


def test_a_site_that_fails_keeps_its_snapshot(cfg, tmp_path, monkeypatch):
    cfg["sitemaps"] = {"site": {"index": "https://example.com/sitemap.xml", "pause": 0}}
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: {"https://example.com/a"})
    sitemaps.diff(cfg, tmp_path)

    def down(*a):
        raise OSError("503")
    monkeypatch.setattr(sitemaps, "site_urls", down)
    assert "error" in sitemaps.diff(cfg, tmp_path)["sites"]["site"]
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: {"https://example.com/a"})
    assert sitemaps.diff(cfg, tmp_path)["sites"]["site"]["new"] == 0


# --- ③ new sites (whois overview, weekly) -----------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures/whois_new_ai_sites.json"   # a real answer, 2026-10-06


def test_new_site_name_terms():
    assert new_sites.name_terms("ai-hailuo.com") == ["hailuo"]
    assert new_sites.name_terms("livephoto.video") == ["livephoto"]
    assert new_sites.name_terms("ai-video-tool.com") == []   # only stop words left


def test_new_sites_drop_domains_ranking_for_a_handful_of_terms(cfg, tmp_path, monkeypatch):
    answer = {"tasks": [{"result": [{"items": json.loads(FIXTURE.read_text())}]}]}
    monkeypatch.setattr(new_sites.aisa, "call_all", lambda calls: ({c["call_id"]: answer for c in calls}, 127000))
    out = new_sites.new_sites(cfg, tmp_path, date(2026, 10, 6))
    domains = [r["domain"] for r in out["sites"]]
    assert "ai-hailuo.com" in domains and "neonwin386.ai" not in domains and "oyo88.ai" not in domains
    assert all(r["new"] for r in out["sites"]) and "hailuo" in [t["term"] for t in out["terms"]]
    again = new_sites.new_sites(cfg, tmp_path, date(2026, 10, 13))
    assert not any(r["new"] for r in again["sites"]) and again["terms"] == []


def test_new_sites_query_needs_both_dates(cfg):
    body = new_sites._calls(cfg["new_sites"], date(2026, 10, 6))[0]["arguments"]["body"][0]
    fields = [f[0] for f in body["filters"] if isinstance(f, list)]
    assert fields[:2] == ["created_datetime", "first_seen"] and body["filters"][0][2].startswith("2026-07-08")


# --- ④ Search Console ---------------------------------------------------------------------

def test_gsc_keeps_queries_the_earlier_window_never_saw():
    recent = [{"query": "hyperframes skill", "page": "/a", "impressions": 4, "clicks": 1},
              {"query": "hyperframes skill", "page": "/b", "impressions": 3, "clicks": 0},
              {"query": "ppt skill", "page": "/c", "impressions": 90, "clicks": 9}]
    fresh = gsc.fresh_queries(recent, [{"query": "ppt skill"}])
    assert fresh == [{"query": "hyperframes skill", "impressions": 7, "clicks": 1, "page": "/a"}]


def test_gsc_is_off_without_a_site(cfg):
    assert gsc.gsc_new(cfg) == {"off": True}


# --- the name judge ------------------------------------------------------------------------

def test_judge_parses_scores_and_drops_what_it_cannot_use():
    answer = 'Here you go:\n[{"i": 0, "score": 0.9}, {"i": 1, "score": "0.1"}, {"i": 7, "score": 1}, {"i": 2, "score": 3}, {"x": 1}]'
    assert judge.parse(answer, 3) == {0: 0.9, 1: 0.1}
    assert judge.parse("sorry", 3) == {}


def test_judge_parse_tolerates_fences_and_object_form():
    # free models sometimes wrap the payload in a markdown fence
    fenced = '```json\n[{"i": 0, "score": 0.9}, {"i": 1, "score": 0.2}]\n```'
    assert judge.parse(fenced, 3) == {0: 0.9, 1: 0.2}
    # ...or answer in object form {"index": score} (observed in the wild)
    assert judge.parse('{"0": 0.9, "1": 0.2}', 3) == {0: 0.9, 1: 0.2}
    # stray brackets in prose must not swallow the real span
    mixed = 'scores [per run] look like [{i}] — actually: [{"i": 1, "score": 0.7}]'
    assert judge.parse(mixed, 3) == {1: 0.7}
    # strictness kept: out-of-range scores still dropped even in object form
    assert judge.parse('{"0": 1.5, "1": 0.4}', 3) == {1: 0.4}
    # wrapper envelopes — models that nest the array under a key must still parse
    assert judge.parse('{"scores": [{"i": 0, "score": 0.9}]}', 3) == {0: 0.9}
    assert judge.parse('{"data": {"results": [{"i": 1, "score": 0.3}]}}', 3) == {1: 0.3}
    # a leading non-payload fence must not hide a later ```json block
    two_fences = '```thinking\nlet me reason\n```\n```json\n[{"i": 0, "score": 0.7}]\n```'
    assert judge.parse(two_fences, 3) == {0: 0.7}
    # a draft score array — in a thinking fence or in prose — must not win
    # over the final payload (reasoning models deliberate before answering)
    assert judge.parse('```thinking\npreliminary: [{"i": 0, "score": 0.1}]\n```\n'
                       'final: [{"i": 0, "score": 0.9}]', 3) == {0: 0.9}
    assert judge.parse('preliminary: [{"i": 0, "score": 0.1}] — final: '
                       '[{"i": 0, "score": 0.9}]', 3) == {0: 0.9}


def test_judge_scores_rows_in_chunks(monkeypatch):
    rows = [{"term": f"t{i}", "examples": []} for i in range(judge.CHUNK + 2)]
    sizes = []

    def chat(model, system, user):
        n = len(json.loads(user))
        sizes.append(n)
        return json.dumps([{"i": i, "score": 0.8} for i in range(n)])
    monkeypatch.setattr(judge.aisa, "chat", chat)
    assert judge.score(rows, "m") == {"judged": judge.CHUNK + 2} and sizes == [judge.CHUNK, 2]
    assert all(r["named"] == 0.8 for r in rows)


def test_judge_failure_leaves_rows_unscored(monkeypatch):
    def chat(*a):
        raise judge.aisa.AisaError("HTTP 402: no credit")
    monkeypatch.setattr(judge.aisa, "chat", chat)
    rows = [{"term": "strata", "examples": []}]
    assert "error" in judge.score(rows, "m") and "named" not in rows[0]


# --- merging the signals ---------------------------------------------------------------------

def _no_judge(monkeypatch):
    monkeypatch.setattr(report.judge, "score", lambda rows, model: {"judged": 0})


def test_leads_merge_a_term_seen_by_two_signals(cfg, monkeypatch):
    _no_judge(monkeypatch)
    blocks = {"x": {"terms": [{"term": "strata", "recent": 2, "note": "X: 2 accounts"}]},
              "breakout": {"terms": [{"term": "strata", "recent": 1, "note": "GitHub breakout",
                                      "variants": ["strata", "niko1221 strata"]}]},
              "sitemap": {"terms": [{"term": "other", "recent": 3, "lift": 4.0}]}}
    rows, _ = report.leads(cfg, blocks)
    assert rows[0]["term"] == "strata" and rows[0]["kinds"] == ["x", "breakout"]
    assert rows[0]["variants"] == ["strata", "niko1221 strata"]
    assert rows[1]["signals"] == ["③ sitemaps: 3 places (×4.0)"]


def test_names_lead_and_known_terms_are_not_recorded(cfg, monkeypatch):
    cfg["known_terms"] = ["Remotion"]

    def score(rows, model):
        for r in rows:
            r["named"] = {"dashboard": 0.1, "remotion skill": 0.9, "hyperframes": 0.95}[r["term"]]
        return {"judged": len(rows)}
    monkeypatch.setattr(report.judge, "score", score)
    blocks = {"sitemap": {"terms": [{"term": t, "recent": 3, "lift": 2.0} for t in ("dashboard", "remotion skill", "hyperframes")]}}
    rows, _ = report.leads(cfg, blocks)
    assert [r["term"] for r in report.named(cfg, rows)] == ["hyperframes", "remotion skill"]
    assert [r["term"] for r in report.to_record(cfg, rows)] == ["hyperframes"]


def test_known_terms_match_either_way(cfg):
    cfg["known_terms"] = ["ppt skills"]
    assert config.known(cfg, "ppt") == "ppt skills" and config.known(cfg, "best ppt skills") == "ppt skills"
    assert config.known(cfg, "hyperframes") == ""


# --- the store ---------------------------------------------------------------------------------

def _lead(term, kinds, **more):
    return {"term": term, "kinds": kinds, "signals": [f"note {term}"], **more}


def test_store_upserts_and_merges(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["x"], named=0.9, variants=["strata"])], date(2026, 9, 25))
    store.record(conn, [_lead("strata", ["breakout"], variants=["strata", "niko strata"])], date(2026, 9, 26))
    row = conn.execute("SELECT first_seen, last_seen, signals, named, variants FROM leads").fetchone()
    assert row[:2] == ("2026-09-25", "2026-09-26") and json.loads(row[2]) == ["breakout", "x"]
    assert row[3] == 0.9 and json.loads(row[4]) == ["strata", "niko strata"]


def test_store_due_verdict_and_summary(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["x", "breakout"]), _lead("flop", ["sitemap"])], date(2026, 9, 25))
    store.record(conn, [_lead("young", ["hf"])], date(2026, 10, 8))
    today = date(2026, 10, 10)
    assert [d[0] for d in store.due(conn, today, 14, 5)] == ["flop", "strata"]
    store.save_verdict(conn, "strata", "hit", 2, {}, today)
    store.save_verdict(conn, "flop", "miss", None, {}, today)
    assert store.due(conn, today, 14, 5) == []
    summ = store.summary(conn, today)
    assert summ["pending"] == 1 and summ["by_signal"]["x"] == {"checked": 1, "hit": 1, "median_lead": 2}
    assert summ["by_signal"]["sitemap"] == {"checked": 1, "hit": 0, "median_lead": None}


def test_store_recent_sources_are_only_x_and_hf(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("qwen3.9", ["x"]), _lead("strata", ["breakout"])], date(2026, 10, 10))
    assert [r["term"] for r in store.recent_sources(conn, date(2026, 10, 10))] == ["qwen3.9"]


# --- ⑤ review: did it rise, judged against its own baseline ----------------------------------

SEEN = date(2026, 9, 25)


def _series(values_by_day):
    start = date(2026, 9, 19)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values_by_day)]


def test_review_a_word_with_an_older_meaning_does_not_count_as_rising():
    # 'strata' alone, 2026-09-19..10-02, as DataForSEO returned it: high baseline throughout
    assert not review.rose(_series([40, 67, 70, 67, 76, 80, 68, 62, 70, 100, 89, 98, 36, 33]), SEEN)["rose"]


def test_review_a_new_name_rising_from_zero_counts_and_dates_the_rise():
    out = review.rose(_series([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 100, 0, 0]), SEEN)
    assert out["rose"] and out["first_rise"] == "2026-09-28"


def test_review_all_zero_is_no_data():
    assert review.rose(_series([0] * 14), SEEN)["no_data"]


def test_review_verdict_takes_the_earliest_variant_that_rose():
    per = {"strata": {"rose": False, "no_data": False},
           "niko1221 strata": {"rose": True, "first_rise": "2026-09-27"},
           "niko strata": {"rose": True, "first_rise": "2026-09-30"}}
    assert review.verdict(per, SEEN) == ("hit", 2)


def test_review_verdict_miss_and_no_data():
    assert review.verdict({"a": {"rose": False, "no_data": False}}, SEEN) == ("miss", None)
    assert review.verdict({"a": {"rose": False, "no_data": True}}, SEEN) == ("no_data", None)


def test_review_sends_one_trends_task_per_call():
    # DataForSEO's live endpoint runs only the first task of a body
    due = [("strata", date(2026, 9, 25), ["strata", "niko1221 strata", "niko strata", "a fourth"]), ("veo 3 free", date(2026, 10, 3), [])]
    calls, plan = review.plan_calls(due, date(2026, 10, 17))
    assert len(calls) == 4 and all(len(c["arguments"]["body"]) == 1 for c in calls)
    assert [c["arguments"]["body"][0]["keywords"] for c in calls[:3]] == [["strata"], ["niko1221 strata"], ["niko strata"]]
    assert calls[3]["arguments"]["body"][0]["keywords"] == ["veo 3 free"]
    assert calls[0]["arguments"]["body"][0]["date_to"] == "2026-10-09" and calls[3]["arguments"]["body"][0]["date_to"] == "2026-10-17"


def test_review_stores_the_verdict(cfg, tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("niko strata", ["breakout"])], SEEN)
    points = [{"date_from": (date(2026, 9, 19) + timedelta(days=i)).isoformat(), "values": [v]}
              for i, v in enumerate([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 100, 0, 0])]
    answer = {"tasks": [{"result": [{"items": [{"data": points}]}]}]}
    monkeypatch.setattr(review.aisa, "call_all", lambda calls: ({c["call_id"]: answer for c in calls}, 1200))
    out = review.review(cfg, conn, date(2026, 10, 10))
    assert out["checked"] == [{"term": "niko strata", "first_seen": "2026-09-25", "verdict": "hit", "lead_days": 3}]
    assert conn.execute("SELECT verdict, lead_days FROM leads").fetchone() == ("hit", 3)


# --- the two commands ---------------------------------------------------------------------------

def _write_config(tmp_path, **more):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"x_accounts": {"labs": ["a"]}, **more}))
    return path


def test_sources_run_writes_no_report_when_nothing_is_new(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "terms": []})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    out = tmp_path / "sources.md"
    assert cli.main(["sources", "--config", str(_write_config(tmp_path)), "--state", str(tmp_path / "s"), "--out", str(out)]) == 0
    assert not out.exists()


def test_sources_run_reports_and_keeps_a_new_name(tmp_path, monkeypatch):
    _no_judge(monkeypatch)
    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "cost_micros": 72000, "terms": [
        {"term": "qwen3.9", "recent": 2, "note": "X: 2 accounts, 900 likes", "examples": ["@Alibaba_Qwen: Qwen3.9 is out"],
         "posts": [{"account": "Alibaba_Qwen", "text": "Qwen3.9 is out", "url": "https://x.com/p/1", "likes": 900}]}]})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    out, state = tmp_path / "sources.md", tmp_path / "s"
    cli.main(["sources", "--config", str(_write_config(tmp_path)), "--state", str(state), "--out", str(out)])
    text = out.read_text()
    assert "**qwen3.9**" in text and "$0.072" in text and "[post](https://x.com/p/1)" in text
    assert store.connect(state).execute("SELECT term FROM leads").fetchall() == [("qwen3.9",)]


def test_daily_run_writes_a_report_when_every_signal_is_down(tmp_path, monkeypatch):
    def down(*a, **k):
        raise OSError("network is down")
    for mod, fn in ((cli.sources, "x_new"), (cli.sources, "hf_new"), (cli.breakout, "breakouts"), (cli.sitemaps, "diff")):
        monkeypatch.setattr(mod, fn, down)
    out = tmp_path / "report.md"
    assert cli.main(["daily", "--config", str(_write_config(tmp_path)), "--state", str(tmp_path / "s"), "--out", str(out),
                     "--skip", "newsites,review"]) == 0
    text = out.read_text()
    assert text.count("network is down") >= 4 and out.with_suffix(".json").exists()


def test_daily_report_in_chinese(tmp_path, monkeypatch):
    _no_judge(monkeypatch)
    monkeypatch.setattr(cli.breakout, "breakouts", lambda cfg, state: {"searched": 120, "first_run": False, "terms": [], "repos": [
        {"repo": "Niko1221/Strata", "stars": 7734, "velocity": 859, "delta": 600, "age_days": 9.0, "description": "Qwen3.8", "new": True}]})
    out = tmp_path / "report.md"
    cli.main(["daily", "--config", str(_write_config(tmp_path, lang="zh")), "--state", str(tmp_path / "s"), "--out", str(out),
              "--skip", "x,hf,sitemaps,newsites,review"])
    text = out.read_text()
    assert "## 新词雷达" in text and "🆕 [Niko1221/Strata]" in text and "| +600 |" in text


# --- topics and mail -----------------------------------------------------------------------------

def test_topic_section_lists_matching_leads_repos_and_queries(cfg):
    cfg["topics"] = {"AI video": r"\bvideo"}
    lead = [{"term": "reelmimic", "examples": ["edenfunf/reelmimic: Show it a video you love"], "signals": ["② breakout"]},
            {"term": "strata", "examples": ["Niko1221/Strata: Qwen3.8 on a gaming PC"], "signals": ["② breakout"]}]
    blocks = {"breakout": {"repos": [{"repo": "a/motion-video-kit", "description": "video kit", "stars": 900, "velocity": 300},
                                     {"repo": "Niko1221/Strata", "description": "Qwen3.8", "stars": 7700, "velocity": 870}]}}
    found = {"queries": [{"query": "claude video skills", "impressions": 6, "page": "/best/x/"},
                         {"query": "best ai scraping tools", "impressions": 3, "page": "/best/web-scraping/"}]}
    text = "\n".join(report.topic_lines(cfg, lead, blocks, found))
    assert "reelmimic" in text and "motion-video-kit" in text and "claude video skills" in text
    assert "strata" not in text.lower() and "scraping" not in text


def test_every_text_has_both_languages():
    assert all(len(pair) == 2 and all(pair) for pair in i18n.T.values())


def test_mail_html_renders_tables():
    out = notify.html("| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<table style=" in out and ">1</td>" in out
    # Gmail drops a <style> block outside <head>: every tag carries its own style.
    assert "<style>" not in out and '<td style="border:1px solid' in out


def test_mail_without_recipient_is_skipped(monkeypatch):
    monkeypatch.delenv("RADAR_EMAIL_TO", raising=False)
    assert notify.send("s", "x").startswith("mail skipped")


def test_mail_recipients_tolerate_how_people_type_them():
    assert notify.recipients('"a@b.com"') == ["a@b.com"]
    assert notify.recipients("a@b.com，c@d.cn") == ["a@b.com", "c@d.cn"]
    assert notify.recipients("<a@b.com>; c@d.cn\n") == ["a@b.com", "c@d.cn"]


# --- the generic terms behind the names ---------------------------------------------------------

from radar import usecases  # noqa: E402

TODAY = date(2026, 10, 12)


def _daily(values, end=TODAY - timedelta(days=1)):
    """A daily series ending yesterday, as Trends returns it; its last two days are unsettled."""
    start = end - timedelta(days=len(values) - 1)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def test_usecases_reads_new_names_first_then_those_due_again():
    seen = {"old": {"first": "2026-10-08", "runs": 1}, "done": {"first": "2026-10-01", "runs": 2},
            "fresh": {"first": "2026-10-11", "runs": 1}}
    assert usecases.due(["a", "old", "b"], seen, TODAY, 5, 3) == ["a", "b", "old"]
    assert usecases.due(["a", "b", "c"], seen, TODAY, 2, 3) == ["a", "b"]


def test_usecases_parse_drops_the_name_itself_sentences_and_repeats():
    answer = json.dumps({"use_cases": [
        {"phrase": "AI Image Detector", "evidence": "check if any image is AI generated", "brand": False},
        {"phrase": "synthid detector online", "evidence": "x", "brand": True},
        {"phrase": "detector", "evidence": "x"},
        {"phrase": "check whether a picture was made by a model today", "evidence": "x"},
        {"phrase": "ai image detector", "evidence": "again"},
        {"phrase": "photoshop alternative", "evidence": "free photoshop clone", "brand": True}]})
    out = usecases.parse("Sure:\n" + answer, "synthid detector")
    assert [(u["phrase"], u["brand"]) for u in out] == [("ai image detector", False), ("photoshop alternative", True)]
    assert usecases.parse("no json here", "x") == []


def test_trends_last_two_days_are_not_a_rise():
    # "ai image detector" on 2026-10-12: flat for 88 days, then 0 and 100 on the two unsettled days
    s = usecases.shape(_daily([5] * 88 + [0, 100]), TODAY)
    assert s["now"] == "flat" and not s["rose"]


def test_a_single_day_spike_on_a_small_term_is_not_a_rise():
    s = usecases.shape(_daily([0] * 80 + [100] + [0] * 9), TODAY)   # "debloat macos"
    assert not s["rose"]


def test_a_real_rise_is_dated_and_still_rising():
    # "synthid detector": nothing, then 9, 53, 40 on the last settled days, then the unsettled two
    s = usecases.shape(_daily([1] * 85 + [9, 53, 40, 59, 100]), TODAY)
    assert s["rose"] and s["new"] and s["now"] == "rising" and s["started"] == "2026-10-07" and s["peak_day"] == "2026-10-08"


def test_a_rise_that_came_back_down_is_fading():
    s = usecases.shape(_daily([1] * 70 + [60, 100, 70, 30, 10] + [5] * 15), TODAY)
    assert s["rose"] and s["now"] == "fading"


def test_searches_a_day_follow_the_terms_own_volume_along_its_curve():
    # 30,000 searches in August at level 10; level 20 in the last settled week: twice as many a day
    points = _daily([10] * 81 + [20] * 9)
    assert usecases.own_per_day(points, 30000, "2026-08", TODAY) == 2000
    assert usecases.own_per_day(points, 0, "2026-08", TODAY) is None


def test_the_benchmark_is_sized_from_terms_whose_volume_is_known():
    def row(volume, term_level, bench_level):
        return {"volume": volume, "month": "2026-08", "_pair": [(d, term_level, bench_level) for d, _ in _daily([0] * 90)]}
    # a term with 3,000 searches a day at a third of the benchmark's level puts the benchmark at 9,000
    assert usecases.bench_daily([row(90000, 10, 30), row(900, 1, 50)]) == 9000   # the 900/month term is too small to count
    assert usecases.bench_daily([row(900, 1, 50)]) is None


def test_decisions():
    up, flat, down = {"now": "rising"}, {"now": "flat"}, {"now": "fading"}
    assert usecases.decide(up, 5000, 300) == "prioritize" and usecases.decide(up, 40, 300) == "watch"
    assert usecases.decide(flat, 5000, 300) == "watch" and usecases.decide(flat, 40, 300) == "skip"
    assert usecases.decide(down, 5000, 300) == "skip" and usecases.decide({"no_data": True}, None, 300) == "watch"


def _fake_usecase_network(monkeypatch):
    rise = _daily([1] * 85 + [9, 53, 40, 59, 100])
    flat = _daily([20] * 90)
    monkeypatch.setattr(usecases, "evidence", lambda names, repos, today, texts=None: (
        {n: {"posts": [{"text": "check if any image is AI generated", "likes": 900, "by": "a", "url": ""}], "readme": "", "repo": ""} for n in names}, 3000))
    monkeypatch.setattr(usecases, "propose", lambda model, name, ev: [
        {"phrase": "ai image detector", "evidence": "check if any image is AI generated", "brand": False},
        {"phrase": "photoshop alternative", "evidence": "free clone", "brand": True},
        {"phrase": "nobody searches this", "evidence": "x", "brand": False}])
    monkeypatch.setattr(usecases, "screen", lambda model, name, ev, cands: cands)
    monkeypatch.setattr(usecases, "volumes", lambda phrases, loc: (
        {"ai image detector": {"volume": 60000, "month": "2026-08"}, "photoshop alternative": {"volume": 6000, "month": "2026-08"},
         "nobody searches this": {"volume": 0, "month": ""}}, 90000))
    monkeypatch.setattr(usecases, "fetch_series", lambda keywords, bench, today: (
        {k: {"solo": rise if k == "synthid detector" else flat,
             "pair": [(d, v, 40) for d, v in (rise if k == "synthid detector" else flat)]} for k in keywords}, 7200))


def test_usecases_run_builds_one_table_and_reads_a_name_twice(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    assert uc["proposed"] == 3 and uc["relevant"] == 3 and uc["kept"] == 2 and "nobody searches this" not in rows
    assert rows["synthid detector"]["kind"] == "name" and rows["synthid detector"]["decision"] == "prioritize"
    assert rows["ai image detector"]["per_day"] == 2000 and rows["ai image detector"]["decision"] == "watch"
    assert rows["photoshop alternative"]["brand"] and rows["photoshop alternative"]["decision"] == "skip"
    assert uc["bench_daily"] == 2200 and uc["cost_micros"] == 100200   # the median of 4,000 and 400
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=1))["names"] == []
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=3))["names"] == ["synthid detector"]
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=9))["names"] == []


def test_keyword_table_marks_brands_and_generic_terms_become_leads(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "| **synthid detector** | name |" in text and "**prioritize**" in text
    assert "| **photoshop alternative** ⚠ | generic | synthid detector |" in text and "someone's brand" in text
    assert "“check if any image is AI generated”" in text
    leads = report.usecase_leads(uc)
    assert [x["term"] for x in leads] == ["ai image detector", "photoshop alternative"] and leads[0]["kinds"] == ["usecase"]


def test_review_ignores_the_unsettled_days(cfg, tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("ai image detector", ["usecase"])], date(2026, 9, 28))
    points = [{"date_from": d.isoformat(), "values": [v]} for d, v in _daily([5] * 43 + [0, 100])]
    answer = {"tasks": [{"result": [{"items": [{"data": points}]}]}]}
    monkeypatch.setattr(review.aisa, "call_all", lambda calls: ({c["call_id"]: answer for c in calls}, 1200))
    assert review.review(cfg, conn, TODAY)["checked"][0]["verdict"] == "miss"


def test_screen_drops_loosely_related_keywords_and_keeps_all_when_the_call_fails(monkeypatch):
    cands = [{"phrase": p, "brand": False} for p in ("ai image detector", "chatgpt alternative", "ai audio detector")]
    ev = {"posts": [{"text": "check if any image is AI generated", "likes": 900, "by": "a"}], "readme": ""}
    monkeypatch.setattr(usecases.aisa, "chat", lambda *a, **k: '[{"i": 0, "score": 0.95}, {"i": 1, "score": 0.2}, {"i": 2, "score": 0.7}]')
    kept = usecases.relevant(usecases.screen("m", "synthid detector", ev, cands))
    assert [(c["phrase"], c["relevance"]) for c in kept] == [("ai image detector", 0.95), ("ai audio detector", 0.7)]

    def down(*a, **k):
        raise usecases.aisa.AisaError("HTTP 500")
    monkeypatch.setattr(usecases.aisa, "chat", down)
    assert usecases.screen("m", "synthid detector", ev, cands) == cands


def test_momentum_is_the_last_settled_week_against_the_week_before():
    assert usecases.shape(_daily([10] * 76 + [20] * 7 + [40] * 7 + [0, 100]), TODAY)["momentum"] == 2.0
    assert usecases.shape(_daily([0] * 83 + [40] * 7 + [0, 100]), TODAY)["momentum"] is None   # nothing the week before


# --- the optional second source: what people searched together with a name ----------------------

def test_related_queries_split_into_long_tails_and_candidates():
    related = {"rising": ["ai photo detector", "google synthid detector", "openai", "gpt image 2.5"],
               "top": ["ai detector", "synthid detector", "synthid detector online", "ai photo detector"]}
    tails, cands = usecases.split_related("synthid detector", related)
    assert tails == ["google synthid detector", "synthid detector online"]
    assert [c["phrase"] for c in cands] == ["ai photo detector", "gpt image 2.5", "ai detector"]   # "openai" is one word
    assert cands[0]["source"] == "searched" and "rising" in cands[0]["evidence"]


def test_screen_marks_brands_and_other_names(monkeypatch):
    cands = [{"phrase": "free up mac disk space", "brand": False, "source": "model"},
             {"phrase": "gpt image 2.5", "brand": False, "source": "searched"}]
    monkeypatch.setattr(usecases.aisa, "chat", lambda *a, **k:
                        '[{"i": 0, "score": 0.8, "brand": true, "name": false}, {"i": 1, "score": 0.1, "brand": true, "name": true}]')
    viewed = usecases.screen("m", "removemacai", {"posts": [], "readme": "frees 12 GB"}, cands)
    assert viewed[0]["brand"] and viewed[1]["is_name"]
    assert [c["phrase"] for c in usecases.relevant(viewed)] == ["free up mac disk space"]


def test_with_a_key_related_queries_join_the_table_and_other_names_become_leads(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    monkeypatch.setattr(usecases.goanyapi, "related_queries", lambda name, geo: (
        {"rising": ["ai photo detector", "gpt image 2.5", "synthid detector online"], "top": []}, 6000))
    monkeypatch.setattr(usecases, "screen", lambda model, name, ev, cands: [
        dict(c, relevance=0.9, is_name=c["phrase"] == "gpt image 2.5") for c in cands])
    monkeypatch.setattr(usecases, "volumes", lambda phrases, loc: (
        {p: {"volume": 9000, "month": "2026-08"} for p in phrases if p != "nobody searches this"}, 90000))
    cfg["usecases"]["country_trends"] = False
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    assert rows["ai photo detector"]["source"] == "searched" and "gpt image 2.5" not in rows
    assert uc["cosearch"] == [{"term": "gpt image 2.5", "with": "synthid detector"}]
    assert uc["tails"] == {"synthid detector": ["synthid detector online"]} and uc["cost_micros"] == 106200
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "| **ai photo detector** 🖼 🔎 |" in text and "gpt image 2.5 (with synthid detector)" in text and "synthid detector online" in text
    assert "incomplete today" not in text
    assert {"term": "gpt image 2.5", "kinds": ["cosearch"]}.items() <= report.usecase_leads(uc)[-1].items()


def test_without_a_key_the_table_says_the_source_is_off_and_a_failure_is_reported(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    assert not uc["related"] and "GOANYAPI_API_KEY" in "\n".join(report.keyword_table(cfg, uc))
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")

    def down(name, geo):
        raise usecases.goanyapi.GoAnyError("HTTP 502")
    monkeypatch.setattr(usecases.goanyapi, "related_queries", down)
    cfg["usecases"]["country_trends"] = False
    uc = usecases.run(cfg, tmp_path / "b", ["synthid detector"], {}, TODAY)
    assert uc["related_errors"] == ["synthid detector: HTTP 502"] and uc["kept"] == 2
    assert "HTTP 502" in "\n".join(report.keyword_table(cfg, uc))


def test_other_names_leave_out_old_products_siblings_and_spelling_variants():
    # what the first real run kept beside filmcraft and pdfcraft (2026-10-13)
    viewed = [{"phrase": p, "is_name": True, "rising": True, "established": old} for p, old in (
        ("premiere pro", True), ("davinci resolve", True), ("photocraft github", False), ("gpt image 2.5", False),
        ("adobe artcraft", False), ("artcraft adobe", False))]
    assert [x["term"] for x in usecases.other_names("filmcraft", viewed, ["photocraft", "filmcraft"])] == ["gpt image 2.5", "adobe artcraft"]
    tails, cands = usecases.split_related("pdfcraft", {"rising": ["pdf craft", "pdf-craft review", "merge pdf files"], "top": []})
    assert tails == ["pdf craft", "pdf-craft review"] and [c["phrase"] for c in cands] == ["merge pdf files"]


def test_every_name_gets_the_country_series_and_the_cap_is_on_generic_terms(cfg, monkeypatch):
    rows = [{"phrase": f"name{i}", "kind": "name"} for i in range(5)] + \
           [{"phrase": f"g{v}", "kind": "generic", "volume": v} for v in (27100, 22200, 8100, 6600, 900)]
    cfg["usecases"]["country_max"] = 3
    asked = []
    monkeypatch.setattr(usecases.goanyapi, "daily_series", lambda term, geo: (asked.append(term) or [(TODAY, 1)], 6000))
    out, cost, errors = usecases.country_series(rows, cfg["usecases"])
    assert sorted(asked) == sorted([f"name{i}" for i in range(5)] + ["g27100", "g22200", "g8100"]) and cost == 48000 and not errors


def test_other_names_must_be_rising_and_not_the_same_family():
    viewed = [{"phrase": p, "is_name": True, "rising": r} for p, r in (
        ("gpt image 2.5", True), ("google nano banana", True), ("nano banana pro", True), ("google gemini", False), ("mistral large 4", True))]
    assert [x["term"] for x in usecases.other_names("nano banana 2.1", viewed)] == ["gpt image 2.5", "mistral large 4"]


# --- one country's settled series for the rows that matter -------------------------------------

def _month(values):
    """A 30-day series ending today, as goanyapi.daily_series returns it."""
    return [(TODAY - timedelta(days=len(values) - 1 - i), v) for i, v in enumerate(values)]


def test_an_established_term_80_percent_up_for_five_days_is_a_rise_in_the_country_series():
    # "photoshop alternative" in the US around PhotoCraft: about 50, then 77, 86, 83, 100, 94, and 79 on the partial day
    points = _month([52, 48, 50, 47, 55, 51, 49, 53, 50, 46, 52, 50, 48, 54, 49, 51, 50, 47, 53, 46, 47, 53, 57, 55, 77, 86, 83, 100, 94, 79])
    s = usecases.shape(points, TODAY, usecases.COUNTRY)
    assert s["rose"] and not s["new"] and s["now"] == "rising" and s["started"] == "2026-10-07" and s["momentum"] >= 1.4


def test_a_flat_country_series_stays_flat():
    assert usecases.shape(_month([30, 34, 28, 31] * 7 + [33, 29]), TODAY, usecases.COUNTRY)["now"] == "flat"


def test_names_and_big_generic_terms_take_their_shape_from_the_country(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    cfg["usecases"]["related"] = False
    up = _month([50] * 24 + [77, 86, 83, 100, 94, 79])
    asked = []

    def daily_series(term, geo):
        asked.append(term)
        if term == "synthid detector":
            raise usecases.goanyapi.GoAnyError("HTTP 502")
        return up, 6000
    monkeypatch.setattr(usecases.goanyapi, "daily_series", daily_series)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    # The lookups run three at a time, so the order they are recorded in is the threads' own.
    assert sorted(asked) == ["ai image detector", "photoshop alternative", "synthid detector"]
    assert rows["ai image detector"]["trend_geo"] and rows["ai image detector"]["now"] == "rising"
    assert rows["ai image detector"]["decision"] == "prioritize"      # flat worldwide, up in the US
    assert "trend_geo" not in rows["synthid detector"] and uc["country_errors"] == ["synthid detector: HTTP 502"]
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "still rising (US)" in text and "worldwide data used instead: synthid detector" in text
    assert "This table is incomplete today" in text


def test_small_generic_terms_are_not_worth_a_country_lookup(cfg):
    rows = [{"phrase": "a", "kind": "generic", "volume": 300}, {"phrase": "b", "kind": "generic", "volume": 5000}, {"phrase": "n", "kind": "name"}]
    asked = []
    import radar.goanyapi as g
    orig = g.daily_series
    g.daily_series = lambda term, geo: (asked.append(term) or [], 0)
    try:
        usecases.country_series(rows, cfg["usecases"])
    finally:
        g.daily_series = orig
    assert asked == ["n", "b"]


def test_started_rising_is_the_run_that_holds_the_peak_not_an_earlier_blip():
    # a one-day blip on 10-01, then the real rise from 10-07
    points = _month([20] * 19 + [45] + [20] * 5 + [70, 100, 68, 51, 42])
    s = usecases.shape(points, TODAY, usecases.COUNTRY)
    assert s["rose"] and s["started"] == "2026-10-08" and s["peak_day"] == "2026-10-09"


def test_only_things_are_read_for_generic_terms_not_words_that_surged():
    today = [{"term": "mcd", "kinds": ["github"], "named": 0.75}, {"term": "kr.co.klisten.api", "kinds": ["sitemap"], "named": 0.78},
             {"term": "photocraft", "kinds": ["breakout"], "named": 0.95}, {"term": "dots", "kinds": ["breakout"], "named": 0.65},
             {"term": "hailuo", "kinds": ["newsite", "sitemap"]}]
    recent = [{"term": "qwen image 2.1", "kinds": ["hf"], "named": 0.96}, {"term": "use rea", "kinds": ["x"], "named": 0.72},
              {"term": "photocraft", "kinds": ["x"], "named": 0.9}]
    assert usecases.names_to_read(today, recent) == ["photocraft", "hailuo", "qwen image 2.1"]


def test_a_failed_post_search_costs_one_name_its_posts_not_the_step(monkeypatch):
    def call(calls):
        if "photocraft" in calls[0]["arguments"]["query"]:
            raise usecases.aisa.AisaError("HTTP 504: <html><h1>504 Gateway Time-out</h1></html>")
        return {"s": {"tweets": [{"text": "check any image", "likeCount": 900, "author": {"userName": "a"}}]}}, 3000
    monkeypatch.setattr(usecases.aisa, "call", call)
    monkeypatch.setattr(usecases, "_readme", lambda repo: "PhotoCraft opens real PSD files")
    ev, cost = usecases.evidence(["photocraft", "synthid detector"], {"photocraft": "storytold/photocraft"}, TODAY)
    assert cost == 3000 and ev["photocraft"]["readme"] and ev["photocraft"]["posts"] == []
    assert ev["photocraft"]["error"] == "HTTP 504: 504 Gateway Time-out" and len(ev["synthid detector"]["posts"]) == 1


# --- names someone asks about: the `keywords` command and the MCP tool ---------------------------

def test_keywords_on_request_reads_every_name_now_and_leaves_the_record_alone(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc, text = cli.keyword_table(cfg, tmp_path, ["SynthID Detector ", "synthid detector", ""], {})
    assert uc["names"] == ["synthid detector"] and "| **ai image detector** 🖼 |" in text
    again, _ = cli.keyword_table(cfg, tmp_path, ["synthid detector"], {})
    assert again["names"] == ["synthid detector"]                 # asked again, read again
    assert not (tmp_path / "usecases.json.gz").exists()           # the daily run's read-twice record is untouched
    many, _ = cli.keyword_table(cfg, tmp_path, [f"name {i}" for i in range(20)], {})
    assert len(many["names"]) == usecases.ASKED_MAX


def test_leads_command_lists_what_was_kept(tmp_path, capsys):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["breakout"], named=0.9)], date.today() - timedelta(days=2))
    store.record(conn, [_lead("old one", ["x"])], date.today() - timedelta(days=30))
    data = cli.kept_leads(tmp_path, 7)
    assert [r["term"] for r in data["leads"]] == ["strata"] and data["hit_rates"]["pending"] == 2
    assert cli.main(["leads", "--state", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "strata  [breakout]" in out and "1 leads in the last 7 days" in out


# --- the MCP server -----------------------------------------------------------------------------

def test_mcp_tools_answer_and_never_print(cfg, tmp_path, monkeypatch, capsys):
    import pytest
    pytest.importorskip("mcp")
    from radar import mcp_server
    (tmp_path / "config.json").write_text(json.dumps({"x_accounts": {"labs": ["a"]}}))
    monkeypatch.setenv("RADAR_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RADAR_STATE", str(tmp_path / "state"))
    store.record(store.connect(tmp_path / "state"), [_lead("photocraft", ["breakout"], named=0.95)], date.today())
    assert [r["term"] for r in mcp_server.recent_leads(7)["leads"]] == ["photocraft"]

    _fake_usecase_network(monkeypatch)
    out = mcp_server._keyword_table(["synthid detector"], {})
    assert "| **ai image detector** 🖼 |" in out["table_markdown"] and out["incomplete"] is False and out["cost_usd"] == 0.1

    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "cost_micros": 3000, "terms": [
        {"term": "qwen3.9", "recent": 2, "note": "X: 2 accounts", "examples": ["@a: Qwen3.9 is out"]}]})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    monkeypatch.setattr(report.judge, "score", lambda rows, model: {"judged": 0})
    ran = mcp_server._run("sources", [])
    assert ran["wrote_report"] and "**qwen3.9**" in ran["report_markdown"]
    assert capsys.readouterr().out == ""        # stdout is the protocol channel


def test_skills_have_a_name_and_a_description():
    root = Path(__file__).resolve().parents[1] / "skills"
    for name in ("new-term-radar", "ai-listening", "seo-new-keywords-research"):
        text = (root / name / "SKILL.md").read_text()
        head = re.match(r"---\nname: (\S+)\ndescription: (.+)\n---\n", text)
        assert head and head.group(1) == name and 100 < len(head.group(2)) < 1024
        assert "python -m radar" in text and "AISA_API_KEY" in text
        # Claude Code replaces $0, $1 ... in a skill's text with the user's arguments: "$0.05" came out
        # as "<what the user typed>.05" on the first real run. Amounts are written "0.05 USD".
        assert not re.search(r"\$\d", text)


def test_the_aisa_key_comes_from_the_environment_or_the_cli_login(tmp_path, monkeypatch):
    from radar import aisa
    monkeypatch.setattr(aisa.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("AISA_API_KEY", raising=False)
    import pytest
    with pytest.raises(aisa.AisaError, match="aisa login"):
        aisa.key()
    (tmp_path / ".aisa").mkdir()
    (tmp_path / ".aisa" / "key").write_text("from-cli\n")
    assert aisa.key() == "from-cli"
    monkeypatch.setenv("AISA_API_KEY", "from-env")
    assert aisa.key() == "from-env"


def test_every_manifest_carries_the_same_version():
    # A release is one number in five places; the registry rejects a server.json whose
    # version PyPI does not have.
    import radar
    root = Path(__file__).resolve().parents[1]
    server = json.loads((root / "server.json").read_text())
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text())
    pyproject = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
    assert {pyproject, server["version"], server["packages"][0]["version"], plugin["version"]} == {radar.__version__}
    assert len(server["description"]) <= 100                                  # the registry's limit
    assert f"mcp-name: {server['name']} -->" in (root / "README.md").read_text()   # how the registry checks who owns the PyPI package


def test_the_mcp_command_says_what_is_missing(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    import sys
    import radar
    monkeypatch.delitem(sys.modules, "radar.mcp_server", raising=False)   # an earlier test may have loaded it;
    monkeypatch.delattr(radar, "mcp_server", raising=False)               # the package keeps its own reference
    monkeypatch.setattr(builtins, "__import__", without_mcp)
    assert cli.main(["mcp"]) == 1
    assert "new-term-radar[mcp]" in capsys.readouterr().err


# --- page one: can a new site get in? ------------------------------------------------------------

from radar import competition  # noqa: E402


def _serp(results, ads=False):
    items = [{"type": "ai_overview"}] + [{"type": "organic", "domain": d, "url": u, "title": t, "rank_group": i + 1}
                                         for i, (d, u, t) in enumerate(results)]
    return {"tasks": [{"result": [{"items": items + ([{"type": "paid"}] if ads else [])}]}]}


def test_page_one_verdicts():
    assert competition.verdict(3, None) == "open" and competition.verdict(0, 15) == "open"
    assert competition.verdict(0, 55) == "crowded" and competition.verdict(None, None) == "crowded"
    assert competition.verdict(1, 30) == "mixed" and competition.verdict(None, 30) == "mixed"
    assert competition.page_kind("youtube.com", "https://youtube.com/watch?v=1", "x") == "video"
    assert competition.page_kind("headshotpro.com", "https://headshotpro.com/", "Free AI Headshot Generator") == "tool"
    assert competition.page_kind("headshots.com", "https://headshots.com/best", "16 Best AI Headshot Generators") == "list"


def test_a_crowded_rising_term_is_watched_not_prioritized(cfg, tmp_path, monkeypatch):
    cfg["competition"]["max_rows"] = 8
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    strong = [("canva.com", "https://canva.com/ai", "Canva AI"), ("adobe.com", "https://adobe.com/x", "Adobe")]
    weak = [("tiny1.ai", "https://tiny1.ai/", "Free Tool"), ("tiny2.ai", "https://tiny2.ai/", "Free Tool"),
            ("tiny3.ai", "https://tiny3.ai/x", "A guide")]
    pages = {"crowded term": _serp(strong, ads=True), "open term": _serp(strong + weak)}

    def call(calls):
        c = calls[0]
        if c["call_id"] == "serp":
            return {"serp": pages[c["arguments"]["body"][0]["keyword"]]}, 2000
        items = [{"keyword": k, "keyword_properties": {"keyword_difficulty": 60 if k == "crowded term" else 25}}
                 for k in c["arguments"]["body"][0]["keywords"]]
        return {"kd": {"tasks": [{"result": [{"items": items}]}]}}, 12000
    monkeypatch.setattr(competition.aisa, "call", call)
    rated = {"canva.com": 94, "adobe.com": 96, "tiny1.ai": 8, "tiny2.ai": 12, "tiny3.ai": 21}
    asked = []

    def goany(path, **p):
        if path == "dr":
            asked.append(p["domain"])
            return {"domain_rating": rated[p["domain"]]}, 2000
        if path == "google-intitle":
            return {"intitleCount": 19500}, 4000
        return {"domains": ["paid.example"] if p["keyword"] == "crowded term" else [], "advertisers": []}, 4000
    monkeypatch.setattr(competition.goanyapi, "call", goany)
    rows = [{"phrase": "crowded term", "decision": "prioritize"}, {"phrase": "open term", "decision": "prioritize"},
            {"phrase": "faded", "decision": "skip"}]
    cost, errors = competition.assess(rows, cfg, tmp_path)
    crowded, opened, faded = rows
    assert crowded["decision"] == "watch" and crowded["demoted"] and crowded["competition"]["ads"]
    assert crowded["competition"]["verdict"] == "crowded" and crowded["competition"]["weak"] == 0
    assert opened["decision"] == "prioritize" and opened["competition"]["weak"] == 3
    assert opened["competition"]["weakest"] == {"domain": "tiny1.ai", "dr": 8, "position": 3, "url": "https://tiny1.ai/"}
    assert "competition" not in faded and errors == [] and cost == 2 * 2000 + 12000 + 5 * 2000 + 2 * (4000 + 4000)
    assert crowded["competition"]["built"] == 19500 and crowded["competition"]["paid"] == ["paid.example"]
    competition.assess([{"phrase": "open term", "decision": "watch"}], cfg, tmp_path)
    assert len(asked) == 5                                    # ratings are kept 30 days, not bought again
    uc = {"names": ["x"], "rows": [dict(r, kind="generic", names=["x"], brand=False, new=True, now="rising", peak_day="2026-10-10",
                                       evidence="") for r in rows[:2]],
          "proposed": 2, "relevant": 2, "kept": 2, "benchmark": "gpts", "volume_month": "2026-08", "related": True}
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "crowded · 0 weak · KD 60 · ads · 19,500 pages built · 1 advertisers" in text and "watch ↓" in text
    assert "open · 3 weak · KD 25 · 19,500 pages built |" in text and "advertising on it: paid.example" in text
    assert "tiny1.ai (rated 8) at #3" in text and "Page one: open = three or more" in text
    assert "**crowded term**: no easy slot, even the weakest, canva.com, is rated 94 (#1)" in text



# --- ⑥ revenue: products people pay for ----------------------------------------------------------

from radar import revenue  # noqa: E402


def _product(slug, name, usd, growth=None, website="https://x.example", stealth=False, about="Turns photos into headshots."):
    return {"slug": slug, "name": name, "website": website, "description": about, "category": "ai",
            "revenue": {"last30Days": usd * 100}, "growth30d": growth, "stealthMode": stealth}


def test_revenue_reports_growth_then_a_product_that_left_zero(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTMRR_API_KEY", "k")
    monkeypatch.setattr(revenue.time, "sleep", lambda s: None)
    pages = {"growth-desc": [_product("headshot", "HeadshotKit", 4200, 4500.0, "https://headshotkit.ai"),
                             _product("anon", "Anonymous startup", 9000, 2.0, website="", stealth=True),
                             _product("gpt", "Talent Tech Global Limited", 840, 1.6, "https://gptproto.com")],
             "listed-desc": [_product("baby", "AI Baby Generator", 0, None, "https://aibabygenerator.ai")]}
    monkeypatch.setattr(revenue, "fetch", lambda sort, page, c: pages[sort] if page == 1 else [])
    first = revenue.products(cfg, tmp_path)
    assert first["first_run"] and [r["term"] for r in first["products"]] == ["headshotkit", "gptproto"]
    assert first["products"][0]["growth_text"] == "up from near $0" and first["products"][1]["growth_text"] == "+160%"
    assert first["terms"][0]["about"] == ("https://headshotkit.ai", "Turns photos into headshots.")
    pages["growth-desc"] = []
    pages["listed-desc"] = [_product("baby", "AI Baby Generator", 900, None, "https://aibabygenerator.ai")]
    later = revenue.products(cfg, tmp_path)
    assert [r["term"] for r in later["products"]] == ["ai baby generator"] and later["products"][0]["from_zero"]
    assert later["terms"][0]["note"] == "TrustMRR: from $0 to $900 in 30 days"
    text = "\n".join(report._revenue_lines(later, cfg))
    assert "**$0 last time, earning now**" in text and "forbid republishing" in text


def test_revenue_without_a_key_says_it_is_off(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("TRUSTMRR_API_KEY", raising=False)
    rv = revenue.products(cfg, tmp_path)
    assert rv == {"skipped": True, "no_key": True}
    assert "No TRUSTMRR_API_KEY" in "\n".join(report._revenue_lines(rv, cfg))



# --- what a lead makes, and one-click chances ----------------------------------------------------

from radar import opportunity  # noqa: E402


def _posts(*texts_likes):
    return [{"text": x, "likes": n} for x, n in texts_likes]


def test_format_reads_the_name_first_then_what_most_posts_say():
    assert opportunity.fmt({"term": "veo 3.1", "posts": []}) == "video"
    assert opportunity.fmt({"term": "elevenlabs v4", "posts": _posts(
        ("give your characters a voice", 1500), ("tts that whispers", 500), ("audio and sound effects", 400),
        ("made a video with it", 300))}) == "voice"                           # voice 3, video 1
    assert opportunity.fmt({"term": "claude opus 5.5", "posts": _posts(
        ("it made an animation", 100000), ("great at code", 6000), ("writes well", 6000), ("fast", 3000),
        ("cheap", 3000), ("smart", 1000), ("solid", 900), ("ok", 800))}) == "text"   # one post in eight is not a format
    assert opportunity.fmt({"term": "headshotkit", "about": ("https://headshotkit.ai", "AI headshots from selfies")}) == "image"


def test_one_click_needs_a_tutorial_people_shared():
    assert opportunity.one_click(_posts(("here's how to build this with Opus + Blender + Unreal", 1699))) == "tutorial"
    assert opportunity.one_click(_posts(("here's how I did it", 120))) is None              # nobody shared it
    assert opportunity.one_click(_posts(("8 denoising steps, open weights", 5301))) is None   # an announcement
    assert opportunity.one_click(_posts(("everyone is making these figurines", 4200))) == "trend"


def test_leads_carry_format_marks_and_the_legend(cfg, monkeypatch):
    monkeypatch.setattr(report.judge, "score", lambda rows, model: {"judged": len(rows)})
    blocks = {"x": {"terms": [{"term": "clipforge", "recent": 2, "note": "X", "examples": ["@a: clipforge animates any photo"],
                               "posts": [{"text": "here's how: step 1 upload, step 2 prompt", "likes": 2400}]}]}}
    lead, _ = report.leads(cfg, blocks)
    assert lead[0]["format"] == "video" and lead[0]["one_click"] == "tutorial"
    assert report._marks(lead[0]) == " 🎬 ⚡"



# --- ⑦ asks, pains and tutorial comments ---------------------------------------------------------

from radar import asks, pains  # noqa: E402


def _reddit_post(pid, title, score=12, sub="VideoEditors"):
    return {"id": pid, "title": title, "text": "", "score": score, "comments": 3, "subreddit": sub,
            "url": f"https://www.reddit.com/r/{sub}/{pid}", "created": 0}


def test_asks_turns_reddit_requests_into_sized_jobs(cfg, tmp_path, monkeypatch):
    cfg["asks"]["patterns"] = ['"is there an ai tool"', '"how do i automate"']
    posts = {'"is there an ai tool"': [_reddit_post("a", "Is there an AI tool to add b-roll automatically?", 45),
                                       _reddit_post("b", "Any AI that turns slides into notes?", 35, "college"),
                                       _reddit_post("c", "Rate my setup", 1)],
             '"how do i automate"': [_reddit_post("a", "Is there an AI tool to add b-roll automatically?", 45)]}
    monkeypatch.setattr(asks.social, "reddit", lambda q, **k: (posts[q], 1880))
    monkeypatch.setattr(asks.aisa, "chat", lambda model, system, user: json.dumps(
        [{"i": i, "kind": "request", "job": job, "ai": True} for i, job in enumerate(
            ["add b roll automatically", "convert slides to notes"])]))
    monkeypatch.setattr(asks.usecases, "volumes", lambda phrases, loc: (
        {"add b roll automatically": {"volume": 880}, "convert slides to notes": {"volume": 10}}, 90000))
    first = asks.asks(cfg, tmp_path)
    assert first["read"] == 2 and first["asks"] == 2 and first["unsized"] == 1          # "rate my setup" scored under the floor
    assert [r["phrase"] for r in first["rows"]] == ["add b roll automatically"] and first["cost_micros"] == 2 * 1880 + 90000
    assert first["terms"][0]["note"] == "Reddit: asked 1× (r/VideoEditors)"
    assert [q["phrase"] for q in first["quiet"]] == ["convert slides to notes"]           # asked, too few searches yet
    text = "\n".join(report._asks_lines(first, cfg))
    assert "| **add b roll automatically** | 1 (45) | r/VideoEditors | 880 |" in text and "do not republish" in text
    assert "| convert slides to notes | 1 (35) | r/college |" in text and "ideas rather than keywords" in text
    again = asks.asks(cfg, tmp_path)
    assert again["rows"][0]["again"] and again["terms"] == []                            # reported once


def test_pains_groups_reviews_posts_and_comments_into_gaps(cfg, monkeypatch):
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    ext = "a" * 32

    def serp(q, gl="US"):
        if q.startswith("site:chromewebstore"):
            return [{"url": f"https://chromewebstore.google.com/detail/headshot-ai/{ext}", "title": "x", "rank": 1}], 2000
        return [{"url": "https://www.youtube.com/watch?v=abcdefghijk", "title": "tutorial", "rank": 1}], 2000
    monkeypatch.setattr(pains.goanyapi, "serp", serp)
    monkeypatch.setattr(pains.goanyapi, "chrome_reviews", lambda e, c="": (
        [{"rating": 1, "text": "Needs a photo of your face, no way to upload a group shot", "date": "2026-10-13"},
         {"rating": 5, "text": "Great!", "date": "2026-10-12"}], "", 3000))
    monkeypatch.setattr(pains.social, "reddit", lambda q, **k: ([_reddit_post("r1", "Headshot apps all look plastic")], 1880))
    monkeypatch.setattr(pains.social, "youtube_comments", lambda vid: ([{"text": "Is there a way to do this for a whole team at once?", "likes": 40},
                                                                        {"text": "nice", "likes": 0}], 1000))
    monkeypatch.setattr(pains.aisa, "chat", lambda model, system, user, max_tokens=2500: json.dumps([
        {"gap": "No team or group headshots", "kind": "missing", "items": [0, 2], "phrase": "ai headshots for teams"},
        {"gap": "Results look plastic", "kind": "quality", "items": [1]}]))
    result = pains.run(cfg, "ai headshot")
    assert result["by_source"] == {"chrome": 1, "reddit": 1, "youtube": 1}           # 5 stars, "nice" and repeats are left out
    assert result["gaps"][0]["count"] == 2 and result["gaps"][0]["sources"] == ["chrome", "youtube"]
    text = report.pains(result)
    assert "| No team or group headshots | missing | 2 | ai headshots for teams |" in text and "Trustpilot" in text


def test_tutorial_asks_only_for_leads_marked_one_click(cfg, monkeypatch):
    cfg["tutorials"]["max_leads"] = 3
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    monkeypatch.setattr(pains.goanyapi, "serp", lambda q, gl="US": ([{"url": "https://www.youtube.com/watch?v=abcdefghijk", "title": "t", "rank": 1}], 2000))
    monkeypatch.setattr(pains.social, "youtube_comments", lambda vid: ([{"text": "Can someone make this one click? Too many steps", "likes": 12}], 1000))
    monkeypatch.setattr(pains.aisa, "chat", lambda model, system, user, max_tokens=2500: json.dumps(
        [{"gap": "Too many manual steps", "kind": "wants a tool", "items": [0], "phrase": "one click blender export"}]))
    out = pains.tutorial_asks(cfg, [{"term": "opus blender game", "one_click": "tutorial"}, {"term": "plain", "one_click": None}])
    assert [x["term"] for x in out["leads"]] == ["opus blender game"] and out["cost_micros"] == 2000 + 1000
    text = "\n".join(report._tutorial_lines(out))
    assert "Too many manual steps (wants a tool, 1) → one click blender export" in text


# --- datasrc: provider dispatch + response shaping --------------------------------

from radar import datasrc, aisa as _aisa  # noqa: E402
import pytest as _pytest  # noqa: E402


def _no_creds(monkeypatch):
    for k in ("WEBCAFE_TOKEN", "DATAFORSEO_LOGIN", "DATAFORSEO_PASSWORD",
              "AISA_API_KEY", "RADAR_DATA_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    datasrc._INSTANCES.clear()


def test_datasrc_no_credentials_raises_aisa_error(monkeypatch):
    _no_creds(monkeypatch)
    with _pytest.raises(_aisa.AisaError):
        datasrc.call([{"call_id": "c", "tool": datasrc.DFS_VOLUME, "arguments": {"body": []}}])


def test_datasrc_unknown_provider_name_rejected(monkeypatch):
    _no_creds(monkeypatch)
    with _pytest.raises(_aisa.AisaError):
        datasrc._select("madeup")


def test_datasrc_forced_provider_without_creds_fails_loudly(monkeypatch):
    _no_creds(monkeypatch)
    with _pytest.raises(_aisa.AisaError):
        datasrc._select("webcafe")


def test_datasrc_unmapped_tool_raises_per_call(monkeypatch):
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc._select(None)
    with _pytest.raises(_aisa.AisaError, match="no route"):
        prov.call([{"call_id": "x", "tool": "post_twitter_search", "arguments": {}}])


def test_datasrc_provider_instance_is_cached_across_batches(monkeypatch):
    """call_all() batches through datasrc.call(); a fresh instance per batch
    would reset the rate gate — selection must return the same object."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    a = datasrc._select(None)
    b = datasrc._select(None)
    assert a is b and a.name == "webcafe"


def test_datasrc_range_rounds_up_not_nearest():
    # 44-day window (review's 30d-before+14d-after): nearest would truncate to
    # 30d and silently drop the baseline — must pick the smallest enum ≥ days
    assert datasrc._range_of({"date_from": "2026-08-20", "date_to": "2026-10-03"}) == "90d"
    assert datasrc._range_of({"date_from": "2026-10-01", "date_to": "2026-10-03"}) == "7d"
    assert datasrc._range_of({}) == "90d"  # missing window → default


def test_webcafe_volume_chunks_and_shapes(monkeypatch):
    """Real webcafe shape: data.keywords is a dict-of-dicts {kw: {avg,
    months:[{ym,v}]}} with months oldest→newest; DFS wants newest-first."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()
    calls = []

    def fake_post(endpoint, payload):
        calls.append(len(payload["keywords"]))
        return {"credits": {"charged": 2},
                "data": {"keywords": {
                    f"kw{i}": {"avg": 100 + i,
                               "months": [{"ym": "2026-08", "v": 90},
                                          {"ym": "2026-09", "v": 80}]}
                    if i != 3 else None               # null = no Ads data
                    for i in range(len(payload["keywords"]))}}}, prov.CREDIT_MICROS * 2
    monkeypatch.setattr(prov, "_post", fake_post)
    task, micros = prov._handle(datasrc.DFS_VOLUME,
                                {"body": [{"keywords": [f"k{i}" for i in range(120)]}]})
    assert calls == [50, 50, 20]                     # chunked, not truncated
    assert len(task["result"]) == 117                # 3 nulls dropped (i=3 per chunk)
    row = task["result"][0]
    assert row == {"keyword": "kw0", "search_volume": 100,
                   "monthly_searches": [{"year": 2026, "month": 9, "search_volume": 80},
                                        {"year": 2026, "month": 8, "search_volume": 90}]}
    assert micros == 2 * prov.CREDIT_MICROS * 3


def test_webcafe_trends_normalizes_epoch_and_dates(monkeypatch):
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()
    monkeypatch.setattr(prov, "_post", lambda e, p: (
        {"points": [{"timestamp": 1759190400, "value": 42},
                    {"timestamp": 1759190400000, "value": 9},       # epoch millis
                    {"date": "2026-10-01", "values": [7, 8]},
                    {"values": [1]},                                # no date → skipped
                    {"date": "2026/10/01", "value": 5},             # non-ISO → skipped
                    {"date": 5, "value": 5},                        # tiny epoch → skipped
                    {"date": True, "value": 5},                     # bool → skipped
                    {"date": "2026-10-02", "values": [{"bad": 1}]}]}, 0))
    task, _ = prov._handle(datasrc.DFS_TRENDS, {"body": [{"keywords": ["a", "b"]}]})
    data = task["result"][0]["items"][0]["data"]
    assert data[0] == {"date_from": "2025-09-30", "values": [42]}   # epoch → ISO
    assert data[1] == {"date_from": "2025-09-30", "values": [9]}    # millis → ISO
    assert data[2] == {"date_from": "2026-10-01", "values": [7, 8]}
    assert data[3] == {"date_from": "2026-10-02", "values": [0]}    # unusable values → 0
    assert len(data) == 4                                           # poison points dropped, call survives


def test_webcafe_serp_shapes_organic_link_rows(monkeypatch):
    """Real webcafe shape: data.organic[] uses link/position, not url/rank."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()
    monkeypatch.setattr(prov, "_post", lambda e, p: (
        {"data": {"organic": [{"link": "https://www.medium.com/a", "title": "t", "position": 2},
                              {"title": "no link"},
                              {"link": "https://x.io/b"}]}}, 0))
    task, _ = prov._handle(datasrc.DFS_SERP, {"body": [{"keyword": "k"}]})
    items = task["result"][0]["items"]
    assert len(items) == 2
    assert items[0] == {"type": "organic", "domain": "medium.com",
                        "url": "https://www.medium.com/a", "title": "t", "rank_group": 2}
    assert items[1]["domain"] == "x.io" and items[1]["rank_group"] == 3


def test_webcafe_kd_reads_data_score(monkeypatch):
    """Real webcafe shape: data.score is the 0-100 difficulty."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()
    monkeypatch.setattr(prov, "_post", lambda e, p: (
        {"data": {"keyword": p["keyword"], "score": 42, "level": "中"}}, 0))
    task, _ = prov._handle(datasrc.DFS_KD, {"body": [{"keywords": ["a", "b"]}]})
    items = task["result"][0]["items"]
    assert items == [{"keyword": "a", "keyword_properties": {"keyword_difficulty": 42}},
                     {"keyword": "b", "keyword_properties": {"keyword_difficulty": 42}}]


def test_webcafe_shaping_surprise_becomes_aisa_error(monkeypatch):
    """Any non-AisaError escape from provider internals must be re-wrapped —
    upstream degrade paths only catch AisaError, anything else kills the
    whole usecases block for the day."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()

    def boom(endpoint, payload):
        raise RuntimeError("unexpected payload surprise")
    monkeypatch.setattr(prov, "_post", boom)
    with _pytest.raises(_aisa.AisaError, match="webcafe"):
        prov._handle(datasrc.DFS_VOLUME, {"body": [{"keywords": ["k"]}]})


def test_datasrc_call_without_call_id_rejected(monkeypatch):
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc._select(None)
    with _pytest.raises(_aisa.AisaError, match="call_id"):
        prov.call([{"tool": datasrc.DFS_VOLUME, "arguments": {}}])


def test_datasrc_partial_failure_keeps_billed_spend(monkeypatch):
    """A raise mid-batch must not lose the cost of requests that already billed
    — _spent accumulates per request and folds into the next call's total."""
    _no_creds(monkeypatch)
    monkeypatch.setenv("WEBCAFE_TOKEN", "t")
    prov = datasrc.WebCafe()
    prov.min_interval = 0
    n = [0]

    def fake(url, payload, timeout, **kw):
        n[0] += 1
        if n[0] == 2:
            raise _aisa.AisaError("boom")
        return {"credits": {"charged": 2}, "data": []}
    monkeypatch.setattr(_aisa, "_post", fake)
    with _pytest.raises(_aisa.AisaError):
        prov._handle(datasrc.DFS_KD, {"body": [{"keywords": ["a", "b"]}]})   # 2nd request dies
    _out, micros = prov.call([{"call_id": "c", "tool": datasrc.DFS_SERP,
                               "arguments": {"body": [{}]}}])
    assert micros == prov.CREDIT_MICROS * 2 * 2   # kd's billed request + serp's
