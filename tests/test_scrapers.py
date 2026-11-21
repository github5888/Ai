"""Phase I/II tests: date normalisation, freshness, URL/hash identity, parsers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.scrapers.jobs_scraper import (
    classify_role_family,
    infer_remote,
    is_ai_relevant,
    parse_arbeitnow,
    parse_remoteok,
    parse_rss_jobs,
)
from src.scrapers.news_scraper import extract_article_text, parse_feed
from src.scrapers.product_scraper import (
    infer_startup_name,
    parse_ai_collection_sections,
    parse_markdown_tools,
)
from src.scrapers.research_scraper import parse_daily_papers, split_authors
from src.scrapers.startup_scraper import (
    parse_ai_services_csv,
    parse_builtin_directory,
    parse_github_markdown_companies,
    parse_yc_csv,
    parse_wiki_members,
    parse_wikidata_employees,
)
from src.utils.dates import is_fresh, parse_timestamp, to_iso
from src.utils.hashing import content_hash, url_hash
from src.utils.text import normalize_url, strip_html

NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Date parsing -> ISO-8601 (Data Quality: 25%)
# ---------------------------------------------------------------------------
def test_relative_dates():
    iso, source = parse_timestamp("2 hours ago", ref=NOW)
    assert source == "stated"
    assert iso == "2026-11-21T10:00:00Z"
    iso30, _ = parse_timestamp("30 minutes ago", ref=NOW)
    assert iso30 == "2026-11-21T11:30:00Z"
    iso_y, _ = parse_timestamp("Yesterday", ref=NOW)
    assert iso_y == "2026-11-20T12:00:00Z"
    iso_t, _ = parse_timestamp("Today", ref=NOW)
    assert iso_t == "2026-11-21T12:00:00Z"


def test_absolute_dates():
    iso, _ = parse_timestamp("November 21, 2026")
    assert iso.startswith("2026-11-21")
    iso2, _ = parse_timestamp("Fri, 20 Nov 2026 19:53:41 +0000")
    assert iso2 == "2026-11-20T19:53:41Z"
    iso3, _ = parse_timestamp("2026-11-20T08:15:00Z")
    assert iso3 == "2026-11-20T08:15:00Z"


def test_epoch_seconds_and_milliseconds():
    assert parse_timestamp(1737412800)[0] == "2025-01-20T22:40:00Z"
    assert parse_timestamp(1737412800000)[0] == "2025-01-20T22:40:00Z"


def test_unparseable_returns_none_not_guessed():
    iso, source = parse_timestamp("whenever we finish")
    assert iso is None  # callers apply first-seen heuristic; never fabricated


def test_freshness_window():
    fresh_iso = to_iso(NOW - timedelta(hours=23))
    old_iso = to_iso(NOW - timedelta(hours=25))
    future_iso = to_iso(NOW + timedelta(hours=2))
    assert is_fresh(fresh_iso, NOW, 24) is True
    assert is_fresh(old_iso, NOW, 24) is False
    assert is_fresh(future_iso, NOW, 24) is False  # future-dated junk rejected
    assert is_fresh(None, NOW, 24) is False


# ---------------------------------------------------------------------------
# URL / content identity (dedup across distributed workers)
# ---------------------------------------------------------------------------
def test_normalize_url_strips_tracking():
    a = normalize_url("https://Example.com/Article/?utm_source=x&fbclid=1&id=7")
    b = normalize_url("https://example.com/Article?id=7")
    assert a == b


def test_url_hash_stable_and_content_hash_stable():
    assert url_hash("https://example.com/a?utm_source=x") == url_hash("https://example.com/a")
    h1 = content_hash({"company": "X", "title": "Role"}, "https://example.com/1")
    h2 = content_hash({"title": "Role", "company": "X"}, "https://example.com/1")
    assert h1 == h2  # dict key order must not change identity


# ---------------------------------------------------------------------------
# Startup parsers
# ---------------------------------------------------------------------------
def test_parse_wiki_members_skips_subcategories():
    payload = {"query": {"categorymembers": [
        {"ns": 0, "title": "OpenAI"},
        {"ns": 14, "title": "Category:AI companies in India"},
    ]}}
    records = parse_wiki_members(payload)
    assert len(records) == 1
    assert records[0]["entityName"] == "OpenAI"
    assert records[0]["sourceUrl"].startswith("https://en.wikipedia.org/wiki/OpenAI")


def test_parse_github_markdown_companies_filters_sections():
    markdown = """# Awesome list
## Companies
- [OpenAI](https://openai.com) - builds GPT models
- [Open AI](https://openai.com)
- Random Text Without Link
## Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - transformer paper
- [Some Blog](https://example.com) - a blog post
"""
    records = parse_github_markdown_companies(markdown, "TestList")
    names = [r["entityName"] for r in records]
    assert "OpenAI" in names
    assert "Open AI" in names
    assert "Attention Is All You Need" not in names  # papers section skipped
    assert "Some Blog" not in names                  # stopwords skipped


def test_parse_builtin_directory():
    html = '<html><body><a href="/company/acme-ai">Acme AI</a><a href="/company/x">X Corp</a>' \
           '<a href="/jobs">Jobs</a></body></html>'
    records = parse_builtin_directory(html, "Builtin")
    assert [r["entityName"] for r in records] == ["Acme AI", "X Corp"]


def test_parse_wikidata_employees():
    payload = {"results": {"bindings": [
        {"label": {"value": "OpenAI"}, "item": {"value": "http://www.wikidata.org/entity/Q19867011"},
         "employees": {"value": "1500"}},
        {"label": {"value": "Broken"}, "item": {"value": "http://www.wikidata.org/entity/Q1"},
         "employees": {"value": "not-a-number"}},
    ]}}
    parsed = parse_wikidata_employees(payload)
    assert parsed["OpenAI"]["employeeCount"] == 1500
    assert "Broken" not in parsed


# ---------------------------------------------------------------------------
# Product parsers + pricing classification
# ---------------------------------------------------------------------------
def test_parse_markdown_tools_skips_badges_and_repos():
    markdown = """## Tools
- [ChatGPT](https://openai.com/chatgpt) - conversational AI assistant, free to use
- [![badge](https://img.shields.io/x)](https://shields.io) - badge junk
- [Some Repo](https://github.com/user/repo) - a github repo
- [Midjourney](https://www.midjourney.com/) - AI image generator, paid subscription
"""
    records = parse_markdown_tools(markdown, "TestList")
    names = [r["productName"] for r in records]
    assert "ChatGPT" in names and "Midjourney" in names
    assert all("badge" not in n.lower() for n in names)
    assert all(not r["sourceUrl"].startswith("https://github.com/") for r in records)


def test_infer_startup_name_via_domain_and_description():
    assert infer_startup_name("ChatGPT", "https://openai.com/chatgpt", None) == "OpenAI"
    assert infer_startup_name("Whatever", "https://unknown.io", "built by ElevenLabs",
                              known_startups=["ElevenLabs"]) == "ElevenLabs"
    assert infer_startup_name("Mystery Tool", "https://mystery.io", None) is None


# ---------------------------------------------------------------------------
# Job parsers + role families + AI relevance
# ---------------------------------------------------------------------------
def test_parse_remoteok_skips_legal_notice():
    payload = [
        {"legal": "notice"},  # first element has no position
        {"date": 1763690000, "position": "Senior ML Engineer", "company": "Acme AI",
         "location": "Remote", "tags": ["machine learning"], "url": "https://remoteok.com/x",
         "description": "<p>build AI systems</p>"},
    ]
    records = parse_remoteok(payload)
    assert len(records) == 1
    assert records[0]["company"] == "Acme AI"
    assert records[0]["isRemote"] is True
    assert records[0]["roleFamily"] == "Data & AI"
    assert records[0]["aiRelevant"] is True


def test_parse_arbeitnow_remote_flag():
    payload = {"data": [{
        "slug": "s", "company_name": "Beta", "title": "Frontend Engineer (AI tools)",
        "remote": True, "url": "https://arbeitnow.com/j", "tags": ["react"],
        "created_at": 1763690000, "description": "AI devtools",
    }]}
    records = parse_arbeitnow(payload)
    assert records[0]["isRemote"] is True
    assert records[0]["roleFamily"] == "Engineering"


def test_parse_rss_jobs_company_role_split():
    xml = """<?xml version="1.0"?>
    <rss><channel>
    <item><title>Acme AI: LLM Platform Engineer</title>
    <link>https://weworkremotely.com/j/1</link>
    <pubDate>Fri, 20 Nov 2026 18:00:00 +0000</pubDate>
    <description>Remote role training large language models</description></item>
    </channel></rss>"""
    records = parse_rss_jobs(xml, "We Work Remotely")
    assert records[0]["company"] == "Acme AI"
    assert records[0]["jobTitle"] == "LLM Platform Engineer"
    assert records[0]["publishedAt"] == "2026-11-20T18:00:00Z"


def test_role_family_classification():
    assert classify_role_family("Research Scientist, LLM") == "Research"
    assert classify_role_family("Senior Software Engineer") == "Engineering"
    assert classify_role_family("Product Manager, AI") == "Product"
    assert classify_role_family("Mystery Role") == "Other"


def test_infer_remote_and_ai_relevance():
    assert infer_remote("Backend Dev", "work from home allowed") is True
    assert infer_remote("Backend Dev", "on-site in Pune", board_default=False) is False
    assert is_ai_relevant("Machine Learning Engineer") is True
    assert is_ai_relevant("Accountant") is False


# ---------------------------------------------------------------------------
# News parsing + full-text extraction
# ---------------------------------------------------------------------------
SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Feed</title>
<item>
  <title>Anthropic IPO shatters records</title>
  <link>https://techcrunch.com/2026/11/20/story/</link>
  <pubDate>Fri, 20 Nov 2026 19:53:41 +0000</pubDate>
  <description><![CDATA[<p>Short summary text about the IPO.</p>]]></description>
</item>
</channel></rss>"""


def test_parse_feed_rss():
    records = parse_feed(SAMPLE_RSS, "TechCrunch AI")
    assert len(records) == 1
    assert records[0]["headline"] == "Anthropic IPO shatters records"
    assert records[0]["publishedAt"] == "2026-11-20T19:53:41Z"
    assert records[0]["sourceName"] == "TechCrunch AI"
    assert records[0]["content"].startswith("Short summary")


def test_extract_article_text_prefers_article_tag():
    html = """
    <html><body><nav>nav junk</nav>
    <article><p>First meaningful paragraph of the article body.</p>
    <p>Second meaningful paragraph with more details about the story.</p></article>
    <script>var x = 1;</script></body></html>"""
    text = extract_article_text(html)
    assert "First meaningful paragraph" in text
    assert "nav junk" not in text
    assert "var x" not in text


def test_strip_html_entities():
    assert strip_html("<p>A &amp; B</p>") == "A & B"


def test_parse_yc_csv_filters_ai_and_keeps_team_size():
    csv_text = """id,name,slug,one_liner,long_description,team_size,tags,status
1,CoolAI,coolai,"An AI copilot for doctors","", 42,"[""Artificial Intelligence"", ""Health""]",Active
2,WeedBot,weedbot,"Kills weeds with robots","", 10,"[""Robotics"", ""Agriculture""]",Active
3,LLMBiz,llmbiz,"Machine learning for sales teams","",,"[]",Active
"""
    records = parse_yc_csv(csv_text)
    names = {r["entityName"] for r in records}
    assert names == {"CoolAI", "LLMBiz"}
    cool = next(r for r in records if r["entityName"] == "CoolAI")
    assert cool["employeeCount"] == 42
    assert cool["sourceUrl"] == "https://www.ycombinator.com/companies/coolai"
    assert "YC team size: 42" in (cool["description"] or "")


def test_parse_ai_services_csv_keeps_ranges_verbatim():
    csv_text = """Company_Name,Website,Location,Number of Employees
Smith.ai,https://smith.ai/,Los Altos CA,250 - 999
DataWorks,https://dataworks.io/,Pune,50 - 249
"""
    records = parse_ai_services_csv(csv_text)
    assert len(records) == 2
    assert records[0]["entityName"] == "Smith.ai"
    assert records[0]["employeeCount"] is None          # ranges are never faked into ints
    assert "Employees: 250 - 999" in records[0]["description"]
    assert records[0]["sourceUrl"] == "https://smith.ai/"


def test_split_authors_handles_dump_formats():
    assert split_authors('["A. Author", "B. Author"]') == ["A. Author", "B. Author"]
    assert split_authors("Name One; Name Two") == ["Name One", "Name Two"]
    assert split_authors(None) == []


def test_parse_daily_papers_extracts_github_fields():
    payload = [{
        "paper": {
            "id": "2611.24738",
            "title": " Jeveration: System One Models ",
            "authors": [{"name": "Yuhe Liu"}, {"name": "Xinchao Wang"}, {"name": ""}],
            "publishedAt": "2026-11-18T00:00:00.000Z",
            "githubRepo": "https://github.com/Carol-lyh/Jeveration",
            "githubStars": 72,
        },
    }, {"paper": {}}, {}]
    records = parse_daily_papers(payload)
    assert len(records) == 1
    assert records[0]["arxivId"] == "2611.24738"
    assert records[0]["authors"] == ["Yuhe Liu", "Xinchao Wang"]
    assert records[0]["githubUrl"] == "https://github.com/Carol-lyh/Jeveration"
    assert records[0]["githubStars"] == 72
    assert records[0]["paper_url"] == "https://arxiv.org/abs/2611.24738"


def test_json_roundtrip_for_raw_records():
    payload = {"a": [1, 2], "b": "text"}
    assert json.loads(json.dumps(payload)) == payload


def test_parse_ai_collection_sections():
    markdown = """## Latest Additions to AI Collection
### OneKDP
<img align="left" width="240" src="x.webp" alt="OneKDP">

#### AI book & KDP cover generator with free KDP tools

[Visit](https://thataicollection.com/redirect/onekdp?utm_source=aicollection)

### Frank
#### AI-Powered Customer Research Platform | Frank

[Visit](https://thataicollection.com/redirect/frank)

### Junk "quoted" | Title
#### tagline
[Visit](https://example.com/junk)
"""
    records = parse_ai_collection_sections(markdown, "GitHub:ai-collection")
    names = [r["productName"] for r in records]
    assert "OneKDP" in names and "Frank" in names
    assert not any("Junk" in n for n in names)
    onekdp = next(r for r in records if r["productName"] == "OneKDP")
    assert "free KDP tools" in (onekdp["pricingHint"] or "")


def test_pricing_rules_free_and_combo():
    from src.llm.orchestrator import classify_pricing
    assert classify_pricing("AI book & KDP cover generator with free KDP tools") == "FREE"
    assert classify_pricing("free and paid plans available") == "FREEMIUM"
    assert classify_pricing("from $19 per month") == "PAID"
    assert classify_pricing("enterprise-ready LLM platform") == "ENTERPRISE"
    assert classify_pricing("a chatbot for doctors") is None
