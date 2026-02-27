"""Social media discovery service.

Discovers social media profiles for portfolio companies across 12+ platforms.
Extracts links from HTML (anchors, aria-labels, Schema.org JSON-LD, meta tags),
markdown, and regex fallback. Normalizes URLs, detects platforms, classifies
accounts, and detects blogs.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from bs4 import BeautifulSoup

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import (
    AccountType,
    BlogLink,
    CompanyLogo,
    DiscoveryMethod,
    ExtractionResult,
    Platform,
    ProcessingError,
    SocialMediaLink,
    VerificationStatus,
)
from valuation_tool.services.firecrawl import FirecrawlClient

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Platform Detection
# ---------------------------------------------------------------------------

_PLATFORM_PATTERNS: list[tuple[re.Pattern, Platform]] = [
    (re.compile(r"(?:www\.)?linkedin\.com/"), Platform.linkedin),
    (re.compile(r"(?:www\.)?(?:twitter\.com|x\.com)/"), Platform.twitter),
    (re.compile(r"(?:www\.)?youtube\.com/"), Platform.youtube),
    (re.compile(r"bsky\.app/"), Platform.bluesky),
    (re.compile(r"(?:www\.|m\.)?(?:facebook\.com|fb\.com)/"), Platform.facebook),
    (re.compile(r"(?:www\.)?instagram\.com/"), Platform.instagram),
    (re.compile(r"(?:www\.)?github\.com/"), Platform.github),
    (re.compile(r"(?:www\.)?tiktok\.com/"), Platform.tiktok),
    (re.compile(r"(?:www\.)?medium\.com/|\.medium\.com"), Platform.medium),
    (re.compile(r"mastodon\."), Platform.mastodon),
    (re.compile(r"(?:www\.)?threads\.net/"), Platform.threads),
    (re.compile(r"(?:www\.)?pinterest\.com/"), Platform.pinterest),
]


def detect_platform(url: str) -> Platform | None:
    """Detect social media platform from URL."""
    url_lower = url.lower()
    for pattern, platform in _PLATFORM_PATTERNS:
        if pattern.search(url_lower):
            return platform
    return None


# ---------------------------------------------------------------------------
# URL Normalization
# ---------------------------------------------------------------------------

def normalize_social_url(url: str, platform: Platform | None = None) -> str:
    """Normalize a social media URL to its canonical form."""
    parsed = urlparse(url)

    # Remove www prefix
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.rstrip("/")

    # Platform-specific normalization
    if platform == Platform.github:
        # Normalize to org level: github.com/acme/repo → github.com/acme
        parts = path.strip("/").split("/")
        if len(parts) >= 1:
            path = "/" + parts[0]

    elif platform == Platform.linkedin:
        # Normalize: linkedin.com/company/acme/about/ → linkedin.com/company/acme
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("company", "in"):
            path = "/" + "/".join(parts[:2])

    elif platform == Platform.twitter:
        # Normalize: twitter.com/acme?ref=xyz → twitter.com/acme
        # Also x.com → twitter.com
        host = "twitter.com" if host in ("twitter.com", "x.com") else host
        parts = path.strip("/").split("/")
        if parts:
            path = "/" + parts[0]

    # Rebuild without query/fragment
    return f"https://{host}{path}"


# ---------------------------------------------------------------------------
# Blog Detection
# ---------------------------------------------------------------------------

_BLOG_PATTERNS = [
    (re.compile(r"\.substack\.com"), "substack"),
    (re.compile(r"\.medium\.com"), "medium"),
    (re.compile(r"/blog(?:/|$)"), "company_blog"),
    (re.compile(r"^blog\."), "company_blog"),
]


def detect_blog(url: str) -> str | None:
    """Detect if URL is a blog. Returns blog_type or None."""
    url_lower = url.lower()
    parsed = urlparse(url_lower)
    full = parsed.netloc + parsed.path

    for pattern, blog_type in _BLOG_PATTERNS:
        if pattern.search(full):
            return blog_type
    return None


def normalize_blog_url(url: str, blog_type: str) -> str:
    """Normalize blog URL to hub level (strip article paths)."""
    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")

    if blog_type == "substack":
        return f"https://{host}"
    if blog_type == "medium":
        return f"https://{host}"
    if blog_type == "company_blog":
        path = parsed.path
        # Keep just /blog
        if "/blog" in path:
            idx = path.index("/blog")
            path = path[:idx + 5]
        else:
            path = ""
        return f"https://{host}{path}"
    return f"https://{host}{parsed.path.rstrip('/')}"


# ---------------------------------------------------------------------------
# Link Extraction
# ---------------------------------------------------------------------------

def _extract_links_from_html(html: str) -> list[dict[str, str]]:
    """Extract social media links from HTML using multiple strategies."""
    links: list[dict[str, str]] = []
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: Anchor href tags
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if detect_platform(href):
            region = _detect_html_region(a)
            links.append({"url": href, "region": region})

    # Strategy 2: Schema.org JSON-LD (sameAs)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict):
                same_as = data.get("sameAs", [])
                if isinstance(same_as, list):
                    for url in same_as:
                        if detect_platform(url):
                            links.append({"url": url, "region": "schema_org"})
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Meta tags (twitter:site)
    for meta in soup.find_all("meta"):
        name = meta.get("name", "") or meta.get("property", "")
        content = meta.get("content", "")
        if name == "twitter:site" and content.startswith("@"):
            links.append({
                "url": f"https://twitter.com/{content.lstrip('@')}",
                "region": "meta_tag",
            })

    # Strategy 4: aria-label attributes
    for el in soup.find_all(attrs={"aria-label": True}):
        href = el.get("href")
        if href and detect_platform(href):
            region = _detect_html_region(el)
            links.append({"url": href, "region": region})

    return links


def _extract_links_from_markdown(markdown: str) -> list[dict[str, str]]:
    """Extract social media links from markdown."""
    links: list[dict[str, str]] = []
    # Match markdown links: [text](url)
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", markdown):
        url = match.group(2)
        if detect_platform(url):
            links.append({"url": url, "region": "markdown"})
    return links


def _extract_links_via_regex(content: str) -> list[dict[str, str]]:
    """Regex fallback: extract social URLs from any text."""
    links: list[dict[str, str]] = []
    url_pattern = re.compile(r'https?://[^\s"\'<>,]+')
    for match in url_pattern.finditer(content):
        url = match.group(0).rstrip(")")
        if detect_platform(url):
            links.append({"url": url, "region": "regex"})
    return links


def _detect_html_region(element: Any) -> str:
    """Detect which HTML region an element is in."""
    regions = {"footer", "header", "nav", "aside", "main"}
    current = element
    while current:
        tag_name = getattr(current, "name", None)
        if tag_name in regions:
            return tag_name
        current = getattr(current, "parent", None)
    return "unknown"


# ---------------------------------------------------------------------------
# Account Classification
# ---------------------------------------------------------------------------

def classify_account(
    url: str, platform: Platform, company_name: str, region: str
) -> tuple[AccountType, float]:
    """Classify whether a social link is a company or personal account."""
    if platform == Platform.linkedin:
        if "/company/" in url:
            return AccountType.company, 0.9
        if "/in/" in url:
            return AccountType.personal, 0.9

    # Company handle heuristic: check if company name appears in handle
    parsed = urlparse(url)
    path_lower = parsed.path.lower().replace("-", "").replace("_", "")
    name_lower = company_name.lower().replace(" ", "").replace("-", "")

    confidence = 0.5
    if name_lower and name_lower in path_lower:
        confidence = 0.8

    # Footer/header links are more likely company accounts
    if region in ("footer", "header", "nav"):
        return AccountType.company, min(confidence + 0.1, 1.0)

    return AccountType.unknown, confidence


# ---------------------------------------------------------------------------
# YouTube Resolution
# ---------------------------------------------------------------------------

def resolve_youtube_embed(url: str) -> str | None:
    """Resolve a YouTube embed URL to the channel URL via oEmbed."""
    if "/embed/" not in url:
        return None
    try:
        resp = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("author_url")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Logo Extraction
# ---------------------------------------------------------------------------

def extract_logo_from_html(html: str, company_id: int) -> CompanyLogo | None:
    """Extract company logo from HTML header."""
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("header") or soup

    # Look for logo images
    for img in header.find_all("img"):
        alt = (img.get("alt") or "").lower()
        src = img.get("src", "")
        classes = " ".join(img.get("class", []))

        if any(word in alt for word in ["logo"]) or "logo" in classes.lower():
            return CompanyLogo(
                company_id=company_id,
                image_url=src,
                image_format=_guess_image_format(src),
                extraction_location="header",
            )
    return None


def _guess_image_format(url: str) -> str:
    lower = url.lower()
    for fmt in ("svg", "png", "jpg", "jpeg", "webp", "gif"):
        if f".{fmt}" in lower:
            return fmt
    return "unknown"


# ---------------------------------------------------------------------------
# Main Discovery
# ---------------------------------------------------------------------------

def discover_social_media(
    config: Config,
    db: Database,
    company_id: int | None = None,
    limit: int | None = None,
    batch_size: int = 50,
) -> ExtractionResult:
    """Discover social media links for companies.

    For a single company: uses individual scrape.
    For all companies: uses batch API.
    """
    if company_id:
        company = db.get_company_by_id(company_id)
        if not company or not company.homepage_url:
            return ExtractionResult(skipped=1)
        companies = [company]
    else:
        companies = db.get_all_companies(limit=limit)
        companies = [c for c in companies if c.homepage_url]

    if not companies:
        return ExtractionResult()

    client = FirecrawlClient(config)
    result = ExtractionResult(processed=len(companies))

    if len(companies) == 1:
        # Single company: use individual scrape
        company = companies[0]
        try:
            data = client.scrape_url(company.homepage_url)
            _process_discovery_result(db, company, data, result)
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})
    else:
        # Batch mode
        url_to_company = {c.homepage_url: c for c in companies}
        urls = list(url_to_company.keys())
        try:
            batch_results = client.batch_scrape(urls, batch_size=batch_size)
            for item in batch_results:
                source_url = item.get("metadata", {}).get("sourceURL", item.get("url", ""))
                company = url_to_company.get(source_url)
                if company:
                    try:
                        _process_discovery_result(db, company, item, result)
                    except Exception as exc:
                        result.failed += 1
                        result.errors.append({"company": company.name, "error": str(exc)})
        except Exception as exc:
            result.failed = len(companies)
            result.errors.append({"error": str(exc)})

    return result


def _process_discovery_result(
    db: Database, company: Any, data: dict, result: ExtractionResult
) -> None:
    """Process Firecrawl data for a single company's social discovery."""
    html = data.get("html", "")
    markdown = data.get("markdown", "")

    # Extract links from all sources
    all_links: list[dict[str, str]] = []
    if html:
        all_links.extend(_extract_links_from_html(html))
    if markdown:
        all_links.extend(_extract_links_from_markdown(markdown))
    # Regex fallback for content not caught above
    all_links.extend(_extract_links_via_regex(html or markdown or ""))

    # Deduplicate by normalized URL
    seen_urls: set[str] = set()
    unique_links: list[dict[str, str]] = []
    for link in all_links:
        platform = detect_platform(link["url"])
        if platform:
            normalized = normalize_social_url(link["url"], platform)
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                unique_links.append({**link, "url": normalized, "platform": platform.value})

    # Store social media links
    stored = 0
    for link in unique_links:
        platform = Platform(link["platform"])
        region = link.get("region", "unknown")

        # Check if it's a blog
        blog_type = detect_blog(link["url"])
        if blog_type:
            blog_url = normalize_blog_url(link["url"], blog_type)
            db.store_blog_link(BlogLink(
                company_id=company.id,
                blog_url=urlparse(blog_url).netloc + urlparse(blog_url).path,
                blog_type=blog_type,
                discovery_method=_region_to_method(region),
            ))
            continue

        account_type, account_confidence = classify_account(
            link["url"], platform, company.name, region
        )

        sm_link = SocialMediaLink(
            company_id=company.id,
            platform=platform,
            profile_url=link["url"],
            discovery_method=DiscoveryMethod(_region_to_method(region)),
            html_location=region,
            account_type=account_type,
            account_confidence=account_confidence,
        )
        link_id = db.store_social_media_link(sm_link)
        if link_id:
            stored += 1

    # Try to extract logo
    if html:
        try:
            logo = extract_logo_from_html(html, company.id)
            if logo:
                db.store_company_logo(logo)
        except Exception:
            pass  # Logo extraction failure doesn't block discovery

    result.successful += 1


def _region_to_method(region: str) -> str:
    mapping = {
        "footer": "page_footer",
        "header": "page_header",
        "nav": "page_header",
        "main": "page_content",
        "aside": "page_content",
    }
    return mapping.get(region, "page_content")


# ---------------------------------------------------------------------------
# Full-Site Discovery
# ---------------------------------------------------------------------------

def discover_social_full_site(
    config: Config,
    db: Database,
    company_id: int,
    max_depth: int = 2,
    max_pages: int = 25,
    include_subdomains: bool = True,
) -> ExtractionResult:
    """Discover social media links across an entire website."""
    company = db.get_company_by_id(company_id)
    if not company or not company.homepage_url:
        return ExtractionResult(skipped=1)

    client = FirecrawlClient(config)
    result = ExtractionResult(processed=1)

    try:
        pages = client.crawl_site(
            company.homepage_url,
            max_depth=max_depth,
            max_pages=max_pages,
            include_subdomains=include_subdomains,
        )

        all_links: list[dict[str, str]] = []
        for page in pages:
            html = page.get("html", "")
            markdown = page.get("markdown", "")
            if html:
                all_links.extend(_extract_links_from_html(html))
            if markdown:
                all_links.extend(_extract_links_from_markdown(markdown))
            all_links.extend(_extract_links_via_regex(html or markdown or ""))

        # Deduplicate
        seen_urls: set[str] = set()
        for link in all_links:
            platform = detect_platform(link["url"])
            if platform:
                normalized = normalize_social_url(link["url"], platform)
                if normalized not in seen_urls:
                    seen_urls.add(normalized)
                    account_type, confidence = classify_account(
                        normalized, platform, company.name, link.get("region", "unknown")
                    )
                    db.store_social_media_link(SocialMediaLink(
                        company_id=company.id,
                        platform=platform,
                        profile_url=normalized,
                        discovery_method=DiscoveryMethod.full_site_crawl,
                        html_location=link.get("region", "unknown"),
                        account_type=account_type,
                        account_confidence=confidence,
                    ))

        result.successful = 1
    except Exception as exc:
        result.failed = 1
        result.errors.append({"error": str(exc)})

    return result


# ---------------------------------------------------------------------------
# Batch Discovery (parallel)
# ---------------------------------------------------------------------------

def discover_social_batch(
    config: Config,
    db: Database,
    limit: int | None = None,
    max_workers: int = 5,
) -> ExtractionResult:
    """Run social discovery in parallel with thread pool."""
    companies = db.get_all_companies(limit=limit)
    companies = [c for c in companies if c.homepage_url]

    result = ExtractionResult(processed=len(companies))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                discover_social_media, config, db, company_id=c.id
            ): c
            for c in companies
        }
        for future in as_completed(futures):
            company = futures[future]
            try:
                sub_result = future.result()
                result.successful += sub_result.successful
                result.failed += sub_result.failed
                result.errors.extend(sub_result.errors)
            except Exception as exc:
                result.failed += 1
                result.errors.append({"company": company.name, "error": str(exc)})

    return result
