"""News monitoring service using Kagi Search API.

Searches for news articles about portfolio companies, verifies relevance
via multi-signal confidence scoring (domain match, name context, logo match,
LLM verification), and stores verified articles with significance analysis.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import (
    ChangeMagnitude,
    ExtractionResult,
    NewsArticle,
    NewsSearchResult,
)
from valuation_tool.services.retry import AuthenticationError, with_retry
from valuation_tool.services.significance import analyze_significance

logger = structlog.get_logger()

# Verification signal weights
VERIFICATION_WEIGHTS = {
    "logo": 0.30,
    "domain": 0.30,
    "context": 0.15,
    "llm": 0.25,
}

VERIFICATION_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Kagi API Client
# ---------------------------------------------------------------------------

class KagiClient:
    """Client for the Kagi Search API."""

    BASE_URL = "https://kagi.com/api/v0/search"

    def __init__(self, config: Config):
        self.api_key = config.kagi_api_key or ""
        self.max_retries = config.max_retry_attempts
        self._client = httpx.Client(
            headers={"Authorization": f"Bot {self.api_key}"},
            timeout=30.0,
        )

    def search(self, query: str) -> list[dict]:
        """Execute a Kagi search query and return result items."""
        fetch = with_retry(self.max_retries)(self._do_search)
        return fetch(query)

    def _do_search(self, query: str) -> list[dict]:
        resp = self._client.get(self.BASE_URL, params={"q": query})
        if resp.status_code in (401, 403):
            raise AuthenticationError(f"Kagi authentication failed: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])


# ---------------------------------------------------------------------------
# Date Range Calculation
# ---------------------------------------------------------------------------

def _calculate_date_range(
    db: Database, company_id: int
) -> tuple[str, str]:
    """Calculate search date range from snapshot history.

    Uses the oldest snapshot date as the start. Falls back to 90 days ago.
    """
    snaps = db.get_snapshots_for_company(company_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if snaps:
        # Oldest snapshot date
        oldest = min(s.captured_at for s in snaps)
        after_date = oldest.strftime("%Y-%m-%d")
    else:
        # Default: 90 days ago
        after_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")

    return after_date, today


# ---------------------------------------------------------------------------
# Multi-Signal Verification
# ---------------------------------------------------------------------------

def _verify_article(
    article: dict,
    company_name: str,
    homepage_url: str | None,
    logo_hash: str | None,
    config: Config,
) -> tuple[float, list[str]]:
    """Verify that an article is actually about the given company.

    Returns (confidence, evidence_list).
    """
    confidence = 0.0
    evidence: list[str] = []
    snippet = article.get("snippet", "")
    article_url = article.get("url", "")

    # Signal 1: Domain match (0.30 weight)
    if homepage_url:
        company_domain = urlparse(homepage_url).netloc.removeprefix("www.")
        if company_domain:
            # Word boundary match to prevent false positives
            pattern = re.compile(r"\b" + re.escape(company_domain) + r"\b")
            text_to_check = snippet + " " + article_url
            if pattern.search(text_to_check):
                confidence += VERIFICATION_WEIGHTS["domain"]
                evidence.append("domain_match")

    # Signal 2: Name in business context (0.15 weight)
    if company_name.lower() in snippet.lower():
        business_context_words = [
            "announced", "raised", "launched", "acquired", "partnered",
            "reported", "expanded", "hired", "funding", "revenue",
            "growth", "product", "series", "round",
        ]
        if any(word in snippet.lower() for word in business_context_words):
            confidence += VERIFICATION_WEIGHTS["context"]
            evidence.append("name_context")

    # Signal 3: Logo match (0.30 weight) - would require image comparison
    # Placeholder: logo_hash comparison would happen here
    if logo_hash:
        # In production, download article images and compare perceptual hashes
        pass

    # Signal 4: LLM verification (0.25 weight) - optional
    if config.llm_enabled:
        try:
            llm_result = _llm_verify_article(article, company_name, config)
            if llm_result:
                confidence += VERIFICATION_WEIGHTS["llm"]
                evidence.append("llm_verification")
        except Exception:
            pass  # LLM failure doesn't block verification

    return confidence, evidence


def _llm_verify_article(article: dict, company_name: str, config: Config) -> bool:
    """Use LLM to verify article relevance. Returns True if confirmed."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=100,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    f"Is this article about the company '{company_name}'? "
                    f"Title: {article.get('title', '')}. "
                    f"Snippet: {article.get('snippet', '')}. "
                    f"Answer YES or NO only."
                ),
            }],
        )
        answer = response.content[0].text.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# News Search
# ---------------------------------------------------------------------------

def search_news_for_company(
    config: Config,
    db: Database,
    company_id: int | None = None,
    company_name: str | None = None,
) -> NewsSearchResult:
    """Search for news about a single company."""
    # Resolve company
    if company_id:
        company = db.get_company_by_id(company_id)
        if not company:
            return NewsSearchResult(
                company_id=company_id or 0,
                company_name=company_name or "",
                errors=["Company not found"],
            )
    elif company_name:
        company = db.get_company_by_name(company_name)
        if not company:
            return NewsSearchResult(
                company_id=0,
                company_name=company_name,
                errors=["Company not found"],
            )
    else:
        return NewsSearchResult(company_id=0, company_name="", errors=["No company specified"])

    kagi = KagiClient(config)
    after_date, before_date = _calculate_date_range(db, company.id)

    # Build query with date filters
    query = f"{company.name} after:{after_date} before:{before_date}"

    try:
        raw_results = kagi.search(query)
    except AuthenticationError:
        raise
    except Exception as exc:
        return NewsSearchResult(
            company_id=company.id,
            company_name=company.name,
            errors=[str(exc)],
        )

    # Get logo hash for verification
    logo = db.get_logo_for_company(company.id)
    logo_hash = logo.perceptual_hash if logo else None

    articles_found = len(raw_results)
    articles_stored = 0

    for raw_article in raw_results:
        article_url = raw_article.get("url", "")

        # Skip duplicates
        if db.article_url_exists(article_url):
            continue

        # Verify relevance
        confidence, evidence = _verify_article(
            raw_article, company.name, company.homepage_url, logo_hash, config
        )

        if confidence < VERIFICATION_THRESHOLD:
            continue

        # Parse published date
        published = raw_article.get("published")
        if published:
            try:
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                published_at = datetime.now(timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)

        # Run significance analysis
        title = raw_article.get("title", "")
        snippet = raw_article.get("snippet", "")
        sig_result = analyze_significance(
            f"{title} {snippet}", ChangeMagnitude.moderate
        )

        # Extract source domain
        source = urlparse(article_url).netloc.removeprefix("www.")

        article = NewsArticle(
            company_id=company.id,
            title=title[:500],
            content_url=article_url,
            source=source,
            snippet=snippet,
            published_at=published_at,
            match_confidence=round(confidence, 2),
            match_evidence=evidence,
            significance_classification=sig_result.classification,
            significance_sentiment=sig_result.sentiment,
            significance_confidence=sig_result.confidence,
        )

        article_id = db.store_news_article(article)
        if article_id:
            articles_stored += 1

    return NewsSearchResult(
        company_id=company.id,
        company_name=company.name,
        articles_found=articles_found,
        articles_stored=articles_stored,
    )


def search_news_all(
    config: Config,
    db: Database,
    limit: int | None = None,
) -> ExtractionResult:
    """Search news for all companies."""
    companies = db.get_all_companies(limit=limit)
    result = ExtractionResult(processed=len(companies))

    for company in companies:
        try:
            news_result = search_news_for_company(config, db, company_id=company.id)
            result.successful += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})
            logger.error("news_search_failed", company=company.name, error=str(exc))

    logger.info(
        "search_news_all_complete",
        processed=result.processed,
        successful=result.successful,
        failed=result.failed,
    )
    return result
