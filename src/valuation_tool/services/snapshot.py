"""Snapshot capture service.

Captures website content for portfolio companies using Firecrawl,
computing MD5 checksums, detecting paywalls and auth walls, and
storing results in the database.
"""

from __future__ import annotations

from typing import Any

import structlog

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import (
    ExtractionResult,
    ProcessingError,
    Snapshot,
)
from valuation_tool.services.firecrawl import FirecrawlClient

logger = structlog.get_logger()


def capture_snapshots(
    config: Config,
    db: Database,
    use_batch_api: bool = False,
    batch_size: int = 50,
    timeout: int = 300,
) -> ExtractionResult:
    """Capture website snapshots for all companies with homepage URLs.

    Supports both sequential (one-at-a-time) and batch modes.
    """
    companies = db.get_all_companies()
    companies_with_urls = [c for c in companies if c.homepage_url]

    result = ExtractionResult(processed=len(companies_with_urls))

    if not companies_with_urls:
        return result

    client = FirecrawlClient(config)

    if use_batch_api:
        _capture_batch(client, db, companies_with_urls, result, batch_size, timeout)
    else:
        _capture_sequential(client, db, companies_with_urls, result)

    logger.info(
        "capture_snapshots_complete",
        processed=result.processed,
        successful=result.successful,
        failed=result.failed,
    )
    return result


def _capture_sequential(
    client: FirecrawlClient,
    db: Database,
    companies: list,
    result: ExtractionResult,
) -> None:
    for company in companies:
        try:
            data = client.scrape_url(company.homepage_url)
            snap = _build_snapshot(company.id, company.homepage_url, data)
            db.store_snapshot(snap)
            result.successful += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})
            logger.error(
                "snapshot_capture_failed",
                company_name=company.name,
                company_id=company.id,
                error=str(exc),
            )
            db.store_processing_error(ProcessingError(
                entity_type="company",
                entity_id=company.id,
                error_type=type(exc).__name__,
                error_message=str(exc)[:5000],
            ))


def _capture_batch(
    client: FirecrawlClient,
    db: Database,
    companies: list,
    result: ExtractionResult,
    batch_size: int,
    timeout: int,
) -> None:
    url_to_company = {c.homepage_url: c for c in companies}
    urls = list(url_to_company.keys())

    try:
        batch_results = client.batch_scrape(urls, batch_size=batch_size, timeout=timeout)
    except Exception as exc:
        # Entire batch failed
        result.failed = len(companies)
        result.errors.append({"error": f"Batch failed: {exc}"})
        logger.error("batch_scrape_failed", error=str(exc))
        return

    for item in batch_results:
        source_url = item.get("metadata", {}).get("sourceURL", item.get("url", ""))
        company = url_to_company.get(source_url)
        if not company:
            continue

        try:
            snap = _build_snapshot(company.id, source_url, item)
            db.store_snapshot(snap)
            result.successful += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})
            logger.error("snapshot_store_failed", company=company.name, error=str(exc))

    # Account for URLs not in results (failures)
    returned_urls = {
        item.get("metadata", {}).get("sourceURL", item.get("url", ""))
        for item in batch_results
    }
    for url, company in url_to_company.items():
        if url not in returned_urls:
            result.failed += 1
            result.errors.append({"company": company.name, "error": "Not in batch results"})


def _build_snapshot(company_id: int, url: str, data: dict[str, Any]) -> Snapshot:
    """Build a Snapshot model from Firecrawl response data."""
    metadata = data.get("metadata", {})
    status_code = metadata.get("statusCode", 200)
    markdown = data.get("markdown")
    html = data.get("html")
    error_msg = data.get("error")

    # Detect paywall / auth wall
    has_paywall = bool(metadata.get("paywall"))
    has_auth = bool(metadata.get("authRequired") or metadata.get("auth_required"))

    # HTTP Last-Modified
    http_last_modified = metadata.get("lastModified") or metadata.get("last-modified")

    # Handle error status codes
    if status_code and status_code >= 400:
        error_msg = error_msg or f"HTTP {status_code}"

    # Compute checksum from markdown
    checksum = Snapshot.compute_checksum(markdown) if markdown else None

    return Snapshot(
        company_id=company_id,
        url=url,
        status_code=status_code,
        content_markdown=markdown,
        content_html=html,
        content_checksum=checksum,
        error_message=error_msg,
        has_paywall=has_paywall,
        has_auth_required=has_auth,
        http_last_modified=http_last_modified,
    )
