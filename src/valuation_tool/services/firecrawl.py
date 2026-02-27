"""Firecrawl API client for web scraping and crawling.

Supports individual scrapes, batch scrapes, and full-site crawls.
Always uses only_main_content=False to capture the complete page.
"""

from __future__ import annotations

import math
import time
from typing import Any

import httpx
import structlog

from valuation_tool.config import Config
from valuation_tool.services.retry import with_retry

logger = structlog.get_logger()

MAX_BATCH_SIZE = 1000


class FirecrawlClient:
    """Client for the Firecrawl scraping API."""

    BASE_URL = "https://api.firecrawl.dev/v1"

    def __init__(self, config: Config):
        self.api_key = config.firecrawl_api_key
        self.max_retries = config.max_retry_attempts
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    def scrape_url(self, url: str) -> dict[str, Any]:
        """Scrape a single URL, returning markdown, HTML, and metadata."""
        fetch = with_retry(self.max_retries)(self._do_scrape)
        return fetch(url)

    def _do_scrape(self, url: str) -> dict[str, Any]:
        resp = self._client.post(
            f"{self.BASE_URL}/scrape",
            json={
                "url": url,
                "formats": ["markdown", "html"],
                "onlyMainContent": False,
            },
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    def batch_scrape(
        self, urls: list[str], batch_size: int = 50, timeout: int = 300
    ) -> list[dict[str, Any]]:
        """Scrape multiple URLs using the batch API.

        URLs are split into batches of at most `batch_size` (capped at 1000).
        """
        batch_size = min(batch_size, MAX_BATCH_SIZE)
        all_results: list[dict[str, Any]] = []

        batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

        for batch_urls in batches:
            fetch = with_retry(self.max_retries)(self._do_batch_scrape)
            results = fetch(batch_urls, timeout)
            all_results.extend(results)

        return all_results

    def _do_batch_scrape(self, urls: list[str], timeout: int) -> list[dict]:
        resp = self._client.post(
            f"{self.BASE_URL}/batch/scrape",
            json={
                "urls": urls,
                "formats": ["markdown", "html"],
                "onlyMainContent": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # The batch API may return a job ID for async processing
        if "id" in data and "data" not in data:
            return self._poll_batch_job(data["id"], timeout)

        return data.get("data", [])

    def _poll_batch_job(self, job_id: str, timeout: int) -> list[dict]:
        """Poll for batch job completion."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._client.get(f"{self.BASE_URL}/batch/scrape/{job_id}")
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "completed":
                return data.get("data", [])
            time.sleep(2)
        raise TimeoutError(f"Batch job {job_id} timed out after {timeout}s")

    def crawl_site(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 25,
        include_subdomains: bool = True,
    ) -> list[dict[str, Any]]:
        """Crawl an entire website starting from a URL."""
        fetch = with_retry(self.max_retries)(self._do_crawl)
        return fetch(url, max_depth, max_pages, include_subdomains)

    def _do_crawl(
        self, url: str, max_depth: int, max_pages: int, include_subdomains: bool
    ) -> list[dict]:
        resp = self._client.post(
            f"{self.BASE_URL}/crawl",
            json={
                "url": url,
                "maxDepth": max_depth,
                "limit": max_pages,
                "includeSubdomains": include_subdomains,
                "scrapeOptions": {
                    "formats": ["markdown", "html"],
                    "onlyMainContent": False,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle async crawl job
        if "id" in data and "data" not in data:
            return self._poll_crawl_job(data["id"])

        return data.get("data", [])

    def _poll_crawl_job(self, job_id: str, timeout: int = 300) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._client.get(f"{self.BASE_URL}/crawl/{job_id}")
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "completed":
                return data.get("data", [])
            time.sleep(3)
        raise TimeoutError(f"Crawl job {job_id} timed out")
