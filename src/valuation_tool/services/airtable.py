"""Airtable data extraction service.

Extracts company records from Airtable "Online Presence" and "Portfolio Companies"
tables, normalizes names, validates URLs, and upserts into the local database.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
import structlog

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import Company, ExtractionResult
from valuation_tool.services.retry import AuthenticationError, with_retry

logger = structlog.get_logger()


class AirtableClient:
    """Client for the Airtable REST API."""

    BASE_URL = "https://api.airtable.com/v0"

    def __init__(self, config: Config):
        self.api_key = config.airtable_api_key
        self.base_id = config.airtable_base_id
        self.max_retries = config.max_retry_attempts
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    def _url(self, table: str) -> str:
        return f"{self.BASE_URL}/{self.base_id}/{table}"

    def _fetch_records(self, table: str, params: dict | None = None) -> list[dict]:
        """Fetch all records from a table, handling pagination."""
        all_records: list[dict] = []
        offset: str | None = None
        fetch = with_retry(self.max_retries)(self._do_fetch)

        while True:
            req_params = dict(params or {})
            if offset:
                req_params["offset"] = offset
            data = fetch(table, req_params)
            all_records.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break

        return all_records

    def _do_fetch(self, table: str, params: dict) -> dict:
        resp = self._client.get(self._url(table), params=params)
        if resp.status_code in (401, 403):
            raise AuthenticationError(f"Airtable authentication failed: {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    def get_online_presence_records(self) -> list[dict]:
        return self._fetch_records("Online Presence")

    def resolve_company_name(self, record_id: str) -> str | None:
        """Look up a company name from the Portfolio Companies table."""
        try:
            fetch = with_retry(self.max_retries)(self._do_get_record)
            data = fetch(record_id)
            return data.get("fields", {}).get("Name")
        except Exception as exc:
            logger.warning("resolve_company_name_failed", record_id=record_id, error=str(exc))
            return None

    def _do_get_record(self, record_id: str) -> dict:
        resp = self._client.get(f"{self._url('Portfolio Companies')}/{record_id}")
        resp.raise_for_status()
        return resp.json()


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)
    except Exception:
        return False


def extract_companies(config: Config, db: Database) -> ExtractionResult:
    """Extract companies from Airtable and upsert into the database.

    Only processes records with 'homepage' resource type.
    Normalizes company names and validates homepage URLs.
    """
    client = AirtableClient(config)
    result = ExtractionResult()

    try:
        records = client.get_online_presence_records()
    except AuthenticationError:
        logger.error("airtable_auth_failed")
        raise

    for record in records:
        fields = record.get("fields", {})
        resources = fields.get("resources", "")

        # Only process homepage records
        if resources != "homepage":
            continue

        result.processed += 1

        # Resolve company name
        company_name_ref = fields.get("company_name_ref")
        if isinstance(company_name_ref, list):
            company_name_ref = company_name_ref[0] if company_name_ref else None

        if not company_name_ref:
            result.skipped += 1
            logger.info("record_skipped_no_ref", record_id=record.get("id"))
            continue

        name = client.resolve_company_name(company_name_ref)
        if not name:
            result.skipped += 1
            continue

        # Normalize name
        name = " ".join(name.split()).title()

        # Validate URL
        url = fields.get("url")
        homepage_url = url if url and _is_valid_url(url) else None

        company = Company(
            name=name,
            homepage_url=homepage_url,
            source_sheet="Online Presence",
        )

        try:
            db.upsert_company(company)
            result.successful += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": name, "error": str(exc)})
            logger.error("upsert_failed", company=name, error=str(exc))

    logger.info(
        "extract_companies_complete",
        processed=result.processed,
        successful=result.successful,
        skipped=result.skipped,
        failed=result.failed,
    )
    return result
