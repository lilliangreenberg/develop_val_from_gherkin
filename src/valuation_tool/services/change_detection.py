"""Website change detection and company status analysis.

Detects content changes between successive snapshots, calculates change
magnitude via SequenceMatcher similarity, extracts diffs, and runs
significance analysis. Also analyzes company operational status from
content indicators.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import (
    ChangeMagnitude,
    ChangeRecord,
    CompanyStatus,
    CompanyStatusType,
    ExtractionResult,
    IndicatorSignal,
    StatusIndicator,
)
from valuation_tool.services.significance import analyze_significance

logger = structlog.get_logger()

MAX_COMPARE_LENGTH = 50_000


# ---------------------------------------------------------------------------
# Change Detection
# ---------------------------------------------------------------------------

def _calculate_magnitude(similarity: float) -> ChangeMagnitude:
    """Determine change magnitude from content similarity ratio.

    >= 0.90 → minor
    >= 0.50 → moderate
    <  0.50 → major
    """
    if similarity >= 0.90:
        return ChangeMagnitude.minor
    if similarity >= 0.50:
        return ChangeMagnitude.moderate
    return ChangeMagnitude.major


def _extract_diff_lines(old_content: str, new_content: str) -> str:
    """Extract only added/modified lines from a unified diff."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    added = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
    return "\n".join(added)


def detect_changes(config: Config, db: Database) -> ExtractionResult:
    """Run change detection for all companies with 2+ snapshots."""
    companies = db.get_all_companies()
    result = ExtractionResult()

    for company in companies:
        old_snap, new_snap = db.get_latest_two_snapshots(company.id)

        if not old_snap or not new_snap:
            result.skipped += 1
            continue

        result.processed += 1

        if not old_snap.content_markdown:
            result.skipped += 1
            logger.warning("missing_old_content", company=company.name, company_id=company.id)
            continue

        try:
            old_content = (old_snap.content_markdown or "")[:MAX_COMPARE_LENGTH]
            new_content = (new_snap.content_markdown or "")[:MAX_COMPARE_LENGTH]

            # Compare checksums
            has_changed = old_snap.content_checksum != new_snap.content_checksum

            if has_changed and old_content and new_content:
                similarity = difflib.SequenceMatcher(None, old_content, new_content).ratio()
            else:
                similarity = 1.0 if not has_changed else 0.0

            magnitude = _calculate_magnitude(similarity)

            # Extract diff for significance analysis
            diff_text = ""
            if has_changed and old_content and new_content:
                diff_text = _extract_diff_lines(old_content, new_content)

            # Run significance analysis
            content_for_analysis = diff_text if diff_text else new_content
            sig_result = analyze_significance(content_for_analysis, magnitude)

            record = ChangeRecord(
                company_id=company.id,
                snapshot_id_old=old_snap.id,
                snapshot_id_new=new_snap.id,
                has_changed=has_changed,
                change_magnitude=magnitude,
                checksum_old=old_snap.content_checksum,
                checksum_new=new_snap.content_checksum,
                diff_summary=diff_text[:5000] if diff_text else None,
                matched_keywords=sig_result.matched_keywords,
                matched_categories=sig_result.matched_categories,
                significance_classification=sig_result.classification,
                significance_sentiment=sig_result.sentiment,
                significance_confidence=sig_result.confidence,
            )

            db.store_change_record(record)
            result.successful += 1

        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})
            logger.error("change_detection_failed", company=company.name, error=str(exc))

    logger.info(
        "detect_changes_complete",
        processed=result.processed,
        successful=result.successful,
        failed=result.failed,
        skipped=result.skipped,
    )
    return result


# ---------------------------------------------------------------------------
# Status Analysis
# ---------------------------------------------------------------------------

_COPYRIGHT_PATTERN = re.compile(
    r"(?:copyright|\(c\)|©|&copy;)\s*(?:\d{4}\s*[-–]\s*)?(\d{4})",
    re.IGNORECASE,
)

_ACQUISITION_PATTERNS = [
    re.compile(r"\bacquired\s+by\b", re.IGNORECASE),
    re.compile(r"\bmerged\s+with\b", re.IGNORECASE),
    re.compile(r"\bsold\s+to\b", re.IGNORECASE),
    re.compile(r"\bnow\s+(?:a\s+)?part\s+of\b", re.IGNORECASE),
    re.compile(r"\bis\s+now\s+a\s+subsidiary\s+of\b", re.IGNORECASE),
    re.compile(r"\bis\s+now\s+a\s+division\s+of\b", re.IGNORECASE),
]

_HIRING_PATTERNS = [
    re.compile(r"\bhiring\b", re.IGNORECASE),
    re.compile(r"\bjoin\s+our\s+team\b", re.IGNORECASE),
    re.compile(r"\bcareers?\b", re.IGNORECASE),
    re.compile(r"\bopen\s+positions?\b", re.IGNORECASE),
]


def extract_copyright_year(content: str) -> int | None:
    """Extract the most recent copyright year from content."""
    matches = _COPYRIGHT_PATTERN.findall(content)
    if not matches:
        return None
    years = [int(y) for y in matches]
    return max(years)


def detect_acquisition(content: str) -> bool:
    """Detect acquisition/merger language, filtering false positives."""
    content_lower = content.lower()
    # False positive: "We acquired new customers" / "talent acquisition"
    for pattern in _ACQUISITION_PATTERNS:
        match = pattern.search(content)
        if match:
            # Check context: must not be preceded by "we", "our", "talent", etc.
            start = max(0, match.start() - 30)
            prefix = content_lower[start:match.start()]
            if any(word in prefix for word in ["we ", "our ", "talent ", "customer ", "data "]):
                continue
            return True
    return False


def analyze_status(
    config: Config, db: Database, company_id: int | None = None
) -> ExtractionResult:
    """Analyze operational status for companies based on their latest snapshot."""
    if company_id:
        companies = [db.get_company_by_id(company_id)]
        companies = [c for c in companies if c]
    else:
        companies = db.get_all_companies()

    result = ExtractionResult()

    for company in companies:
        result.processed += 1
        snaps = db.get_snapshots_for_company(company.id, limit=1)
        if not snaps:
            result.skipped += 1
            continue

        latest = snaps[0]
        content = latest.content_markdown or ""
        indicators: list[StatusIndicator] = []

        # Copyright year indicator
        year = extract_copyright_year(content)
        now_year = datetime.now(timezone.utc).year
        if year is not None:
            if year >= now_year - 1:
                indicators.append(StatusIndicator(
                    type="copyright_year", value=str(year), signal=IndicatorSignal.positive
                ))
            elif year >= now_year - 3:
                indicators.append(StatusIndicator(
                    type="copyright_year", value=str(year), signal=IndicatorSignal.neutral
                ))
            else:
                indicators.append(StatusIndicator(
                    type="copyright_year", value=str(year), signal=IndicatorSignal.negative
                ))

        # Acquisition detection
        if detect_acquisition(content):
            indicators.append(StatusIndicator(
                type="acquisition", value="detected", signal=IndicatorSignal.negative
            ))

        # Hiring signals
        for hp in _HIRING_PATTERNS:
            if hp.search(content):
                indicators.append(StatusIndicator(
                    type="hiring_signal", value="detected", signal=IndicatorSignal.positive
                ))
                break

        # HTTP freshness
        if latest.http_last_modified:
            try:
                lm_dt = datetime.fromisoformat(latest.http_last_modified.replace("Z", "+00:00"))
                days_old = (datetime.now(timezone.utc) - lm_dt).days
                if days_old <= 30:
                    indicators.append(StatusIndicator(
                        type="http_freshness", value=f"{days_old} days", signal=IndicatorSignal.positive
                    ))
                elif days_old <= 180:
                    indicators.append(StatusIndicator(
                        type="http_freshness", value=f"{days_old} days", signal=IndicatorSignal.neutral
                    ))
                else:
                    indicators.append(StatusIndicator(
                        type="http_freshness", value=f"{days_old} days", signal=IndicatorSignal.negative
                    ))
            except (ValueError, TypeError):
                pass

        # Calculate confidence and status
        positive_count = sum(1 for i in indicators if i.signal == IndicatorSignal.positive)
        negative_count = sum(1 for i in indicators if i.signal == IndicatorSignal.negative)
        neutral_count = sum(1 for i in indicators if i.signal == IndicatorSignal.neutral)
        total = positive_count + negative_count + neutral_count

        if total == 0:
            confidence = 0.2
        else:
            # Higher confidence when signals agree
            dominant = max(positive_count, negative_count)
            confidence = min(0.4 + 0.2 * dominant, 1.0)

        if confidence >= 0.5 and positive_count > negative_count:
            status = CompanyStatusType.operational
        elif confidence >= 0.5 and negative_count > positive_count:
            status = CompanyStatusType.likely_closed
        elif confidence >= 0.5 and positive_count == negative_count:
            status = CompanyStatusType.uncertain
        else:
            status = CompanyStatusType.uncertain

        company_status = CompanyStatus(
            company_id=company.id,
            status=status,
            confidence=round(confidence, 2),
            indicators=indicators,
        )

        try:
            db.store_company_status(company_status)
            result.successful += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append({"company": company.name, "error": str(exc)})

    return result
