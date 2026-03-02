"""Typer CLI for the investment valuation tool.

All commands load configuration, initialize the database, and delegate
to the appropriate service module.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import structlog
import typer

from valuation_tool.config import Config
from valuation_tool.database import Database

app = typer.Typer(
    name="valuation-tool",
    help="Automated investment valuation and portfolio monitoring tool.",
    add_completion=False,
)

logger = structlog.get_logger()


def _get_config() -> Config:
    try:
        return Config()  # type: ignore[call-arg]
    except Exception as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(1)


def _get_db(config: Config) -> Database:
    db = Database(config.database_path)
    db.init_db()
    return db


# ===================================================================
# Database
# ===================================================================

@app.command()
def init_db() -> None:
    """Initialize the database schema."""
    config = _get_config()
    db = _get_db(config)
    typer.echo("Database initialized successfully.")


@app.command()
def migrate() -> None:
    """Run database migrations."""
    config = _get_config()
    db = _get_db(config)
    db.run_migrations()
    typer.echo("Migrations complete.")


# ===================================================================
# Extract Companies
# ===================================================================

@app.command()
def extract_companies() -> None:
    """Extract companies from Airtable and store in database."""
    from valuation_tool.services.airtable import extract_companies as _extract

    config = _get_config()
    db = _get_db(config)
    result = _extract(config, db)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Skipped: {result.skipped}, "
        f"Failed: {result.failed}"
    )


# ===================================================================
# Capture Snapshots
# ===================================================================

@app.command()
def capture_snapshots(
    use_batch_api: bool = typer.Option(False, "--use-batch-api", help="Use Firecrawl batch API"),
    batch_size: int = typer.Option(50, "--batch-size", help="URLs per batch"),
    timeout: int = typer.Option(300, "--timeout", help="Batch timeout in seconds"),
) -> None:
    """Capture website snapshots for portfolio companies."""
    from valuation_tool.services.snapshot import capture_snapshots as _capture

    config = _get_config()
    db = _get_db(config)
    result = _capture(config, db, use_batch_api=use_batch_api, batch_size=batch_size, timeout=timeout)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Failed: {result.failed}"
    )


# ===================================================================
# Detect Changes
# ===================================================================

@app.command()
def detect_changes() -> None:
    """Detect website content changes between snapshots."""
    from valuation_tool.services.change_detection import detect_changes as _detect

    config = _get_config()
    db = _get_db(config)
    result = _detect(config, db)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Skipped: {result.skipped}, "
        f"Failed: {result.failed}"
    )


# ===================================================================
# Analyze Status
# ===================================================================

@app.command()
def analyze_status(
    company_id: Optional[int] = typer.Option(None, "--company-id", help="Specific company ID"),
) -> None:
    """Analyze operational status for companies."""
    from valuation_tool.services.change_detection import analyze_status as _analyze

    config = _get_config()
    db = _get_db(config)
    result = _analyze(config, db, company_id=company_id)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Skipped: {result.skipped}"
    )


# ===================================================================
# Social Media Discovery
# ===================================================================

@app.command()
def discover_social_media(
    company_id: Optional[int] = typer.Option(None, "--company-id"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    batch_size: int = typer.Option(50, "--batch-size"),
) -> None:
    """Discover social media profiles from company homepages."""
    from valuation_tool.services.social_discovery import discover_social_media as _discover

    config = _get_config()
    db = _get_db(config)
    result = _discover(config, db, company_id=company_id, limit=limit, batch_size=batch_size)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Failed: {result.failed}"
    )


@app.command()
def discover_social_full_site(
    company_id: int = typer.Option(..., "--company-id"),
    max_depth: int = typer.Option(2, "--max-depth"),
    max_pages: int = typer.Option(25, "--max-pages"),
    no_subdomains: bool = typer.Option(False, "--no-subdomains"),
) -> None:
    """Discover social media across an entire company website."""
    from valuation_tool.services.social_discovery import discover_social_full_site as _discover

    config = _get_config()
    db = _get_db(config)
    result = _discover(
        config, db, company_id,
        max_depth=max_depth,
        max_pages=max_pages,
        include_subdomains=not no_subdomains,
    )
    typer.echo(f"Successful: {result.successful}, Failed: {result.failed}")


@app.command()
def discover_social_batch(
    limit: Optional[int] = typer.Option(None, "--limit"),
    max_workers: int = typer.Option(5, "--max-workers"),
) -> None:
    """Run batch social media discovery in parallel."""
    from valuation_tool.services.social_discovery import discover_social_batch as _discover

    config = _get_config()
    db = _get_db(config)
    result = _discover(config, db, limit=limit, max_workers=max_workers)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Failed: {result.failed}"
    )


# ===================================================================
# News Monitoring
# ===================================================================

@app.command()
def search_news(
    company_name: Optional[str] = typer.Option(None, "--company-name"),
    company_id: Optional[int] = typer.Option(None, "--company-id"),
) -> None:
    """Search for news about a specific company."""
    from valuation_tool.services.news_monitoring import search_news_for_company

    config = _get_config()
    db = _get_db(config)
    result = search_news_for_company(config, db, company_id=company_id, company_name=company_name)
    typer.echo(
        f"Company: {result.company_name}, "
        f"Found: {result.articles_found}, "
        f"Stored: {result.articles_stored}"
    )
    if result.errors:
        typer.echo(f"Errors: {result.errors}", err=True)


@app.command()
def search_news_all(
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Search news for all companies."""
    from valuation_tool.services.news_monitoring import search_news_all as _search

    config = _get_config()
    db = _get_db(config)
    result = _search(config, db, limit=limit)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Failed: {result.failed}"
    )


# ===================================================================
# Leadership Extraction
# ===================================================================

@app.command()
def extract_leadership(
    company_id: int = typer.Option(..., "--company-id"),
    headless: Optional[bool] = typer.Option(None, "--headless"),
    profile_dir: Optional[str] = typer.Option(None, "--profile-dir"),
) -> None:
    """Extract leadership for a specific company."""
    from valuation_tool.services.leadership import extract_leadership_for_company

    config = _get_config()
    db = _get_db(config)
    result = extract_leadership_for_company(
        config, db, company_id,
        headless=headless,
        profile_dir=profile_dir,
    )
    typer.echo(
        f"Company: {result.company_name}, "
        f"Leaders found: {result.leaders_found}, "
        f"Method: {result.method_used}"
    )
    if result.leadership_changes:
        for change in result.leadership_changes:
            if change.change_type.value != "no_change":
                typer.echo(f"  Change: {change.change_type.value} - {change.person_name} ({change.title})")


@app.command()
def extract_leadership_all(
    limit: Optional[int] = typer.Option(None, "--limit"),
    max_workers: int = typer.Option(1, "--max-workers"),
) -> None:
    """Extract leadership for all companies."""
    from valuation_tool.services.leadership import extract_leadership_all as _extract

    config = _get_config()
    db = _get_db(config)
    result = _extract(config, db, limit=limit, max_workers=max_workers)
    typer.echo(
        f"Processed: {result.processed}, "
        f"Successful: {result.successful}, "
        f"Failed: {result.failed}"
    )


@app.command()
def check_leadership_changes(
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Re-extract leadership and report critical changes."""
    from valuation_tool.services.leadership import check_leadership_changes as _check

    config = _get_config()
    db = _get_db(config)
    changes = _check(config, db, limit=limit)

    if not changes:
        typer.echo("No critical leadership changes detected.")
    else:
        for change in changes:
            typer.echo(
                f"  {change['company']}: {change['change_type']} - "
                f"{change['person']} ({change['title']})"
            )


# ===================================================================
# Significance Backfill
# ===================================================================

@app.command()
def backfill_significance(
    dry_run: bool = typer.Option(False, "--dry-run"),
    batch_size: int = typer.Option(50, "--batch-size"),
) -> None:
    """Backfill significance analysis for change records missing it."""
    from valuation_tool.services.significance import analyze_significance

    config = _get_config()
    db = _get_db(config)

    total_updated = 0
    while True:
        records = db.get_change_records_with_null_significance(batch_size=batch_size)
        if not records:
            break

        for record in records:
            # Get snapshot content for analysis
            content = ""
            if record.snapshot_id_new:
                snaps = db.get_snapshots_for_company(record.company_id, limit=10)
                for snap in snaps:
                    if snap.id == record.snapshot_id_new:
                        content = snap.content_markdown or ""
                        break

            if not content and record.diff_summary:
                content = record.diff_summary

            sig = analyze_significance(content, record.change_magnitude)

            if dry_run:
                typer.echo(
                    f"  Record {record.id}: {sig.classification.value} "
                    f"({sig.sentiment.value}, confidence={sig.confidence:.2f})"
                )
            else:
                db.update_change_record_significance(
                    record.id,
                    sig.classification.value,
                    sig.sentiment.value,
                    sig.confidence,
                    sig.matched_keywords,
                    sig.matched_categories,
                )

            total_updated += 1

    typer.echo(f"{'Would update' if dry_run else 'Updated'}: {total_updated} records")


# ===================================================================
# Query Commands
# ===================================================================

@app.command()
def show_changes(
    company_name: str = typer.Option(..., "--company-name"),
) -> None:
    """Show change history for a company."""
    config = _get_config()
    db = _get_db(config)

    company = db.get_company_by_name(company_name)
    if not company:
        typer.echo(f"Company '{company_name}' not found.", err=True)
        raise typer.Exit(1)

    records = db.get_change_records_for_company(company.id)
    news = db.get_news_for_company(company.id)

    typer.echo(f"\nChange history for {company.name}:")
    typer.echo("-" * 60)
    for r in records:
        sig = f" [{r.significance_classification.value}]" if r.significance_classification else ""
        typer.echo(
            f"  {r.detected_at.strftime('%Y-%m-%d %H:%M')} | "
            f"Changed: {r.has_changed} | "
            f"Magnitude: {r.change_magnitude.value}{sig}"
        )

    if news:
        typer.echo(f"\nRelated news articles:")
        for article in news:
            typer.echo(f"  {article.published_at.strftime('%Y-%m-%d')} | {article.title}")
            typer.echo(f"    {article.content_url}")


@app.command()
def show_status(
    company_name: str = typer.Option(..., "--company-name"),
) -> None:
    """Show current status for a company."""
    config = _get_config()
    db = _get_db(config)

    company = db.get_company_by_name(company_name)
    if not company:
        typer.echo(f"Company '{company_name}' not found.", err=True)
        raise typer.Exit(1)

    status = db.get_latest_status(company.id)
    if not status:
        typer.echo(f"No status analysis available for '{company_name}'.")
        return

    typer.echo(f"\nStatus for {company.name}:")
    typer.echo(f"  Status: {status.status.value}")
    typer.echo(f"  Confidence: {status.confidence:.2f}")
    if status.indicators:
        typer.echo("  Indicators:")
        for ind in status.indicators:
            typer.echo(f"    - {ind.type}: {ind.value} ({ind.signal.value})")


@app.command()
def list_active(
    days: int = typer.Option(180, "--days"),
) -> None:
    """List companies with recent changes."""
    config = _get_config()
    db = _get_db(config)

    changes = db.get_recent_changes(days=days)
    seen: set[str] = set()
    for c in changes:
        name = c.get("company_name", "")
        if name not in seen:
            seen.add(name)
            typer.echo(f"  {name} (last change: {c.get('detected_at', 'unknown')})")

    if not seen:
        typer.echo("No companies with recent changes.")


@app.command()
def list_inactive(
    days: int = typer.Option(180, "--days"),
) -> None:
    """List companies without recent changes."""
    config = _get_config()
    db = _get_db(config)

    active_changes = db.get_recent_changes(days=days)
    active_names = {c.get("company_name") for c in active_changes}
    all_companies = db.get_all_companies()

    for company in all_companies:
        if company.name not in active_names:
            typer.echo(f"  {company.name}")


@app.command()
def list_significant_changes(
    days: int = typer.Option(180, "--days"),
    sentiment: Optional[str] = typer.Option(None, "--sentiment"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
) -> None:
    """List significant changes, optionally filtered by sentiment."""
    config = _get_config()
    db = _get_db(config)

    changes = db.get_significant_changes(days=days, sentiment=sentiment, min_confidence=min_confidence)
    for c in changes:
        typer.echo(
            f"  {c.get('company_name', '')} | "
            f"{c.get('significance_sentiment', '')} | "
            f"confidence: {c.get('significance_confidence', 0):.2f} | "
            f"{c.get('detected_at', '')}"
        )

    if not changes:
        typer.echo("No significant changes found.")


@app.command()
def list_uncertain_changes(
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """List uncertain changes requiring manual review."""
    config = _get_config()
    db = _get_db(config)

    changes = db.get_uncertain_changes(limit=limit)
    for c in changes:
        typer.echo(
            f"  {c.get('company_name', '')} | "
            f"detected: {c.get('detected_at', '')}"
        )

    if not changes:
        typer.echo("No uncertain changes found.")


if __name__ == "__main__":
    app()
