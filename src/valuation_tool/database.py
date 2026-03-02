"""SQLite database layer with full schema, CRUD, and migration support.

All datetimes are stored as ISO 8601 strings in UTC.
List fields are stored as JSON text.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import structlog

from valuation_tool.models import (
    BlogLink,
    ChangeRecord,
    Company,
    CompanyLeadership,
    CompanyLogo,
    CompanyStatus,
    NewsArticle,
    ProcessingError,
    Snapshot,
    SocialMediaLink,
    StatusIndicator,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    homepage_url TEXT,
    source_sheet TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(name, homepage_url)
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status_code INTEGER,
    content_markdown TEXT,
    content_html TEXT,
    content_checksum TEXT,
    error_message TEXT,
    has_paywall INTEGER NOT NULL DEFAULT 0,
    has_auth_required INTEGER NOT NULL DEFAULT 0,
    http_last_modified TEXT,
    captured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    snapshot_id_old INTEGER REFERENCES snapshots(id),
    snapshot_id_new INTEGER REFERENCES snapshots(id),
    has_changed INTEGER NOT NULL,
    change_magnitude TEXT NOT NULL,
    checksum_old TEXT,
    checksum_new TEXT,
    diff_summary TEXT,
    matched_keywords TEXT,
    matched_categories TEXT,
    significance_classification TEXT,
    significance_sentiment TEXT,
    significance_confidence REAL,
    significance_notes TEXT,
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS company_statuses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    indicators TEXT,
    analyzed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS social_media_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    profile_url TEXT NOT NULL,
    discovery_method TEXT,
    html_location TEXT,
    verification_status TEXT DEFAULT 'unverified',
    account_type TEXT DEFAULT 'unknown',
    account_confidence REAL DEFAULT 0.5,
    similarity_score REAL,
    discovered_at TEXT,
    UNIQUE(company_id, profile_url)
);

CREATE TABLE IF NOT EXISTS blog_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    blog_url TEXT NOT NULL,
    blog_type TEXT DEFAULT 'company_blog',
    is_active INTEGER NOT NULL DEFAULT 1,
    discovery_method TEXT DEFAULT 'page_footer',
    discovered_at TEXT,
    UNIQUE(company_id, blog_url)
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content_url TEXT NOT NULL UNIQUE,
    source TEXT,
    snippet TEXT,
    published_at TEXT,
    match_confidence REAL DEFAULT 0.0,
    match_evidence TEXT,
    significance_classification TEXT,
    significance_sentiment TEXT,
    significance_confidence REAL,
    discovered_at TEXT
);

CREATE TABLE IF NOT EXISTS company_logos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    image_url TEXT,
    image_format TEXT,
    perceptual_hash TEXT,
    extraction_location TEXT,
    extracted_at TEXT,
    UNIQUE(company_id, perceptual_hash)
);

CREATE TABLE IF NOT EXISTS company_leadership (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    person_name TEXT NOT NULL,
    title TEXT NOT NULL,
    linkedin_profile_url TEXT,
    discovery_method TEXT,
    confidence REAL DEFAULT 0.8,
    is_current INTEGER NOT NULL DEFAULT 1,
    discovered_at TEXT,
    UNIQUE(company_id, linkedin_profile_url)
);

CREATE TABLE IF NOT EXISTS processing_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    occurred_at TEXT NOT NULL
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name);
CREATE INDEX IF NOT EXISTS idx_snapshots_company_id ON snapshots(company_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_captured_at ON snapshots(captured_at);
CREATE INDEX IF NOT EXISTS idx_change_records_company_id ON change_records(company_id);
CREATE INDEX IF NOT EXISTS idx_social_media_links_company_id ON social_media_links(company_id);
CREATE INDEX IF NOT EXISTS idx_social_media_links_platform ON social_media_links(platform);
CREATE INDEX IF NOT EXISTS idx_news_articles_company_id ON news_articles(company_id);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_significance ON news_articles(significance_classification);
CREATE INDEX IF NOT EXISTS idx_company_logos_company_id ON company_logos(company_id);
CREATE INDEX IF NOT EXISTS idx_company_logos_perceptual_hash ON company_logos(perceptual_hash);
CREATE INDEX IF NOT EXISTS idx_company_leadership_company_id ON company_leadership(company_id);
CREATE INDEX IF NOT EXISTS idx_company_leadership_title ON company_leadership(title);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_to_dt(val: str | None) -> datetime | None:
    if val is None:
        return None
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _json_dumps(val: list | dict | None) -> str | None:
    if val is None:
        return None
    return json.dumps(val)


def _json_loads(val: str | None) -> list | dict | None:
    if val is None:
        return None
    return json.loads(val)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, db_path: str = "data/companies.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Schema / Migration ---

    def init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.executescript(_INDEX_SQL)
        logger.info("database_initialized", path=self.db_path)

    def run_migrations(self) -> None:
        """Apply incremental schema migrations safely."""
        migrations = [
            self._migration_add_content_checksum,
        ]
        with self.connection() as conn:
            for migration_fn in migrations:
                try:
                    migration_fn(conn)
                except Exception as exc:
                    logger.warning("migration_skipped", migration=migration_fn.__name__, error=str(exc))

    @staticmethod
    def _migration_add_content_checksum(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()}
        if "content_checksum" not in existing:
            conn.execute("ALTER TABLE snapshots ADD COLUMN content_checksum TEXT")

    # =======================================================================
    # Companies CRUD
    # =======================================================================

    def upsert_company(self, company: Company) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO companies (name, homepage_url, source_sheet, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name, homepage_url) DO UPDATE SET
                    source_sheet = excluded.source_sheet,
                    updated_at = excluded.updated_at
                """,
                (
                    company.name,
                    company.homepage_url,
                    company.source_sheet,
                    _dt_to_iso(company.created_at),
                    _dt_to_iso(company.updated_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_company_by_id(self, company_id: int) -> Company | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_company(row)

    def get_company_by_name(self, name: str) -> Company | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM companies WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return self._row_to_company(row)

    def get_all_companies(self, limit: int | None = None) -> list[Company]:
        sql = "SELECT * FROM companies ORDER BY id"
        params: tuple = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_company(r) for r in rows]

    def update_company(self, company_id: int, **kwargs: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [company_id]
        with self.connection() as conn:
            conn.execute(f"UPDATE companies SET {sets} WHERE id = ?", vals)

    def delete_company(self, company_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))

    @staticmethod
    def _row_to_company(row: sqlite3.Row) -> Company:
        return Company(
            id=row["id"],
            name=row["name"],
            homepage_url=row["homepage_url"],
            source_sheet=row["source_sheet"],
            created_at=_iso_to_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_iso_to_dt(row["updated_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Snapshots CRUD
    # =======================================================================

    def store_snapshot(self, snap: Snapshot) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO snapshots
                    (company_id, url, status_code, content_markdown, content_html,
                     content_checksum, error_message, has_paywall, has_auth_required,
                     http_last_modified, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.company_id,
                    snap.url,
                    snap.status_code,
                    snap.content_markdown,
                    snap.content_html,
                    snap.content_checksum,
                    snap.error_message,
                    int(snap.has_paywall),
                    int(snap.has_auth_required),
                    snap.http_last_modified,
                    _dt_to_iso(snap.captured_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_snapshots_for_company(self, company_id: int, limit: int | None = None) -> list[Snapshot]:
        sql = "SELECT * FROM snapshots WHERE company_id = ? ORDER BY captured_at DESC"
        params: list = [company_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def get_latest_two_snapshots(self, company_id: int) -> tuple[Snapshot | None, Snapshot | None]:
        snaps = self.get_snapshots_for_company(company_id, limit=2)
        new = snaps[0] if len(snaps) > 0 else None
        old = snaps[1] if len(snaps) > 1 else None
        return old, new

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
        return Snapshot(
            id=row["id"],
            company_id=row["company_id"],
            url=row["url"],
            status_code=row["status_code"],
            content_markdown=row["content_markdown"],
            content_html=row["content_html"],
            content_checksum=row["content_checksum"],
            error_message=row["error_message"],
            has_paywall=bool(row["has_paywall"]),
            has_auth_required=bool(row["has_auth_required"]),
            http_last_modified=row["http_last_modified"],
            captured_at=_iso_to_dt(row["captured_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Change Records CRUD
    # =======================================================================

    def store_change_record(self, record: ChangeRecord) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO change_records
                    (company_id, snapshot_id_old, snapshot_id_new, has_changed, change_magnitude,
                     checksum_old, checksum_new, diff_summary, matched_keywords, matched_categories,
                     significance_classification, significance_sentiment, significance_confidence,
                     significance_notes, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.company_id,
                    record.snapshot_id_old,
                    record.snapshot_id_new,
                    int(record.has_changed),
                    record.change_magnitude.value,
                    record.checksum_old,
                    record.checksum_new,
                    record.diff_summary,
                    _json_dumps(record.matched_keywords),
                    _json_dumps(record.matched_categories),
                    record.significance_classification.value if record.significance_classification else None,
                    record.significance_sentiment.value if record.significance_sentiment else None,
                    record.significance_confidence,
                    record.significance_notes,
                    _dt_to_iso(record.detected_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_change_records_for_company(self, company_id: int) -> list[ChangeRecord]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM change_records WHERE company_id = ? ORDER BY detected_at",
                (company_id,),
            ).fetchall()
        return [self._row_to_change_record(r) for r in rows]

    def get_change_records_with_null_significance(self, batch_size: int = 50) -> list[ChangeRecord]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM change_records WHERE significance_classification IS NULL LIMIT ?",
                (batch_size,),
            ).fetchall()
        return [self._row_to_change_record(r) for r in rows]

    def update_change_record_significance(
        self,
        record_id: int,
        classification: str,
        sentiment: str,
        confidence: float,
        keywords: list[str],
        categories: list[str],
        notes: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE change_records SET
                    significance_classification = ?,
                    significance_sentiment = ?,
                    significance_confidence = ?,
                    matched_keywords = ?,
                    matched_categories = ?,
                    significance_notes = ?
                WHERE id = ?
                """,
                (
                    classification,
                    sentiment,
                    confidence,
                    _json_dumps(keywords),
                    _json_dumps(categories),
                    notes,
                    record_id,
                ),
            )

    def get_recent_changes(self, days: int = 180) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cr.*, c.name as company_name
                FROM change_records cr
                JOIN companies c ON c.id = cr.company_id
                WHERE cr.has_changed = 1
                  AND cr.detected_at >= datetime('now', ? || ' days')
                ORDER BY cr.detected_at DESC
                """,
                (f"-{days}",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_significant_changes(
        self,
        days: int = 180,
        sentiment: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        sql = """
            SELECT cr.*, c.name as company_name
            FROM change_records cr
            JOIN companies c ON c.id = cr.company_id
            WHERE cr.significance_classification = 'significant'
              AND cr.detected_at >= datetime('now', ? || ' days')
              AND COALESCE(cr.significance_confidence, 0) >= ?
        """
        params: list[Any] = [f"-{days}", min_confidence]
        if sentiment:
            sql += " AND cr.significance_sentiment = ?"
            params.append(sentiment)
        sql += " ORDER BY cr.detected_at DESC"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_uncertain_changes(self, limit: int | None = None) -> list[dict]:
        sql = """
            SELECT cr.*, c.name as company_name
            FROM change_records cr
            JOIN companies c ON c.id = cr.company_id
            WHERE cr.significance_classification = 'uncertain'
            ORDER BY cr.detected_at DESC
        """
        params: tuple = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_change_record(row: sqlite3.Row) -> ChangeRecord:
        return ChangeRecord(
            id=row["id"],
            company_id=row["company_id"],
            snapshot_id_old=row["snapshot_id_old"],
            snapshot_id_new=row["snapshot_id_new"],
            has_changed=bool(row["has_changed"]),
            change_magnitude=row["change_magnitude"],
            checksum_old=row["checksum_old"],
            checksum_new=row["checksum_new"],
            diff_summary=row["diff_summary"],
            matched_keywords=_json_loads(row["matched_keywords"]) or [],
            matched_categories=_json_loads(row["matched_categories"]) or [],
            significance_classification=row["significance_classification"],
            significance_sentiment=row["significance_sentiment"],
            significance_confidence=row["significance_confidence"],
            significance_notes=row["significance_notes"],
            detected_at=_iso_to_dt(row["detected_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Company Statuses CRUD
    # =======================================================================

    def store_company_status(self, status: CompanyStatus) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO company_statuses (company_id, status, confidence, indicators, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    status.company_id,
                    status.status.value,
                    status.confidence,
                    _json_dumps([ind.model_dump() for ind in status.indicators]),
                    _dt_to_iso(status.analyzed_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_latest_status(self, company_id: int) -> CompanyStatus | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM company_statuses WHERE company_id = ? ORDER BY analyzed_at DESC LIMIT 1",
                (company_id,),
            ).fetchone()
        if row is None:
            return None
        indicators_raw = _json_loads(row["indicators"]) or []
        return CompanyStatus(
            id=row["id"],
            company_id=row["company_id"],
            status=row["status"],
            confidence=row["confidence"],
            indicators=[StatusIndicator(**ind) for ind in indicators_raw],
            analyzed_at=_iso_to_dt(row["analyzed_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Social Media Links CRUD
    # =======================================================================

    def store_social_media_link(self, link: SocialMediaLink) -> int | None:
        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO social_media_links
                        (company_id, platform, profile_url, discovery_method, html_location,
                         verification_status, account_type, account_confidence, similarity_score,
                         discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link.company_id,
                        link.platform.value,
                        link.profile_url,
                        link.discovery_method.value if link.discovery_method else None,
                        link.html_location,
                        link.verification_status.value,
                        link.account_type.value,
                        link.account_confidence,
                        link.similarity_score,
                        _dt_to_iso(link.discovered_at),
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # UNIQUE constraint: silently skip duplicates
            return None

    def get_social_links_for_company(
        self, company_id: int, platform: str | None = None
    ) -> list[SocialMediaLink]:
        sql = "SELECT * FROM social_media_links WHERE company_id = ?"
        params: list = [company_id]
        if platform:
            sql += " AND platform = ?"
            params.append(platform)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_social_link(r) for r in rows]

    @staticmethod
    def _row_to_social_link(row: sqlite3.Row) -> SocialMediaLink:
        return SocialMediaLink(
            id=row["id"],
            company_id=row["company_id"],
            platform=row["platform"],
            profile_url=row["profile_url"],
            discovery_method=row["discovery_method"] or "page_footer",
            html_location=row["html_location"],
            verification_status=row["verification_status"] or "unverified",
            account_type=row["account_type"] or "unknown",
            account_confidence=row["account_confidence"] or 0.5,
            similarity_score=row["similarity_score"],
            discovered_at=_iso_to_dt(row["discovered_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Blog Links CRUD
    # =======================================================================

    def store_blog_link(self, link: BlogLink) -> int | None:
        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO blog_links (company_id, blog_url, blog_type, is_active,
                                            discovery_method, discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link.company_id,
                        link.blog_url,
                        link.blog_type,
                        int(link.is_active),
                        link.discovery_method,
                        _dt_to_iso(link.discovered_at),
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    # =======================================================================
    # News Articles CRUD
    # =======================================================================

    def store_news_article(self, article: NewsArticle) -> int | None:
        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO news_articles
                        (company_id, title, content_url, source, snippet, published_at,
                         match_confidence, match_evidence, significance_classification,
                         significance_sentiment, significance_confidence, discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.company_id,
                        article.title,
                        article.content_url,
                        article.source,
                        article.snippet,
                        _dt_to_iso(article.published_at),
                        article.match_confidence,
                        _json_dumps(article.match_evidence),
                        article.significance_classification.value if article.significance_classification else None,
                        article.significance_sentiment.value if article.significance_sentiment else None,
                        article.significance_confidence,
                        _dt_to_iso(article.discovered_at),
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate content_url
            return None

    def get_news_for_company(self, company_id: int) -> list[NewsArticle]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM news_articles WHERE company_id = ? ORDER BY published_at DESC",
                (company_id,),
            ).fetchall()
        return [self._row_to_news_article(r) for r in rows]

    def article_url_exists(self, content_url: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM news_articles WHERE content_url = ?", (content_url,)
            ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_news_article(row: sqlite3.Row) -> NewsArticle:
        return NewsArticle(
            id=row["id"],
            company_id=row["company_id"],
            title=row["title"],
            content_url=row["content_url"],
            source=row["source"],
            snippet=row["snippet"],
            published_at=_iso_to_dt(row["published_at"]) or datetime.now(timezone.utc),
            match_confidence=row["match_confidence"] or 0.0,
            match_evidence=_json_loads(row["match_evidence"]) or [],
            significance_classification=row["significance_classification"],
            significance_sentiment=row["significance_sentiment"],
            significance_confidence=row["significance_confidence"],
            discovered_at=_iso_to_dt(row["discovered_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Company Logos CRUD
    # =======================================================================

    def store_company_logo(self, logo: CompanyLogo) -> int | None:
        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO company_logos
                        (company_id, image_url, image_format, perceptual_hash,
                         extraction_location, extracted_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        logo.company_id,
                        logo.image_url,
                        logo.image_format,
                        logo.perceptual_hash,
                        logo.extraction_location,
                        _dt_to_iso(logo.extracted_at),
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_logo_for_company(self, company_id: int) -> CompanyLogo | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM company_logos WHERE company_id = ? ORDER BY extracted_at DESC LIMIT 1",
                (company_id,),
            ).fetchone()
        if row is None:
            return None
        return CompanyLogo(
            id=row["id"],
            company_id=row["company_id"],
            image_url=row["image_url"],
            image_format=row["image_format"],
            perceptual_hash=row["perceptual_hash"],
            extraction_location=row["extraction_location"],
            extracted_at=_iso_to_dt(row["extracted_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Company Leadership CRUD
    # =======================================================================

    def store_leadership(self, leader: CompanyLeadership) -> int | None:
        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO company_leadership
                        (company_id, person_name, title, linkedin_profile_url,
                         discovery_method, confidence, is_current, discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        leader.company_id,
                        leader.person_name,
                        leader.title,
                        leader.linkedin_profile_url,
                        leader.discovery_method.value if leader.discovery_method else None,
                        leader.confidence,
                        int(leader.is_current),
                        _dt_to_iso(leader.discovered_at),
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_leadership_for_company(
        self, company_id: int, current_only: bool = False
    ) -> list[CompanyLeadership]:
        sql = "SELECT * FROM company_leadership WHERE company_id = ?"
        params: list = [company_id]
        if current_only:
            sql += " AND is_current = 1"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_leadership(r) for r in rows]

    def mark_leadership_not_current(self, company_id: int, linkedin_url: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE company_leadership SET is_current = 0 WHERE company_id = ? AND linkedin_profile_url = ?",
                (company_id, linkedin_url),
            )

    @staticmethod
    def _row_to_leadership(row: sqlite3.Row) -> CompanyLeadership:
        return CompanyLeadership(
            id=row["id"],
            company_id=row["company_id"],
            person_name=row["person_name"],
            title=row["title"],
            linkedin_profile_url=row["linkedin_profile_url"],
            discovery_method=row["discovery_method"] or "playwright_scrape",
            confidence=row["confidence"] or 0.8,
            is_current=bool(row["is_current"]),
            discovered_at=_iso_to_dt(row["discovered_at"]) or datetime.now(timezone.utc),
        )

    # =======================================================================
    # Processing Errors CRUD
    # =======================================================================

    def store_processing_error(self, error: ProcessingError) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO processing_errors
                    (entity_type, entity_id, error_type, error_message, retry_count, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    error.entity_type,
                    error.entity_id,
                    error.error_type,
                    error.error_message,
                    error.retry_count,
                    _dt_to_iso(error.occurred_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]
