"""Tests for database layer."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from valuation_tool.database import Database
from valuation_tool.models import (
    BlogLink,
    ChangeRecord,
    ChangeMagnitude,
    Company,
    CompanyLeadership,
    CompanyLogo,
    CompanyStatus,
    CompanyStatusType,
    IndicatorSignal,
    LeadershipDiscoveryMethod,
    NewsArticle,
    Platform,
    ProcessingError,
    Snapshot,
    SocialMediaLink,
    StatusIndicator,
)


@pytest.fixture
def db(tmp_path):
    """Create a fresh database for each test."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    database.init_db()
    return database


class TestSchemaInitialization:
    def test_all_tables_created(self, db):
        expected_tables = {
            "companies", "snapshots", "change_records", "company_statuses",
            "social_media_links", "blog_links", "news_articles",
            "company_logos", "company_leadership", "processing_errors",
        }
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        actual = {r["name"] for r in rows}
        assert expected_tables.issubset(actual)

    def test_indexes_created(self, db):
        expected_indexes = {
            "idx_companies_name",
            "idx_snapshots_company_id",
            "idx_snapshots_captured_at",
            "idx_change_records_company_id",
            "idx_social_media_links_company_id",
            "idx_social_media_links_platform",
            "idx_news_articles_company_id",
            "idx_news_articles_published_at",
            "idx_news_articles_significance",
            "idx_company_logos_company_id",
            "idx_company_logos_perceptual_hash",
            "idx_company_leadership_company_id",
            "idx_company_leadership_title",
        }
        with db.connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        actual = {r["name"] for r in rows}
        assert expected_indexes.issubset(actual)

    def test_idempotent_init(self, db):
        """Schema initialization is idempotent."""
        # Insert data
        db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        # Re-init
        db.init_db()
        # Data should still exist
        company = db.get_company_by_name("Test Corp")
        assert company is not None

    def test_wal_mode(self, db):
        with db.connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        with db.connection() as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestCompanyCRUD:
    def test_upsert_and_retrieve(self, db):
        company_id = db.upsert_company(
            Company(name="Test Corp", homepage_url="https://test.com")
        )
        assert company_id > 0
        company = db.get_company_by_id(company_id)
        assert company.name == "Test Corp"
        assert company.homepage_url == "https://test.com"

    def test_get_by_name(self, db):
        db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        company = db.get_company_by_name("Acme Corp")
        assert company is not None

    def test_upsert_no_duplicate(self, db):
        db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        companies = db.get_all_companies()
        acme_count = sum(1 for c in companies if c.name == "Acme Corp" and c.homepage_url == "https://acme.com")
        assert acme_count == 1

    def test_same_name_different_url_allowed(self, db):
        db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.io"))
        companies = db.get_all_companies()
        acme_count = sum(1 for c in companies if c.name == "Acme Corp")
        assert acme_count == 2

    def test_update_company(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        db.update_company(cid, homepage_url="https://new-test.com")
        company = db.get_company_by_id(cid)
        assert company.homepage_url == "https://new-test.com"

    def test_delete_company(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        db.delete_company(cid)
        assert db.get_company_by_id(cid) is None

    def test_cascade_delete(self, db):
        """Deleting a company cascades to all related records."""
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_snapshot(Snapshot(
            company_id=cid, url="https://acme.com", content_markdown="test"
        ))
        db.store_social_media_link(SocialMediaLink(
            company_id=cid, platform=Platform.twitter,
            profile_url="https://twitter.com/acme"
        ))
        db.delete_company(cid)
        assert db.get_snapshots_for_company(cid) == []
        assert db.get_social_links_for_company(cid) == []


class TestSnapshotsCRUD:
    def test_store_and_retrieve(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        sid = db.store_snapshot(Snapshot(
            company_id=cid, url="https://test.com",
            status_code=200, content_markdown="Hello",
            content_checksum="d41d8cd98f00b204e9800998ecf8427e",
        ))
        snaps = db.get_snapshots_for_company(cid)
        assert len(snaps) == 1
        assert snaps[0].content_markdown == "Hello"

    def test_latest_two_snapshots(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        db.store_snapshot(Snapshot(company_id=cid, url="https://test.com", content_markdown="old"))
        db.store_snapshot(Snapshot(company_id=cid, url="https://test.com", content_markdown="new"))
        old, new = db.get_latest_two_snapshots(cid)
        assert old is not None
        assert new is not None

    def test_foreign_key_violation(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.store_snapshot(Snapshot(
                company_id=999, url="https://test.com", content_markdown="test"
            ))


class TestChangeRecordsCRUD:
    def test_store_and_retrieve(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        db.store_change_record(ChangeRecord(
            company_id=cid, has_changed=True,
            change_magnitude=ChangeMagnitude.moderate,
            matched_keywords=["funding", "series a"],
            matched_categories=["funding_investment"],
        ))
        records = db.get_change_records_for_company(cid)
        assert len(records) == 1
        assert records[0].matched_keywords == ["funding", "series a"]

    def test_json_serialization_roundtrip(self, db):
        cid = db.upsert_company(Company(name="Test Corp", homepage_url="https://test.com"))
        db.store_change_record(ChangeRecord(
            company_id=cid, has_changed=True,
            change_magnitude=ChangeMagnitude.minor,
            matched_keywords=["funding", "series a"],
        ))
        records = db.get_change_records_for_company(cid)
        assert isinstance(records[0].matched_keywords, list)
        assert records[0].matched_keywords == ["funding", "series a"]


class TestSocialMediaLinksCRUD:
    def test_store_and_filter(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_social_media_link(SocialMediaLink(
            company_id=cid, platform=Platform.twitter,
            profile_url="https://twitter.com/acme"
        ))
        db.store_social_media_link(SocialMediaLink(
            company_id=cid, platform=Platform.linkedin,
            profile_url="https://linkedin.com/company/acme"
        ))
        all_links = db.get_social_links_for_company(cid)
        assert len(all_links) == 2
        twitter_links = db.get_social_links_for_company(cid, platform="twitter")
        assert len(twitter_links) == 1

    def test_unique_constraint(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_social_media_link(SocialMediaLink(
            company_id=cid, platform=Platform.twitter,
            profile_url="https://twitter.com/acme"
        ))
        # Duplicate should be silently skipped
        result = db.store_social_media_link(SocialMediaLink(
            company_id=cid, platform=Platform.twitter,
            profile_url="https://twitter.com/acme"
        ))
        assert result is None
        assert len(db.get_social_links_for_company(cid)) == 1


class TestNewsArticlesCRUD:
    def test_store_and_retrieve(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_news_article(NewsArticle(
            company_id=cid, title="Acme Raises $50M",
            content_url="https://techcrunch.com/acme-50m",
        ))
        articles = db.get_news_for_company(cid)
        assert len(articles) == 1
        assert articles[0].source == "techcrunch.com"

    def test_unique_content_url(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_news_article(NewsArticle(
            company_id=cid, title="Article 1",
            content_url="https://example.com/article-1",
        ))
        result = db.store_news_article(NewsArticle(
            company_id=cid, title="Article 1 Dupe",
            content_url="https://example.com/article-1",
        ))
        assert result is None

    def test_article_url_exists(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_news_article(NewsArticle(
            company_id=cid, title="Test",
            content_url="https://example.com/exists",
        ))
        assert db.article_url_exists("https://example.com/exists") is True
        assert db.article_url_exists("https://example.com/missing") is False


class TestLeadershipCRUD:
    def test_store_and_retrieve(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Jane Smith", title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
        ))
        leaders = db.get_leadership_for_company(cid)
        assert len(leaders) == 1
        assert leaders[0].person_name == "Jane Smith"

    def test_current_only_filter(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Jane", title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
            is_current=True,
        ))
        db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Bob", title="CTO",
            linkedin_profile_url="https://linkedin.com/in/bob",
            is_current=True,
        ))
        current = db.get_leadership_for_company(cid, current_only=True)
        assert len(current) == 2

    def test_mark_not_current(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Jane", title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
        ))
        db.mark_leadership_not_current(cid, "https://linkedin.com/in/jane")
        current = db.get_leadership_for_company(cid, current_only=True)
        assert len(current) == 0
        all_leaders = db.get_leadership_for_company(cid)
        assert len(all_leaders) == 1  # Record still exists

    def test_unique_constraint(self, db):
        cid = db.upsert_company(Company(name="Acme Corp", homepage_url="https://acme.com"))
        db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Jane", title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
        ))
        result = db.store_leadership(CompanyLeadership(
            company_id=cid, person_name="Jane Updated", title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
        ))
        assert result is None


class TestProcessingErrors:
    def test_store(self, db):
        eid = db.store_processing_error(ProcessingError(
            entity_type="company", entity_id=42,
            error_type="FirecrawlTimeout",
            error_message="Request timed out after 30s",
            retry_count=2,
        ))
        assert eid > 0


class TestCompanyStatus:
    def test_store_and_retrieve(self, db):
        cid = db.upsert_company(Company(name="Active Corp", homepage_url="https://active.com"))
        db.store_company_status(CompanyStatus(
            company_id=cid,
            status=CompanyStatusType.operational,
            confidence=0.85,
            indicators=[
                StatusIndicator(type="copyright_year", value="2026", signal=IndicatorSignal.positive),
            ],
        ))
        status = db.get_latest_status(cid)
        assert status is not None
        assert status.status == CompanyStatusType.operational
        assert status.confidence == 0.85
        assert len(status.indicators) == 1
        assert status.indicators[0].type == "copyright_year"


class TestDatetimeHandling:
    def test_iso8601_roundtrip(self, db):
        cid = db.upsert_company(Company(name="DT Corp", homepage_url="https://dt.com"))
        company = db.get_company_by_id(cid)
        assert company.created_at.tzinfo is not None

    def test_utc_timezone(self, db):
        cid = db.upsert_company(Company(name="UTC Corp", homepage_url="https://utc.com"))
        db.store_snapshot(Snapshot(
            company_id=cid, url="https://utc.com", content_markdown="test"
        ))
        snaps = db.get_snapshots_for_company(cid)
        assert snaps[0].captured_at.tzinfo is not None
