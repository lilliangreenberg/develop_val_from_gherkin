"""Tests for Pydantic models."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from valuation_tool.models import (
    AccountType,
    ChangeMagnitude,
    ChangeRecord,
    Company,
    CompanyLeadership,
    CompanyStatusType,
    DiscoveryMethod,
    ExtractionResult,
    IndicatorSignal,
    KeywordMatch,
    LeadershipChangeType,
    LeadershipDiscoveryMethod,
    NewsArticle,
    Platform,
    ProcessingError,
    SignificanceClassification,
    SignificanceSentiment,
    Snapshot,
    SocialMediaLink,
    VerificationStatus,
)


class TestCompany:
    def test_create_valid(self):
        c = Company(name="Acme Corp", homepage_url="https://www.acme.com")
        assert c.name == "Acme Corp"

    def test_name_normalized(self):
        c = Company(name="  acme   corp  ", homepage_url=None)
        assert c.name == "Acme Corp"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Company(name="", homepage_url=None)


class TestSnapshot:
    def test_valid_snapshot(self):
        s = Snapshot(
            company_id=1,
            url="https://example.com",
            status_code=200,
            content_markdown="Hello",
        )
        assert s.content_markdown == "Hello"

    @pytest.mark.parametrize("code,valid", [
        (200, True), (404, True), (100, True), (599, True),
        (99, False), (600, False),
    ])
    def test_status_code_range(self, code, valid):
        if valid:
            s = Snapshot(company_id=1, url="https://x.com", status_code=code, content_markdown="ok")
            assert s.status_code == code
        else:
            with pytest.raises(ValidationError):
                Snapshot(company_id=1, url="https://x.com", status_code=code, content_markdown="ok")

    def test_requires_content_or_error(self):
        with pytest.raises(ValidationError):
            Snapshot(company_id=1, url="https://x.com", status_code=200)

    def test_error_message_sufficient(self):
        s = Snapshot(company_id=1, url="https://x.com", error_message="Not found")
        assert s.error_message == "Not found"

    def test_checksum_md5(self):
        content = "Hello World"
        expected = hashlib.md5(content.encode()).hexdigest()
        assert Snapshot.compute_checksum(content) == expected

    @pytest.mark.parametrize("checksum,valid", [
        ("d41d8cd98f00b204e9800998ecf8427e", True),
        ("D41D8CD98F00B204E9800998ECF8427E", True),
        ("not-a-valid-checksum", False),
        ("d41d8cd98f00b204", False),
    ])
    def test_checksum_format(self, checksum, valid):
        if valid:
            s = Snapshot(
                company_id=1, url="https://x.com",
                content_markdown="ok", content_checksum=checksum
            )
            assert s.content_checksum == checksum.lower()
        else:
            with pytest.raises(ValidationError):
                Snapshot(
                    company_id=1, url="https://x.com",
                    content_markdown="ok", content_checksum=checksum
                )

    def test_future_captured_at_rejected(self):
        with pytest.raises(ValidationError):
            Snapshot(
                company_id=1, url="https://x.com",
                content_markdown="ok",
                captured_at=datetime.now(timezone.utc) + timedelta(days=1),
            )


class TestChangeRecord:
    def test_magnitude_enum(self):
        for val in ["minor", "moderate", "major"]:
            assert ChangeMagnitude(val)

    def test_significance_enum(self):
        for val in ["significant", "insignificant", "uncertain"]:
            assert SignificanceClassification(val)

    def test_sentiment_enum(self):
        for val in ["positive", "negative", "neutral", "mixed"]:
            assert SignificanceSentiment(val)

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            ChangeRecord(
                company_id=1, has_changed=True,
                change_magnitude=ChangeMagnitude.minor,
                significance_confidence=1.5,
            )


class TestCompanyStatus:
    def test_status_enum(self):
        for val in ["operational", "likely_closed", "uncertain"]:
            assert CompanyStatusType(val)

    def test_indicator_signal_enum(self):
        for val in ["positive", "negative", "neutral"]:
            assert IndicatorSignal(val)


class TestSocialMediaLink:
    def test_platform_enum_has_13(self):
        expected = {
            "linkedin", "twitter", "youtube", "bluesky", "facebook",
            "instagram", "github", "tiktok", "medium", "mastodon",
            "threads", "pinterest", "blog",
        }
        assert {p.value for p in Platform} == expected

    def test_discovery_method_enum(self):
        for val in ["page_footer", "page_header", "page_content", "full_site_crawl"]:
            assert DiscoveryMethod(val)

    def test_verification_status_enum(self):
        for val in ["logo_matched", "unverified", "manually_reviewed", "flagged"]:
            assert VerificationStatus(val)

    def test_account_type_enum(self):
        for val in ["company", "personal", "unknown"]:
            assert AccountType(val)

    def test_similarity_score_range(self):
        with pytest.raises(ValidationError):
            SocialMediaLink(company_id=1, platform=Platform.twitter,
                          profile_url="https://twitter.com/x", similarity_score=1.5)


class TestNewsArticle:
    def test_source_extracted_from_url(self):
        a = NewsArticle(
            company_id=1,
            title="Test Article",
            content_url="https://techcrunch.com/2026/01/acme-funding",
        )
        assert a.source == "techcrunch.com"

    def test_invalid_url_rejected(self):
        with pytest.raises(ValidationError):
            NewsArticle(company_id=1, title="Test", content_url="not-a-url")

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            NewsArticle(company_id=1, title="", content_url="https://example.com/article")

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            NewsArticle(company_id=1, title="Test",
                       content_url="https://example.com/a", match_confidence=1.5)


class TestCompanyLeadership:
    def test_personal_profile_required(self):
        with pytest.raises(ValidationError):
            CompanyLeadership(
                company_id=1, person_name="Jane",
                title="CEO",
                linkedin_profile_url="https://linkedin.com/company/acme",
            )

    def test_valid_personal_profile(self):
        l = CompanyLeadership(
            company_id=1, person_name="Jane",
            title="CEO",
            linkedin_profile_url="https://linkedin.com/in/jane",
        )
        assert l.linkedin_profile_url == "https://linkedin.com/in/jane"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            CompanyLeadership(company_id=1, person_name="", title="CEO")

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            CompanyLeadership(company_id=1, person_name="Jane", title="")

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            CompanyLeadership(company_id=1, person_name="Jane", title="CEO", confidence=1.5)

    def test_discovery_method_enum(self):
        for val in ["playwright_scrape", "kagi_search"]:
            assert LeadershipDiscoveryMethod(val)

    def test_change_type_enum(self):
        expected = {
            "ceo_departure", "founder_departure", "cto_departure", "coo_departure",
            "executive_departure", "new_ceo", "new_leadership", "no_change",
        }
        assert {ct.value for ct in LeadershipChangeType} == expected


class TestProcessingError:
    def test_pascal_case_required(self):
        pe = ProcessingError(
            entity_type="company", entity_id=42,
            error_type="FirecrawlTimeout",
            error_message="Request timed out after 30s",
            retry_count=2,
        )
        assert pe.error_type == "FirecrawlTimeout"

    def test_non_pascal_case_rejected(self):
        with pytest.raises(ValidationError):
            ProcessingError(
                entity_type="company", entity_id=42,
                error_type="not_pascal_case",
                error_message="some error",
            )
