"""Pydantic models for all domain entities.

Covers: companies, snapshots, change records, company statuses,
social media links, blog links, news articles, company logos,
company leadership, and processing errors.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChangeMagnitude(str, Enum):
    minor = "minor"
    moderate = "moderate"
    major = "major"


class SignificanceClassification(str, Enum):
    significant = "significant"
    insignificant = "insignificant"
    uncertain = "uncertain"


class SignificanceSentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    mixed = "mixed"


class CompanyStatusType(str, Enum):
    operational = "operational"
    likely_closed = "likely_closed"
    uncertain = "uncertain"


class IndicatorSignal(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Platform(str, Enum):
    linkedin = "linkedin"
    twitter = "twitter"
    youtube = "youtube"
    bluesky = "bluesky"
    facebook = "facebook"
    instagram = "instagram"
    github = "github"
    tiktok = "tiktok"
    medium = "medium"
    mastodon = "mastodon"
    threads = "threads"
    pinterest = "pinterest"
    blog = "blog"


class DiscoveryMethod(str, Enum):
    page_footer = "page_footer"
    page_header = "page_header"
    page_content = "page_content"
    full_site_crawl = "full_site_crawl"


class VerificationStatus(str, Enum):
    logo_matched = "logo_matched"
    unverified = "unverified"
    manually_reviewed = "manually_reviewed"
    flagged = "flagged"


class AccountType(str, Enum):
    company = "company"
    personal = "personal"
    unknown = "unknown"


class LeadershipDiscoveryMethod(str, Enum):
    playwright_scrape = "playwright_scrape"
    kagi_search = "kagi_search"


class LeadershipChangeType(str, Enum):
    ceo_departure = "ceo_departure"
    founder_departure = "founder_departure"
    cto_departure = "cto_departure"
    coo_departure = "coo_departure"
    executive_departure = "executive_departure"
    new_ceo = "new_ceo"
    new_leadership = "new_leadership"
    no_change = "no_change"


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

class Company(BaseModel):
    id: int | None = None
    name: str = Field(..., min_length=1)
    homepage_url: str | None = None
    source_sheet: str = "Online Presence"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        return " ".join(v.split()).title()


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot(BaseModel):
    id: int | None = None
    company_id: int
    url: str
    status_code: int | None = None
    content_markdown: str | None = None
    content_html: str | None = None
    content_checksum: str | None = None
    error_message: str | None = None
    has_paywall: bool = False
    has_auth_required: bool = False
    http_last_modified: str | None = None
    captured_at: datetime = Field(default_factory=utcnow)

    @field_validator("status_code")
    @classmethod
    def _valid_status_code(cls, v: int | None) -> int | None:
        if v is not None and (v < 100 or v > 599):
            raise ValueError("status_code must be between 100 and 599")
        return v

    @field_validator("content_checksum")
    @classmethod
    def _valid_checksum(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.lower()
            if not re.match(r"^[0-9a-f]{32}$", v):
                raise ValueError("content_checksum must be a 32-char lowercase hex MD5")
        return v

    @field_validator("captured_at")
    @classmethod
    def _not_future(cls, v: datetime) -> datetime:
        now = datetime.now(timezone.utc)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v > now:
            raise ValueError("captured_at must not be in the future")
        return v

    @model_validator(mode="after")
    def _require_content_or_error(self) -> Snapshot:
        if not self.content_markdown and not self.content_html and not self.error_message:
            raise ValueError("Snapshot must have content_markdown, content_html, or error_message")
        return self

    @staticmethod
    def compute_checksum(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Change Record
# ---------------------------------------------------------------------------

class ChangeRecord(BaseModel):
    id: int | None = None
    company_id: int
    snapshot_id_old: int | None = None
    snapshot_id_new: int | None = None
    has_changed: bool
    change_magnitude: ChangeMagnitude = ChangeMagnitude.minor
    checksum_old: str | None = None
    checksum_new: str | None = None
    diff_summary: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    matched_categories: list[str] = Field(default_factory=list)
    significance_classification: SignificanceClassification | None = None
    significance_sentiment: SignificanceSentiment | None = None
    significance_confidence: float | None = None
    significance_notes: str | None = None
    detected_at: datetime = Field(default_factory=utcnow)

    @field_validator("significance_confidence")
    @classmethod
    def _valid_confidence(cls, v: float | None) -> float | None:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("significance_confidence must be between 0 and 1")
        return v


# ---------------------------------------------------------------------------
# Company Status & Indicators
# ---------------------------------------------------------------------------

class StatusIndicator(BaseModel):
    type: str
    value: str | None = None
    signal: IndicatorSignal


class CompanyStatus(BaseModel):
    id: int | None = None
    company_id: int
    status: CompanyStatusType
    confidence: float = Field(ge=0, le=1)
    indicators: list[StatusIndicator] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Social Media Link
# ---------------------------------------------------------------------------

class SocialMediaLink(BaseModel):
    id: int | None = None
    company_id: int
    platform: Platform
    profile_url: str = Field(..., min_length=1)
    discovery_method: DiscoveryMethod = DiscoveryMethod.page_footer
    html_location: str | None = None
    verification_status: VerificationStatus = VerificationStatus.unverified
    account_type: AccountType = AccountType.unknown
    account_confidence: float = Field(default=0.5, ge=0, le=1)
    similarity_score: float | None = Field(default=None, ge=0, le=1)
    discovered_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Blog Link
# ---------------------------------------------------------------------------

class BlogLink(BaseModel):
    id: int | None = None
    company_id: int
    blog_url: str = Field(..., min_length=1)
    blog_type: str = "company_blog"
    is_active: bool = True
    discovery_method: str = "page_footer"
    discovered_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# News Article
# ---------------------------------------------------------------------------

class NewsArticle(BaseModel):
    id: int | None = None
    company_id: int
    title: str = Field(..., min_length=1, max_length=500)
    content_url: str
    source: str | None = None
    snippet: str | None = None
    published_at: datetime = Field(default_factory=utcnow)
    match_confidence: float = Field(default=0.0, ge=0, le=1)
    match_evidence: list[str] = Field(default_factory=list)
    significance_classification: SignificanceClassification | None = None
    significance_sentiment: SignificanceSentiment | None = None
    significance_confidence: float | None = None
    discovered_at: datetime = Field(default_factory=utcnow)

    @field_validator("content_url")
    @classmethod
    def _valid_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("content_url must be a valid URL")
        return v

    @model_validator(mode="after")
    def _extract_source(self) -> NewsArticle:
        if not self.source and self.content_url:
            parsed = urlparse(self.content_url)
            self.source = parsed.netloc.removeprefix("www.")
        return self


# ---------------------------------------------------------------------------
# Company Logo
# ---------------------------------------------------------------------------

class CompanyLogo(BaseModel):
    id: int | None = None
    company_id: int
    image_url: str | None = None
    image_format: str | None = None
    perceptual_hash: str | None = None
    extraction_location: str | None = None
    extracted_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Company Leadership
# ---------------------------------------------------------------------------

class CompanyLeadership(BaseModel):
    id: int | None = None
    company_id: int
    person_name: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    linkedin_profile_url: str | None = None
    discovery_method: LeadershipDiscoveryMethod = LeadershipDiscoveryMethod.playwright_scrape
    confidence: float = Field(default=0.8, ge=0, le=1)
    is_current: bool = True
    discovered_at: datetime = Field(default_factory=utcnow)

    @field_validator("linkedin_profile_url")
    @classmethod
    def _must_be_personal_profile(cls, v: str | None) -> str | None:
        if v is not None and "/in/" not in v:
            raise ValueError("linkedin_profile_url must be a personal profile containing '/in/'")
        return v


# ---------------------------------------------------------------------------
# Processing Error
# ---------------------------------------------------------------------------

class ProcessingError(BaseModel):
    id: int | None = None
    entity_type: str
    entity_id: int | None = None
    error_type: str = Field(..., min_length=1, max_length=100)
    error_message: str = Field(..., min_length=1, max_length=5000)
    retry_count: int = 0
    occurred_at: datetime = Field(default_factory=utcnow)

    @field_validator("error_type")
    @classmethod
    def _pascal_case(cls, v: str) -> str:
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", v):
            raise ValueError("error_type must be in PascalCase format")
        return v


# ---------------------------------------------------------------------------
# Keyword Match (used by significance analysis)
# ---------------------------------------------------------------------------

class KeywordMatch(BaseModel):
    keyword: str
    position: int = 0
    context_before: str = ""
    context_after: str = ""
    category: str = ""
    is_negated: bool = False
    is_false_positive: bool = False


# ---------------------------------------------------------------------------
# Significance Result
# ---------------------------------------------------------------------------

class SignificanceResult(BaseModel):
    classification: SignificanceClassification
    sentiment: SignificanceSentiment = SignificanceSentiment.neutral
    confidence: float = Field(ge=0, le=1)
    matched_keywords: list[str] = Field(default_factory=list)
    matched_categories: list[str] = Field(default_factory=list)
    notes: str | None = None


# ---------------------------------------------------------------------------
# LLM Validation Result
# ---------------------------------------------------------------------------

class LLMValidationResult(BaseModel):
    classification: SignificanceClassification
    sentiment: SignificanceSentiment
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""
    validated_keywords: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Leadership Change
# ---------------------------------------------------------------------------

class LeadershipChange(BaseModel):
    change_type: LeadershipChangeType
    person_name: str
    title: str
    severity: str = "notable"
    confidence: float = Field(default=0.8, ge=0, le=1)
    significance_classification: SignificanceClassification | None = None
    significance_sentiment: SignificanceSentiment | None = None
    significance_confidence: float | None = None


# ---------------------------------------------------------------------------
# Extraction / Command Results
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """Generic result for batch operations."""
    processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)


class LeadershipExtractionResult(BaseModel):
    company_id: int
    company_name: str
    leaders_found: int = 0
    method_used: str = ""
    leadership_changes: list[LeadershipChange] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class NewsSearchResult(BaseModel):
    company_id: int
    company_name: str
    articles_found: int = 0
    articles_stored: int = 0
    errors: list[str] = Field(default_factory=list)
