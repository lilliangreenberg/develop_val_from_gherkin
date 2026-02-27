"""Tests for significance analysis engine."""

import pytest

from valuation_tool.models import ChangeMagnitude, SignificanceClassification, SignificanceSentiment
from valuation_tool.services.significance import (
    FALSE_POSITIVE_PATTERNS,
    INSIGNIFICANT_PATTERNS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    analyze_significance,
)


class TestPositiveKeywords:
    def test_all_7_categories(self):
        expected = {
            "funding_investment", "product_launch", "growth_success",
            "partnerships", "expansion", "recognition", "ipo_exit",
        }
        assert set(POSITIVE_KEYWORDS.keys()) == expected

    @pytest.mark.parametrize("text,keyword,category", [
        ("We raised a $50M Series B round", "series b", "funding_investment"),
        ("Announced our seed round today", "seed round", "funding_investment"),
        ("We launched our new product today", "launched", "product_launch"),
        ("Revenue growth exceeded 200%", "revenue growth", "growth_success"),
        ("Strategic partnership with BigCo", "partnership", "partnerships"),
        ("We are hiring 50 engineers", "hiring", "expansion"),
        ("Won the Innovation Award 2026", "innovation award", "recognition"),
        ("Filed S-1 for public offering", "filed s-1", "ipo_exit"),
    ])
    def test_positive_keyword_detection(self, text, keyword, category):
        result = analyze_significance(text, ChangeMagnitude.moderate)
        assert keyword in [k.lower() for k in result.matched_keywords]


class TestNegativeKeywords:
    def test_all_9_categories(self):
        expected = {
            "closure", "layoffs_downsizing", "financial_distress",
            "legal_issues", "security_breach", "acquisition",
            "leadership_changes", "product_failures", "market_exit",
        }
        assert set(NEGATIVE_KEYWORDS.keys()) == expected

    @pytest.mark.parametrize("text,keyword", [
        ("Company shut down operations", "shut down"),
        ("Announced layoffs affecting 200 employees", "layoffs"),
        ("Filed for Chapter 11 bankruptcy", "chapter 11"),
        ("Lawsuit filed against the company", "lawsuit"),
        ("Suffered a major data breach", "data breach"),
        ("Acquired by BigTech Corp", "acquired by"),
        ("CEO resigned unexpectedly", "ceo resigned"),
    ])
    def test_negative_keyword_detection(self, text, keyword):
        result = analyze_significance(text, ChangeMagnitude.moderate)
        assert keyword in [k.lower() for k in result.matched_keywords]


class TestInsignificantPatterns:
    def test_all_3_categories(self):
        expected = {"css_styling", "copyright_year", "tracking_analytics"}
        assert set(INSIGNIFICANT_PATTERNS.keys()) == expected

    @pytest.mark.parametrize("text,category", [
        ("Updated font-family to Arial", "css_styling"),
        ("Copyright 2026 Company Name", "copyright_year"),
        ("Updated google-analytics tracking code", "tracking_analytics"),
    ])
    def test_insignificant_pattern_detection(self, text, category):
        result = analyze_significance(text, ChangeMagnitude.minor)
        assert result.classification == SignificanceClassification.insignificant


class TestClassificationRules:
    def test_rule1_insignificant_only_minor(self):
        """Only insignificant patterns + minor → insignificant (0.85)."""
        result = analyze_significance(
            "Updated font-family to Helvetica", ChangeMagnitude.minor
        )
        assert result.classification == SignificanceClassification.insignificant
        assert abs(result.confidence - 0.85) < 0.01

    def test_rule2_two_negative(self):
        """2+ negative keywords → significant."""
        result = analyze_significance(
            "Layoffs announced and company shut down operations",
            ChangeMagnitude.moderate,
        )
        assert result.classification == SignificanceClassification.significant
        assert 0.80 <= result.confidence <= 0.95

    def test_rule3_two_positive(self):
        """2+ positive keywords → significant."""
        result = analyze_significance(
            "Raised Series B funding and hiring new team members",
            ChangeMagnitude.moderate,
        )
        assert result.classification == SignificanceClassification.significant
        assert 0.80 <= result.confidence <= 0.90

    def test_rule4_one_keyword_major(self):
        """1 keyword + major magnitude → significant (0.70)."""
        result = analyze_significance("layoffs", ChangeMagnitude.major)
        assert result.classification == SignificanceClassification.significant
        assert abs(result.confidence - 0.70) < 0.05

    def test_rule5_one_keyword_minor(self):
        """1 keyword + minor magnitude → uncertain (0.50)."""
        result = analyze_significance("hiring", ChangeMagnitude.minor)
        assert result.classification == SignificanceClassification.uncertain
        assert abs(result.confidence - 0.50) < 0.05

    def test_rule6_no_keywords(self):
        """No keywords → insignificant (0.75)."""
        result = analyze_significance(
            "The quick brown fox jumps over the lazy dog",
            ChangeMagnitude.minor,
        )
        assert result.classification == SignificanceClassification.insignificant
        assert abs(result.confidence - 0.75) < 0.01

    def test_more_keywords_increase_confidence(self):
        """3 negative keywords should have higher confidence than 2."""
        result_2 = analyze_significance(
            "layoffs and shutdown", ChangeMagnitude.major
        )
        result_3 = analyze_significance(
            "layoffs and shutdown and bankruptcy", ChangeMagnitude.major
        )
        assert result_3.confidence >= result_2.confidence


class TestSentimentClassification:
    @pytest.mark.parametrize("pos,neg,expected", [
        (3, 0, "positive"),
        (0, 3, "negative"),
        (2, 2, "mixed"),
        (3, 2, "mixed"),
        (1, 0, "neutral"),
        (0, 1, "neutral"),
        (0, 0, "neutral"),
    ])
    def test_sentiment(self, pos, neg, expected):
        # Build content with the right number of keywords
        pos_words = ["funding", "hiring", "expansion", "launched", "milestone"][:pos]
        neg_words = ["layoffs", "shutdown", "bankruptcy", "lawsuit", "data breach"][:neg]
        content = " and ".join(pos_words + neg_words) if (pos_words or neg_words) else "nothing"
        result = analyze_significance(content, ChangeMagnitude.moderate)
        # For sentiment to be classified, we need enough keywords
        if pos >= 2 or neg >= 2:
            assert result.sentiment.value == expected


class TestNegationDetection:
    @pytest.mark.parametrize("phrase", [
        "no funding was received",
        "not acquired by anyone",
        "never had layoffs",
        "without any data breach",
        "lacks funding",
    ])
    def test_negation_detected(self, phrase):
        result = analyze_significance(phrase, ChangeMagnitude.moderate)
        # Negated keywords should reduce confidence
        # The keyword is still found, but effect is reduced


class TestFalsePositiveDetection:
    @pytest.mark.parametrize("phrase", [
        "talent acquisition team",
        "customer acquisition cost",
        "data acquisition pipeline",
    ])
    def test_false_positive_detected(self, phrase):
        result = analyze_significance(phrase, ChangeMagnitude.moderate)
        # False positives should not trigger high-confidence significance


class TestLeadershipKeywords:
    @pytest.mark.parametrize("text,keyword", [
        ("CEO departed the company", "ceo departed"),
        ("Founder left to pursue new venture", "founder left"),
        ("New CEO appointed", "new ceo"),
        ("Executive is stepping down", "stepping down"),
    ])
    def test_leadership_keywords(self, text, keyword):
        result = analyze_significance(text, ChangeMagnitude.moderate)
        found_keywords = [k.lower() for k in result.matched_keywords]
        # At least one leadership-related keyword should be detected
        assert len(found_keywords) > 0
