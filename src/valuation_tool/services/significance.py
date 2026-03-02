"""Significance analysis engine.

Classifies business significance of website changes and news articles
using keyword detection, negation handling, false positive detection,
and optional LLM validation.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from valuation_tool.models import (
    ChangeMagnitude,
    KeywordMatch,
    LLMValidationResult,
    SignificanceClassification,
    SignificanceResult,
    SignificanceSentiment,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Keyword Dictionaries
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: dict[str, list[str]] = {
    "funding_investment": [
        "series a", "series b", "series c", "series d", "series e",
        "seed round", "funding round", "raised", "funding",
        "venture capital", "valuation", "investment round",
        "pre-seed", "angel round",
    ],
    "product_launch": [
        "launched", "launch", "beta release", "general availability",
        "new product", "product release", "now available", "go live",
    ],
    "growth_success": [
        "revenue growth", "milestone", "profitable", "record revenue",
        "year over year growth", "customer growth", "user growth",
    ],
    "partnerships": [
        "partnership", "joint venture", "strategic alliance",
        "collaboration", "partner",
    ],
    "expansion": [
        "new office", "hiring", "expansion", "international expansion",
        "new market", "headcount growth", "opening new",
    ],
    "recognition": [
        "innovation award", "top 10", "top 50", "best places to work",
        "award", "recognized",
    ],
    "ipo_exit": [
        "filed s-1", "nasdaq", "nyse", "ipo", "public offering",
        "going public", "spac",
    ],
}

NEGATIVE_KEYWORDS: dict[str, list[str]] = {
    "closure": [
        "shut down", "shutdown", "ceased operations", "winding down",
        "closing", "closed", "dissolved",
    ],
    "layoffs_downsizing": [
        "layoffs", "laid off", "lays off", "workforce reduction",
        "job cuts", "downsizing", "restructuring", "rif",
    ],
    "financial_distress": [
        "chapter 11", "chapter 7", "bankruptcy", "insolvent",
        "cash crunch", "default", "debt crisis",
    ],
    "legal_issues": [
        "lawsuit", "investigation", "settlement", "sued",
        "regulatory action", "fine", "penalty",
    ],
    "security_breach": [
        "data breach", "hacked", "ransomware", "cyber attack",
        "security incident", "compromised",
    ],
    "acquisition": [
        "acquired by", "merged with", "sold to",
        "acquisition", "takeover",
    ],
    "leadership_changes": [
        "ceo resigned", "ceo departed", "founder left", "cto left",
        "stepping down", "leadership transition", "new ceo",
        "retired", "resigned",
    ],
    "product_failures": [
        "product recall", "discontinued", "sunset",
        "end of life", "deprecate",
    ],
    "market_exit": [
        "exiting market", "market withdrawal", "divesting",
        "pulling out of",
    ],
}

INSIGNIFICANT_PATTERNS: dict[str, list[str]] = {
    "css_styling": [
        "font-family", "font-size", "background-color", "color:",
        "margin:", "padding:", "border:", "text-align",
    ],
    "copyright_year": [
        "copyright", "all rights reserved", "©",
    ],
    "tracking_analytics": [
        "google-analytics", "gtag", "tracking code", "analytics",
        "pixel", "facebook pixel",
    ],
}

# False positive patterns for acquisition keywords
FALSE_POSITIVE_PATTERNS: list[str] = [
    "talent acquisition",
    "customer acquisition",
    "data acquisition",
    "user acquisition",
    "funding opportunities",
    "funding sources",
    "self-funded",
]

NEGATION_WORDS = {"no", "not", "never", "without", "lacks"}


# ---------------------------------------------------------------------------
# Keyword Detection
# ---------------------------------------------------------------------------

def _find_keywords(content: str, keyword_dict: dict[str, list[str]]) -> list[KeywordMatch]:
    """Find all keyword matches in content with context."""
    content_lower = content.lower()
    matches: list[KeywordMatch] = []

    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            idx = 0
            while True:
                pos = content_lower.find(keyword, idx)
                if pos == -1:
                    break
                start = max(0, pos - 50)
                end = min(len(content), pos + len(keyword) + 50)

                match = KeywordMatch(
                    keyword=keyword,
                    position=pos,
                    context_before=content[start:pos],
                    context_after=content[pos + len(keyword):end],
                    category=category,
                    is_negated=_is_negated(content_lower, pos),
                    is_false_positive=_is_false_positive(content_lower, pos, keyword),
                )
                matches.append(match)
                idx = pos + len(keyword)

    return matches


def _is_negated(content_lower: str, keyword_pos: int) -> bool:
    """Check if keyword is preceded by a negation word within ~20 chars."""
    prefix = content_lower[max(0, keyword_pos - 20):keyword_pos].strip()
    words = prefix.split()
    for word in words:
        if word in NEGATION_WORDS:
            return True
    return False


def _is_false_positive(content_lower: str, keyword_pos: int, keyword: str) -> bool:
    """Check if keyword match is a known false positive."""
    # Get a wider context window
    start = max(0, keyword_pos - 30)
    end = min(len(content_lower), keyword_pos + len(keyword) + 30)
    context = content_lower[start:end]

    for fp_pattern in FALSE_POSITIVE_PATTERNS:
        if fp_pattern in context:
            return True
    return False


def _find_insignificant_patterns(content: str) -> list[KeywordMatch]:
    """Find insignificant pattern matches in content."""
    content_lower = content.lower()
    matches: list[KeywordMatch] = []

    for category, patterns in INSIGNIFICANT_PATTERNS.items():
        for pattern in patterns:
            if pattern in content_lower:
                matches.append(KeywordMatch(
                    keyword=pattern,
                    category=category,
                ))

    return matches


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_sentiment(
    positive_count: int, negative_count: int
) -> SignificanceSentiment:
    """Determine sentiment from keyword polarity counts."""
    if positive_count >= 2 and negative_count >= 2:
        return SignificanceSentiment.mixed
    if positive_count >= 2:
        return SignificanceSentiment.positive
    if negative_count >= 2:
        return SignificanceSentiment.negative
    return SignificanceSentiment.neutral


def analyze_significance(
    content: str,
    magnitude: ChangeMagnitude = ChangeMagnitude.minor,
) -> SignificanceResult:
    """Perform keyword-based significance analysis on content.

    Classification rules:
    1. Only insignificant patterns + minor magnitude → insignificant (0.85)
    2. 2+ negative keywords → significant (0.80-0.95)
    3. 2+ positive keywords → significant (0.80-0.90)
    4. 1 keyword + major magnitude → significant (0.70)
    5. 1 keyword + minor magnitude → uncertain (0.50)
    6. No keywords → insignificant (0.75)
    """
    positive_matches = _find_keywords(content, POSITIVE_KEYWORDS)
    negative_matches = _find_keywords(content, NEGATIVE_KEYWORDS)
    insignificant_matches = _find_insignificant_patterns(content)

    # Filter out negated and false positive matches for counting
    effective_positive = [m for m in positive_matches if not m.is_negated and not m.is_false_positive]
    effective_negative = [m for m in negative_matches if not m.is_negated and not m.is_false_positive]

    pos_count = len(effective_positive)
    neg_count = len(effective_negative)
    total_keyword_count = pos_count + neg_count

    all_keywords = [m.keyword for m in effective_positive + effective_negative]
    all_categories = list({m.category for m in effective_positive + effective_negative})

    # Calculate confidence adjustments for negated/false-positive matches
    negated_count = sum(1 for m in positive_matches + negative_matches if m.is_negated)
    fp_count = sum(1 for m in positive_matches + negative_matches if m.is_false_positive)

    sentiment = _classify_sentiment(pos_count, neg_count)

    # Rule 1: Only insignificant patterns with minor magnitude
    if total_keyword_count == 0 and insignificant_matches and magnitude == ChangeMagnitude.minor:
        return SignificanceResult(
            classification=SignificanceClassification.insignificant,
            sentiment=SignificanceSentiment.neutral,
            confidence=0.85,
            matched_keywords=[m.keyword for m in insignificant_matches],
            matched_categories=[m.category for m in insignificant_matches],
        )

    # Rule 2: 2+ negative keywords
    if neg_count >= 2:
        confidence = min(0.80 + 0.05 * (neg_count - 2), 0.95)
        # Reduce confidence for negated/fp matches
        confidence -= 0.20 * negated_count + 0.30 * fp_count
        confidence = max(0.0, min(1.0, confidence))
        return SignificanceResult(
            classification=SignificanceClassification.significant,
            sentiment=sentiment,
            confidence=confidence,
            matched_keywords=all_keywords,
            matched_categories=all_categories,
        )

    # Rule 3: 2+ positive keywords
    if pos_count >= 2:
        confidence = min(0.80 + 0.03 * (pos_count - 2), 0.90)
        confidence -= 0.20 * negated_count + 0.30 * fp_count
        confidence = max(0.0, min(1.0, confidence))
        return SignificanceResult(
            classification=SignificanceClassification.significant,
            sentiment=sentiment,
            confidence=confidence,
            matched_keywords=all_keywords,
            matched_categories=all_categories,
        )

    # Rule 4: 1 keyword + major magnitude
    if total_keyword_count >= 1 and magnitude == ChangeMagnitude.major:
        confidence = 0.70
        confidence -= 0.20 * negated_count + 0.30 * fp_count
        confidence = max(0.0, min(1.0, confidence))
        return SignificanceResult(
            classification=SignificanceClassification.significant,
            sentiment=sentiment,
            confidence=confidence,
            matched_keywords=all_keywords,
            matched_categories=all_categories,
        )

    # Rule 5: 1 keyword + minor/moderate magnitude
    if total_keyword_count >= 1:
        confidence = 0.50
        confidence -= 0.20 * negated_count + 0.30 * fp_count
        confidence = max(0.0, min(1.0, confidence))
        return SignificanceResult(
            classification=SignificanceClassification.uncertain,
            sentiment=sentiment,
            confidence=confidence,
            matched_keywords=all_keywords,
            matched_categories=all_categories,
        )

    # Rule 6: No keywords
    return SignificanceResult(
        classification=SignificanceClassification.insignificant,
        sentiment=SignificanceSentiment.neutral,
        confidence=0.75,
        matched_keywords=[],
        matched_categories=[],
    )


# ---------------------------------------------------------------------------
# LLM Validation (Optional)
# ---------------------------------------------------------------------------

async def validate_with_llm(
    content: str,
    keyword_result: SignificanceResult,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> LLMValidationResult | None:
    """Use LLM to validate keyword-based classification.

    Returns None on failure (keyword classification used as fallback).
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic_not_installed")
        return None

    prompt = f"""Analyze the following website content change for business significance.

Content excerpt:
{content[:2000]}

Detected keywords: {', '.join(keyword_result.matched_keywords)}
Keyword-based classification: {keyword_result.classification.value}
Keyword-based sentiment: {keyword_result.sentiment.value}

Respond in JSON format:
{{
    "classification": "significant" | "insignificant" | "uncertain",
    "sentiment": "positive" | "negative" | "neutral" | "mixed",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "validated_keywords": ["keywords that are genuine signals"],
    "false_positives": ["keywords that are false positives"]
}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.content[0].text
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return LLMValidationResult(**data)
    except Exception as exc:
        logger.warning("llm_validation_failed", error=str(exc))

    return None
