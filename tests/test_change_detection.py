"""Tests for change detection and status analysis."""

import pytest

from valuation_tool.models import ChangeMagnitude
from valuation_tool.services.change_detection import (
    _calculate_magnitude,
    detect_acquisition,
    extract_copyright_year,
)


class TestChangeMagnitude:
    @pytest.mark.parametrize("similarity,expected", [
        (0.95, ChangeMagnitude.minor),
        (0.90, ChangeMagnitude.minor),
        (0.75, ChangeMagnitude.moderate),
        (0.50, ChangeMagnitude.moderate),
        (0.30, ChangeMagnitude.major),
        (0.10, ChangeMagnitude.major),
        (0.00, ChangeMagnitude.major),
    ])
    def test_magnitude_thresholds(self, similarity, expected):
        assert _calculate_magnitude(similarity) == expected

    def test_boundary_090_is_minor(self):
        assert _calculate_magnitude(0.90) == ChangeMagnitude.minor

    def test_boundary_089_is_moderate(self):
        assert _calculate_magnitude(0.89) == ChangeMagnitude.moderate

    def test_boundary_049_is_major(self):
        assert _calculate_magnitude(0.49) == ChangeMagnitude.major


class TestCopyrightYearExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("(c) 2025 Company Name", 2025),
        ("(C) 2026 Company Name", 2026),
        ("Copyright 2025 Company Name", 2025),
        ("&copy; 2024-2026 Company Name", 2026),
        ("All content copyright 2025", 2025),
    ])
    def test_extract_copyright_year(self, text, expected):
        assert extract_copyright_year(text) == expected

    def test_no_copyright_marker(self):
        """Do not match bare years without copyright marker."""
        assert extract_copyright_year("Founded in 2020") is None


class TestAcquisitionDetection:
    @pytest.mark.parametrize("text,detected", [
        ("acquired by BigTech", True),
        ("merged with Partner Corp", True),
        ("sold to Buyer Inc", True),
        ("now part of Parent Co", True),
        ("is now a subsidiary of Parent Co", True),
        ("is now a division of Parent Co", True),
        ("Product X is now available", False),
        ("We acquired new customers", False),
    ])
    def test_acquisition_detection(self, text, detected):
        assert detect_acquisition(text) == detected
