"""Tests for leadership extraction service."""

import pytest

from valuation_tool.models import (
    CompanyLeadership,
    LeadershipChangeType,
    LeadershipDiscoveryMethod,
)
from valuation_tool.services.leadership import (
    detect_leadership_changes,
    extract_leadership_title,
    is_leadership_title,
    normalize_title,
    rank_title,
)


class TestTitleDetection:
    @pytest.mark.parametrize("title,expected", [
        ("CEO", True),
        ("Chief Executive Officer", True),
        ("Founder", True),
        ("Co-Founder", True),
        ("Cofounder", True),
        ("CTO", True),
        ("Chief Technology Officer", True),
        ("COO", True),
        ("Chief Operating Officer", True),
        ("CFO", True),
        ("Chief Financial Officer", True),
        ("CMO", True),
        ("Chief Marketing Officer", True),
        ("Chief People Officer", True),
        ("Chief Product Officer", True),
        ("CRO", True),
        ("CSO", True),
        ("President", True),
        ("Managing Director", True),
        ("General Manager", True),
        ("VP of Engineering", True),
        ("Vice President", True),
        ("Software Engineer", False),
        ("Product Manager", False),
        ("Data Scientist", False),
        ("Marketing Manager", False),
        ("Sales Representative", False),
    ])
    def test_is_leadership_title(self, title, expected):
        assert is_leadership_title(title) == expected

    @pytest.mark.parametrize("title", ["ceo", "CEO", "Ceo", "CHIEF EXECUTIVE OFFICER"])
    def test_case_insensitive(self, title):
        assert is_leadership_title(title) is True

    def test_embedded_title(self):
        assert is_leadership_title("CEO at Acme Corp") is True
        extracted = extract_leadership_title("CEO at Acme Corp")
        assert extracted == "CEO"

    @pytest.mark.parametrize("title", [
        "Chief Revenue Officer",
        "Chief Strategy Officer",
        "Chief Data Officer",
        "Chief Information Officer",
    ])
    def test_generic_chief_pattern(self, title):
        assert is_leadership_title(title) is True

    @pytest.mark.parametrize("title", [
        "VP of Engineering",
        "VP Engineering",
        "VP of Product",
        "VP Product",
        "Vice President of Engineering",
    ])
    def test_vp_patterns(self, title):
        assert is_leadership_title(title) is True


class TestTitleNormalization:
    @pytest.mark.parametrize("input_title,normalized", [
        ("Chief Executive Officer", "CEO"),
        ("Chief Technology Officer", "CTO"),
        ("Chief Operating Officer", "COO"),
        ("Chief Financial Officer", "CFO"),
        ("Cofounder", "Co-Founder"),
        ("co founder", "Co-Founder"),
        ("Co-founder", "Co-Founder"),
    ])
    def test_normalize(self, input_title, normalized):
        assert normalize_title(input_title) == normalized


class TestTitleRanking:
    @pytest.mark.parametrize("title_a,title_b,comparison", [
        ("CEO", "CTO", "higher"),
        ("Founder", "Co-Founder", "higher"),
        ("Co-Founder", "CTO", "higher"),
        ("CTO", "VP of Engineering", "higher"),
        ("CEO", "CEO", "equal"),
    ])
    def test_ranking(self, title_a, title_b, comparison):
        rank_a = rank_title(title_a)
        rank_b = rank_title(title_b)
        if comparison == "higher":
            assert rank_a < rank_b  # Lower rank = more senior
        elif comparison == "equal":
            assert rank_a == rank_b


class TestLeadershipChangeDetection:
    def _make_leader(self, name, title, url):
        return CompanyLeadership(
            company_id=1, person_name=name, title=title,
            linkedin_profile_url=url,
        )

    def test_ceo_departure(self):
        previous = [self._make_leader("Jane Smith", "CEO", "https://linkedin.com/in/jane-smith")]
        current = []
        changes = detect_leadership_changes(previous, current)
        assert any(c.change_type == LeadershipChangeType.ceo_departure for c in changes)
        assert any(c.severity == "critical" for c in changes)
        assert any(c.confidence == 0.95 for c in changes)

    def test_founder_departure(self):
        previous = [self._make_leader("Bob", "Founder", "https://linkedin.com/in/bob")]
        current = []
        changes = detect_leadership_changes(previous, current)
        assert any(c.change_type == LeadershipChangeType.founder_departure for c in changes)

    def test_new_ceo_arrival(self):
        previous = []
        current = [{"name": "Bob Wilson", "title": "CEO", "profile_url": "https://linkedin.com/in/bob"}]
        changes = detect_leadership_changes(previous, current)
        assert any(c.change_type == LeadershipChangeType.new_ceo for c in changes)

    def test_new_leadership(self):
        previous = []
        current = [{"name": "Alice Chen", "title": "CTO", "profile_url": "https://linkedin.com/in/alice"}]
        changes = detect_leadership_changes(previous, current)
        assert any(c.change_type == LeadershipChangeType.new_leadership for c in changes)

    def test_no_change(self):
        leader = self._make_leader("Jane", "CEO", "https://linkedin.com/in/jane")
        previous = [leader]
        current = [{"name": "Jane", "title": "CEO", "profile_url": "https://linkedin.com/in/jane"}]
        changes = detect_leadership_changes(previous, current)
        assert any(c.change_type == LeadershipChangeType.no_change for c in changes)
        assert any(c.confidence == 0.75 for c in changes)

    def test_profile_url_matching(self):
        """Match leaders by LinkedIn URL, not name."""
        previous = [self._make_leader("Jane Smith", "CEO", "https://linkedin.com/in/jane-smith")]
        current = [{"name": "Jane A Smith", "title": "CEO", "profile_url": "https://linkedin.com/in/jane-smith"}]
        changes = detect_leadership_changes(previous, current)
        # Should be no change since URL matches
        departures = [c for c in changes if "departure" in c.change_type.value]
        assert len(departures) == 0

    def test_mixed_departures_and_arrivals(self):
        previous = [self._make_leader("Old CEO", "CEO", "https://linkedin.com/in/old-ceo")]
        current = [{"name": "New CEO", "title": "CEO", "profile_url": "https://linkedin.com/in/new-ceo"}]
        changes = detect_leadership_changes(previous, current)
        types = {c.change_type for c in changes}
        assert LeadershipChangeType.ceo_departure in types
        assert LeadershipChangeType.new_ceo in types


class TestLeadershipEnums:
    def test_discovery_methods(self):
        assert LeadershipDiscoveryMethod.playwright_scrape.value == "playwright_scrape"
        assert LeadershipDiscoveryMethod.kagi_search.value == "kagi_search"

    def test_change_types(self):
        expected = {
            "ceo_departure", "founder_departure", "cto_departure", "coo_departure",
            "executive_departure", "new_ceo", "new_leadership", "no_change",
        }
        assert {ct.value for ct in LeadershipChangeType} == expected
