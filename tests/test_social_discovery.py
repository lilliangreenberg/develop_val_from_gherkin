"""Tests for social media discovery service."""

import pytest

from valuation_tool.models import Platform
from valuation_tool.services.social_discovery import (
    classify_account,
    detect_blog,
    detect_platform,
    normalize_blog_url,
    normalize_social_url,
)


class TestPlatformDetection:
    @pytest.mark.parametrize("url,expected", [
        ("https://linkedin.com/company/acme", Platform.linkedin),
        ("https://www.linkedin.com/in/john-doe", Platform.linkedin),
        ("https://twitter.com/acmecorp", Platform.twitter),
        ("https://x.com/acmecorp", Platform.twitter),
        ("https://youtube.com/@acme", Platform.youtube),
        ("https://youtube.com/channel/UC123abc", Platform.youtube),
        ("https://youtube.com/c/AcmeCorp", Platform.youtube),
        ("https://bsky.app/profile/acme.bsky.social", Platform.bluesky),
        ("https://facebook.com/acmecorp", Platform.facebook),
        ("https://fb.com/acmecorp", Platform.facebook),
        ("https://m.facebook.com/acmecorp", Platform.facebook),
        ("https://instagram.com/acmecorp", Platform.instagram),
        ("https://github.com/acme", Platform.github),
        ("https://tiktok.com/@acme", Platform.tiktok),
        ("https://medium.com/@acme", Platform.medium),
        ("https://acme.medium.com", Platform.medium),
        ("https://mastodon.social/@acme", Platform.mastodon),
        ("https://threads.net/@acme", Platform.threads),
        ("https://pinterest.com/acme", Platform.pinterest),
    ])
    def test_detect_platform(self, url, expected):
        assert detect_platform(url) == expected

    @pytest.mark.parametrize("url", [
        "https://www.google.com",
        "https://www.acme.com/about",
        "https://docs.acme.com",
        "mailto:info@acme.com",
    ])
    def test_non_social_rejected(self, url):
        assert detect_platform(url) is None


class TestURLNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("https://github.com/acme/repo-name", "https://github.com/acme"),
        ("https://linkedin.com/company/acme/about/", "https://linkedin.com/company/acme"),
        ("https://twitter.com/acme/", "https://twitter.com/acme"),
        ("https://www.twitter.com/acme", "https://twitter.com/acme"),
        ("https://twitter.com/acme?ref=website", "https://twitter.com/acme"),
    ])
    def test_normalize(self, raw, expected):
        platform = detect_platform(raw)
        assert normalize_social_url(raw, platform) == expected

    def test_github_repo_to_org(self):
        result = normalize_social_url(
            "https://github.com/acme/awesome-project", Platform.github
        )
        assert result == "https://github.com/acme"

    def test_linkedin_trailing_path(self):
        result = normalize_social_url(
            "https://linkedin.com/company/acme/jobs/", Platform.linkedin
        )
        assert result == "https://linkedin.com/company/acme"


class TestBlogDetection:
    @pytest.mark.parametrize("url,expected_type", [
        ("https://blog.acme.com", "company_blog"),
        ("https://www.acme.com/blog", "company_blog"),
        ("https://acme.medium.com", "medium"),
        ("https://acme.substack.com", "substack"),
    ])
    def test_detect_blog(self, url, expected_type):
        assert detect_blog(url) == expected_type

    def test_non_blog(self):
        assert detect_blog("https://www.acme.com/about") is None


class TestBlogNormalization:
    @pytest.mark.parametrize("post_url,hub_url", [
        ("https://blog.acme.com/2024/01/my-post", "https://blog.acme.com"),
        ("https://www.acme.com/blog/category/post", "https://acme.com/blog"),
        ("https://acme.substack.com/p/article-title", "https://acme.substack.com"),
    ])
    def test_normalize_blog_url(self, post_url, hub_url):
        blog_type = detect_blog(post_url)
        assert blog_type is not None
        result = normalize_blog_url(post_url, blog_type)
        assert result == hub_url


class TestAccountClassification:
    def test_linkedin_company(self):
        acct_type, conf = classify_account(
            "https://linkedin.com/company/acme", Platform.linkedin, "Acme Corp", "footer"
        )
        assert acct_type.value == "company"

    def test_linkedin_personal(self):
        acct_type, conf = classify_account(
            "https://linkedin.com/in/john-doe", Platform.linkedin, "Acme Corp", "main"
        )
        assert acct_type.value == "personal"

    def test_name_in_handle(self):
        _, conf_match = classify_account(
            "https://twitter.com/acmecorp", Platform.twitter, "Acme Corp", "footer"
        )
        _, conf_no_match = classify_account(
            "https://twitter.com/randomhandle", Platform.twitter, "Acme Corp", "main"
        )
        assert conf_match > conf_no_match
