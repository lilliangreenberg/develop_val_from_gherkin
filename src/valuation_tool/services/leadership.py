"""LinkedIn leadership extraction service.

Primary: Playwright-based LinkedIn People tab scraping.
Fallback: Kagi search for leadership profiles.
Includes title detection, normalization, ranking, and change detection.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import structlog

from valuation_tool.config import Config
from valuation_tool.database import Database
from valuation_tool.models import (
    CompanyLeadership,
    ExtractionResult,
    LeadershipChange,
    LeadershipChangeType,
    LeadershipDiscoveryMethod,
    LeadershipExtractionResult,
    SignificanceClassification,
    SignificanceSentiment,
)
from valuation_tool.services.retry import LinkedInBlockedError

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Title Detection & Normalization
# ---------------------------------------------------------------------------

_TITLE_CANONICALIZATION = {
    "chief executive officer": "CEO",
    "chief technology officer": "CTO",
    "chief operating officer": "COO",
    "chief financial officer": "CFO",
    "chief marketing officer": "CMO",
    "chief people officer": "Chief People Officer",
    "chief product officer": "Chief Product Officer",
    "chief revenue officer": "CRO",
    "chief strategy officer": "CSO",
    "chief data officer": "Chief Data Officer",
    "chief information officer": "Chief Information Officer",
    "cofounder": "Co-Founder",
    "co founder": "Co-Founder",
    "co-founder": "Co-Founder",
}

# Leadership title patterns (order matters for matching)
_LEADERSHIP_PATTERNS = [
    re.compile(r"\b(CEO|CTO|COO|CFO|CMO|CRO|CSO)\b", re.IGNORECASE),
    re.compile(r"\bChief\s+\w+\s+Officer\b", re.IGNORECASE),
    re.compile(r"\b(?:Co-?[Ff]ounder|Cofounder)\b", re.IGNORECASE),
    re.compile(r"\bFounder\b", re.IGNORECASE),
    re.compile(r"\bPresident\b", re.IGNORECASE),
    re.compile(r"\bManaging\s+Director\b", re.IGNORECASE),
    re.compile(r"\bGeneral\s+Manager\b", re.IGNORECASE),
    re.compile(r"\bVP\s+(?:of\s+)?\w+\b", re.IGNORECASE),
    re.compile(r"\bVice\s+President(?:\s+of\s+\w+)?\b", re.IGNORECASE),
]

# Title ranking (lower = more senior)
_TITLE_RANK = {
    "CEO": 1,
    "Founder": 2,
    "Co-Founder": 3,
    "President": 4,
    "CTO": 5,
    "COO": 5,
    "CFO": 5,
    "CMO": 6,
    "CRO": 6,
    "CSO": 6,
    "Managing Director": 7,
    "General Manager": 8,
}


def is_leadership_title(title: str) -> bool:
    """Check if a title indicates a leadership position."""
    for pattern in _LEADERSHIP_PATTERNS:
        if pattern.search(title):
            return True
    return False


def extract_leadership_title(title: str) -> str | None:
    """Extract the leadership title portion from a longer string."""
    for pattern in _LEADERSHIP_PATTERNS:
        match = pattern.search(title)
        if match:
            return match.group(0)
    return None


def normalize_title(title: str) -> str:
    """Normalize a title to its canonical form."""
    title_lower = title.lower().strip()
    if title_lower in _TITLE_CANONICALIZATION:
        return _TITLE_CANONICALIZATION[title_lower]
    # Try extracting the leadership part
    extracted = extract_leadership_title(title)
    if extracted:
        extracted_lower = extracted.lower()
        if extracted_lower in _TITLE_CANONICALIZATION:
            return _TITLE_CANONICALIZATION[extracted_lower]
        return extracted
    return title


def rank_title(title: str) -> int:
    """Get the seniority rank of a title (lower = more senior)."""
    normalized = normalize_title(title)
    # Check direct match
    if normalized in _TITLE_RANK:
        return _TITLE_RANK[normalized]
    # VP-level
    if "VP" in normalized.upper() or "Vice President" in normalized:
        return 9
    # Chief X Officer
    if "Chief" in normalized:
        return 6
    return 99


# ---------------------------------------------------------------------------
# Playwright LinkedIn Extraction
# ---------------------------------------------------------------------------

async def _extract_via_playwright(
    linkedin_url: str,
    headless: bool = False,
    profile_dir: str = "data/linkedin_profile",
    auth_wait_timeout: int = 120,
) -> list[dict[str, str]]:
    """Extract employee cards from LinkedIn People tab using Playwright.

    Returns list of dicts with keys: name, title, profile_url.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise LinkedInBlockedError("Playwright not installed")

    people_url = linkedin_url.rstrip("/") + "/people/"
    employees: list[dict[str, str]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            profile_dir,
            headless=headless,
        )
        page = await browser.new_page()

        try:
            await page.goto(people_url, wait_until="domcontentloaded", timeout=30000)

            # Check for auth wall
            if await _detect_auth_wall(page):
                logger.info("linkedin_auth_wall_detected", url=people_url)
                # Wait for manual login
                try:
                    await page.wait_for_selector(
                        '[data-test-id="org-people"],.org-people',
                        timeout=auth_wait_timeout * 1000,
                    )
                except Exception:
                    raise LinkedInBlockedError("Auth wall - login timeout")

            # Check for CAPTCHA / rate limiting
            if await _detect_captcha(page):
                raise LinkedInBlockedError("CAPTCHA detected")

            if await _detect_rate_limit(page):
                raise LinkedInBlockedError("Rate limit detected")

            # Scroll to load more cards
            for _ in range(3):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)

            # Extract employee cards using multiple selector strategies
            employees = await _extract_employee_cards(page)

        finally:
            await browser.close()

    return employees


async def _detect_auth_wall(page: Any) -> bool:
    """Check if LinkedIn requires login."""
    sign_in = await page.query_selector('a[href*="login"], .sign-in-form, #session_key')
    return sign_in is not None


async def _detect_captcha(page: Any) -> bool:
    """Check for CAPTCHA challenge."""
    captcha = await page.query_selector(
        '#captcha-internal, .captcha-module, [data-captcha]'
    )
    return captcha is not None


async def _detect_rate_limit(page: Any) -> bool:
    """Check for rate limiting."""
    content = await page.content()
    return "too many requests" in content.lower()


async def _extract_employee_cards(page: Any) -> list[dict[str, str]]:
    """Extract employee data from LinkedIn People tab DOM."""
    employees: list[dict[str, str]] = []

    # Primary selectors
    selectors = [
        '.org-people-profile-card',
        '[data-test-id="people-card"]',
        '.artdeco-entity-lockup',
    ]

    for selector in selectors:
        cards = await page.query_selector_all(selector)
        if cards:
            for card in cards:
                name_el = await card.query_selector(
                    '.org-people-profile-card__profile-title, '
                    '.artdeco-entity-lockup__title, '
                    '[data-anonymize="person-name"]'
                )
                title_el = await card.query_selector(
                    '.artdeco-entity-lockup__subtitle, '
                    '.org-people-profile-card__designation'
                )
                link_el = await card.query_selector('a[href*="/in/"]')

                name = (await name_el.inner_text()).strip() if name_el else ""
                title = (await title_el.inner_text()).strip() if title_el else ""
                profile_url = (await link_el.get_attribute("href")) if link_el else ""

                if name and title:
                    employees.append({
                        "name": name,
                        "title": title,
                        "profile_url": profile_url or "",
                    })
            break

    # Fallback: extract all /in/ links
    if not employees:
        links = await page.query_selector_all('a[href*="/in/"]')
        for link in links:
            href = await link.get_attribute("href") or ""
            text = (await link.inner_text()).strip()
            if text and href:
                employees.append({
                    "name": text,
                    "title": "",
                    "profile_url": href,
                })

    return employees


# ---------------------------------------------------------------------------
# Kagi Search Fallback
# ---------------------------------------------------------------------------

def _extract_via_kagi(
    config: Config, company_name: str
) -> list[dict[str, str]]:
    """Search for leadership profiles via Kagi search.

    Executes 3 parallel queries for CEO, founder, and CTO.
    """
    from valuation_tool.services.news_monitoring import KagiClient

    kagi = KagiClient(config)

    queries = [
        f'"{company_name}" CEO linkedin.com/in',
        f'"{company_name}" founder linkedin.com/in',
        f'"{company_name}" CTO linkedin.com/in',
    ]

    all_results: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(kagi.search, q): q for q in queries}
        for future in as_completed(futures):
            try:
                results = future.result()
                for item in results:
                    parsed = _parse_kagi_leadership_result(item)
                    if parsed:
                        all_results.append(parsed)
            except Exception as exc:
                logger.warning("kagi_leadership_query_failed", error=str(exc))

    # Deduplicate by LinkedIn profile URL
    seen_urls: set[str] = set()
    unique: list[dict[str, str]] = []
    for result in all_results:
        url = result.get("profile_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(result)

    return unique


def _parse_kagi_leadership_result(item: dict) -> dict[str, str] | None:
    """Parse a leadership record from a Kagi search result."""
    url = item.get("url", "")
    title_text = item.get("title", "")

    # Must be a personal LinkedIn profile
    if "/in/" not in url or "/company/" in url:
        return None

    # Parse "Name - Title - Company | LinkedIn" format
    parts = re.split(r"\s*[-–|]\s*", title_text)
    if len(parts) >= 2:
        person_name = parts[0].strip()
        job_title = parts[1].strip()

        if person_name and job_title:
            return {
                "name": person_name,
                "title": job_title,
                "profile_url": url,
            }

    return None


# ---------------------------------------------------------------------------
# Leadership Change Detection
# ---------------------------------------------------------------------------

def detect_leadership_changes(
    previous: list[CompanyLeadership],
    current: list[dict[str, str]],
) -> list[LeadershipChange]:
    """Compare previous and current leadership to detect changes.

    Matches by LinkedIn profile URL, not by name.
    """
    changes: list[LeadershipChange] = []

    prev_urls = {l.linkedin_profile_url: l for l in previous if l.linkedin_profile_url}
    curr_urls = {r.get("profile_url", ""): r for r in current if r.get("profile_url")}

    # Departures: in previous but not in current
    for url, leader in prev_urls.items():
        if url not in curr_urls:
            change_type = _departure_type(leader.title)
            severity = "critical" if change_type in (
                LeadershipChangeType.ceo_departure,
                LeadershipChangeType.founder_departure,
                LeadershipChangeType.cto_departure,
                LeadershipChangeType.coo_departure,
            ) else "notable"

            changes.append(LeadershipChange(
                change_type=change_type,
                person_name=leader.person_name,
                title=leader.title,
                severity=severity,
                confidence=0.95 if severity == "critical" else 0.80,
                significance_classification=SignificanceClassification.significant if severity == "critical" else None,
                significance_sentiment=SignificanceSentiment.negative if severity == "critical" else None,
                significance_confidence=0.95 if severity == "critical" else None,
            ))

    # Arrivals: in current but not in previous
    for url, person in curr_urls.items():
        if url not in prev_urls:
            title_norm = normalize_title(person.get("title", ""))

            if title_norm == "CEO":
                change_type = LeadershipChangeType.new_ceo
                severity = "notable"
            else:
                change_type = LeadershipChangeType.new_leadership
                severity = "notable"

            changes.append(LeadershipChange(
                change_type=change_type,
                person_name=person.get("name", ""),
                title=person.get("title", ""),
                severity=severity,
                confidence=0.80,
            ))

    # No changes
    if not changes:
        changes.append(LeadershipChange(
            change_type=LeadershipChangeType.no_change,
            person_name="",
            title="",
            severity="insignificant",
            confidence=0.75,
        ))

    return changes


def _departure_type(title: str) -> LeadershipChangeType:
    """Map a title to the appropriate departure change type."""
    norm = normalize_title(title).upper()
    if norm == "CEO":
        return LeadershipChangeType.ceo_departure
    if norm in ("FOUNDER", "CO-FOUNDER"):
        return LeadershipChangeType.founder_departure
    if norm == "CTO":
        return LeadershipChangeType.cto_departure
    if norm == "COO":
        return LeadershipChangeType.coo_departure
    return LeadershipChangeType.executive_departure


# ---------------------------------------------------------------------------
# Leadership Manager (Orchestrator)
# ---------------------------------------------------------------------------

def extract_leadership_for_company(
    config: Config,
    db: Database,
    company_id: int,
    headless: bool | None = None,
    profile_dir: str | None = None,
) -> LeadershipExtractionResult:
    """Extract leadership for a single company.

    Strategy: Playwright first (if LinkedIn URL available), then Kagi fallback.
    """
    company = db.get_company_by_id(company_id)
    if not company:
        return LeadershipExtractionResult(
            company_id=company_id,
            company_name="",
            errors=["Company not found"],
        )

    result = LeadershipExtractionResult(
        company_id=company.id,
        company_name=company.name,
    )

    headless = headless if headless is not None else config.linkedin_headless
    profile_dir = profile_dir or config.linkedin_profile_dir

    # Look up LinkedIn company URL
    social_links = db.get_social_links_for_company(company.id, platform="linkedin")
    linkedin_url = None
    for link in social_links:
        if "/company/" in link.profile_url:
            linkedin_url = link.profile_url
            break

    # Get previous leadership for change detection
    previous_leaders = db.get_leadership_for_company(company.id, current_only=True)

    leaders: list[dict[str, str]] = []
    method_used = ""

    # Try Playwright first if LinkedIn URL available
    if linkedin_url:
        try:
            import asyncio
            leaders = asyncio.get_event_loop().run_until_complete(
                _extract_via_playwright(linkedin_url, headless, profile_dir)
            )
            method_used = "playwright_scrape"
        except LinkedInBlockedError:
            logger.info("playwright_blocked", company=company.name, falling_back="kagi")
        except Exception as exc:
            logger.warning("playwright_failed", company=company.name, error=str(exc))

    # Kagi fallback
    if not leaders and config.kagi_available:
        try:
            leaders = _extract_via_kagi(config, company.name)
            method_used = "kagi_search"
        except Exception as exc:
            result.errors.append(f"Kagi fallback failed: {exc}")
            logger.error("kagi_fallback_failed", company=company.name, error=str(exc))

    # Filter to leadership titles only
    leadership_leaders = [
        l for l in leaders if is_leadership_title(l.get("title", ""))
    ]

    # Store leaders
    confidence = 0.8 if method_used == "playwright_scrape" else 0.6
    for leader_data in leadership_leaders:
        profile_url = leader_data.get("profile_url", "")
        if profile_url and "/in/" in profile_url:
            leadership = CompanyLeadership(
                company_id=company.id,
                person_name=leader_data["name"],
                title=normalize_title(leader_data["title"]),
                linkedin_profile_url=profile_url,
                discovery_method=LeadershipDiscoveryMethod(method_used) if method_used else LeadershipDiscoveryMethod.playwright_scrape,
                confidence=confidence,
            )
            db.store_leadership(leadership)

    # Detect changes
    changes = detect_leadership_changes(previous_leaders, leadership_leaders)
    result.leadership_changes = changes

    # Apply changes: mark departed leaders as not current
    for change in changes:
        if change.change_type in (
            LeadershipChangeType.ceo_departure,
            LeadershipChangeType.founder_departure,
            LeadershipChangeType.cto_departure,
            LeadershipChangeType.coo_departure,
            LeadershipChangeType.executive_departure,
        ):
            # Find the matching previous leader
            for prev in previous_leaders:
                if prev.person_name == change.person_name:
                    if prev.linkedin_profile_url:
                        db.mark_leadership_not_current(company.id, prev.linkedin_profile_url)

    result.leaders_found = len(leadership_leaders)
    result.method_used = method_used

    if changes and any(c.severity == "critical" for c in changes):
        logger.warning(
            "critical_leadership_change",
            company=company.name,
            changes=[c.model_dump() for c in changes if c.severity == "critical"],
        )

    return result


# ---------------------------------------------------------------------------
# Batch Extraction
# ---------------------------------------------------------------------------

def extract_leadership_all(
    config: Config,
    db: Database,
    limit: int | None = None,
    max_workers: int = 1,
) -> ExtractionResult:
    """Extract leadership for all companies.

    Defaults to sequential (1 worker) for Playwright safety.
    """
    companies = db.get_all_companies(limit=limit)
    result = ExtractionResult(processed=len(companies))

    if max_workers <= 1:
        for company in companies:
            try:
                sub = extract_leadership_for_company(config, db, company.id)
                result.successful += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append({"company": company.name, "error": str(exc)})
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extract_leadership_for_company, config, db, c.id
                ): c
                for c in companies
            }
            for future in as_completed(futures):
                company = futures[future]
                try:
                    future.result()
                    result.successful += 1
                except Exception as exc:
                    result.failed += 1
                    result.errors.append({"company": company.name, "error": str(exc)})

    logger.info(
        "extract_leadership_all_complete",
        processed=result.processed,
        successful=result.successful,
        failed=result.failed,
    )
    return result


def check_leadership_changes(
    config: Config,
    db: Database,
    limit: int | None = None,
) -> list[dict]:
    """Re-extract leadership and report only critical changes."""
    companies = db.get_all_companies(limit=limit)
    critical_changes: list[dict] = []

    for company in companies:
        try:
            sub = extract_leadership_for_company(config, db, company.id)
            for change in sub.leadership_changes:
                if change.severity == "critical":
                    critical_changes.append({
                        "company": company.name,
                        "change_type": change.change_type.value,
                        "person": change.person_name,
                        "title": change.title,
                    })
        except Exception as exc:
            logger.error("check_changes_failed", company=company.name, error=str(exc))

    return critical_changes
