@feature-005 @leadership-extraction
Feature: LinkedIn Leadership Extraction
  As a portfolio monitoring operator
  I want to extract CEO, founder, and C-level profiles from LinkedIn company pages
  So that I can track leadership changes that may indicate significant business events

  Background:
    Given the system is configured with valid environment variables
    And the database is initialized with the required schema

  # ==========================================================================
  # Playwright LinkedIn Extraction (Primary Method)
  # ==========================================================================

  @happy-path @playwright
  Scenario: Successfully extract leadership from LinkedIn People tab
    Given company "Acme Corp" with id 42 has a LinkedIn company URL in social_media_links
    And the LinkedIn People tab for "Acme Corp" contains employee cards:
      | name        | title                    | profile_url                        |
      | Jane Smith  | CEO                      | https://linkedin.com/in/jane-smith |
      | Bob Wilson  | CTO                      | https://linkedin.com/in/bob-wilson |
      | Alice Chen  | VP of Engineering        | https://linkedin.com/in/alice-chen |
      | Dave Brown  | Software Engineer        | https://linkedin.com/in/dave-brown |
    When I run "extract-leadership" with "--company-id 42"
    Then 3 leadership records should be stored for "Acme Corp"
    And the records should include "Jane Smith" with title "CEO"
    And the records should include "Bob Wilson" with title "CTO"
    And the records should include "Alice Chen" with title "VP of Engineering"
    And "Dave Brown" should be filtered out as a non-leadership title
    And all records should have discovery_method "playwright_scrape"
    And all records should have confidence 0.8

  @playwright @headless
  Scenario: Run Playwright extraction in headless mode
    Given company "Acme Corp" with id 42 has a LinkedIn company URL
    When I run "extract-leadership" with "--company-id 42 --headless"
    Then the browser should launch in headless mode

  @playwright @persistent-session
  Scenario: Reuse persistent browser session from previous login
    Given a LinkedIn session exists in the profile directory "data/linkedin_profile"
    When I run "extract-leadership" with "--company-id 42"
    Then the browser should use the saved session cookies
    And no manual login should be required

  @playwright @auth-wall
  Scenario: Detect auth wall and wait for manual login
    Given the LinkedIn page shows an auth wall (sign-in required)
    And no saved session exists
    When Playwright navigates to the People tab
    Then an auth wall should be detected
    And the browser should wait up to 120 seconds for manual login
    And after login the session should be saved to the profile directory

  @playwright @captcha
  Scenario: Fall back to Kagi when LinkedIn shows CAPTCHA
    Given the LinkedIn page shows a CAPTCHA challenge
    When Playwright navigates to the People tab
    Then a LinkedInBlockedError should be raised
    And the system should fall back to Kagi search

  @playwright @rate-limit
  Scenario: Fall back to Kagi when LinkedIn rate limits
    Given the LinkedIn page shows a "too many requests" error
    When Playwright navigates to the People tab
    Then a LinkedInBlockedError should be raised
    And the system should fall back to Kagi search

  @playwright @scroll-loading
  Scenario: Scroll page to load additional employee cards
    Given the LinkedIn People tab initially shows 10 cards
    And scrolling reveals 5 more cards
    When Playwright extracts employee data
    Then the browser should scroll at least 3 times
    And all 15 cards should be processed

  @playwright @selector-fallback
  Scenario: Use fallback selectors when primary selectors fail
    Given the LinkedIn DOM structure uses a different layout than expected
    And the primary employee card selectors do not match
    When Playwright attempts to extract employee data
    Then fallback selectors should be tried
    And profile links with "/in/" should be extracted as a last resort

  @playwright @custom-profile-dir
  Scenario: Use custom browser profile directory
    When I run "extract-leadership" with "--company-id 42 --profile-dir /custom/path"
    Then the browser should use "/custom/path" as the profile directory

  # ==========================================================================
  # Kagi Search Fallback (Secondary Method)
  # ==========================================================================

  @kagi-fallback
  Scenario: Fall back to Kagi search when Playwright is blocked
    Given company "Acme Corp" with id 42
    And Playwright extraction fails with LinkedInBlockedError
    And the Kagi API returns search results for leadership queries:
      | query                              | results                                           |
      | "Acme Corp" CEO linkedin.com/in    | Jane Smith - CEO - Acme Corp, linkedin.com/in/jane |
      | "Acme Corp" founder linkedin.com/in| Bob Wilson - Founder - Acme Corp, linkedin.com/in/bob |
      | "Acme Corp" CTO linkedin.com/in    | Alice Chen - CTO - Acme Corp, linkedin.com/in/alice  |
    When leadership extraction falls back to Kagi
    Then 3 leadership records should be stored
    And all records should have discovery_method "kagi_search"
    And all records should have confidence 0.6

  @kagi-fallback @no-linkedin-url
  Scenario: Go directly to Kagi when company has no LinkedIn URL
    Given company "Startup Inc" with id 55 has no LinkedIn company URL in social_media_links
    When I run "extract-leadership" with "--company-id 55"
    Then Playwright should not be attempted
    And Kagi search should be used directly

  @kagi-fallback @parallel-queries
  Scenario: Execute 3 parallel Kagi queries for different leadership roles
    Given a Kagi search fallback is triggered for "Acme Corp"
    When the Kagi search is performed
    Then 3 separate queries should be executed:
      | query                                   |
      | "Acme Corp" CEO linkedin.com/in         |
      | "Acme Corp" founder linkedin.com/in     |
      | "Acme Corp" CTO linkedin.com/in         |
    And the queries should run in parallel using ThreadPoolExecutor

  @kagi-fallback @deduplication
  Scenario: Deduplicate Kagi results by LinkedIn profile URL
    Given Kagi search returns "Jane Smith" in both CEO and founder query results
    And both results have profile URL "https://linkedin.com/in/jane-smith"
    When results are aggregated
    Then only 1 record for "Jane Smith" should be stored

  @kagi-fallback @result-parsing
  Scenario: Parse leadership data from Kagi search result titles
    Given a Kagi search result has:
      | field   | value                                       |
      | title   | John Doe - CEO - Acme Corp \| LinkedIn      |
      | url     | https://linkedin.com/in/john-doe            |
      | snippet | John Doe is the CEO of Acme Corp since 2023 |
    When the result is parsed
    Then the person_name should be "John Doe"
    And the title should be "CEO"
    And the linkedin_profile_url should be "https://linkedin.com/in/john-doe"

  @kagi-fallback @filter-company-pages
  Scenario: Filter out LinkedIn company pages from Kagi results
    Given a Kagi search result has URL "https://linkedin.com/company/acme"
    When the result is parsed
    Then the result should be rejected because it is a company page, not a personal profile

  @kagi-fallback @empty-results
  Scenario: Handle Kagi returning no leadership results
    Given all 3 Kagi queries return 0 relevant results
    When leadership extraction falls back to Kagi
    Then 0 leadership records should be stored
    And the result should report 0 leaders found

  # ==========================================================================
  # Title Detection
  # ==========================================================================

  @title-detection
  Scenario Outline: Detect leadership titles
    Given a person has the title "<title>"
    When leadership title detection is performed
    Then the title should be classified as <is_leadership>

    Examples:
      | title                        | is_leadership     |
      | CEO                          | leadership        |
      | Chief Executive Officer      | leadership        |
      | Founder                      | leadership        |
      | Co-Founder                   | leadership        |
      | Cofounder                    | leadership        |
      | CTO                          | leadership        |
      | Chief Technology Officer     | leadership        |
      | COO                          | leadership        |
      | Chief Operating Officer      | leadership        |
      | CFO                          | leadership        |
      | Chief Financial Officer      | leadership        |
      | CMO                          | leadership        |
      | Chief Marketing Officer      | leadership        |
      | Chief People Officer         | leadership        |
      | Chief Product Officer        | leadership        |
      | CRO                          | leadership        |
      | CSO                          | leadership        |
      | President                    | leadership        |
      | Managing Director            | leadership        |
      | General Manager              | leadership        |
      | VP of Engineering            | leadership        |
      | Vice President               | leadership        |
      | Software Engineer            | not leadership    |
      | Product Manager              | not leadership    |
      | Data Scientist               | not leadership    |
      | Marketing Manager            | not leadership    |
      | Sales Representative         | not leadership    |

  @title-detection @case-insensitive
  Scenario Outline: Title detection is case-insensitive
    Given a person has the title "<title>"
    When leadership title detection is performed
    Then the title should be classified as leadership

    Examples:
      | title                    |
      | ceo                      |
      | CEO                      |
      | Ceo                      |
      | CHIEF EXECUTIVE OFFICER  |

  @title-detection @embedded
  Scenario: Detect leadership title within longer text
    Given a person has the title "CEO at Acme Corp"
    When leadership title detection is performed
    Then the title should be classified as leadership
    And the extracted title should be "CEO"

  @title-detection @generic-pattern
  Scenario Outline: Detect generic Chief X Officer pattern
    Given a person has the title "<title>"
    When leadership title detection is performed
    Then the title should be classified as leadership

    Examples:
      | title                    |
      | Chief Revenue Officer    |
      | Chief Strategy Officer   |
      | Chief Data Officer       |
      | Chief Information Officer|

  @title-detection @vp-pattern
  Scenario Outline: Detect VP title patterns
    Given a person has the title "<title>"
    When leadership title detection is performed
    Then the title should be classified as leadership

    Examples:
      | title                           |
      | VP of Engineering               |
      | VP Engineering                  |
      | VP of Product                   |
      | VP Product                      |
      | Vice President of Engineering   |

  @title-normalization
  Scenario Outline: Normalize titles to canonical form
    Given a title "<input_title>" needs normalization
    When title normalization is performed
    Then the normalized title should be "<normalized>"

    Examples:
      | input_title                | normalized  |
      | Chief Executive Officer    | CEO         |
      | Chief Technology Officer   | CTO         |
      | Chief Operating Officer    | COO         |
      | Chief Financial Officer    | CFO         |
      | Cofounder                  | Co-Founder  |
      | co founder                 | Co-Founder  |
      | Co-founder                 | Co-Founder  |

  @title-ranking
  Scenario Outline: Rank titles by seniority
    Given two leaders with titles "<title_a>" and "<title_b>"
    When title ranking is performed
    Then "<title_a>" should rank <comparison> "<title_b>"

    Examples:
      | title_a    | title_b     | comparison         |
      | CEO        | CTO         | higher than        |
      | Founder    | Co-Founder  | higher than        |
      | Co-Founder | CTO         | higher than        |
      | CTO        | VP of Engineering | higher than   |
      | CEO        | CEO         | equal to           |

  # ==========================================================================
  # Leadership Change Detection
  # ==========================================================================

  @change-detection @departure @critical
  Scenario Outline: Detect critical leadership departures
    Given company "Acme Corp" previously had a leader:
      | person_name | title    | linkedin_profile_url              |
      | Jane Smith  | <title>  | https://linkedin.com/in/jane-smith|
    And the current extraction does not find "Jane Smith"
    When leadership change detection is performed
    Then a "<change_type>" change should be detected
    And the severity should be "critical"
    And the confidence should be 0.95

    Examples:
      | title     | change_type         |
      | CEO       | ceo_departure       |
      | Founder   | founder_departure   |
      | CTO       | cto_departure       |
      | COO       | coo_departure       |

  @change-detection @arrival
  Scenario: Detect new CEO arrival
    Given company "Acme Corp" had no CEO in previous extraction
    And the current extraction finds:
      | person_name | title | linkedin_profile_url              |
      | Bob Wilson  | CEO   | https://linkedin.com/in/bob-wilson|
    When leadership change detection is performed
    Then a "new_ceo" change should be detected
    And the severity should be "notable"
    And the confidence should be 0.80

  @change-detection @new-leadership
  Scenario: Detect new non-CEO leadership arrival
    Given the current extraction finds a new leader:
      | person_name | title | linkedin_profile_url                |
      | Alice Chen  | CTO   | https://linkedin.com/in/alice-chen  |
    And this person was not in the previous extraction
    When leadership change detection is performed
    Then a "new_leadership" change should be detected

  @change-detection @mixed
  Scenario: Detect both departures and arrivals
    Given company "Acme Corp" previously had:
      | person_name | title | linkedin_profile_url               |
      | Old CEO     | CEO   | https://linkedin.com/in/old-ceo    |
    And the current extraction finds:
      | person_name | title | linkedin_profile_url               |
      | New CEO     | CEO   | https://linkedin.com/in/new-ceo    |
    When leadership change detection is performed
    Then a "ceo_departure" change should be detected for "Old CEO"
    And a "new_ceo" change should be detected for "New CEO"
    And the overall sentiment should be "mixed"

  @change-detection @no-change
  Scenario: Report no changes when leadership is stable
    Given company "Acme Corp" had the same leadership in both previous and current extraction
    When leadership change detection is performed
    Then the change type should be "no_change"
    And the significance should be "insignificant"
    And the confidence should be 0.75

  @change-detection @profile-url-matching
  Scenario: Match leaders by LinkedIn profile URL not by name
    Given company "Acme Corp" previously had:
      | person_name | title | linkedin_profile_url               |
      | Jane Smith  | CEO   | https://linkedin.com/in/jane-smith |
    And the current extraction finds the same URL with a slightly different name:
      | person_name  | title | linkedin_profile_url               |
      | Jane A Smith | CEO   | https://linkedin.com/in/jane-smith |
    When leadership change detection is performed
    Then no departure should be detected
    And no new arrival should be detected

  @change-detection @is-current-flag
  Scenario: Mark departed leaders as not current
    Given company "Acme Corp" had leader "Jane Smith" (CEO) marked as is_current=True
    And "Jane Smith" is detected as departed
    When leadership changes are applied
    Then "Jane Smith" should be updated to is_current=False in the database
    And the record should remain in the database for historical tracking

  @change-detection @significance-integration
  Scenario: Leadership changes integrate with significance analysis
    Given a CEO departure is detected for company "Acme Corp"
    When the leadership change summary is built
    Then the significance_classification should be "significant"
    And the significance_sentiment should be "negative"
    And the significance_confidence should be 0.95
    And the change should be logged at WARNING level

  # ==========================================================================
  # Leadership Manager Orchestrator
  # ==========================================================================

  @orchestrator @playwright-first
  Scenario: Orchestrator tries Playwright first then falls back to Kagi
    Given company "Acme Corp" with id 42 has a LinkedIn company URL
    And Playwright extraction succeeds
    When I run "extract-leadership" with "--company-id 42"
    Then Playwright should be used as the primary method
    And Kagi should not be invoked

  @orchestrator @fallback-chain
  Scenario: Orchestrator falls back to Kagi after Playwright failure
    Given company "Acme Corp" with id 42 has a LinkedIn company URL
    And Playwright extraction raises LinkedInBlockedError
    And Kagi search returns valid results
    When I run "extract-leadership" with "--company-id 42"
    Then the result should show method_used as "kagi_search"
    And leaders from Kagi should be stored

  @orchestrator @linkedin-url-lookup
  Scenario: Look up LinkedIn company URL from social_media_links table
    Given company "Acme Corp" with id 42 has social media links:
      | platform | profile_url                           |
      | linkedin | https://linkedin.com/company/acme     |
      | twitter  | https://twitter.com/acme              |
    When the orchestrator looks up the LinkedIn company URL
    Then it should use "https://linkedin.com/company/acme"

  @orchestrator @duplicate-handling
  Scenario: Handle duplicate leadership records gracefully
    Given the database already contains a leadership record for:
      | company_id | linkedin_profile_url               |
      | 42         | https://linkedin.com/in/jane-smith |
    When the orchestrator attempts to store the same leader again
    Then the UNIQUE constraint should be handled silently
    And no error should be raised

  @orchestrator @result-format
  Scenario: Return structured result from single company extraction
    When leadership extraction completes for company "Acme Corp" (id 42)
    Then the result should contain:
      | field               | type         |
      | company_id          | integer      |
      | company_name        | string       |
      | leaders_found       | integer      |
      | method_used         | string       |
      | leadership_changes  | list         |
      | errors              | list         |

  # ==========================================================================
  # Batch Leadership Extraction
  # ==========================================================================

  @batch @extract-all
  Scenario: Batch extract leadership for all companies
    Given the database contains 5 companies
    When I run the "extract-leadership-all" command
    Then leadership extraction should be attempted for all 5 companies
    And aggregate results should be printed

  @batch @limit
  Scenario: Limit batch extraction to first N companies
    Given the database contains 50 companies
    When I run "extract-leadership-all" with "--limit 10"
    Then only 10 companies should be processed

  @batch @sequential-default
  Scenario: Batch extraction defaults to sequential processing (1 worker)
    When I run "extract-leadership-all" without "--max-workers"
    Then companies should be processed sequentially with 1 worker
    And this is safe for Playwright browser extraction

  @batch @parallel-workers
  Scenario: Batch extraction with parallel workers
    When I run "extract-leadership-all" with "--max-workers 4"
    Then up to 4 companies should be processed in parallel

  @batch @error-isolation
  Scenario: Individual company failure does not abort batch
    Given the database contains 3 companies
    And extraction fails for company "Bad Corp" but succeeds for others
    When I run the "extract-leadership-all" command
    Then 2 companies should have successful extractions
    And 1 company should be reported as failed
    And the batch should complete with aggregate results

  # ==========================================================================
  # Check Leadership Changes Command
  # ==========================================================================

  @check-changes
  Scenario: Re-extract and report only critical changes
    Given the database contains existing leadership records
    And re-extraction detects a CEO departure at "Startup X"
    When I run the "check-leadership-changes" command
    Then the output should report the CEO departure at "Startup X"
    And non-critical changes should not be highlighted

  @check-changes @no-changes
  Scenario: Report when no critical changes detected
    Given the database contains existing leadership records
    And re-extraction shows stable leadership across all companies
    When I run the "check-leadership-changes" command
    Then the output should say "No critical leadership changes detected"

  @check-changes @limit
  Scenario: Limit leadership change check to first N companies
    Given the database contains 50 companies
    When I run "check-leadership-changes" with "--limit 10"
    Then only 10 companies should be re-extracted and compared

  # ==========================================================================
  # Model Validation
  # ==========================================================================

  @validation @leadership-model
  Scenario: CompanyLeadership requires linkedin_profile_url with /in/ path
    Given a leadership record with linkedin_profile_url "https://linkedin.com/company/acme"
    When the model is validated
    Then the validation should fail because company pages are not personal profiles

  @validation @leadership-model
  Scenario: CompanyLeadership person_name must be non-empty
    Given a leadership record with person_name ""
    When the model is validated
    Then the validation should fail

  @validation @leadership-model
  Scenario: CompanyLeadership title must be non-empty
    Given a leadership record with title ""
    When the model is validated
    Then the validation should fail

  @validation @leadership-model
  Scenario: CompanyLeadership confidence must be between 0 and 1
    Given a leadership record with confidence 1.5
    When the model is validated
    Then the validation should fail

  @validation @leadership-model
  Scenario: LeadershipDiscoveryMethod enum values
    Then the LeadershipDiscoveryMethod enum should include:
      | method            |
      | playwright_scrape |
      | kagi_search       |

  @validation @leadership-model
  Scenario: LeadershipChangeType enum values
    Then the LeadershipChangeType enum should include:
      | change_type          |
      | ceo_departure        |
      | founder_departure    |
      | cto_departure        |
      | coo_departure        |
      | executive_departure  |
      | new_ceo              |
      | new_leadership       |
      | no_change            |

  @validation @leadership-model @database-constraint
  Scenario: UNIQUE constraint on company_id and linkedin_profile_url
    Given the database contains a leadership record for company 42 with URL "https://linkedin.com/in/jane"
    When another record for company 42 with the same URL is inserted
    Then a UNIQUE constraint violation should occur
