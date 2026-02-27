@feature-004 @news-monitoring
Feature: News Monitoring with Kagi Search
  As a portfolio monitoring operator
  I want to search for news articles about portfolio companies and verify their relevance
  So that I can track significant business events reported in the media

  Background:
    Given the system is configured with valid environment variables
    And the KAGI_API_KEY is configured
    And the database is initialized with the required schema

  # ==========================================================================
  # Single Company News Search
  # ==========================================================================

  @happy-path @search-news @by-name
  Scenario: Search news for a company by name
    Given the database contains company "Modal Labs" with id 42
    And the Kagi Search API returns 3 articles for "Modal Labs":
      | title                                | url                                  | published                |
      | Modal Labs Raises Series B           | https://techcrunch.com/modal-b       | 2026-01-15T10:00:00Z     |
      | Modal Labs Launches New GPU Feature  | https://venturebeat.com/modal-gpu    | 2026-01-20T10:00:00Z     |
      | AI Infrastructure Market Growing     | https://reuters.com/ai-infra         | 2026-01-25T10:00:00Z     |
    And company verification passes for 2 of the 3 articles
    When I run "search-news" with "--company-name Modal Labs"
    Then the command should complete successfully
    And 2 verified articles should be stored in the database
    And each stored article should have significance analysis populated
    And the output should report 3 found and 2 stored

  @happy-path @search-news @by-id
  Scenario: Search news for a company by ID
    Given the database contains company "Acme Corp" with id 10
    And the Kagi Search API returns articles for "Acme Corp"
    When I run "search-news" with "--company-id 10"
    Then the search should be performed using "Acme Corp" as the query

  @search-news @date-range @with-snapshots
  Scenario: Calculate date range from snapshot history
    Given company "Acme Corp" has snapshots captured at:
      | captured_at          |
      | 2025-06-01T00:00:00Z |
      | 2025-09-01T00:00:00Z |
      | 2026-01-01T00:00:00Z |
    When news search is performed for "Acme Corp"
    Then the Kagi API should be called with after_date "2025-06-01"
    And the Kagi API should be called with before_date set to today

  @search-news @date-range @no-snapshots
  Scenario: Default to 90-day lookback when no snapshots exist
    Given company "New Corp" has no snapshots in the database
    When news search is performed for "New Corp"
    Then the Kagi API should be called with after_date set to 90 days ago

  @search-news @duplicate-detection
  Scenario: Skip articles with duplicate URLs
    Given the database already contains a news article with URL "https://techcrunch.com/existing-article"
    And the Kagi API returns an article with the same URL
    When news search is performed
    Then the duplicate article should be skipped
    And the article count should not increase

  @search-news @no-results
  Scenario: Handle search returning no articles
    Given the Kagi Search API returns 0 results for "Obscure Corp"
    When I run "search-news" with "--company-name Obscure Corp"
    Then the output should report 0 articles found
    And 0 articles should be stored

  @search-news @all-fail-verification
  Scenario: Handle all articles failing verification
    Given the Kagi API returns 5 articles for "Common Name LLC"
    And all 5 articles fail company verification with confidence below 0.40
    When I run "search-news" with "--company-name Common Name LLC"
    Then 0 articles should be stored in the database
    And the output should report 5 found but 0 verified

  # ==========================================================================
  # Multi-Signal Company Verification
  # ==========================================================================

  @verification @domain-match
  Scenario: Verify article via domain match
    Given company "Acme Corp" has homepage URL "https://www.acme.com"
    And an article from "https://techcrunch.com/acme-funding" mentions "acme.com" in its content
    When company verification is performed
    Then the domain_match signal should contribute 30% weight to confidence

  @verification @domain-match @word-boundary
  Scenario: Domain matching uses word boundaries to prevent false positives
    Given company "AI Corp" has homepage URL "https://www.ai.com"
    And an article mentions "email.ai.common" but not "ai.com" as a standalone domain
    When company verification is performed
    Then the domain_match signal should not trigger a match

  @verification @name-context
  Scenario: Verify article via company name in business context
    Given an article snippet contains "Modal Labs announced a new funding round"
    When company verification is performed for "Modal Labs"
    Then the name_context signal should contribute 15% weight to confidence

  @verification @logo-similarity
  Scenario: Verify article via logo perceptual hash comparison
    Given company "Acme Corp" has a stored logo with a perceptual hash
    And the article contains a logo image with similarity score 0.90
    When company verification is performed
    Then the logo_match signal should contribute 30% weight to confidence

  @verification @llm
  Scenario: Verify article via LLM when enabled
    Given LLM verification is enabled with a valid ANTHROPIC_API_KEY
    And the LLM confirms the article is about "Acme Corp"
    When company verification is performed
    Then the llm_verification signal should contribute 25% weight to confidence

  @verification @threshold
  Scenario Outline: Article passes verification at or above 40% confidence threshold
    Given an article has the following verification signals:
      | signal        | matches | weight |
      | domain_match  | <dom>   | 0.30   |
      | name_context  | <name>  | 0.15   |
      | logo_match    | <logo>  | 0.30   |
      | llm_verify    | <llm>   | 0.25   |
    When weighted confidence is calculated
    Then the total confidence should be <confidence>
    And the article should <result>

    Examples:
      | dom   | name  | logo  | llm   | confidence | result          |
      | true  | true  | false | false | 0.45       | pass verification |
      | true  | false | false | true  | 0.55       | pass verification |
      | false | true  | false | false | 0.15       | fail verification |
      | false | false | false | false | 0.00       | fail verification |
      | true  | true  | true  | true  | 1.00       | pass verification |

  @verification @evidence
  Scenario: Build human-readable evidence list
    Given an article passes domain match and name context verification
    When the evidence list is built
    Then the evidence should include "domain_match" and "name_context"
    And the evidence list should be stored with the article

  @verification @weights
  Scenario: Use default verification weights
    When the default verification weights are loaded
    Then the weights should be:
      | signal  | weight |
      | logo    | 0.30   |
      | domain  | 0.30   |
      | context | 0.15   |
      | llm     | 0.25   |

  # ==========================================================================
  # Batch News Search
  # ==========================================================================

  @happy-path @search-news-all
  Scenario: Search news for all companies
    Given the database contains 3 companies
    And the Kagi API returns articles for each company
    When I run the "search-news-all" command
    Then news should be searched for all 3 companies
    And an aggregate summary should be printed

  @search-news-all @limit
  Scenario: Limit batch news search to first N companies
    Given the database contains 50 companies
    When I run "search-news-all" with "--limit 10"
    Then only the first 10 companies should be searched

  @search-news-all @error-isolation
  Scenario: Individual company search failure does not abort batch
    Given the database contains 3 companies
    And the Kagi API fails for company "Broken Corp" but succeeds for others
    When I run the "search-news-all" command
    Then articles for the 2 successful companies should be stored
    And the error for "Broken Corp" should be logged
    And the batch should complete with a summary

  # ==========================================================================
  # Kagi API Integration
  # ==========================================================================

  @kagi-api @query-format
  Scenario: Format Kagi search query with date filters
    Given company "Modal Labs" needs news search from 2025-06-01 to 2026-02-01
    When the Kagi API query is constructed
    Then the query should include "Modal Labs"
    And the query should include "after:2025-06-01"
    And the query should include "before:2026-02-01"

  @kagi-api @response-parsing
  Scenario: Parse Kagi search response into article models
    Given the Kagi API returns a response with:
      """
      {
        "data": [
          {
            "title": "Acme Raises $50M",
            "url": "https://techcrunch.com/acme-50m",
            "snippet": "Acme Corp today announced...",
            "published": "2026-01-15T10:30:00Z"
          }
        ]
      }
      """
    When the response is parsed
    Then a NewsArticle should be created with:
      | field        | value                              |
      | title        | Acme Raises $50M                   |
      | content_url  | https://techcrunch.com/acme-50m    |
      | source       | techcrunch.com                     |

  @kagi-api @missing-date
  Scenario: Use current datetime when article has no published date
    Given the Kagi API returns an article without a "published" field
    When the response is parsed
    Then the published_at should default to the current datetime

  @kagi-api @auth-failure
  Scenario: Handle Kagi API authentication failure
    Given the KAGI_API_KEY is invalid
    When a Kagi search is attempted
    Then an authentication error should be raised
    And the error should not be retried

  @kagi-api @rate-limit
  Scenario: Retry on Kagi API rate limiting
    Given the Kagi API returns HTTP 429 on the first request
    And the Kagi API succeeds on the second request
    When a Kagi search is attempted
    Then the request should be retried with exponential backoff
    And the search results should be returned successfully

  @kagi-api @server-error
  Scenario: Retry on Kagi API server errors
    Given the Kagi API returns HTTP 500 on the first request
    And the Kagi API succeeds on the retry
    When a Kagi search is attempted
    Then the request should be retried

  @kagi-api @bearer-auth
  Scenario: Kagi API authentication uses Bearer token
    When a Kagi search request is made
    Then the Authorization header should be "Bot {api_key}"

  # ==========================================================================
  # News Article Significance Analysis
  # ==========================================================================

  @news-significance
  Scenario: Analyze news article for business significance
    Given a verified news article with title "Acme Corp Raises $100M Series C"
    And the article snippet contains "funding" and "series c"
    When significance analysis is performed on the article
    Then the significance_classification should be "significant"
    And the significance_sentiment should be "positive"
    And the matched_keywords should include "funding" and "series c"

  @news-significance @insignificant
  Scenario: Classify routine press mention as insignificant
    Given a verified news article with title "Top 50 Startups to Watch"
    And the article snippet mentions the company in passing without significant keywords
    When significance analysis is performed on the article
    Then the significance_classification should be "insignificant"

  @news-significance @integration
  Scenario: News articles appear in show-changes output
    Given company "Acme Corp" has both change records and news articles
    When I run the "show-changes" command for "Acme Corp"
    Then the output should include both website changes and related news articles

  # ==========================================================================
  # News Article Model Validation
  # ==========================================================================

  @validation @news-article-model
  Scenario: News article title must be non-empty with max 500 characters
    Given a news article with an empty title
    When the model is validated
    Then the validation should fail

  @validation @news-article-model
  Scenario: News article content_url must be a valid URL
    Given a news article with content_url "not-a-url"
    When the model is validated
    Then the validation should fail

  @validation @news-article-model
  Scenario: News article match_confidence must be between 0 and 1
    Given a news article with match_confidence 1.5
    When the model is validated
    Then the validation should fail

  @validation @news-article-model
  Scenario: News article content_url must be globally unique
    Given a news article with content_url "https://example.com/article" already exists
    When another article with the same content_url is stored
    Then a UNIQUE constraint violation should occur
    And the duplicate should be rejected

  @validation @news-article-model
  Scenario: Source is extracted from article URL domain
    Given a news article with content_url "https://techcrunch.com/2026/01/acme-funding"
    When the article is created
    Then the source should be "techcrunch.com"
