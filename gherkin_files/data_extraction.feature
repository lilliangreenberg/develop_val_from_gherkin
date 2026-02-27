@feature-001 @data-extraction
Feature: Data Extraction and Snapshot Capture
  As a portfolio monitoring operator
  I want to extract company data from Airtable and capture website snapshots
  So that I can build a database of portfolio companies with current website content

  Background:
    Given the system is configured with valid environment variables
    And the database is initialized with the required schema

  # ==========================================================================
  # Company Extraction from Airtable
  # ==========================================================================

  @happy-path @extract-companies
  Scenario: Successfully extract companies from Airtable
    Given the Airtable base contains the following "Online Presence" records:
      | url                       | resources | company_name_ref |
      | https://www.acme.com      | homepage  | rec_acme_001     |
      | https://www.techcorp.io   | homepage  | rec_tech_002     |
      | https://www.startup.ai    | homepage  | rec_start_003    |
    And the "Portfolio Companies" table resolves the following names:
      | record_id      | company_name |
      | rec_acme_001   | Acme Corp    |
      | rec_tech_002   | TechCorp     |
      | rec_start_003  | Startup AI   |
    When I run the "extract-companies" command
    Then the command should complete successfully
    And 3 companies should be stored in the database
    And the company "Acme Corp" should have homepage_url "https://www.acme.com"
    And the company "TechCorp" should have homepage_url "https://www.techcorp.io"
    And the output should contain a summary with processed count 3

  @extract-companies @upsert
  Scenario: Upsert existing companies without creating duplicates
    Given the database already contains a company:
      | name      | homepage_url           | source_sheet      |
      | Acme Corp | https://www.acme.com   | Online Presence   |
    And the Airtable base contains a record for "Acme Corp" with URL "https://www.acme.com"
    When I run the "extract-companies" command
    Then only 1 company named "Acme Corp" should exist in the database
    And the company's updated_at timestamp should be refreshed

  @extract-companies @unhappy-path
  Scenario: Skip records with missing company name reference
    Given the Airtable base contains an "Online Presence" record with:
      | url                  | resources | company_name_ref |
      | https://orphan.com   | homepage  |                  |
    When I run the "extract-companies" command
    Then the record should be skipped
    And the output should report 1 skipped record
    And 0 companies should be stored from that record

  @extract-companies @unhappy-path
  Scenario: Store company with NULL homepage URL when URL is invalid
    Given the Airtable base contains a record for "Bad URL Corp" with URL "not-a-valid-url"
    When I run the "extract-companies" command
    Then the company "Bad URL Corp" should be stored with a NULL homepage_url

  @extract-companies @unhappy-path
  Scenario: Handle Airtable API authentication failure
    Given the AIRTABLE_API_KEY is invalid
    When I run the "extract-companies" command
    Then the command should fail with an authentication error
    And no companies should be stored in the database

  @extract-companies @unhappy-path @retry
  Scenario: Retry on transient Airtable API errors
    Given the Airtable API returns HTTP 500 on the first request
    And the Airtable API succeeds on the second request with 3 companies
    When I run the "extract-companies" command
    Then the retry logic should attempt the request again
    And 3 companies should be stored in the database

  @extract-companies @filtering
  Scenario: Only extract records with "homepage" resource type
    Given the Airtable base contains "Online Presence" records:
      | url                      | resources       | company_name_ref |
      | https://www.acme.com     | homepage        | rec_acme_001     |
      | https://twitter.com/acme | social_media    | rec_acme_001     |
      | https://blog.acme.com    | blog            | rec_acme_001     |
    When I run the "extract-companies" command
    Then only the record with resources "homepage" should be processed
    And 1 company should be stored in the database

  @extract-companies @validation
  Scenario: Company names are normalized on extraction
    Given the Airtable base resolves a company name as "  acme   corp  "
    When I run the "extract-companies" command
    Then the company should be stored with name "Acme Corp"

  # ==========================================================================
  # Sequential Snapshot Capture
  # ==========================================================================

  @happy-path @capture-snapshots @sequential
  Scenario: Successfully capture snapshots sequentially
    Given the database contains the following companies with homepage URLs:
      | name      | homepage_url           |
      | Acme Corp | https://www.acme.com   |
      | TechCorp  | https://www.techcorp.io|
    And the Firecrawl API returns valid content for all URLs
    When I run the "capture-snapshots" command
    Then 2 snapshots should be stored in the database
    And each snapshot should have content_markdown populated
    And each snapshot should have content_html populated
    And each snapshot should have a valid content_checksum

  @capture-snapshots @critical-invariant
  Scenario: Firecrawl is always called with only_main_content set to False
    Given the database contains a company with homepage URL "https://www.example.com"
    When I run the "capture-snapshots" command
    Then the Firecrawl API should be called with only_main_content set to False
    And the Firecrawl API should be called with formats "markdown" and "html"

  @capture-snapshots @checksum
  Scenario: Content checksum is computed as lowercase hex MD5
    Given the Firecrawl API returns markdown content "Hello World" for a company
    When the snapshot is captured
    Then the content_checksum should be a 32-character lowercase hexadecimal string
    And the content_checksum should be the MD5 hash of "Hello World"

  @capture-snapshots @unhappy-path
  Scenario: Handle website returning HTTP 404
    Given the database contains a company with homepage URL "https://www.gone.com"
    And the Firecrawl API returns status code 404 for "https://www.gone.com"
    When I run the "capture-snapshots" command
    Then a snapshot should be stored with status_code 404
    And the snapshot should have an error_message populated

  @capture-snapshots @unhappy-path
  Scenario: Handle Firecrawl API timeout for a single URL
    Given the database contains companies:
      | name    | homepage_url             |
      | Good Co | https://www.good.com     |
      | Slow Co | https://www.slow.com     |
    And the Firecrawl API times out for "https://www.slow.com"
    And the Firecrawl API returns valid content for "https://www.good.com"
    When I run the "capture-snapshots" command
    Then a snapshot for "Good Co" should be stored successfully
    And a processing error should be logged for "Slow Co"
    And the output should report 1 successful and 1 failed capture

  @capture-snapshots @paywall
  Scenario: Detect paywall on captured website
    Given the Firecrawl API detects a paywall for "https://www.paywalled.com"
    When the snapshot is captured for that URL
    Then the snapshot should have has_paywall set to True

  @capture-snapshots @auth-wall
  Scenario: Detect authentication wall on captured website
    Given the Firecrawl API detects an auth wall for "https://www.protected.com"
    When the snapshot is captured for that URL
    Then the snapshot should have has_auth_required set to True

  @capture-snapshots @http-headers
  Scenario: Store HTTP Last-Modified header from snapshot
    Given the Firecrawl API returns a Last-Modified header of "2025-06-15T10:30:00Z"
    When the snapshot is captured
    Then the snapshot should have http_last_modified set to "2025-06-15T10:30:00Z"

  @capture-snapshots @no-homepage
  Scenario: Skip companies without homepage URLs
    Given the database contains companies:
      | name       | homepage_url           |
      | Has URL Co | https://www.hasurl.com |
      | No URL Co  |                        |
    When I run the "capture-snapshots" command
    Then only 1 snapshot should be captured for "Has URL Co"
    And "No URL Co" should be skipped

  # ==========================================================================
  # Batch Snapshot Capture
  # ==========================================================================

  @happy-path @capture-snapshots @batch
  Scenario: Successfully capture snapshots using batch API
    Given the database contains 5 companies with homepage URLs
    And the Firecrawl batch API is available
    When I run the "capture-snapshots" command with option "--use-batch-api"
    Then the Firecrawl batch API should be called instead of individual scrapes
    And 5 snapshots should be stored in the database
    And each snapshot should have only_main_content set to False in the request

  @capture-snapshots @batch @batch-size
  Scenario Outline: Batch API respects configurable batch size
    Given the database contains <company_count> companies with homepage URLs
    When I run "capture-snapshots" with "--use-batch-api --batch-size <batch_size>"
    Then the URLs should be grouped into batches of <batch_size>
    And <expected_batches> batch API calls should be made

    Examples:
      | company_count | batch_size | expected_batches |
      | 10            | 5          | 2                |
      | 25            | 10         | 3                |
      | 100           | 50         | 2                |
      | 3             | 20         | 1                |

  @capture-snapshots @batch @error-isolation
  Scenario: Individual URL failure does not abort batch
    Given the database contains 5 companies with homepage URLs
    And the Firecrawl batch API fails for 1 URL but succeeds for 4
    When I run the "capture-snapshots" command with option "--use-batch-api"
    Then 4 snapshots should be stored successfully
    And 1 processing error should be logged
    And the output should report 4 successful and 1 failed

  @capture-snapshots @batch @retry
  Scenario: Retry batch API on rate limiting
    Given the database contains 3 companies with homepage URLs
    And the Firecrawl batch API returns HTTP 429 on the first attempt
    And the Firecrawl batch API succeeds on the second attempt
    When I run the "capture-snapshots" command with option "--use-batch-api"
    Then the batch should be retried with exponential backoff
    And 3 snapshots should be stored successfully

  @capture-snapshots @batch @timeout
  Scenario: Handle batch API timeout
    Given the database contains 3 companies with homepage URLs
    And the Firecrawl batch API times out after the configured timeout
    When I run "capture-snapshots" with "--use-batch-api --timeout 60"
    Then a timeout error should be logged
    And the batch should be retried per retry configuration

  @capture-snapshots @batch @max-size
  Scenario: Enforce maximum batch size of 1000
    Given the database contains 1500 companies with homepage URLs
    When I run "capture-snapshots" with "--use-batch-api --batch-size 1000"
    Then each batch should contain at most 1000 URLs
    And at least 2 batch API calls should be made

  # ==========================================================================
  # Snapshot Model Validation
  # ==========================================================================

  @validation @snapshot-model
  Scenario Outline: Validate snapshot status code range
    Given a snapshot is being created with status_code <status_code>
    When the snapshot model is validated
    Then the validation should <result>

    Examples:
      | status_code | result  |
      | 200         | pass    |
      | 404         | pass    |
      | 100         | pass    |
      | 599         | pass    |
      | 99          | fail    |
      | 600         | fail    |

  @validation @snapshot-model
  Scenario: Snapshot requires at least one content field or error message
    Given a snapshot is being created with:
      | content_markdown | content_html | error_message |
      |                  |              |               |
    When the snapshot model is validated
    Then the validation should fail with a message about requiring content or error

  @validation @snapshot-model
  Scenario: Snapshot captured_at must not be in the future
    Given a snapshot is being created with captured_at set to tomorrow
    When the snapshot model is validated
    Then the validation should fail with a message about future timestamp

  @validation @checksum-format
  Scenario Outline: Validate checksum format
    Given a snapshot has content_checksum "<checksum>"
    When the snapshot model is validated
    Then the validation should <result>

    Examples:
      | checksum                          | result |
      | d41d8cd98f00b204e9800998ecf8427e  | pass   |
      | D41D8CD98F00B204E9800998ECF8427E  | pass   |
      | not-a-valid-checksum              | fail   |
      | d41d8cd98f00b204                  | fail   |
