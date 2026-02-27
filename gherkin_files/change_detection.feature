@feature-002 @change-detection
Feature: Website Change Detection and Status Analysis
  As a portfolio monitoring operator
  I want to detect content changes between website snapshots and analyze company status
  So that I can identify significant business events and track company health

  Background:
    Given the system is configured with valid environment variables
    And the database is initialized with the required schema

  # ==========================================================================
  # Change Detection
  # ==========================================================================

  @happy-path @detect-changes
  Scenario: Detect content changes between two snapshots
    Given the database contains company "Acme Corp" with 2 snapshots:
      | snapshot | content_markdown                    | content_checksum                 |
      | old      | Welcome to Acme Corp                | aaa1111111111111111111111111111a |
      | new      | Welcome to Acme Corp - Now Hiring!  | bbb2222222222222222222222222222b |
    When I run the "detect-changes" command
    Then a change record should be created for "Acme Corp"
    And the change record should have has_changed set to True
    And the change record should have checksum_old "aaa1111111111111111111111111111a"
    And the change record should have checksum_new "bbb2222222222222222222222222222b"
    And the change record should have significance analysis populated

  @detect-changes @no-change
  Scenario: Record no change when checksums are identical
    Given the database contains company "Stable Corp" with 2 snapshots:
      | snapshot | content_checksum                 |
      | old      | abc1234567890abcdef1234567890abcd |
      | new      | abc1234567890abcdef1234567890abcd |
    When I run the "detect-changes" command
    Then a change record should be created for "Stable Corp"
    And the change record should have has_changed set to False
    And the change record should have change_magnitude "minor"

  @detect-changes @skip
  Scenario: Skip companies with fewer than 2 snapshots
    Given the database contains company "New Corp" with only 1 snapshot
    When I run the "detect-changes" command
    Then no change record should be created for "New Corp"

  @detect-changes @magnitude
  Scenario Outline: Calculate change magnitude based on content similarity
    Given company "Test Corp" has old content with similarity ratio <similarity> to new content
    When change detection is performed for "Test Corp"
    Then the change magnitude should be "<magnitude>"

    Examples:
      | similarity | magnitude |
      | 0.95       | minor     |
      | 0.90       | minor     |
      | 0.75       | moderate  |
      | 0.50       | moderate  |
      | 0.30       | major     |
      | 0.10       | major     |
      | 0.00       | major     |

  @detect-changes @magnitude @boundary
  Scenario: Minor magnitude boundary at exactly 0.90 similarity
    Given company "Boundary Corp" has old content with similarity ratio 0.90 to new content
    When change detection is performed for "Boundary Corp"
    Then the change magnitude should be "minor"

  @detect-changes @magnitude @boundary
  Scenario: Moderate magnitude just below 0.90 similarity
    Given company "Boundary Corp" has old content with similarity ratio 0.89 to new content
    When change detection is performed for "Boundary Corp"
    Then the change magnitude should be "moderate"

  @detect-changes @magnitude @boundary
  Scenario: Major magnitude boundary at exactly below 0.50 similarity
    Given company "Boundary Corp" has old content with similarity ratio 0.49 to new content
    When change detection is performed for "Boundary Corp"
    Then the change magnitude should be "major"

  @detect-changes @truncation
  Scenario: Truncate content comparison at 50000 characters
    Given company "Long Content Corp" has snapshots with content exceeding 50000 characters
    When change detection is performed
    Then only the first 50000 characters of each snapshot should be compared
    And the change record should still be created with a valid magnitude

  @detect-changes @diff-extraction
  Scenario: Extract only added lines from content diff
    Given company "Diff Corp" has old content "Line A\nLine B\nLine C"
    And the new content is "Line A\nLine B Modified\nLine C\nLine D Added"
    When change detection is performed
    Then the diff should contain only the added and modified lines
    And the diff should be used for significance keyword analysis

  @detect-changes @unhappy-path
  Scenario: Handle missing snapshot content gracefully
    Given company "Missing Corp" has 2 snapshots but the old snapshot has NULL content_markdown
    When I run the "detect-changes" command
    Then "Missing Corp" should be skipped with an error logged
    And processing should continue for remaining companies

  @detect-changes @ordering
  Scenario: Always compare the two most recent snapshots
    Given company "Multi Corp" has 4 snapshots captured at:
      | captured_at          |
      | 2025-01-01T00:00:00Z |
      | 2025-02-01T00:00:00Z |
      | 2025-03-01T00:00:00Z |
      | 2025-04-01T00:00:00Z |
    When I run the "detect-changes" command
    Then the change detection should compare the March and April snapshots

  # ==========================================================================
  # Status Analysis
  # ==========================================================================

  @happy-path @analyze-status
  Scenario: Analyze company as operational with positive indicators
    Given company "Active Corp" has a latest snapshot with content containing:
      """
      Copyright 2026 Active Corp. All rights reserved.
      We are hiring! Join our team.
      """
    And the snapshot has HTTP Last-Modified within the last 30 days
    When I run the "analyze-status" command
    Then the status for "Active Corp" should be "operational"
    And the confidence should be greater than 0.7

  @analyze-status @likely-closed
  Scenario: Detect likely closed company with negative indicators
    Given company "Gone Corp" has a latest snapshot with content containing:
      """
      This company has been acquired by BigTech Inc.
      Gone Corp is now part of BigTech.
      """
    When I run the "analyze-status" command
    Then the status for "Gone Corp" should be "likely_closed"
    And the indicators should include an acquisition signal

  @analyze-status @uncertain
  Scenario: Classify company as uncertain with mixed signals
    Given company "Mixed Corp" has a latest snapshot with content containing:
      """
      Copyright 2020 Mixed Corp.
      """
    And the snapshot has HTTP Last-Modified older than 180 days
    When I run the "analyze-status" command
    Then the status for "Mixed Corp" should be "uncertain"
    And the confidence should be less than 0.7

  @analyze-status @copyright-year
  Scenario Outline: Extract copyright year from various patterns
    Given the snapshot content contains "<copyright_text>"
    When copyright year extraction is performed
    Then the extracted year should be <expected_year>

    Examples:
      | copyright_text                     | expected_year |
      | (c) 2025 Company Name              | 2025          |
      | (C) 2026 Company Name              | 2026          |
      | Copyright 2025 Company Name        | 2025          |
      | &copy; 2024-2026 Company Name      | 2026          |
      | All content copyright 2025         | 2025          |

  @analyze-status @copyright-year @negative
  Scenario: Do not match bare years without copyright marker
    Given the snapshot content contains "Founded in 2020"
    When copyright year extraction is performed
    Then no copyright year should be extracted

  @analyze-status @acquisition-detection
  Scenario Outline: Detect acquisition keywords
    Given the snapshot content contains "<text>"
    When acquisition detection is performed
    Then acquisition should be <detected>

    Examples:
      | text                                          | detected     |
      | acquired by BigTech                           | detected     |
      | merged with Partner Corp                      | detected     |
      | sold to Buyer Inc                             | detected     |
      | now part of Parent Co                         | detected     |
      | is now a subsidiary of Parent Co              | detected     |
      | is now a division of Parent Co                | detected     |
      | Product X is now available                    | not detected |
      | We acquired new customers                     | not detected |

  @analyze-status @confidence
  Scenario Outline: Calculate confidence from indicators
    Given a company has <positive> positive, <negative> negative, and <neutral> neutral indicators
    When confidence is calculated
    Then the confidence should be approximately <expected_confidence>

    Examples:
      | positive | negative | neutral | expected_confidence |
      | 2        | 0        | 0       | 0.8                 |
      | 0        | 2        | 0       | 0.8                 |
      | 1        | 0        | 1       | 0.6                 |
      | 0        | 0        | 1       | 0.2                 |

  @analyze-status @status-rules
  Scenario Outline: Determine status from confidence and signals
    Given the confidence is <confidence> with <positive> positive and <negative> negative signals
    When status determination is performed
    Then the status should be "<expected_status>"

    Examples:
      | confidence | positive | negative | expected_status |
      | 0.8        | 2        | 0        | operational     |
      | 0.8        | 0        | 2        | likely_closed   |
      | 0.5        | 2        | 1        | operational     |
      | 0.5        | 1        | 2        | likely_closed   |
      | 0.5        | 1        | 1        | uncertain       |
      | 0.3        | 1        | 0        | uncertain       |

  @analyze-status @http-freshness
  Scenario: HTTP Last-Modified header contributes to status analysis
    Given company "Fresh Corp" has a snapshot with HTTP Last-Modified 5 days ago
    When status analysis is performed
    Then the indicators should include a positive HTTP freshness signal

  @analyze-status @http-freshness @stale
  Scenario: Stale HTTP Last-Modified header is a negative signal
    Given company "Stale Corp" has a snapshot with HTTP Last-Modified 200 days ago
    When status analysis is performed
    Then the indicators should include a negative HTTP freshness signal

  # ==========================================================================
  # Query Commands
  # ==========================================================================

  @query @show-changes
  Scenario: Show change history for a company
    Given company "Acme Corp" has the following change records:
      | detected_at          | has_changed | change_magnitude | significance_classification |
      | 2025-01-15T10:00:00Z | true        | moderate         | significant                 |
      | 2025-02-15T10:00:00Z | false       | minor            |                             |
      | 2025-03-15T10:00:00Z | true        | major            | significant                 |
    When I run the "show-changes" command for "Acme Corp"
    Then the output should display all 3 change records
    And the output should include significance data for significant changes
    And the output should include related news articles if any exist

  @query @show-status
  Scenario: Show current status for a company
    Given company "Acme Corp" has status "operational" with confidence 0.85
    And the status has indicators:
      | type           | value | signal   |
      | copyright_year | 2026  | positive |
    When I run the "show-status" command for "Acme Corp"
    Then the output should display status "operational"
    And the output should display confidence 0.85
    And the output should list the indicators

  @query @list-active
  Scenario: List companies with recent changes
    Given the following companies have change records:
      | company    | detected_at          | has_changed |
      | Active Co  | 2026-01-15T10:00:00Z | true        |
      | Stale Co   | 2025-01-15T10:00:00Z | true        |
      | Recent Co  | 2026-02-01T10:00:00Z | true        |
    When I run the "list-active" command with "--days 180"
    Then the output should list "Active Co" and "Recent Co"
    And the output should not list "Stale Co"

  @query @list-inactive
  Scenario: List companies without recent changes
    Given the following companies have change records:
      | company    | detected_at          | has_changed |
      | Active Co  | 2026-01-15T10:00:00Z | true        |
      | Stale Co   | 2025-01-15T10:00:00Z | true        |
    When I run the "list-inactive" command with "--days 180"
    Then the output should list "Stale Co"
    And the output should not list "Active Co"

  @query @list-significant-changes
  Scenario: List significant changes filtered by sentiment
    Given the following significant change records exist:
      | company    | significance_sentiment | significance_confidence | detected_at          |
      | Good Corp  | positive               | 0.85                    | 2026-01-01T10:00:00Z |
      | Bad Corp   | negative               | 0.90                    | 2026-01-15T10:00:00Z |
      | Mixed Corp | mixed                  | 0.75                    | 2026-02-01T10:00:00Z |
    When I run "list-significant-changes" with "--days 180 --sentiment positive"
    Then the output should list only "Good Corp"

  @query @list-significant-changes @min-confidence
  Scenario: Filter significant changes by minimum confidence
    Given the following significant change records exist:
      | company      | significance_confidence |
      | High Conf Co | 0.90                    |
      | Low Conf Co  | 0.55                    |
      | Mid Conf Co  | 0.70                    |
    When I run "list-significant-changes" with "--min-confidence 0.7"
    Then the output should list "High Conf Co" and "Mid Conf Co"
    And the output should not list "Low Conf Co"

  @query @list-uncertain-changes
  Scenario: List uncertain changes requiring manual review
    Given the following change records exist:
      | company       | significance_classification |
      | Uncertain Co  | uncertain                   |
      | Clear Co      | significant                 |
      | Also Unclear  | uncertain                   |
    When I run the "list-uncertain-changes" command
    Then the output should list "Uncertain Co" and "Also Unclear"
    And the output should not list "Clear Co"

  @query @list-uncertain-changes @limit
  Scenario: Limit uncertain changes output
    Given 100 change records with classification "uncertain" exist
    When I run "list-uncertain-changes" with "--limit 10"
    Then at most 10 records should be displayed

  # ==========================================================================
  # Change Record Model Validation
  # ==========================================================================

  @validation @change-record-model
  Scenario: Change magnitude enum values
    Given a change record is being created
    Then the change_magnitude must be one of "minor", "moderate", "major"

  @validation @change-record-model
  Scenario: Significance classification enum values
    Given a change record has significance analysis
    Then the significance_classification must be one of "significant", "insignificant", "uncertain"

  @validation @change-record-model
  Scenario: Significance sentiment enum values
    Given a change record has significance analysis
    Then the significance_sentiment must be one of "positive", "negative", "neutral", "mixed"

  @validation @change-record-model
  Scenario: Significance confidence must be between 0 and 1
    Given a change record has significance_confidence of 1.5
    When the change record model is validated
    Then the validation should fail

  @validation @company-status-model
  Scenario: Company status type enum values
    Given a company status is being created
    Then the status must be one of "operational", "likely_closed", "uncertain"

  @validation @company-status-model
  Scenario: Status indicator signals
    Given a status indicator is being created
    Then the signal must be one of "positive", "negative", "neutral"
