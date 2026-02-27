@cross-cutting @error-handling
Feature: Error Handling and Retry Logic
  As a portfolio monitoring operator
  I want the system to handle errors gracefully with retries and error isolation
  So that transient failures do not require manual intervention and batch operations are resilient

  Background:
    Given the system is configured with valid environment variables
    And the database is initialized with the required schema

  # ==========================================================================
  # Retry Logic
  # ==========================================================================

  @retry @exponential-backoff
  Scenario: Retry with exponential backoff on transient errors
    Given the retry configuration has MAX_RETRY_ATTEMPTS set to 3
    And an API call fails with a ConnectionError on the first 2 attempts
    And the API call succeeds on the 3rd attempt
    When the operation is executed
    Then the request should be retried 2 times
    And the backoff delays should follow exponential pattern starting at 2 seconds
    And the maximum backoff should not exceed 10 seconds

  @retry @backoff-timing
  Scenario Outline: Exponential backoff timing
    Given the retry multiplier is 1 with min 2 seconds and max 10 seconds
    When retry attempt <attempt> occurs
    Then the wait time should be approximately <wait_seconds> seconds

    Examples:
      | attempt | wait_seconds |
      | 1       | 2            |
      | 2       | 4            |
      | 3       | 8            |
      | 4       | 10           |

  @retry @max-attempts
  Scenario: Stop retrying after max attempts exhausted
    Given the retry configuration has MAX_RETRY_ATTEMPTS set to 2
    And an API call fails on all attempts with ConnectionError
    When the operation is executed
    Then the request should be attempted 2 times total
    And the error should be logged after all retries exhausted
    And the operation should fail with the last error

  @retry @configurable-attempts
  Scenario Outline: Max retry attempts is configurable
    Given MAX_RETRY_ATTEMPTS is set to <max_attempts>
    And an API call always fails with a transient error
    When the operation is executed
    Then the total number of attempts should be <max_attempts>

    Examples:
      | max_attempts |
      | 0            |
      | 1            |
      | 2            |
      | 3            |
      | 5            |

  # ==========================================================================
  # Retryable Error Conditions
  # ==========================================================================

  @retry @retryable-errors
  Scenario Outline: Retry on transient error types
    Given an API call fails with a "<error_type>" error
    When the retry decorator evaluates the error
    Then the error should be retryable

    Examples:
      | error_type      |
      | ConnectionError |
      | TimeoutError    |
      | OSError         |

  @retry @retryable-http
  Scenario Outline: Retry on specific HTTP status codes
    Given an API call returns HTTP status <status_code>
    When the retry decorator evaluates the response
    Then the error should be retryable

    Examples:
      | status_code |
      | 429         |
      | 500         |
      | 502         |
      | 503         |
      | 504         |

  @retry @non-retryable
  Scenario Outline: Do not retry on non-transient errors
    Given an API call returns HTTP status <status_code>
    When the retry decorator evaluates the response
    Then the error should not be retryable
    And the operation should fail immediately

    Examples:
      | status_code |
      | 401         |
      | 403         |
      | 404         |

  @retry @validation-errors
  Scenario: Do not retry on data validation errors
    Given a Pydantic validation error occurs during data processing
    When the retry decorator evaluates the error
    Then the error should not be retryable
    And the record should be logged and skipped

  # ==========================================================================
  # Error Categories
  # ==========================================================================

  @error-category @transient-network
  Scenario: Handle transient network errors with retry
    Given the Firecrawl API is temporarily unreachable due to network issues
    When a snapshot capture is attempted
    Then the request should be retried with exponential backoff
    And the error should be classified as "Transient Network"

  @error-category @rate-limiting
  Scenario: Handle rate limiting with longer backoff
    Given the Kagi API returns HTTP 429 (Too Many Requests)
    When a news search is attempted
    Then the request should be retried with exponential backoff
    And the backoff should accommodate the rate limit reset time

  @error-category @auth-failure
  Scenario: Handle authentication failure without retry
    Given the Airtable API returns HTTP 401 (Unauthorized)
    When a company extraction is attempted
    Then the operation should fail immediately without retrying
    And the error should be classified as "Auth Failure"
    And a clear error message should indicate invalid credentials

  @error-category @data-validation
  Scenario: Handle data validation errors by logging and skipping
    Given a company record from Airtable has an invalid field value
    When the record is processed
    Then the validation error should be logged
    And the record should be skipped
    And processing should continue with the next record

  @error-category @api-error
  Scenario: Handle API errors by logging and skipping
    Given the Firecrawl API returns HTTP 500 for a specific URL
    And the URL has exhausted its retry attempts
    When processing continues
    Then the error should be logged with the URL context
    And the operation should be skipped for that URL
    And the batch should continue processing remaining URLs

  @error-category @database-error
  Scenario: Handle database constraint violations
    Given a duplicate company record triggers a UNIQUE constraint violation
    When the insert is attempted
    Then the transaction should handle the constraint error
    And the duplicate should be skipped or upserted
    And no data corruption should occur

  # ==========================================================================
  # Batch Error Isolation
  # ==========================================================================

  @batch-isolation @snapshot-capture
  Scenario: Individual snapshot failure does not abort batch
    Given a batch of 10 URLs is being captured
    And URL 3 fails with a TimeoutError after all retries
    And URL 7 fails with a ConnectionError after all retries
    When the batch snapshot capture completes
    Then 8 snapshots should be stored successfully
    And 2 errors should be accumulated in the error list
    And the summary should report 8 successful and 2 failed

  @batch-isolation @social-discovery
  Scenario: Individual company discovery failure does not abort batch
    Given 5 companies are being processed for social media discovery
    And discovery fails for company 3 due to Firecrawl API error
    When the batch discovery completes
    Then 4 companies should have discovery results
    And 1 error should be reported
    And all errors should be logged via structlog with context

  @batch-isolation @news-search
  Scenario: Individual company news search failure does not abort batch
    Given 3 companies are being searched for news
    And the Kagi API fails for "Problem Corp" after all retries
    When the batch news search completes
    Then news for 2 companies should be processed
    And 1 error should be reported for "Problem Corp"

  @batch-isolation @leadership-extraction
  Scenario: Individual leadership extraction failure does not abort batch
    Given 5 companies are being processed for leadership extraction
    And extraction fails for company 2 due to LinkedIn blocking
    And Kagi fallback also fails for company 2
    When the batch extraction completes
    Then 4 companies should have extraction results
    And 1 company should be reported as failed

  @batch-isolation @error-accumulation
  Scenario: Batch operations accumulate errors for summary reporting
    Given a batch operation processes 20 items
    And 3 items fail with different errors
    When the batch completes
    Then the errors list should contain 3 error entries
    And each error should include the item identifier and error details
    And the summary should include processed, successful, failed, and skipped counts

  # ==========================================================================
  # ProcessingError Tracking
  # ==========================================================================

  @processing-error @storage
  Scenario: Store processing errors for debugging
    Given a snapshot capture fails for company with id 42
    When the error is recorded
    Then a processing_errors record should be created with:
      | field         | value                              |
      | entity_type   | company                            |
      | entity_id     | 42                                 |
      | error_type    | FirecrawlTimeout                   |
      | error_message | Request timed out after 30s        |
      | retry_count   | 2                                  |

  @processing-error @entity-types
  Scenario Outline: Processing errors track different entity types
    Given a "<entity_type>" operation fails for entity with id <entity_id>
    When the error is recorded
    Then the processing error should have entity_type "<entity_type>"

    Examples:
      | entity_type | entity_id |
      | company     | 42        |
      | snapshot    | 101       |

  @processing-error @validation
  Scenario: ProcessingError model validates error_type format
    Given a processing error is being created
    Then the error_type should be in PascalCase format
    And the error_type should be between 1 and 100 characters
    And the error_message should be between 1 and 5000 characters

  @processing-error @retry-count
  Scenario: ProcessingError tracks retry attempts
    Given an operation fails after 2 retry attempts
    When the processing error is recorded
    Then the retry_count should be 2

  # ==========================================================================
  # Structured Logging
  # ==========================================================================

  @logging @structured
  Scenario: Errors logged with structured context via structlog
    Given an operation fails for company "Acme Corp" with id 42
    When the error is logged
    Then the log entry should include:
      | field        | value             |
      | company_name | Acme Corp         |
      | company_id   | 42                |
      | error_type   | the error class   |
      | error_message| the error details |

  @logging @retry-logging
  Scenario: Each retry attempt is logged
    Given an operation fails and retries 3 times
    When the retries are executed
    Then each retry should be logged with the attempt number
    And the final success or failure should be logged

  @logging @batch-summary
  Scenario: Batch operations log completion summary
    Given a batch operation processes 50 items with 3 failures
    When the batch completes
    Then a summary log should include:
      | field     | value |
      | processed | 50    |
      | successful| 47    |
      | failed    | 3     |

  # ==========================================================================
  # Graceful Degradation
  # ==========================================================================

  @graceful-degradation @llm-unavailable
  Scenario: System works without LLM when Anthropic API is unavailable
    Given LLM_VALIDATION_ENABLED is "true"
    And the Anthropic API is unreachable
    When significance analysis is performed
    Then keyword-based classification should be used as fallback
    And the system should not crash

  @graceful-degradation @kagi-unavailable
  Scenario: Leadership extraction works without Kagi
    Given the KAGI_API_KEY is not configured
    And Playwright LinkedIn extraction succeeds
    When leadership extraction is performed
    Then the extraction should succeed using Playwright only
    And no Kagi fallback should be attempted

  @graceful-degradation @partial-content
  Scenario: Snapshot stored even with partial content
    Given the Firecrawl API returns markdown but no HTML for a URL
    When the snapshot is captured
    Then the snapshot should be stored with content_markdown populated
    And content_html should be NULL
    And no error should be raised
