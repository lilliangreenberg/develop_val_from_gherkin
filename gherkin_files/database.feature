@cross-cutting @database
Feature: Database Schema, Constraints, and Operations
  As a system developer
  I want the SQLite database to enforce data integrity via schema constraints
  So that the system maintains consistent, non-duplicated data across all operations

  Background:
    Given a SQLite database at the configured DATABASE_PATH

  # ==========================================================================
  # Schema Initialization
  # ==========================================================================

  @schema @initialization
  Scenario: Initialize database with complete schema
    Given no database file exists at the configured path
    When Database.init_db() is called
    Then the database should be created with all required tables:
      | table              |
      | companies          |
      | snapshots          |
      | change_records     |
      | company_statuses   |
      | social_media_links |
      | blog_links         |
      | news_articles      |
      | company_logos       |
      | company_leadership |
      | processing_errors  |

  @schema @indexes
  Scenario: All required indexes are created
    When Database.init_db() is called
    Then the following indexes should exist:
      | index_name                              | table              | columns              |
      | idx_companies_name                      | companies          | name                 |
      | idx_snapshots_company_id                | snapshots          | company_id           |
      | idx_snapshots_captured_at               | snapshots          | captured_at          |
      | idx_change_records_company_id           | change_records     | company_id           |
      | idx_social_media_links_company_id       | social_media_links | company_id           |
      | idx_social_media_links_platform         | social_media_links | platform             |
      | idx_news_articles_company_id            | news_articles      | company_id           |
      | idx_news_articles_published_at          | news_articles      | published_at         |
      | idx_news_articles_significance          | news_articles      | significance_classification |
      | idx_company_logos_company_id             | company_logos      | company_id           |
      | idx_company_logos_perceptual_hash        | company_logos      | perceptual_hash      |
      | idx_company_leadership_company_id        | company_leadership | company_id           |
      | idx_company_leadership_title             | company_leadership | title                |

  @schema @idempotent
  Scenario: Schema initialization is idempotent
    Given the database already exists with all tables
    When Database.init_db() is called again
    Then the operation should complete without errors
    And existing data should not be affected

  # ==========================================================================
  # UNIQUE Constraints
  # ==========================================================================

  @constraints @unique @companies
  Scenario: Companies table enforces UNIQUE on name and homepage_url
    Given a company "Acme Corp" with homepage_url "https://www.acme.com" exists
    When another record with name "Acme Corp" and homepage_url "https://www.acme.com" is inserted
    Then a UNIQUE constraint violation should occur
    And the duplicate should be rejected or handled via upsert

  @constraints @unique @companies @different-url
  Scenario: Same company name with different URL is allowed
    Given a company "Acme Corp" with homepage_url "https://www.acme.com" exists
    When a record with name "Acme Corp" and homepage_url "https://acme.io" is inserted
    Then the insert should succeed
    And both records should exist in the database

  @constraints @unique @social-media-links
  Scenario: Social media links enforce UNIQUE on company_id and profile_url
    Given a social media link exists for company 1 with profile_url "https://twitter.com/acme"
    When another link for company 1 with profile_url "https://twitter.com/acme" is inserted
    Then a UNIQUE constraint violation should occur

  @constraints @unique @social-media-links @different-company
  Scenario: Same social URL for different companies is allowed
    Given a social media link exists for company 1 with profile_url "https://twitter.com/acme"
    When a link for company 2 with profile_url "https://twitter.com/acme" is inserted
    Then the insert should succeed

  @constraints @unique @news-articles
  Scenario: News articles enforce globally UNIQUE content_url
    Given a news article with content_url "https://techcrunch.com/article-1" exists for company 1
    When another article with the same content_url is inserted for company 2
    Then a UNIQUE constraint violation should occur
    And the duplicate URL should be rejected regardless of company

  @constraints @unique @blog-links
  Scenario: Blog links enforce UNIQUE on company_id and blog_url
    Given a blog link exists for company 1 with blog_url "https://blog.acme.com"
    When another blog link for company 1 with blog_url "https://blog.acme.com" is inserted
    Then a UNIQUE constraint violation should occur

  @constraints @unique @company-logos
  Scenario: Company logos enforce UNIQUE on company_id and perceptual_hash
    Given a logo exists for company 1 with perceptual_hash "abc123"
    When another logo for company 1 with perceptual_hash "abc123" is inserted
    Then a UNIQUE constraint violation should occur

  @constraints @unique @company-leadership
  Scenario: Company leadership enforces UNIQUE on company_id and linkedin_profile_url
    Given a leadership record exists for company 1 with linkedin_profile_url "https://linkedin.com/in/jane"
    When another record for company 1 with the same URL is inserted
    Then a UNIQUE constraint violation should occur

  # ==========================================================================
  # Foreign Key Cascade
  # ==========================================================================

  @constraints @foreign-key @cascade
  Scenario: Deleting a company cascades to all related records
    Given company "Acme Corp" with id 1 exists with the following related records:
      | table              | count |
      | snapshots          | 3     |
      | change_records     | 2     |
      | social_media_links | 5     |
      | blog_links         | 1     |
      | news_articles      | 4     |
      | company_logos       | 1     |
      | company_leadership | 3     |
    When company with id 1 is deleted
    Then all related records in dependent tables should be deleted
    And no orphaned records should remain

  @constraints @foreign-key @snapshots
  Scenario: Snapshots reference valid company_id
    Given no company with id 999 exists in the database
    When a snapshot with company_id 999 is inserted
    Then a foreign key constraint violation should occur

  @constraints @foreign-key @change-records
  Scenario: Change records reference valid snapshot IDs
    Given snapshots with ids 1 and 2 exist
    When a change record references snapshot_id_old=1 and snapshot_id_new=2
    Then the insert should succeed
    When a change record references snapshot_id_old=999 (non-existent)
    Then a foreign key constraint violation should occur

  # ==========================================================================
  # JSON Serialization
  # ==========================================================================

  @json-serialization
  Scenario Outline: List fields are stored as JSON text in SQLite
    Given a record in table "<table>" has a "<field>" value of <python_value>
    When the record is stored in the database
    Then the "<field>" column should contain '<json_text>'

    Examples:
      | table          | field              | python_value                   | json_text                          |
      | change_records | matched_keywords   | ["funding", "series a"]        | ["funding", "series a"]            |
      | change_records | matched_categories | ["funding_investment"]         | ["funding_investment"]             |
      | news_articles  | match_evidence     | ["domain_match", "name_context"]| ["domain_match", "name_context"]  |

  @json-serialization @indicators
  Scenario: StatusIndicator list serialized as JSON in company_statuses
    Given a company status has indicators:
      | type           | value | signal   |
      | copyright_year | 2026  | positive |
    When the status is stored in the database
    Then the indicators column should contain a JSON array of indicator objects

  @json-serialization @read-back
  Scenario: JSON fields are deserialized correctly when read
    Given a change record is stored with matched_keywords as '["funding", "series a"]'
    When the change record is read from the database
    Then the matched_keywords should be a Python list ["funding", "series a"]

  # ==========================================================================
  # Datetime Handling
  # ==========================================================================

  @datetime
  Scenario: Datetimes stored as ISO 8601 strings
    Given a company is created with created_at "2026-02-15T10:30:00+00:00"
    When the record is stored in the database
    Then the created_at column should contain an ISO 8601 formatted string

  @datetime @utc
  Scenario: All datetimes use UTC timezone
    Given a snapshot is captured at the current time
    When the captured_at timestamp is stored
    Then the timestamp should be in UTC timezone

  @datetime @read-back
  Scenario: ISO 8601 strings are parsed back to Python datetime
    Given a snapshot has captured_at stored as "2026-02-15T10:30:00+00:00"
    When the snapshot is read from the database
    Then the captured_at should be a Python datetime with UTC timezone

  # ==========================================================================
  # CRUD Operations
  # ==========================================================================

  @crud @company
  Scenario: Company CRUD operations
    When a company is created with name "Test Corp" and homepage_url "https://test.com"
    Then the company should be retrievable by id
    And the company should be retrievable by name
    When the company is updated with a new homepage_url "https://new-test.com"
    Then the updated URL should be reflected in the database
    When the company is deleted
    Then the company should no longer exist in the database

  @crud @snapshot
  Scenario: Store and retrieve snapshots for a company
    Given company "Acme Corp" with id 1 exists
    When 3 snapshots are stored for company 1
    Then querying snapshots for company 1 should return 3 records
    And the snapshots should be ordered by captured_at

  @crud @social-media-link
  Scenario: Store and retrieve social media links for a company
    Given company "Acme Corp" with id 1 exists
    When 5 social media links are stored for company 1
    Then querying social links for company 1 should return 5 records
    And filtering by platform "twitter" should return only Twitter links

  @crud @news-article
  Scenario: Store and retrieve news articles for a company
    Given company "Acme Corp" with id 1 exists
    When 3 news articles are stored for company 1
    Then querying articles for company 1 should return 3 records
    And the articles should include significance classification data

  @crud @leadership
  Scenario: Store and retrieve leadership records for a company
    Given company "Acme Corp" with id 1 exists
    When 2 leadership records are stored for company 1
    Then querying leadership for company 1 should return 2 records
    And querying current leadership should return only is_current=True records

  @crud @leadership @mark-not-current
  Scenario: Mark departed leaders as not current
    Given company "Acme Corp" has a leadership record for "Jane Smith" with is_current=True
    When mark_not_current is called for "Jane Smith" at company 1
    Then "Jane Smith" should have is_current=False
    And the record should still exist in the database

  @crud @upsert
  Scenario: Upsert company updates existing record
    Given company "Acme Corp" with homepage_url "https://acme.com" exists with id 1
    When an upsert is performed for "Acme Corp" with homepage_url "https://acme.com"
    Then the existing record should be updated (not duplicated)
    And only 1 company with name "Acme Corp" and URL "https://acme.com" should exist

  # ==========================================================================
  # Migration Support
  # ==========================================================================

  @migration
  Scenario: Migrations are applied incrementally
    Given the database was created with an older schema version
    When the migration scripts are executed
    Then new columns should be added via ALTER TABLE
    And new tables should be created if they do not exist
    And existing data should be preserved

  @migration @idempotent
  Scenario: Migrations check if already applied before executing
    Given a migration to add column "content_checksum" to snapshots has already run
    When the same migration script is executed again
    Then the migration should detect the column already exists
    And no error should be raised

  @migration @transaction
  Scenario: Migrations run within transactions for safety
    Given a migration script is being executed
    When a migration step fails midway
    Then the transaction should be rolled back
    And the database should remain in its pre-migration state

  # ==========================================================================
  # Data Integrity
  # ==========================================================================

  @integrity @checksums
  Scenario: Checksums are always lowercase hex MD5 (32 characters)
    Given a snapshot is stored with content_checksum "D41D8CD98F00B204E9800998ECF8427E"
    When the checksum is stored
    Then the stored value should be lowercase "d41d8cd98f00b204e9800998ecf8427e"
    And the length should be exactly 32 characters

  @integrity @not-null
  Scenario Outline: Required columns reject NULL values
    Given an insert into table "<table>" with "<column>" set to NULL
    When the insert is attempted
    Then a NOT NULL constraint violation should occur

    Examples:
      | table              | column           |
      | companies          | name             |
      | companies          | source_sheet     |
      | companies          | created_at       |
      | snapshots          | company_id       |
      | snapshots          | url              |
      | snapshots          | captured_at      |
      | change_records     | company_id       |
      | change_records     | has_changed      |
      | change_records     | change_magnitude |
      | social_media_links | company_id       |
      | social_media_links | platform         |
      | social_media_links | profile_url      |
      | news_articles      | company_id       |
      | news_articles      | title            |
      | news_articles      | content_url      |
      | company_leadership | company_id       |
      | company_leadership | person_name      |
      | company_leadership | title            |

  @integrity @autoincrement
  Scenario: All tables use INTEGER PRIMARY KEY AUTOINCREMENT
    When a new record is inserted without specifying an id
    Then the id should be automatically assigned
    And the id should be unique and incrementing

  @integrity @wal-mode
  Scenario: SQLite uses WAL mode for better read concurrency
    When the database is opened
    Then the journal mode should support concurrent readers
