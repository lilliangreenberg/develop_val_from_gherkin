@cross-cutting @configuration
Feature: Configuration and Environment
  As a system administrator
  I want to configure the application via environment variables
  So that the system connects to the correct APIs and uses appropriate settings

  # ==========================================================================
  # Required Environment Variables
  # ==========================================================================

  @required-config @happy-path
  Scenario: Application starts with all required environment variables set
    Given the .env file contains:
      | variable          | value                |
      | AIRTABLE_API_KEY  | pat.test_key_123     |
      | AIRTABLE_BASE_ID  | appABC123DEF456      |
      | FIRECRAWL_API_KEY | fc-test_key_456      |
    When the configuration is loaded
    Then the Config model should validate successfully
    And all required API clients should be initializable

  @required-config @missing
  Scenario Outline: Application fails to start when required variable is missing
    Given the .env file is missing the variable "<variable>"
    When the configuration is loaded
    Then a validation error should be raised
    And the error should indicate that "<variable>" is required

    Examples:
      | variable          |
      | AIRTABLE_API_KEY  |
      | AIRTABLE_BASE_ID  |
      | FIRECRAWL_API_KEY |

  @required-config @empty
  Scenario: Empty string for required API key is rejected
    Given the .env file contains AIRTABLE_API_KEY with value ""
    When the configuration is loaded
    Then a validation error should be raised for min_length constraint

  # ==========================================================================
  # Airtable Base ID Validation
  # ==========================================================================

  @validation @base-id
  Scenario Outline: Validate Airtable base ID format
    Given AIRTABLE_BASE_ID is set to "<base_id>"
    When the configuration is loaded
    Then the validation should <result>

    Examples:
      | base_id             | result |
      | appABC123DEF456     | pass   |
      | appAbCdEfGhIjKlMnO  | pass   |
      | app12345            | pass   |
      | tblABC123           | fail   |
      | ABC123DEF456        | fail   |
      | app                 | fail   |
      | app!@#$%            | fail   |

  # ==========================================================================
  # Optional Variables with Defaults
  # ==========================================================================

  @optional-config @defaults
  Scenario: Optional variables use correct defaults when not set
    Given the .env file contains only required variables
    When the configuration is loaded
    Then the following defaults should be applied:
      | variable               | default_value              |
      | DATABASE_PATH          | data/companies.db          |
      | LOG_LEVEL              | INFO                       |
      | MAX_RETRY_ATTEMPTS     | 2                          |
      | LLM_VALIDATION_ENABLED | false                      |
      | LLM_MODEL              | claude-haiku-4-5-20251001  |
      | LINKEDIN_HEADLESS      | false                      |
      | LINKEDIN_PROFILE_DIR   | data/linkedin_profile      |

  @optional-config @database-path
  Scenario: Custom database path is used when configured
    Given DATABASE_PATH is set to "/custom/path/mydb.sqlite"
    When the configuration is loaded
    Then the database path should be "/custom/path/mydb.sqlite"

  @optional-config @database-path @auto-create
  Scenario: Parent directory for database is auto-created if missing
    Given DATABASE_PATH is set to "new_data/my_companies.db"
    And the directory "new_data" does not exist
    When the configuration is loaded
    Then the parent directory "new_data" should be created automatically

  @optional-config @log-level
  Scenario Outline: Configure log level
    Given LOG_LEVEL is set to "<level>"
    When the configuration is loaded
    Then the log level should be "<level>"

    Examples:
      | level   |
      | DEBUG   |
      | INFO    |
      | WARNING |
      | ERROR   |

  @optional-config @retry-attempts
  Scenario Outline: Validate MAX_RETRY_ATTEMPTS range
    Given MAX_RETRY_ATTEMPTS is set to <value>
    When the configuration is loaded
    Then the validation should <result>

    Examples:
      | value | result |
      | 0     | pass   |
      | 1     | pass   |
      | 2     | pass   |
      | 5     | pass   |
      | -1    | fail   |
      | 6     | fail   |
      | 100   | fail   |

  # ==========================================================================
  # LLM Configuration
  # ==========================================================================

  @llm-config
  Scenario: LLM features disabled by default
    Given LLM_VALIDATION_ENABLED is not set in the .env file
    When the configuration is loaded
    Then LLM validation should be disabled
    And no ANTHROPIC_API_KEY should be required

  @llm-config @enabled
  Scenario: LLM features enabled with valid API key
    Given the .env file contains:
      | variable               | value                     |
      | LLM_VALIDATION_ENABLED | true                      |
      | ANTHROPIC_API_KEY      | sk-test-key-123           |
      | LLM_MODEL              | claude-haiku-4-5-20251001 |
    When the configuration is loaded
    Then LLM validation should be enabled
    And the LLM model should be "claude-haiku-4-5-20251001"

  @llm-config @custom-model
  Scenario: Custom LLM model can be specified
    Given LLM_MODEL is set to "claude-sonnet-4-20250514"
    When the configuration is loaded
    Then the LLM model should be "claude-sonnet-4-20250514"

  # ==========================================================================
  # Kagi Configuration
  # ==========================================================================

  @kagi-config
  Scenario: Kagi API key is optional
    Given KAGI_API_KEY is not set in the .env file
    When the configuration is loaded
    Then the configuration should load successfully
    And news monitoring features should be unavailable

  @kagi-config @configured
  Scenario: Kagi API key enables news monitoring
    Given KAGI_API_KEY is set to "test-kagi-key-123"
    When the configuration is loaded
    Then news monitoring features should be available

  # ==========================================================================
  # LinkedIn Configuration
  # ==========================================================================

  @linkedin-config @headless
  Scenario: LinkedIn browser headless mode is configurable
    Given LINKEDIN_HEADLESS is set to "true"
    When the configuration is loaded
    Then the LinkedIn browser should run in headless mode

  @linkedin-config @headed-default
  Scenario: LinkedIn browser defaults to headed mode for manual login
    Given LINKEDIN_HEADLESS is not set in the .env file
    When the configuration is loaded
    Then the LinkedIn browser should run in headed mode (headless=false)

  @linkedin-config @profile-dir
  Scenario: Custom LinkedIn profile directory is configurable
    Given LINKEDIN_PROFILE_DIR is set to "/custom/linkedin/profile"
    When the configuration is loaded
    Then the LinkedIn browser profile directory should be "/custom/linkedin/profile"

  # ==========================================================================
  # Config Loading via pydantic-settings
  # ==========================================================================

  @config-loading @env-file
  Scenario: Configuration loaded from .env file
    Given a .env file exists in the project root with valid configuration
    When the Config model is instantiated
    Then values should be read from the .env file
    And the file should be read with UTF-8 encoding

  @config-loading @case-insensitive
  Scenario: Environment variable names are case-insensitive
    Given the .env file contains "airtable_api_key=test_key"
    When the configuration is loaded
    Then the AIRTABLE_API_KEY should be "test_key"

  @config-loading @extra-ignore
  Scenario: Unknown environment variables are ignored
    Given the .env file contains an unknown variable "CUSTOM_VAR=value"
    When the configuration is loaded
    Then the configuration should load successfully
    And no error should be raised for the unknown variable
