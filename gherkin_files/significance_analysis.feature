@cross-cutting @significance-analysis
Feature: Significance Analysis System
  As a portfolio monitoring operator
  I want website changes and news articles to be automatically classified by business significance
  So that I can prioritize my review of important events and filter out noise

  Background:
    Given the significance analysis system is initialized
    And the keyword dictionaries are loaded

  # ==========================================================================
  # Positive Keyword Detection
  # ==========================================================================

  @keyword-detection @positive
  Scenario Outline: Detect positive keywords in content
    Given the content contains the text "<text>"
    When keyword analysis is performed
    Then the keyword "<keyword>" should be detected
    And the category should be "<category>"

    Examples:
      | text                                      | keyword          | category            |
      | We raised a $50M Series B round           | series b         | funding_investment  |
      | Announced our seed round today            | seed round       | funding_investment  |
      | Venture capital firm led the round         | venture capital  | funding_investment  |
      | Company valuation reached $1B             | valuation        | funding_investment  |
      | We launched our new product today          | launched         | product_launch      |
      | Beta release now available                 | beta release     | product_launch      |
      | General availability announced             | general availability | product_launch   |
      | Revenue growth exceeded 200%               | revenue growth   | growth_success      |
      | We hit a major milestone this quarter      | milestone        | growth_success      |
      | Company is now profitable                  | profitable       | growth_success      |
      | Strategic partnership with BigCo           | partnership      | partnerships        |
      | Joint venture announced today              | joint venture    | partnerships        |
      | Opening new office in London               | new office       | expansion           |
      | We are hiring 50 engineers                 | hiring           | expansion           |
      | International expansion into Asia          | expansion        | expansion           |
      | Won the Innovation Award 2026              | innovation award | recognition         |
      | Named to top 10 startups list              | top 10           | recognition         |
      | Filed S-1 for public offering              | filed s-1        | ipo_exit            |
      | Going public on NASDAQ                     | nasdaq           | ipo_exit            |

  @keyword-detection @positive @categories
  Scenario: All 7 positive keyword categories are active
    When the positive keyword dictionary is loaded
    Then it should contain keywords in these categories:
      | category            |
      | funding_investment  |
      | product_launch      |
      | growth_success      |
      | partnerships        |
      | expansion           |
      | recognition         |
      | ipo_exit            |

  # ==========================================================================
  # Negative Keyword Detection
  # ==========================================================================

  @keyword-detection @negative
  Scenario Outline: Detect negative keywords in content
    Given the content contains the text "<text>"
    When keyword analysis is performed
    Then the keyword "<keyword>" should be detected
    And the category should be "<category>"

    Examples:
      | text                                      | keyword            | category            |
      | Company shut down operations               | shut down          | closure             |
      | Ceased operations effective immediately     | ceased operations  | closure             |
      | Winding down the business                  | winding down       | closure             |
      | Announced layoffs affecting 200 employees  | layoffs            | layoffs_downsizing  |
      | Workforce reduction of 30%                 | workforce reduction| layoffs_downsizing  |
      | Job cuts across all departments            | job cuts           | layoffs_downsizing  |
      | Filed for Chapter 11 bankruptcy            | chapter 11         | financial_distress  |
      | Company is insolvent                       | insolvent          | financial_distress  |
      | Facing a cash crunch                       | cash crunch        | financial_distress  |
      | Lawsuit filed against the company          | lawsuit            | legal_issues        |
      | Under federal investigation                | investigation      | legal_issues        |
      | Reached settlement in lawsuit              | settlement         | legal_issues        |
      | Suffered a major data breach               | data breach        | security_breach     |
      | Systems hacked by attackers                | hacked             | security_breach     |
      | Ransomware attack disrupted services       | ransomware         | security_breach     |
      | Acquired by BigTech Corp                   | acquired by        | acquisition         |
      | Merged with competitor                     | merged with        | acquisition         |
      | CEO resigned unexpectedly                  | ceo resigned       | leadership_changes  |
      | Founder left the company                   | founder left       | leadership_changes  |
      | Executive is stepping down                 | stepping down      | leadership_changes  |

  @keyword-detection @negative @categories
  Scenario: All 9 negative keyword categories are active
    When the negative keyword dictionary is loaded
    Then it should contain keywords in these categories:
      | category            |
      | closure             |
      | layoffs_downsizing  |
      | financial_distress  |
      | legal_issues        |
      | security_breach     |
      | acquisition         |
      | leadership_changes  |
      | product_failures    |
      | market_exit         |

  # ==========================================================================
  # Insignificant Pattern Detection
  # ==========================================================================

  @keyword-detection @insignificant
  Scenario Outline: Detect insignificant patterns in content
    Given the content contains the text "<text>"
    When keyword analysis is performed
    Then the match should be classified as an insignificant pattern
    And the category should be "<category>"

    Examples:
      | text                                       | category            |
      | Updated font-family to Arial               | css_styling         |
      | Changed background-color to #fff           | css_styling         |
      | Copyright 2026 Company Name                | copyright_year      |
      | All rights reserved                        | copyright_year      |
      | Updated google-analytics tracking code     | tracking_analytics  |
      | Added gtag conversion pixel                | tracking_analytics  |

  @keyword-detection @insignificant @categories
  Scenario: All 3 insignificant pattern categories are active
    When the insignificant pattern dictionary is loaded
    Then it should contain patterns in these categories:
      | category            |
      | css_styling         |
      | copyright_year      |
      | tracking_analytics  |

  # ==========================================================================
  # Classification Rules
  # ==========================================================================

  @classification @rule-1
  Scenario: Rule 1 - Only insignificant patterns with minor magnitude is INSIGNIFICANT
    Given the content matches only insignificant patterns
    And the change magnitude is "minor"
    When significance classification is performed
    Then the classification should be "insignificant"
    And the confidence should be approximately 0.85

  @classification @rule-2
  Scenario: Rule 2 - Two or more negative keywords is SIGNIFICANT
    Given the content contains keywords "layoffs" and "shut down"
    And both are negative keywords
    When significance classification is performed
    Then the classification should be "significant"
    And the confidence should be between 0.80 and 0.95

  @classification @rule-3
  Scenario: Rule 3 - Two or more positive keywords is SIGNIFICANT
    Given the content contains keywords "funding" and "expansion"
    And both are positive keywords
    When significance classification is performed
    Then the classification should be "significant"
    And the confidence should be between 0.80 and 0.90

  @classification @rule-4
  Scenario: Rule 4 - One keyword with major magnitude is SIGNIFICANT
    Given the content contains 1 negative keyword "layoffs"
    And the change magnitude is "major"
    When significance classification is performed
    Then the classification should be "significant"
    And the confidence should be approximately 0.70

  @classification @rule-5
  Scenario: Rule 5 - One keyword with minor magnitude is UNCERTAIN
    Given the content contains 1 positive keyword "hiring"
    And the change magnitude is "minor"
    When significance classification is performed
    Then the classification should be "uncertain"
    And the confidence should be approximately 0.50

  @classification @rule-6
  Scenario: Rule 6 - No keywords detected is INSIGNIFICANT
    Given the content contains no recognized keywords
    When significance classification is performed
    Then the classification should be "insignificant"
    And the confidence should be approximately 0.75

  @classification @precedence
  Scenario: Multiple keywords increase confidence
    Given the content contains 3 negative keywords: "layoffs", "shut down", "bankruptcy"
    When significance classification is performed
    Then the classification should be "significant"
    And the confidence should be higher than for 2 keywords

  # ==========================================================================
  # Sentiment Classification
  # ==========================================================================

  @sentiment
  Scenario Outline: Classify sentiment based on keyword polarity
    Given the content contains <positive_count> positive and <negative_count> negative keywords
    When sentiment classification is performed
    Then the sentiment should be "<expected_sentiment>"

    Examples:
      | positive_count | negative_count | expected_sentiment |
      | 3              | 0              | positive           |
      | 0              | 3              | negative           |
      | 2              | 2              | mixed              |
      | 3              | 2              | mixed              |
      | 1              | 0              | neutral            |
      | 0              | 1              | neutral            |
      | 0              | 0              | neutral            |

  @sentiment @mixed-threshold
  Scenario: Mixed sentiment requires 2+ keywords from each polarity
    Given the content contains 2 positive keywords and 2 negative keywords
    When sentiment classification is performed
    Then the sentiment should be "mixed"

  @sentiment @neutral-threshold
  Scenario: Fewer than 2 total keywords results in neutral sentiment
    Given the content contains 1 positive keyword and 0 negative keywords
    When sentiment classification is performed
    Then the sentiment should be "neutral"

  # ==========================================================================
  # Negation Detection
  # ==========================================================================

  @negation
  Scenario Outline: Detect negated keywords
    Given the content contains the phrase "<phrase>"
    When negation detection is performed
    Then the keyword should be flagged as negated
    And the confidence should be reduced by 20%

    Examples:
      | phrase                      |
      | no funding was received     |
      | not acquired by anyone      |
      | never had layoffs           |
      | without any data breach     |
      | lacks funding               |

  @negation @context
  Scenario: Negated keywords do not flip classification
    Given the content contains "The company was not acquired" as the only keyword match
    When significance classification is performed
    Then the keyword "acquired" should be flagged as negated
    And the overall confidence should be lower than non-negated

  @negation @patterns
  Scenario Outline: Recognize negation word patterns
    Given the text "<prefix> funding" appears in the content
    When negation detection is performed
    Then the keyword "funding" should be flagged as negated

    Examples:
      | prefix   |
      | no       |
      | not      |
      | never    |
      | without  |
      | lacks    |

  # ==========================================================================
  # False Positive Detection
  # ==========================================================================

  @false-positive
  Scenario Outline: Detect false positive keyword matches
    Given the content contains the phrase "<phrase>"
    When false positive detection is performed
    Then the match should be flagged as a false positive
    And the confidence should be reduced by 30%

    Examples:
      | phrase                    |
      | talent acquisition team   |
      | customer acquisition cost |
      | data acquisition pipeline |
      | funding opportunities     |
      | funding sources           |
      | self-funded startup       |

  @false-positive @talent-acquisition
  Scenario: "Talent acquisition" is not flagged as company acquisition
    Given the content contains "Our talent acquisition team is growing"
    When keyword analysis is performed
    Then "acquisition" should be flagged as a false positive
    And the content should not be classified as a company acquisition event

  @false-positive @customer-acquisition
  Scenario: "Customer acquisition" is a marketing term, not M&A
    Given the content contains "Customer acquisition cost decreased by 20%"
    When keyword analysis is performed
    Then "acquisition" should be flagged as a false positive

  @false-positive @combined-reduction
  Scenario: Negation and false positive effects are cumulative
    Given a keyword match is both negated and a false positive
    When confidence adjustment is applied
    Then the total confidence reduction should reflect both adjustments

  # ==========================================================================
  # Keyword Match Model
  # ==========================================================================

  @keyword-match @model
  Scenario: KeywordMatch captures context around the keyword
    Given the keyword "funding" is found at position 50 in content
    When a KeywordMatch is created
    Then it should include:
      | field          | value                    |
      | keyword        | funding                  |
      | position       | 50                       |
      | context_before | up to 50 chars before    |
      | context_after  | up to 50 chars after     |
      | is_negated     | false                    |
      | is_false_positive | false                 |

  # ==========================================================================
  # LLM Validation (Optional)
  # ==========================================================================

  @llm-validation @enabled
  Scenario: LLM validates keyword-based classification when enabled
    Given LLM_VALIDATION_ENABLED is set to "true"
    And ANTHROPIC_API_KEY is configured
    And keyword analysis classifies content as "significant" with confidence 0.80
    When LLM validation is performed
    Then the LLM should receive the content excerpt and detected keywords
    And the LLM result should take precedence over keyword-only classification

  @llm-validation @override
  Scenario: LLM can override keyword classification
    Given keyword analysis classifies content as "significant"
    And the LLM determines the change is actually "insignificant" with reasoning
    When LLM validation completes
    Then the final classification should be "insignificant"
    And the LLM reasoning should be stored in significance_notes

  @llm-validation @fallback
  Scenario: Keyword classification used when LLM fails
    Given LLM_VALIDATION_ENABLED is set to "true"
    And keyword analysis classifies content as "significant" with confidence 0.80
    And the LLM API call fails with a network error
    When LLM validation is attempted
    Then the keyword-based classification should be used as fallback
    And the final classification should remain "significant"

  @llm-validation @disabled
  Scenario: LLM validation is not invoked when disabled
    Given LLM_VALIDATION_ENABLED is set to "false"
    When significance analysis is performed
    Then no LLM API calls should be made
    And only keyword-based classification should be used

  @llm-validation @deterministic
  Scenario: LLM calls use temperature 0.0 for deterministic results
    Given LLM validation is enabled
    When the LLM API is called
    Then the temperature parameter should be 0.0
    And max_tokens should be 500

  @llm-validation @result-model
  Scenario: LLM returns structured validation result
    When the LLM validation completes successfully
    Then the result should contain:
      | field               | type                |
      | classification      | SignificanceClassification |
      | sentiment           | SignificanceSentiment      |
      | confidence          | float (0.0-1.0)           |
      | reasoning           | string                    |
      | validated_keywords  | list of strings           |
      | false_positives     | list of strings           |

  # ==========================================================================
  # Backfill Support
  # ==========================================================================

  @backfill
  Scenario: Backfill significance for existing change records
    Given the database contains 10 change records with NULL significance fields
    And each change record has associated snapshot content
    When I run the "backfill-significance" command
    Then all 10 records should have significance fields populated
    And the output should report 10 records backfilled

  @backfill @dry-run
  Scenario: Preview backfill without updating records
    Given the database contains 5 change records with NULL significance fields
    When I run "backfill-significance" with "--dry-run"
    Then no records should be updated in the database
    And the output should preview what would be updated

  @backfill @batch-size
  Scenario: Backfill processes records in configurable batches
    Given the database contains 250 change records with NULL significance fields
    When I run "backfill-significance" with "--batch-size 50"
    Then records should be processed in batches of 50

  @backfill @skip-populated
  Scenario: Backfill skips records that already have significance data
    Given the database contains change records:
      | id | significance_classification |
      | 1  | significant                 |
      | 2  |                             |
      | 3  | insignificant               |
      | 4  |                             |
    When I run the "backfill-significance" command
    Then only records 2 and 4 should be processed
    And records 1 and 3 should be skipped

  # ==========================================================================
  # Leadership Significance Keywords
  # ==========================================================================

  @leadership-keywords
  Scenario Outline: Detect leadership change keywords
    Given the content contains the text "<text>"
    When keyword analysis is performed
    Then the keyword "<keyword>" should be detected in the "leadership_changes" category

    Examples:
      | text                               | keyword              |
      | CEO departed the company           | ceo departed         |
      | Founder left to pursue new venture | founder left         |
      | CTO left the organization          | cto left             |
      | New CEO appointed                  | new ceo              |
      | Leadership transition announced    | leadership transition|
      | Executive is stepping down         | stepping down        |
      | Board member retired               | retired              |
      | VP resigned from position          | resigned             |

  # ==========================================================================
  # Shared Across Features
  # ==========================================================================

  @shared @website-changes
  Scenario: Significance analysis applied to website change detection
    Given a change record is created for company "Acme Corp"
    And the diff content contains "Series B funding of $50M"
    When significance analysis is performed on the change record
    Then the change record should have significance_classification "significant"
    And the matched_keywords should include "series b" and "funding"

  @shared @news-articles
  Scenario: Significance analysis applied to news articles
    Given a news article about "Acme Corp" with title "Acme Corp Lays Off 30% of Staff"
    When significance analysis is performed on the article
    Then the article should have significance_classification "significant"
    And the significance_sentiment should be "negative"
    And the matched_keywords should include "lays off"
