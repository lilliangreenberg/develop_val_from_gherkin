@feature-003 @social-media-discovery
Feature: Social Media Discovery
  As a portfolio monitoring operator
  I want to discover social media profiles for portfolio companies across 12 platforms
  So that I can track their online presence comprehensively

  Background:
    Given the system is configured with valid environment variables
    And the database is initialized with the required schema

  # ==========================================================================
  # Homepage Discovery (Default, Cost-Optimized)
  # ==========================================================================

  @happy-path @homepage-discovery
  Scenario: Discover social media links from company homepage
    Given the database contains company "Acme Corp" with homepage URL "https://www.acme.com"
    And the Firecrawl API returns HTML for "https://www.acme.com" containing:
      """
      <footer>
        <a href="https://linkedin.com/company/acme">LinkedIn</a>
        <a href="https://twitter.com/acme">Twitter</a>
        <a href="https://github.com/acme">GitHub</a>
      </footer>
      """
    When I run the "discover-social-media" command
    Then 3 social media links should be stored for "Acme Corp"
    And the links should include platform "linkedin" with URL containing "linkedin.com/company/acme"
    And the links should include platform "twitter" with URL containing "twitter.com/acme"
    And the links should include platform "github" with URL containing "github.com/acme"

  @homepage-discovery @batch-api
  Scenario: Use batch API for homepage discovery
    Given the database contains 5 companies with homepage URLs
    When I run the "discover-social-media" command with default batch size 50
    Then the Firecrawl batch API should be called with all 5 URLs in one batch
    And all homepages should be scraped with only_main_content set to False

  @homepage-discovery @batch-size
  Scenario Outline: Configurable batch size for homepage discovery
    Given the database contains <company_count> companies with homepage URLs
    When I run "discover-social-media" with "--batch-size <batch_size>"
    Then URLs should be grouped into batches of at most <batch_size>

    Examples:
      | company_count | batch_size |
      | 100           | 50         |
      | 200           | 100        |
      | 10            | 50         |

  @homepage-discovery @single-company
  Scenario: Discover social media for a single company by ID
    Given the database contains company "Acme Corp" with id 42
    When I run "discover-social-media" with "--company-id 42"
    Then only "Acme Corp" should be processed
    And the batch API should be bypassed for single company mode

  @homepage-discovery @limit
  Scenario: Limit the number of companies processed
    Given the database contains 50 companies with homepage URLs
    When I run "discover-social-media" with "--limit 10"
    Then only 10 companies should be processed

  @homepage-discovery @no-homepage
  Scenario: Skip companies without homepage URLs
    Given the database contains companies:
      | name       | homepage_url           |
      | Has URL Co | https://www.hasurl.com |
      | No URL Co  |                        |
    When I run the "discover-social-media" command
    Then "No URL Co" should be skipped
    And only "Has URL Co" should be processed

  @homepage-discovery @no-links
  Scenario: Handle company with no social media links on homepage
    Given company "Minimal Corp" has a homepage with no social media links
    When social media discovery is performed for "Minimal Corp"
    Then 0 social media links should be stored for "Minimal Corp"
    And the discovery result should report empty discovered links

  # ==========================================================================
  # Platform Detection
  # ==========================================================================

  @platform-detection
  Scenario Outline: Detect platform from URL pattern
    Given a URL "<url>" is found on a company homepage
    When platform detection is performed
    Then the detected platform should be "<platform>"

    Examples:
      | url                                          | platform  |
      | https://linkedin.com/company/acme            | linkedin  |
      | https://www.linkedin.com/in/john-doe         | linkedin  |
      | https://twitter.com/acmecorp                 | twitter   |
      | https://x.com/acmecorp                       | twitter   |
      | https://youtube.com/@acme                    | youtube   |
      | https://youtube.com/channel/UC123abc         | youtube   |
      | https://youtube.com/c/AcmeCorp               | youtube   |
      | https://bsky.app/profile/acme.bsky.social    | bluesky   |
      | https://facebook.com/acmecorp                | facebook  |
      | https://fb.com/acmecorp                      | facebook  |
      | https://m.facebook.com/acmecorp              | facebook  |
      | https://instagram.com/acmecorp               | instagram |
      | https://github.com/acme                      | github    |
      | https://tiktok.com/@acme                     | tiktok    |
      | https://medium.com/@acme                     | medium    |
      | https://acme.medium.com                      | medium    |
      | https://mastodon.social/@acme                | mastodon  |
      | https://threads.net/@acme                    | threads   |
      | https://pinterest.com/acme                   | pinterest |

  @platform-detection @non-social
  Scenario Outline: Reject non-social-media URLs
    Given a URL "<url>" is found on a company homepage
    When platform detection is performed
    Then the URL should not be classified as a social media link

    Examples:
      | url                                    |
      | https://www.google.com                 |
      | https://www.acme.com/about             |
      | https://docs.acme.com                  |
      | mailto:info@acme.com                   |

  # ==========================================================================
  # Link Extraction Strategies
  # ==========================================================================

  @link-extraction @markdown
  Scenario: Extract social links from markdown content
    Given the Firecrawl markdown output contains:
      """
      Follow us on [Twitter](https://twitter.com/acme) and
      [LinkedIn](https://linkedin.com/company/acme).
      """
    When link extraction is performed on the markdown
    Then 2 social media URLs should be extracted

  @link-extraction @html-href
  Scenario: Extract social links from HTML anchor tags
    Given the HTML content contains:
      """
      <a href="https://twitter.com/acme">Follow us</a>
      <a href="https://linkedin.com/company/acme">Connect</a>
      """
    When link extraction is performed on the HTML
    Then 2 social media URLs should be extracted

  @link-extraction @schema-org
  Scenario: Extract social links from Schema.org JSON-LD
    Given the HTML content contains Schema.org structured data:
      """
      <script type="application/ld+json">
      {
        "@type": "Organization",
        "name": "Acme Corp",
        "sameAs": [
          "https://twitter.com/acme",
          "https://linkedin.com/company/acme",
          "https://facebook.com/acme"
        ]
      }
      </script>
      """
    When link extraction is performed on the HTML
    Then the 3 sameAs URLs should be extracted as social media links

  @link-extraction @meta-tags
  Scenario: Extract social links from meta tags
    Given the HTML content contains meta tags:
      """
      <meta name="twitter:site" content="@acmecorp">
      <meta property="og:url" content="https://www.acme.com">
      """
    When link extraction is performed on the HTML
    Then the Twitter handle "@acmecorp" should be extracted

  @link-extraction @aria-labels
  Scenario: Extract social links from aria-label attributes
    Given the HTML content contains:
      """
      <a href="https://twitter.com/acme" aria-label="Follow us on Twitter">
        <svg>...</svg>
      </a>
      """
    When link extraction is performed on the HTML
    Then the Twitter URL should be extracted from the aria-labeled element

  @link-extraction @regex-fallback
  Scenario: Extract social links via regex pattern matching
    Given the HTML content contains social URLs embedded in JavaScript:
      """
      <script>var socialLinks = ["https://twitter.com/acme"];</script>
      """
    When regex-based link extraction is performed
    Then the Twitter URL should be extracted via pattern matching

  # ==========================================================================
  # HTML Region Detection
  # ==========================================================================

  @html-region
  Scenario Outline: Detect HTML region for discovered links
    Given a social media link is found within a "<tag>" element
    When HTML region detection is performed
    Then the html_location should be "<region>"

    Examples:
      | tag    | region  |
      | footer | footer  |
      | header | header  |
      | nav    | nav     |
      | aside  | aside   |
      | main   | main    |

  @html-region @unknown
  Scenario: Set unknown region when tag is not identifiable
    Given a social media link is found outside standard semantic elements
    When HTML region detection is performed
    Then the html_location should be "unknown"

  # ==========================================================================
  # URL Normalization
  # ==========================================================================

  @url-normalization
  Scenario Outline: Normalize social media URLs to canonical form
    Given a raw social media URL "<raw_url>" is discovered
    When URL normalization is performed
    Then the normalized URL should be "<normalized_url>"

    Examples:
      | raw_url                                          | normalized_url                          |
      | https://github.com/acme/repo-name                | https://github.com/acme                 |
      | https://linkedin.com/company/acme/about/         | https://linkedin.com/company/acme       |
      | https://twitter.com/acme/                         | https://twitter.com/acme                |
      | https://www.twitter.com/acme                      | https://twitter.com/acme                |
      | https://twitter.com/acme?ref=website              | https://twitter.com/acme                |

  @url-normalization @github
  Scenario: Normalize GitHub repo URLs to org level
    Given a GitHub URL "https://github.com/acme/awesome-project" is discovered
    When URL normalization is performed
    Then the normalized URL should be "https://github.com/acme"

  @url-normalization @linkedin
  Scenario: Normalize LinkedIn company URLs with trailing paths
    Given a LinkedIn URL "https://linkedin.com/company/acme/jobs/" is discovered
    When URL normalization is performed
    Then the normalized URL should be "https://linkedin.com/company/acme"

  # ==========================================================================
  # Deduplication
  # ==========================================================================

  @deduplication
  Scenario: Deduplicate links found multiple times on same page
    Given the HTML contains 3 links to "https://twitter.com/acme" in different locations
    When social media discovery is performed
    Then only 1 social media link for "https://twitter.com/acme" should be stored

  @deduplication @database-constraint
  Scenario: Database UNIQUE constraint prevents duplicate links per company
    Given the database already contains a social media link for company 1:
      | platform | profile_url                      |
      | twitter  | https://twitter.com/acme         |
    When discovery finds the same URL "https://twitter.com/acme" again
    Then the duplicate should be silently skipped
    And only 1 record should exist for that company and URL

  @deduplication @cross-page
  Scenario: Deduplicate links discovered across multiple pages
    Given full-site crawl discovers "https://twitter.com/acme" on 5 different pages
    When link aggregation is performed
    Then only 1 unique link should be stored for that URL

  # ==========================================================================
  # Account Classification
  # ==========================================================================

  @account-classification
  Scenario: Classify link found in footer as company account
    Given a LinkedIn link is found in the footer region of company "Acme Corp"
    And the link URL is "https://linkedin.com/company/acme"
    When account classification is performed
    Then the account_type should be "company"
    And the account_confidence should be high

  @account-classification @personal
  Scenario: Classify personal LinkedIn profile link
    Given a LinkedIn link "https://linkedin.com/in/john-doe" is found in main content
    When account classification is performed
    Then the account_type should be "personal"

  @account-classification @name-matching
  Scenario: Higher confidence when company name appears in account handle
    Given company "Acme Corp" has a discovered Twitter link "https://twitter.com/acmecorp"
    When account classification is performed
    Then the account_confidence should be higher than for a non-matching handle

  # ==========================================================================
  # Logo Extraction and Verification
  # ==========================================================================

  @logo-extraction
  Scenario: Extract company logo from homepage
    Given company "Acme Corp" has a homepage with a logo image in the header
    When logo extraction is performed
    Then a CompanyLogo should be stored with:
      | field               | value                 |
      | image_format        | a valid format        |
      | perceptual_hash     | a non-empty string    |
      | extraction_location | header                |

  @logo-verification
  Scenario: Verify social media link via logo comparison
    Given company "Acme Corp" has a stored logo with perceptual hash "abc123"
    And a social media profile has a logo with perceptual hash similarity score 0.92
    When logo verification is performed
    Then the link verification_status should be "logo_matched"
    And the similarity_score should be 0.92

  @logo-verification @no-match
  Scenario: Mark link as unverified when logo does not match
    Given company "Acme Corp" has a stored logo
    And a social media profile has a very different logo with similarity score 0.20
    When logo verification is performed
    Then the link verification_status should be "unverified"

  @logo-extraction @failure
  Scenario: Continue discovery when logo extraction fails
    Given logo extraction fails for company "Acme Corp"
    When social media discovery is performed
    Then the social media links should still be stored
    And the verification_status should be "unverified" for all links

  # ==========================================================================
  # Blog Detection
  # ==========================================================================

  @blog-detection
  Scenario Outline: Detect blog links from various patterns
    Given a URL "<url>" is found on company "Acme Corp" homepage
    When blog detection is performed
    Then the URL should be detected as a blog with type "<blog_type>"

    Examples:
      | url                             | blog_type     |
      | https://blog.acme.com           | company_blog  |
      | https://www.acme.com/blog       | company_blog  |
      | https://acme.medium.com         | medium        |
      | https://acme.substack.com       | substack      |

  @blog-detection @normalization
  Scenario Outline: Normalize blog URLs to hub level
    Given a blog post URL "<post_url>" is found
    When blog URL normalization is performed
    Then the normalized blog URL should be "<hub_url>"

    Examples:
      | post_url                                   | hub_url                    |
      | https://blog.acme.com/2024/01/my-post      | https://blog.acme.com      |
      | https://www.acme.com/blog/category/post     | https://www.acme.com/blog  |
      | https://acme.substack.com/p/article-title   | https://acme.substack.com  |

  @blog-detection @storage
  Scenario: Store blog links in blog_links table
    Given a blog URL "https://blog.acme.com" is discovered for company "Acme Corp"
    When the blog link is stored
    Then a blog_links record should exist with:
      | field            | value         |
      | blog_type        | company_blog  |
      | blog_url         | blog.acme.com |
      | is_active        | true          |
      | discovery_method | page_footer   |

  # ==========================================================================
  # Full-Site Discovery
  # ==========================================================================

  @full-site-discovery
  Scenario: Discover social media across entire website
    Given company "Acme Corp" with id 42 has homepage URL "https://www.acme.com"
    And the Firecrawl crawl API returns 10 pages for "https://www.acme.com"
    And 3 of those pages contain social media links
    When I run "discover-social-full-site" with "--company-id 42"
    Then social media links from all 3 pages should be aggregated
    And the discovery_method should be "full_site_crawl" for all links

  @full-site-discovery @depth
  Scenario: Respect maximum crawl depth
    When I run "discover-social-full-site" with "--company-id 42 --max-depth 2"
    Then the Firecrawl crawl API should be called with max_depth 2

  @full-site-discovery @max-pages
  Scenario: Respect maximum pages limit
    When I run "discover-social-full-site" with "--company-id 42 --max-pages 25"
    Then the Firecrawl crawl API should be called with max_pages 25

  @full-site-discovery @subdomains
  Scenario: Include subdomains in full-site crawl by default
    When I run "discover-social-full-site" with "--company-id 42"
    Then the crawl should include subdomains

  @full-site-discovery @no-subdomains
  Scenario: Exclude subdomains when requested
    When I run "discover-social-full-site" with "--company-id 42 --no-subdomains"
    Then the crawl should exclude subdomains

  # ==========================================================================
  # YouTube Video Resolution
  # ==========================================================================

  @youtube-resolution
  Scenario: Resolve YouTube embed URL to channel URL
    Given a YouTube embed URL "https://youtube.com/embed/VIDEO123" is found
    And the YouTube oEmbed API returns author_url "https://www.youtube.com/@acmechannel"
    When YouTube video resolution is performed
    Then the resolved URL should be "https://www.youtube.com/@acmechannel"

  @youtube-resolution @timeout
  Scenario: Handle YouTube oEmbed API timeout
    Given a YouTube embed URL is found
    And the YouTube oEmbed API times out
    When YouTube video resolution is performed
    Then the embed URL should be stored as-is without resolution

  # ==========================================================================
  # Batch Social Discovery
  # ==========================================================================

  @batch-discovery
  Scenario: Run batch discovery with parallel workers
    Given the database contains 20 companies with homepage URLs
    When I run "discover-social-batch" with "--limit 20 --max-workers 5"
    Then up to 5 companies should be processed in parallel
    And all results should be stored in the database

  @batch-discovery @worker-config
  Scenario: Configurable worker count for batch processing
    When I run "discover-social-batch" with "--max-workers 10"
    Then the ThreadPoolExecutor should use 10 workers

  # ==========================================================================
  # Social Media Link Model Validation
  # ==========================================================================

  @validation @social-link-model
  Scenario: Platform enum includes all 13 supported platforms
    Then the Platform enum should include:
      | platform  |
      | linkedin  |
      | twitter   |
      | youtube   |
      | bluesky   |
      | facebook  |
      | instagram |
      | github    |
      | tiktok    |
      | medium    |
      | mastodon  |
      | threads   |
      | pinterest |
      | blog      |

  @validation @social-link-model
  Scenario: Discovery method enum values
    Then the DiscoveryMethod enum should include:
      | method           |
      | page_footer      |
      | page_header      |
      | page_content     |
      | full_site_crawl  |

  @validation @social-link-model
  Scenario: Verification status enum values
    Then the VerificationStatus enum should include:
      | status             |
      | logo_matched       |
      | unverified         |
      | manually_reviewed  |
      | flagged            |

  @validation @social-link-model
  Scenario: Account type enum values
    Then the AccountType enum should include:
      | type    |
      | company |
      | personal|
      | unknown |

  @validation @social-link-model
  Scenario: Similarity score must be between 0 and 1
    Given a social media link has similarity_score 1.5
    When the model is validated
    Then the validation should fail
