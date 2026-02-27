"""Retry logic with exponential backoff and error classification.

Uses tenacity for retries. Classifies errors as retryable or non-retryable
based on exception type and HTTP status codes.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import httpx
import structlog
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger()

T = TypeVar("T")

# HTTP status codes that are retryable (transient server errors and rate limiting)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that are non-retryable (client errors)
NON_RETRYABLE_STATUS_CODES = {401, 403, 404}


class AuthenticationError(Exception):
    """Raised when an API returns 401/403."""


class LinkedInBlockedError(Exception):
    """Raised when LinkedIn blocks the scraper (CAPTCHA, rate limit)."""


def is_retryable_error(exc: BaseException) -> bool:
    """Determine if an exception is transient and worth retrying."""
    # Never retry validation errors
    if isinstance(exc, (ValidationError, ValueError, TypeError)):
        return False

    # Never retry auth errors
    if isinstance(exc, AuthenticationError):
        return False

    # Transient network errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # HTTP response errors
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES

    return False


def _log_retry(retry_state: RetryCallState) -> None:
    """Log each retry attempt."""
    logger.warning(
        "retry_attempt",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
    )


def with_retry(max_attempts: int = 2) -> Callable:
    """Create a tenacity retry decorator with exponential backoff.

    Backoff: min=2s, max=10s, multiplier=1.
    Only retries on transient errors.
    """
    if max_attempts <= 0:
        # No retries — just call once
        def no_retry_decorator(func: Callable[..., T]) -> Callable[..., T]:
            return func
        return no_retry_decorator

    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=_log_retry,
        reraise=True,
    )


def classify_error(exc: BaseException) -> str:
    """Classify an error into a human-readable category."""
    if isinstance(exc, AuthenticationError):
        return "Auth Failure"
    if isinstance(exc, TimeoutError):
        return "Timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "Transient Network"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return "Rate Limiting"
        if code in (401, 403):
            return "Auth Failure"
        return "API Error"
    if isinstance(exc, (ValidationError, ValueError)):
        return "Data Validation"
    return "Unknown"
