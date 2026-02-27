"""Tests for retry logic and error classification."""

import pytest

from pydantic import ValidationError

from valuation_tool.services.retry import (
    AuthenticationError,
    LinkedInBlockedError,
    classify_error,
    is_retryable_error,
)


class TestRetryableErrors:
    @pytest.mark.parametrize("error_type", [
        ConnectionError,
        TimeoutError,
        OSError,
    ])
    def test_transient_errors_retryable(self, error_type):
        assert is_retryable_error(error_type("test")) is True

    def test_validation_error_not_retryable(self):
        try:
            raise ValueError("bad value")
        except ValueError as exc:
            assert is_retryable_error(exc) is False

    def test_auth_error_not_retryable(self):
        assert is_retryable_error(AuthenticationError("bad key")) is False


class TestErrorClassification:
    def test_auth_failure(self):
        assert classify_error(AuthenticationError("bad")) == "Auth Failure"

    def test_connection_error(self):
        assert classify_error(ConnectionError("lost")) == "Transient Network"

    def test_timeout(self):
        assert classify_error(TimeoutError("slow")) == "Timeout"

    def test_validation_error(self):
        assert classify_error(ValueError("bad data")) == "Data Validation"
