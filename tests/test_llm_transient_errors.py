from types import SimpleNamespace

from src.llm.errors import (
    classify_litellm_transient_error,
    litellm_retry_after_seconds,
)


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, retry_after: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers={"Retry-After": retry_after} if retry_after else {},
        )


def test_rate_limit_and_server_errors_are_transient() -> None:
    assert classify_litellm_transient_error(
        ProviderError("too many requests", status_code=429)
    )
    assert classify_litellm_transient_error(
        ProviderError("upstream unavailable", status_code=503)
    )


def test_balance_or_hard_quota_error_is_not_retried_even_when_status_is_429() -> None:
    assert not classify_litellm_transient_error(
        ProviderError("insufficient_balance", status_code=429)
    )
    assert not classify_litellm_transient_error(
        ProviderError("insufficient_quota", status_code=429)
    )
    assert not classify_litellm_transient_error(
        ProviderError("account balance insufficient", status_code=429)
    )


def test_retry_after_hint_is_extracted_and_capped() -> None:
    assert litellm_retry_after_seconds(
        ProviderError("rate limit", status_code=429, retry_after="17.5")
    ) == 17.5
    assert litellm_retry_after_seconds(
        ProviderError("rate limit", status_code=429, retry_after="3600")
    ) == 900.0
