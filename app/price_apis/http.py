"""The HTTP details every price API shares: timeouts, which errors are worth retrying, and how errors are reported."""

import random
from collections.abc import Callable

import requests

from app.redaction import redact_api_tokens

# seconds to wait for the API to accept the connection, and then for each read of its answer, so a server that stops
# answering fails the attempt (as a requests.Timeout, which is retried) instead of hanging the run
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0
# the most characters of an error answer's body quoted in a log line
ERROR_DETAIL_LIMIT = 200

# a price request that meets a transient error (see 'is_transient') is tried up to RETRY_ATTEMPTS times, waiting
# about RETRY_BASE_DELAY seconds before the first retry and twice as long before each later one
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0
# the longest wait, whatever the API's Retry-After asks for, so a spent hourly limit cannot stall a run
RETRY_MAX_DELAY = 60.0


def response_of(error: Exception) -> requests.Response | None:
    """The HTTP response behind an error, if any: requests' HTTPError carries it."""
    return getattr(error, 'response', None)


def error_detail(error: Exception) -> str:
    """
    The error, followed by what the API said in the body of an error answer (its JSON 'detail', or else the text),
    shortened to ERROR_DETAIL_LIMIT characters. Every api_token= value is masked, wherever it appears.
    """
    return redact_api_tokens(_error_detail(error))


def _error_detail(error: Exception) -> str:
    response = response_of(error)
    if response is None:
        return str(error)
    try:
        body = response.json()
        detail = body.get('detail') if isinstance(body, dict) else None
    except ValueError:
        detail = None
    detail = ' '.join(str(detail if detail is not None else response.text).split())
    if len(detail) > ERROR_DETAIL_LIMIT:
        detail = detail[: ERROR_DETAIL_LIMIT - 1] + '…'
    return f'{error}: {detail}' if detail else str(error)


def is_transient(error: Exception) -> bool:
    """True for errors worth retrying: a 429, a 5xx, or a connection that failed or timed out."""
    if isinstance(error, requests.ConnectionError | requests.Timeout):
        return True
    response = response_of(error)
    return response is not None and (response.status_code == 429 or response.status_code >= 500)


def retry_delay(error: Exception, attempt: int, rand: Callable[[], float] = random.random) -> float:
    """
    Seconds to wait before retrying after 'attempt' (1 for the first) failed with 'error'.

    Honours a Retry-After given in seconds, up to RETRY_MAX_DELAY. Otherwise waits RETRY_BASE_DELAY doubled for each
    earlier attempt, times a random factor between 0.5 and 1.5, so concurrent retries do not line up.
    """
    response = response_of(error)
    retry_after = response.headers.get('Retry-After') if response is not None else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), RETRY_MAX_DELAY)
        except ValueError:
            pass  # an HTTP date, which neither API sends: use the backoff
    return min(RETRY_BASE_DELAY * 2 ** (attempt - 1) * (0.5 + rand()), RETRY_MAX_DELAY)
