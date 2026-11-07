"""
Keeps API keys out of everything the screener writes: log lines, tracebacks and messages.

redact() is the one function that does it. RedactFilter applies it to every log record on the handlers it is attached
to (config/0-stdout.json), and redacting_excepthook to a traceback that escapes main(), which reaches stderr outside
logging. The dashboard's runner redacts the screener's output again, as a second line of defence only.
"""

import logging
import os
import re
import sys
import traceback
from collections.abc import Iterable
from types import TracebackType

# the config keys whose values are secrets; their names, not their values, live in app.config
SECRET_KEYS = ('TIINGO_API_KEY', 'EODHD_API_KEY')

MASK = '***'

# EODHD takes its key in the URL: whatever follows api_token=, up to the next &, whitespace or quote
_API_TOKEN = re.compile(r'(api_token=)[^&\s\'"]*')

_registered: set[str] = set()


def register_secrets(values: Iterable[str | None]) -> None:
    """Adds secret values to redact from now on, beside those of SECRET_KEYS in the environment; skips empty ones."""
    _registered.update(value for value in values if value)


def redact_api_tokens(text: str) -> str:
    """Replaces every api_token= value in 'text' with MASK, whatever the value."""
    return _API_TOKEN.sub(rf'\g<1>{MASK}', text)


def redact(text: str, *extra: str | None) -> str:
    """
    Replaces with MASK every api_token= value in 'text', every registered secret, the current value of each of
    SECRET_KEYS in the environment, and any of 'extra'.
    """
    secrets = set(_registered)
    secrets.update(os.environ.get(key) or '' for key in SECRET_KEYS)
    secrets.update(value or '' for value in extra)
    secrets.discard('')
    # longest first, so a secret containing another is masked whole
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return redact_api_tokens(text)


class RedactFilter(logging.Filter):
    """
    A handler filter that redacts each record in place before it is formatted: its message with the arguments merged
    in, its traceback and its stack. It never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            # the formatter uses exc_text as it stands rather than formatting exc_info again
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


def redacting_excepthook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    """A sys.excepthook that prints the traceback to stderr as the default one does, redacted."""
    sys.stderr.write(redact(''.join(traceback.format_exception(exc_type, exc, tb))))
    sys.stderr.flush()
