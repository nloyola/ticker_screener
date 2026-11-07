import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from dotenv import dotenv_values

TIINGO_API_KEY = 'TIINGO_API_KEY'
EODHD_API_KEY = 'EODHD_API_KEY'
DB_NAME = 'DB_NAME'
# the price API to fetch prices from; a command that requires it also requires that API's key
PRICE_API = 'PRICE_API'
KEYS = (TIINGO_API_KEY, EODHD_API_KEY, DB_NAME, PRICE_API)

TIINGO = 'tiingo'
EODHD = 'eodhd'
# the PRICE_API values, each with the key its API needs
PRICE_API_KEYS = {TIINGO: TIINGO_API_KEY, EODHD: EODHD_API_KEY}
DEFAULT_PRICE_API = TIINGO


class ConfigError(Exception):
    """A required config key is missing, or PRICE_API is not a known API; the message names what is wrong."""


@dataclass(frozen=True)
class Config:
    """The config values; a key the command did not require may be None."""

    db_name: str | None
    tiingo_api_key: str | None
    eodhd_api_key: str | None
    # one of PRICE_API_KEYS, when the command requires PRICE_API
    price_api: str | None = None


def load_config(require: Iterable[str]) -> Config:
    """
    Reads the config and raises ConfigError naming every key in 'require' that is missing or empty.

    Requiring PRICE_API requires a known price API (tiingo when unset; any other value is refused, never replaced by
    the default) and that API's key (PRICE_API_KEYS); the other API's key is not required.

    Each key takes the first non-empty value from: the environment, a .env piped on stdin, the .env file in the
    working directory. The file, when there is one, is always read, so a PRICE_API set only there applies even when
    the environment holds every key (the container images hold no .env). stdin is read only when it is not a terminal
    and the environment lacks a required key: when the environment already holds them all (the containers set them),
    a non-interactive run would otherwise block reading an open stdin. The environment is never changed.
    """
    required = tuple(require)
    unknown = [key for key in required if key not in KEYS]
    if unknown:
        raise ValueError(f'unknown config keys: {", ".join(unknown)}')

    values = {key: os.getenv(key) or None for key in KEYS}
    sources = []
    if not all(values[key] for key in _required_keys(required, values)):
        if sys.stdin is not None and not sys.stdin.isatty():
            sources.append(dotenv_values(stream=sys.stdin))
    sources.append(dotenv_values('.env'))
    for source in sources:
        for key in KEYS:
            values[key] = values[key] or source.get(key) or None

    price_api = None
    if PRICE_API in required:
        price_api = values[PRICE_API] or DEFAULT_PRICE_API
        if price_api not in PRICE_API_KEYS:
            raise ConfigError(f"{PRICE_API} is '{price_api}', not one of: {', '.join(PRICE_API_KEYS)}")

    missing = [key for key in _required_keys(required, values) if not values[key]]
    if missing:
        raise ConfigError(f'missing required config: {", ".join(missing)} (set in the environment or in .env)')

    return Config(
        db_name=values[DB_NAME],
        tiingo_api_key=values[TIINGO_API_KEY],
        eodhd_api_key=values[EODHD_API_KEY],
        price_api=price_api,
    )


def _required_keys(required: tuple[str, ...], values: dict[str, str | None]) -> list[str]:
    """The keys that must have a value: PRICE_API stands for the key of the API it names, which has a default."""
    keys = [key for key in required if key != PRICE_API]
    if PRICE_API in required:
        api_key = PRICE_API_KEYS.get(values[PRICE_API] or DEFAULT_PRICE_API)
        if api_key and api_key not in keys:
            keys.append(api_key)
    return keys
