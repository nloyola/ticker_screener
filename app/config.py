import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from dotenv import dotenv_values

TIINGO_API_KEY = 'TIINGO_API_KEY'
DB_NAME = 'DB_NAME'
KEYS = (TIINGO_API_KEY, DB_NAME)


class ConfigError(Exception):
    """A required config key is missing; the message names every missing key."""


@dataclass(frozen=True)
class Config:
    """The config values; a key the command did not require may be None."""

    db_name: str | None
    tiingo_api_key: str | None


def load_config(require: Iterable[str]) -> Config:
    """
    Reads the config and raises ConfigError naming every key in 'require' that is missing or empty.

    Each key takes the first non-empty value from: the environment, a .env piped on stdin, the .env file in the
    working directory. When the environment already holds every required key (the containers set them), it is the
    whole configuration and neither stdin nor the file is read: a non-interactive run would otherwise block reading
    an open stdin. stdin is read only when it is not a terminal. The environment is never changed.
    """
    required = tuple(require)
    unknown = [key for key in required if key not in KEYS]
    if unknown:
        raise ValueError(f'unknown config keys: {", ".join(unknown)}')

    values = {key: os.getenv(key) or None for key in KEYS}
    if not all(values[key] for key in required):
        sources = []
        if sys.stdin is not None and not sys.stdin.isatty():
            sources.append(dotenv_values(stream=sys.stdin))
        sources.append(dotenv_values('.env'))
        for source in sources:
            for key in KEYS:
                values[key] = values[key] or source.get(key) or None

    missing = [key for key in required if not values[key]]
    if missing:
        raise ConfigError(f'missing required config: {", ".join(missing)} (set in the environment or in .env)')

    return Config(db_name=values[DB_NAME], tiingo_api_key=values[TIINGO_API_KEY])
