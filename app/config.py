import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from dotenv import load_dotenv

TIINGO_API_KEY = 'TIINGO_API_KEY'
DB_NAME = 'DB_NAME'
KEYS = (TIINGO_API_KEY, DB_NAME)


class ConfigError(Exception):
    """A required config key is missing; the message names every missing key."""


@dataclass(frozen=True)
class Config:
    db_name: str | None
    tiingo_api_key: str | None


def load_config(require: Iterable[str]) -> Config:
    """
    Reads the config and raises ConfigError naming every key in 'require' that is missing or empty.

    When the environment already holds every required key (the containers set them), it is the whole configuration
    and stdin is left alone: a non-interactive run would otherwise block reading an open stdin. Otherwise a .env
    piped on stdin is read when stdin is not a terminal, and the .env file when it is. Neither overrides a key the
    environment already holds.
    """
    required = tuple(require)
    unknown = [key for key in required if key not in KEYS]
    if unknown:
        raise ValueError(f'unknown config keys: {", ".join(unknown)}')

    if not all(os.getenv(key) for key in required):
        if sys.stdin is not None and not sys.stdin.isatty():
            load_dotenv(stream=sys.stdin)
        else:
            load_dotenv()

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ConfigError(f'missing required config: {", ".join(missing)} (set in the environment or in .env)')

    return Config(db_name=os.getenv(DB_NAME), tiingo_api_key=os.getenv(TIINGO_API_KEY))
