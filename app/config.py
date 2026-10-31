import os
import sys

from dotenv import load_dotenv

_REQUIRED_KEYS = ('TIINGO_API_KEY', 'DB_NAME')

# When the environment already holds every key (the container sets them), it is
# the whole configuration and stdin is left alone: a non-interactive run would
# otherwise block reading an open stdin. Otherwise read a .env piped on stdin,
# or the .env file.
if not all(os.getenv(key) for key in _REQUIRED_KEYS):
    if not sys.stdin.isatty():
        load_dotenv(stream=sys.stdin)
    else:
        load_dotenv()


class Config:
    @staticmethod
    def get_tiingo_api_key():
        return os.getenv('TIINGO_API_KEY')

    @staticmethod
    def get_db_name():
        return os.getenv('DB_NAME')

    @staticmethod
    def get_config():
        """Get all config values as a dictionary."""
        result = {
            'TIINGO_API_KEY': Config.get_tiingo_api_key(),
            'DB_NAME': Config.get_db_name(),
        }

        optional_keys = []

        for key in optional_keys:
            val = os.getenv(key)
            if val is not None:
                result[key] = val

        return result
