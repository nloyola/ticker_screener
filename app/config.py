import os
import sys

from dotenv import load_dotenv

# is there input from stdin?
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
