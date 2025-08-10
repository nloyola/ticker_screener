import datetime as _dt
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Enable TIMESTAMP <-> datetime conversion via ISO 8601
sqlite3.register_adapter(_dt.datetime, lambda dt: dt.isoformat(timespec='seconds'))
sqlite3.register_converter('TIMESTAMP', lambda b: _dt.datetime.fromisoformat(b.decode()))


class SQLite:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,  # use TIMESTAMP converter
            isolation_level=None,  # autocommit mode; we'll BEGIN manually
        )
        con.row_factory = sqlite3.Row
        try:
            con.execute('BEGIN')
            yield con
            con.execute('COMMIT')
        except:  # noqa: E722
            con.execute('ROLLBACK')
            raise
        finally:
            con.close()


def init_db(db_path: str):
    schema = Path('schema.sql').read_text(encoding='utf-8')
    con = sqlite3.connect(db_path)
    try:
        con.executescript(schema)
    finally:
        con.close()
