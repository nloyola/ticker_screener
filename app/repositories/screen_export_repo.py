from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from app.container import Container
from app.db import connect
from app.markets import market_sql
from app.models import SIGNAL_KIND, StockCriteria
from app.repositories.stock_criteria_repo import COLUMNS as CRITERIA_COLUMNS

# the stock_criteria columns StockCriteria holds: date_created is the row's session instead, and the dates are TEXT,
# read back as dates
_CRITERIA_FIELDS = tuple(c for c in CRITERIA_COLUMNS if c != 'date_created')
_DATE_FIELDS = ('breakout_date', 'swing_low_date')

# the follow-up events (FollowUpEvent.event), each with the signal_track column holding its date
FAILED_BREAKOUT = 'failed_breakout'
RETEST = 'retest'
_EVENT_DATES = {FAILED_BREAKOUT: 'failure_date', RETEST: 'retest_date'}


def _date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


@dataclass(frozen=True)
class DayRange:
    """A ticker's high and low on its scored session, as price_data holds them; None where it holds none."""

    high: float | None
    low: float | None


@dataclass(frozen=True)
class ExportRow:
    """
    One ticker with a status on its market's session, as stock_criteria holds it: its criteria, that session, the
    names of its subsector and sector (None when it is in none), the id of its record in the signal archive under
    the screener version asked for (None when it has none, as a listing that is not its issuer's primary) and its
    high and low on that session (price_data).
    """

    criteria: StockCriteria
    session_date: date
    sector: str | None
    subsector: str | None
    archive_id: int | None
    day: DayRange


@dataclass(frozen=True)
class FollowUpEvent:
    """
    A signal's breakout failure or retest (signal_track), as decided on one of the sessions asked for: the signal's
    archive id, ticker, market and session, the breakout level its tracking judged the event against (in the path's
    basis), the event (FAILED_BREAKOUT or RETEST) and its session, and that session's index in the signal's path with
    its close, low, volume and prior 20-day average volume (signal_price), None where the path has no such session.
    """

    archive_id: int
    ticker: str
    market: str
    breakout_date: date
    breakout_level: float | None
    event: str
    event_date: date
    sessions_after_breakout: int | None
    close: float | None
    low: float | None
    volume: int | None
    avg_volume_20d: float | None


class ScreenExportRepo:
    """
    The stored values the screen export (app/screen_export.py) reports for the sessions a run scored: the
    stock_criteria rows with a status and the follow-up events of the archived signals. It only reads.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def rows(self, sessions: Mapping[str, date], screener_version: str) -> list[ExportRow]:
        """
        Each stock_criteria row with a status of each market in 'sessions' dated by that market's session, with its
        archive id under 'screener_version'; a row of an earlier session, as a market another run scored, is left out.
        """
        columns = ', '.join(f'sc.{c}' for c in ('id', *_CRITERIA_FIELDS, 'date_created'))
        sql = f"""
        SELECT {columns}, se.sector AS sector_name, su.subsector AS subsector_name, s.id AS archive_id,
               p.high AS day_high, p.low AS day_low
          FROM stock_criteria sc
          LEFT JOIN subsector su ON su.id = sc.subsector_id
          LEFT JOIN sector se ON se.id = su.sector_id
          LEFT JOIN signal s ON s.ticker = sc.ticker AND s.session_date = sc.date_created AND s.screener_version = ?
          LEFT JOIN price_data p ON p.ticker = sc.ticker AND p.date = sc.date_created
         WHERE sc.status IS NOT NULL AND {market_sql('sc.ticker')} = ? AND sc.date_created = ?
         ORDER BY sc.ticker
        """
        found: list[ExportRow] = []
        with connect(self.db_path) as con:
            for market, session_date in sessions.items():
                for row in con.execute(sql, (screener_version, market, session_date.isoformat())):
                    values = {c: row[c] for c in _CRITERIA_FIELDS}
                    values.update({c: _date(values[c]) for c in _DATE_FIELDS})
                    found.append(
                        ExportRow(
                            criteria=StockCriteria(id=row['id'], **values),
                            session_date=row['date_created'],
                            sector=row['sector_name'],
                            subsector=row['subsector_name'],
                            archive_id=row['archive_id'],
                            day=DayRange(row['day_high'], row['day_low']),
                        )
                    )
        return found

    def follow_ups(self, windows: Mapping[str, tuple[date, date]], screener_version: str) -> list[FollowUpEvent]:
        """
        Each breakout failure and retest of a signal (kind signal) of 'screener_version', live or backfill, of each
        market in 'windows' that happened from the first to the last session of that market's window, by session,
        ticker and event.
        """
        found: list[FollowUpEvent] = []
        with connect(self.db_path) as con:
            for market, (first, last) in windows.items():
                for event, column in _EVENT_DATES.items():
                    rows = con.execute(
                        f"""
                        SELECT s.id, s.ticker, s.market, s.session_date, t.breakout_level, t.{column} AS event_date,
                               p.session_index, p.close, p.low, p.volume, p.avg_volume_20d
                          FROM signal_track t
                          JOIN signal s ON s.id = t.signal_id
                          LEFT JOIN signal_price p ON p.signal_id = s.id AND p.session_date = t.{column}
                         WHERE s.kind = ? AND s.screener_version = ? AND s.market = ? AND t.{column} BETWEEN ? AND ?
                        """,
                        (SIGNAL_KIND, screener_version, market, first.isoformat(), last.isoformat()),
                    )
                    found.extend(
                        FollowUpEvent(
                            archive_id=row['id'],
                            ticker=row['ticker'],
                            market=row['market'],
                            breakout_date=date.fromisoformat(row['session_date']),
                            breakout_level=row['breakout_level'],
                            event=event,
                            event_date=date.fromisoformat(row['event_date']),
                            sessions_after_breakout=row['session_index'],
                            close=row['close'],
                            low=row['low'],
                            volume=row['volume'],
                            avg_volume_20d=row['avg_volume_20d'],
                        )
                        for row in rows
                    )
        return sorted(found, key=lambda e: (e.event_date, e.ticker, e.event))
