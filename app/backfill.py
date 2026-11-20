"""
Seeding the signal archive from the stored prices: every session the price window can fully score is replayed with
Analyzer.score_as_of, and the records of every kind it keeps are archived as backfill.
"""

import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from app.analyzer import (
    BACKFILL,
    EMA_MIN_PERIODS,
    LONG_SMA_DAYS,
    SCREENER_VERSION,
    SMA_RISING_DAYS,
    Analyzer,
    archive_kind,
    describe_records,
    rule_settings,
    trading_days_between,
    utc_now,
)
from app.container import Container
from app.markets import MARKETS, UnsupportedTicker, last_complete_session, market_of
from app.models import SIGNAL_KIND
from app.provenance import git_commit
from app.repositories.signal_repo import SignalLink

if TYPE_CHECKING:
    from app.repositories.signal_repo import SignalRepo

logger = logging.getLogger(__name__)

# A session is replayable when its market has enough sessions on or before it for the longest indicators: the
# 200-day SMA with the trading days its rising check reaches back, and the 21-week EMA, empty until EMA_MIN_PERIODS
# weekly closes, with one more week for its rising check. An earlier session would archive signals whose secondary
# indicators are left empty only because the window starts too late, which a live run never sees.
REPLAY_MIN_SESSIONS = LONG_SMA_DAYS + SMA_RISING_DAYS
REPLAY_MIN_WEEKS = EMA_MIN_PERIODS + 1


@dataclass(frozen=True)
class BackfillResult:
    """
    How many sessions the backfill replayed, signals it archived and later signals it relinked, and the records it
    archived of each kind (KINDS in app/models.py), the signals among them.
    """

    sessions: int
    signals: int
    relinked: int
    records: Mapping[str, int] = field(default_factory=dict)


def replayable_sessions(dates: Mapping[str, pd.Index], started_at: datetime) -> dict[date, tuple[str, ...]]:
    """
    Each replayable session, oldest first, with the markets it is replayable in: from each market's session dates
    ('dates', the days any of its tickers has a price), those with REPLAY_MIN_SESSIONS sessions and REPLAY_MIN_WEEKS
    calendar weeks (Monday to Sunday) on or before them, whose candles were final at 'started_at'.
    """
    sessions: dict[date, list[str]] = {}
    for market in MARKETS:
        market_dates = sorted(set(dates.get(market, [])))
        last = last_complete_session(market, started_at)
        weeks: set[tuple[int, int]] = set()
        for count, session in enumerate(market_dates, start=1):
            weeks.add(session.isocalendar()[:2])
            if session > last:
                break
            if count >= REPLAY_MIN_SESSIONS and len(weeks) >= REPLAY_MIN_WEEKS:
                sessions.setdefault(session, []).append(market)
    return {session: tuple(sessions[session]) for session in sorted(sessions)}


def market_dates(prices: pd.DataFrame) -> dict[str, pd.Index]:
    """Each market's session dates: the days any of its tickers has a price, leaving out a ticker of no market."""
    by_market: dict[str, set[date]] = {market: set() for market in MARKETS}
    for ticker, group_df in prices.groupby(level='ticker'):
        try:
            market = market_of(str(ticker))
        except UnsupportedTicker:
            continue
        by_market[market].update(group_df.index.get_level_values('date'))
    return {market: pd.Index(sorted(days)) for market, days in by_market.items()}


@contextmanager
def quiet(name: str) -> Iterator[None]:
    """Keeps the named logger's info lines (one per ticker scored) out of a replay of many sessions."""
    named = logging.getLogger(name)
    level = named.level
    named.setLevel(logging.WARNING)
    try:
        yield
    finally:
        named.setLevel(level)


class Backfill:
    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.analyzer: Analyzer = container.analyzer
        self.signal_repo: SignalRepo = container.signal_repo
        # the clock that decides which sessions are complete, timezone aware
        self.now = now

    def run(self) -> BackfillResult:
        """
        Replays every replayable session (replayable_sessions) oldest first, as a run on it would have scored it,
        and archives the records of every kind it keeps (archive_kind) as backfill, with the current rules and commit
        and no run. Oldest first, each new signal links to the ticker's previous one, and each continuation to its
        breakout. A record already archived for the ticker, session and version, live or backfill, is left as it was,
        so a second backfill adds nothing.

        A signal archived before the backfill, on a session after one it archived, is relinked to the ticker's
        nearest earlier signal (relink). Prints one line per session and a summary.
        """
        prices = self.analyzer.load_price_data()
        if prices is None:
            print('backfill: no price data, nothing to replay')
            return BackfillResult(0, 0, 0)
        sessions = replayable_sessions(market_dates(prices), self.now())
        settings = json.dumps(rule_settings(), sort_keys=True)
        commit = git_commit()
        non_primary = self.analyzer.issuer_repo.non_primary()
        exchanges = self.analyzer.security_state_repo.exchanges()
        archived: Counter[str] = Counter()
        tickers: set[str] = set()
        for session, markets in sessions.items():
            with quiet('app.analyzer'):
                scored = self.analyzer.score_as_of(session, prices, markets)
                stored = self.analyzer.archive_signals(
                    scored, None, settings, commit, source=BACKFILL, non_primary=non_primary, exchanges=exchanges
                )
            archived.update(stored)
            if stored[SIGNAL_KIND]:
                tickers.update(s.criteria.ticker for s in scored if archive_kind(s.criteria) == SIGNAL_KIND)
            print(
                f'{session} ({", ".join(markets)}): scored {len(scored)} ticker{"" if len(scored) == 1 else "s"}, '
                f'archived {describe_records(stored)}'
            )
        relinked = self.relink(tickers, lambda ticker: prices.xs(ticker, level='ticker').index)
        print(
            f'backfill: replayed {len(sessions)} session{"" if len(sessions) == 1 else "s"}, '
            f'archived {describe_records(archived)}, '
            f'relinked {relinked} later signal{"" if relinked == 1 else "s"}'
        )
        return BackfillResult(len(sessions), archived[SIGNAL_KIND], relinked, dict(archived))

    def relink(self, tickers: set[str], dates_of: Callable[[str], pd.Index]) -> int:
        """
        Links each signal of 'tickers' archived under the current screener version whose previous signal is no longer
        the ticker's latest signal of that version on an earlier session, as SignalRepo.previous finds it, to that
        one, with the trading days since it counted from the ticker's price dates ('dates_of', oldest first). A
        signal of another version is left as it is. Returns how many it relinked.
        """
        changed: list[SignalLink] = []
        by_ticker: dict[str, list[SignalLink]] = {}
        for link in self.signal_repo.links(sorted(tickers), SCREENER_VERSION):
            by_ticker.setdefault(link.ticker, []).append(link)
        for ticker, links in by_ticker.items():
            dates = dates_of(ticker)
            previous: SignalLink | None = None
            for i, link in enumerate(links):
                # the latest signal on an earlier session: the last of the session before this one's
                earlier = [other for other in links[:i] if other.session_date < link.session_date]
                previous = earlier[-1] if earlier else None
                previous_id = previous.id if previous else None
                if link.previous_signal_id == previous_id:
                    continue
                days = trading_days_between(dates, previous.session_date, link.session_date) if previous else None
                changed.append(replace(link, previous_signal_id=previous_id, trading_days_since_previous=days))
        relinked = self.signal_repo.relink(changed)
        if relinked:
            logger.info(f'relinked {relinked} signal{"" if relinked == 1 else "s"} to an earlier backfill signal')
        return relinked
