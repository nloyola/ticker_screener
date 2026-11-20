"""
The screen export (the parent's issue #70, "Export design"): one JSON file per session a live run scored,
screen_<as_of>.json in SCREEN_EXPORT_DIR, holding the run's header, one row per ticker with a status and the
follow-up events of the archived signals, every value exactly as the run stored it, so tools work from the values the
screener used rather than the rounded ones the PDF shows. build_export is a pure function of what the repositories
read; ScreenExport reads them and writes the file, replacing one of the same as-of date.
"""

import json
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from app.analyzer import (
    CONFIRMED,
    EXTENDED,
    HIGH_CONVICTION,
    PRE_BREAKOUT,
    SCREENER_VERSION,
    SECONDARY_INDICATORS,
    SIGNAL_STATUSES,
    rule_settings,
)
from app.container import Container
from app.export_settings import EXPORT_VERSION, WIDE_RISK_PCT, export_settings
from app.markets import CANADA, MARKETS, US, UnsupportedTicker, exchange_of, market_of
from app.models import StockCriteria
from app.provenance import git_commit
from app.repositories.screen_export_repo import DayRange, ExportRow, FollowUpEvent

if TYPE_CHECKING:
    from app.repositories.issuer_repo import IssuerRepo
    from app.repositories.screen_export_repo import ScreenExportRepo
    from app.repositories.security_state_repo import SecurityStateRepo
    from app.services.sector_service import SectorService

logger = logging.getLogger(__name__)

# each market's listings' currency, which every price in a row is in
CURRENCIES = {US: 'USD', CANADA: 'CAD'}
# the suffix a US ticker takes in EODHD's format; a Canadian ticker already carries its exchange's
US_SUFFIX = '.US'
# the statuses in the home page's order, which technical_rank follows
STATUS_ORDER = (HIGH_CONVICTION, CONFIRMED, EXTENDED, PRE_BREAKOUT)
# the basis of a row's invalidation level: the level its breakout run broke
BREAKOUT_LEVEL_BASIS = 'breakout_level'

# the values each secondary check reports, by its SECONDARY_INDICATORS name, under the export's names
SECONDARY_VALUES: dict[str, dict[str, str]] = {
    'rsi': {'rsi': 'rsi'},
    'macd': {
        'macd': 'macd',
        'signal': 'signal',
        'macd_hist': 'macd_hist',
        'macd_hist_prev': 'macd_hist_prev',
        'macd_days_above': 'macd_days_above',
        'macd_above_signal': 'macd_above_signal',
        'macd_crossover': 'macd_crossover',
        'macd_hist_rising': 'macd_hist_rising',
    },
    'relative_strength': {
        'return_1m_pct': 'return_1m_pct',
        'return_3m_pct': 'return_3m_pct',
        'rs_market_1m': 'rs_market_1m',
        'rs_market_3m': 'rs_market_3m',
        'rs_sector_1m': 'rs_sector_1m',
        'rs_sector_3m': 'rs_sector_3m',
        'outperforms_market_1m': 'outperforms_market_1m',
        'outperforms_market_3m': 'outperforms_market_3m',
        'outperforms_sector_1m': 'outperforms_sector_1m',
        'outperforms_sector_3m': 'outperforms_sector_3m',
    },
    'moving_average_alignment': {
        'sma_20': 'dma_20',
        'sma_20_15d_ago': 'dma_20_prev_15d',
        'sma_200': 'dma_200',
        'sma_200_15d_ago': 'dma_200_prev_15d',
        'price_gt_20dma': 'price_gt_20dma',
        'price_gt_50dma': 'price_gt_50dma',
        'price_gt_200dma': 'price_gt_200dma',
        'dma_20_rising': 'dma_20_rising',
        'dma_50_rising': 'dma_50_rising',
        'dma_200_rising': 'dma_200_rising',
    },
    'consolidation_quality': {
        'consolidation_range_pct': 'consolidation_range_pct',
        'consolidation_tight': 'consolidation_tight',
        'consolidation_vol_ratio': 'consolidation_vol_ratio',
        'consolidation_vol_declining': 'consolidation_vol_declining',
    },
    'closing_strength': {
        'high': 'high',
        'low': 'low',
        'close_position': 'close_position',
        'close_upper_quarter': 'close_upper_quarter',
    },
    'ema_21_alignment': {
        'ema_21': 'ema_21',
        'ema_21_5d_ago': 'ema_21_prev_5d',
        'ema_21w': 'ema_21w',
        'ema_21w_1w_ago': 'ema_21w_prev_1w',
        'price_gt_21ema': 'price_gt_21ema',
        'ema_21_rising': 'ema_21_rising',
        'price_gt_21wema': 'price_gt_21wema',
        'ema_21w_rising': 'ema_21w_rising',
    },
}

# the stock_criteria flags, stored as 0 and 1, which the export reports as booleans
_FLAGS = frozenset(
    {
        'macd_above_signal',
        'macd_crossover',
        'macd_hist_rising',
        'outperforms_market_1m',
        'outperforms_market_3m',
        'outperforms_sector_1m',
        'outperforms_sector_3m',
        'price_gt_20dma',
        'price_gt_50dma',
        'price_gt_200dma',
        'dma_20_rising',
        'dma_50_rising',
        'dma_200_rising',
        'consolidation_tight',
        'consolidation_vol_declining',
        'close_upper_quarter',
        'price_gt_21ema',
        'ema_21_rising',
        'price_gt_21wema',
        'ema_21w_rising',
    }
)


@dataclass(frozen=True)
class RunHeader:
    """What the export's header says about the run, beside what build_export derives from the rows."""

    run_time: datetime
    # each market's session the run scored, with how many of its tickers it scored
    sessions: Mapping[str, tuple[date, int]]
    data_source: str
    git_commit: str | None
    run_id: int | None
    # how many tickers the sectors page lists in each market
    listed: Mapping[str, int]


def export_file_name(as_of: date) -> str:
    """The export's file name for the session 'as_of'."""
    return f'screen_{as_of.isoformat()}.json'


def eodhd_ticker(ticker: str) -> str:
    """'ticker' in EODHD's format: a US ticker with US_SUFFIX ('BILL.US'), a Canadian one as stored ('STN.TO')."""
    return f'{ticker}{US_SUFFIX}' if market_of(ticker) == US else ticker


def signal_id(session_date: date, ticker: str) -> str:
    """The export's id of a ticker's row on a session: '<session_date>:<EODHD ticker>'."""
    return f'{session_date.isoformat()}:{eodhd_ticker(ticker)}'


def _flag(value: int | None) -> bool | None:
    return None if value is None else bool(value)


def _float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _pct_below(close: float | None, level: float | None) -> float | None:
    """How far 'close' is above 'level', in percent of the close; None without either or with a close of 0."""
    if close is None or level is None or close == 0:
        return None
    return (close - level) / close * 100


def _lookback(sc: StockCriteria) -> int | None:
    if sc.breakout_60d == 1:
        return 60
    if sc.breakout_20d == 1:
        return 20
    return None


def _margin_met(sc: StockCriteria) -> bool | None:
    if sc.breakout_level is None or sc.breakout_margin_pct is None or sc.required_margin_pct is None:
        return None
    return sc.breakout_margin_pct >= sc.required_margin_pct


def _breakout_pass(sc: StockCriteria) -> bool | None:
    if sc.breakout_20d is None and sc.breakout_60d is None:
        return None
    return sc.breakout_20d == 1 or sc.breakout_60d == 1


def _secondary_value(sc: StockCriteria, day: DayRange, column: str) -> object:
    if column == 'high':
        return _float(day.high)
    if column == 'low':
        return _float(day.low)
    value = getattr(sc, column)
    if column in _FLAGS:
        return _flag(value)
    if column == 'macd_days_above':
        return _int(value)
    return _float(value)


def _secondary(sc: StockCriteria, day: DayRange) -> dict:
    checks = []
    for i, (name, passes) in enumerate(SECONDARY_INDICATORS.items(), start=1):
        values = {out: _secondary_value(sc, day, column) for out, column in SECONDARY_VALUES[name].items()}
        checks.append({'id': f'S{i}', 'name': name, 'pass': passes(sc), 'values': values})
    return {'checks': checks, 'score': _int(sc.secondary_passed)}


def _rank_key(row: ExportRow) -> tuple:
    sc = row.criteria
    return (STATUS_ORDER.index(sc.status), -(sc.secondary_passed or 0), sc.ticker)


def export_row(
    row: ExportRow,
    rank: int | None,
    issuer: str | None,
    primary: bool,
    exchange: str | None,
    day: DayRange,
) -> dict:
    """One row of the export ('rows'), from its stored criteria, its rank, issuer, exchange and the day's range."""
    sc = row.criteria
    market = market_of(sc.ticker)
    level = _float(sc.breakout_level)
    pct_to_invalidation = _float(sc.invalidation_distance_pct)
    swing = _float(sc.swing_low)
    return {
        'signal_id': signal_id(row.session_date, sc.ticker),
        'ticker': eodhd_ticker(sc.ticker),
        'screener_ticker': sc.ticker,
        'archive_id': row.archive_id,
        'market': market,
        'exchange': exchange,
        'currency': CURRENCIES[market],
        'sector': row.sector,
        'subsector': row.subsector,
        'issuer': issuer,
        'primary_listing': primary,
        'status': sc.status,
        'session_date': row.session_date.isoformat(),
        'mandatory': {
            'breakout': {
                'pass': _breakout_pass(sc),
                'close': _float(sc.price),
                'high_20d': _float(sc.high_20d),
                'high_60d': _float(sc.high_60d),
                'breakout_20d': _flag(sc.breakout_20d),
                'breakout_60d': _flag(sc.breakout_60d),
                'lookback_cleared': _lookback(sc),
                'breakout_level': level,
                'breakout_margin_pct': _float(sc.breakout_margin_pct),
                'required_margin_pct': _float(sc.required_margin_pct),
                'average_daily_range_pct': _float(sc.average_daily_range_pct),
                'margin_met': _margin_met(sc),
            },
            'volume': {
                'pass': _flag(sc.breakout_confirmed),
                'volume': _int(sc.volume),
                'avg_volume_20d': _float(sc.avg_vol_20d),
                'volume_ratio': _float(sc.volume_ratio),
            },
            'above_sma_50': {'pass': _flag(sc.price_gt_50dma), 'sma_50': _float(sc.dma_50)},
            'sma_50_rising': {
                'pass': _flag(sc.dma_50_rising),
                'sma_50': _float(sc.dma_50),
                'sma_50_15d_ago': _float(sc.dma_50_prev_15d),
            },
            'core_score': _int(sc.core_criteria_score),
        },
        'secondary': _secondary(sc, day),
        'context': {
            'dist_high_20d_pct': _float(sc.dist_high_20d_pct),
            'dist_high_60d_pct': _float(sc.dist_high_60d_pct),
            'near_resistance': _flag(sc.near_resistance),
            'high_120d': _float(sc.high_120d),
            'dist_high_120d_pct': _float(sc.dist_high_120d_pct),
            'in_120d_tier': sc.status in SIGNAL_STATUSES and sc.breakout_120d == 1,
            'avg_dollar_volume': _float(sc.avg_dollar_volume),
        },
        'risk': {
            'invalidation_level': level,
            'invalidation_basis': None if level is None else BREAKOUT_LEVEL_BASIS,
            'pct_to_invalidation': pct_to_invalidation,
            'swing_low': swing,
            'swing_low_date': _iso(sc.swing_low_date),
            'pct_to_swing_low': _pct_below(_float(sc.price), swing),
            'wide_risk': None if level is None or pct_to_invalidation is None else pct_to_invalidation > WIDE_RISK_PCT,
        },
        'technical_rank': rank,
        'breakout_date': _iso(sc.breakout_date),
        'days_since_breakout': _int(sc.sessions_since_breakout),
    }


def follow_up_entry(event: FollowUpEvent) -> dict:
    """One entry of the export's 'follow_up', from a signal's stored failure or retest."""
    return {
        'signal_id': signal_id(event.breakout_date, event.ticker),
        'archive_id': event.archive_id,
        'ticker': eodhd_ticker(event.ticker),
        'breakout_date': event.breakout_date.isoformat(),
        'breakout_level': _float(event.breakout_level),
        'event': event.event,
        'event_date': event.event_date.isoformat(),
        'sessions_after_breakout': event.sessions_after_breakout,
        'close': _float(event.close),
        'low': _float(event.low),
        'volume': _int(event.volume),
        'avg_volume_20d': _float(event.avg_volume_20d),
    }


def _time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')


def build_export(
    header: RunHeader,
    rows: Sequence[ExportRow],
    follow_ups: Sequence[FollowUpEvent],
    issuers: Mapping[str, str],
    exchanges: Mapping[str, str],
    day_ranges: Mapping[str, DayRange],
) -> dict:
    """
    The export of one run as a JSON-ready object: its header ('header', with the status counts and the universe
    derived here), its 'rows', ranked, and its 'follow_up'. 'issuers' maps each issuer's listings, upper case, to its
    primary (IssuerRepo.listings), 'exchanges' a bare US ticker, upper case, to the exchange the security master
    records, and 'day_ranges' each row's ticker to its session's high and low.
    """

    def is_primary(row: ExportRow) -> bool:
        ticker = row.criteria.ticker.upper()
        return issuers.get(ticker, ticker) == ticker

    primaries = sorted((r for r in rows if is_primary(r)), key=_rank_key)
    others = sorted((r for r in rows if not is_primary(r)), key=_rank_key)
    out_rows = [_row(row, rank, issuers, exchanges, day_ranges) for rank, row in enumerate(primaries, start=1)]
    out_rows.extend(_row(row, None, issuers, exchanges, day_ranges) for row in others)
    counts = Counter(r.criteria.status for r in rows)
    sessions = {market: _iso(header.sessions[market][0]) if market in header.sessions else None for market in MARKETS}
    return {
        'run': {
            'export_version': EXPORT_VERSION,
            'run_time': _time(header.run_time),
            'as_of': max(d for d, _ in header.sessions.values()).isoformat(),
            'sessions': sessions,
            'data_source': header.data_source,
            'screener_version': SCREENER_VERSION,
            'git_commit': header.git_commit,
            'run_id': header.run_id,
            'rule_settings': rule_settings(),
            'export_settings': export_settings(),
            'universe': {
                market: {
                    'listed': header.listed.get(market, 0),
                    'scored': header.sessions[market][1] if market in header.sessions else 0,
                }
                for market in MARKETS
            },
            'status_counts': {status: counts.get(status, 0) for status in STATUS_ORDER},
        },
        'rows': out_rows,
        'follow_up': [follow_up_entry(e) for e in follow_ups],
    }


def _row(
    row: ExportRow,
    rank: int | None,
    issuers: Mapping[str, str],
    exchanges: Mapping[str, str],
    day_ranges: Mapping[str, DayRange],
) -> dict:
    ticker = row.criteria.ticker
    issuer = issuers.get(ticker.upper())
    return export_row(
        row,
        rank=rank,
        issuer=issuer,
        primary=rank is not None,
        exchange=exchange_of(ticker) or exchanges.get(ticker.upper()),
        day=day_ranges.get(ticker, DayRange(None, None)),
    )


def write_atomically(directory: Path, name: str, content: str) -> Path:
    """
    Writes 'content' to 'name' in 'directory', created when missing, through a temporary file in the same directory
    renamed over it, so a reader never sees a half-written file. Returns the file's path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=f'.{name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def utc_now() -> datetime:
    return datetime.now(UTC)


class ScreenExport:
    """
    Writes the screen export of a live run (run-refresh) once its scores, archive and signal tracking are stored, from
    what the repositories read back, to the directory SCREEN_EXPORT_DIR names.
    """

    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.directory = Path(container.screen_export_dir)
        self.repo: ScreenExportRepo = container.screen_export_repo
        self.issuer_repo: IssuerRepo = container.issuer_repo
        self.security_state_repo: SecurityStateRepo = container.security_state_repo
        self.sector_service: SectorService = container.sector_service
        self.now = now

    def write(self, sessions: Mapping[str, tuple[date, int]], run_id: int | None, data_source: str) -> Path | None:
        """
        Writes the export of the run that scored 'sessions' (each market's session with how many of its tickers it
        scored) from 'data_source', the PRICE_API it fetched from, as the stock_analyzer screener_run 'run_id'.
        Returns the file written, or None, writing nothing, when the run scored no session.
        """
        if not sessions:
            logger.info('screen export: no session scored, no file written')
            return None
        dates = {market: session_date for market, (session_date, _) in sessions.items()}
        rows = self.repo.rows(dates, SCREENER_VERSION)
        header = RunHeader(
            run_time=self.now(),
            sessions=sessions,
            data_source=data_source,
            git_commit=git_commit(),
            run_id=run_id,
            listed=self._listed(),
        )
        export = build_export(
            header,
            rows,
            self.repo.follow_ups(dates, SCREENER_VERSION),
            self.issuer_repo.listings(),
            self.security_state_repo.exchanges(),
            self.repo.day_ranges((r.criteria.ticker, r.session_date) for r in rows),
        )
        content = json.dumps(export, ensure_ascii=False, allow_nan=False, indent=1) + '\n'
        path = write_atomically(self.directory, export_file_name(date.fromisoformat(export['run']['as_of'])), content)
        logger.info(f'screen export: wrote {path}, {len(rows)} rows and {len(export["follow_up"])} follow-up events')
        return path

    def _listed(self) -> dict[str, int]:
        """How many distinct tickers the sectors page lists in each market, leaving out one no market takes."""
        listed: dict[str, set[str]] = {market: set() for market in MARKETS}
        for sector in self.sector_service.get_all():
            for sub in sector.subsectors:
                for ticker in sub.tickers:
                    try:
                        listed[market_of(ticker)].add(ticker)
                    except UnsupportedTicker:
                        continue
        return {market: len(tickers) for market, tickers in listed.items()}
