PRAGMA journal_mode = WAL;

-- metadata table for simple key/value app state
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- keep updated_at fresh on updates
CREATE TRIGGER IF NOT EXISTS trg_metadata_updated_at
AFTER UPDATE ON metadata
FOR EACH ROW BEGIN
  UPDATE metadata
  SET updated_at = CURRENT_TIMESTAMP
  WHERE key = NEW.key;
END;

CREATE TABLE IF NOT EXISTS ticker (
  ticker TEXT PRIMARY KEY,
  last REAL,
  prev_close REAL,
  volume INTEGER,
  bid_price REAL,
  ask_price REAL,
  timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticker_volume ON ticker(volume);
CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker(timestamp);

CREATE TABLE IF NOT EXISTS sector (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       sector TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       UNIQUE(sector)
);

CREATE INDEX IF NOT EXISTS idx_sector_secto ON sector(sector);

CREATE TABLE IF NOT EXISTS subsector (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       sector_id INTEGER,
       subsector TEXT,
       tickers TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       FOREIGN KEY (sector_id) REFERENCES sector(id)
           ON UPDATE CASCADE
           ON DELETE CASCADE
);

-- date stays TEXT, a YYYY-MM-DD string: declared DATE, the dashboard's Go driver would scan it as a time.
-- The CHECK accepts exactly the real calendar dates in that shape, since date() always returns YYYY-MM-DD
-- or NULL. The upgrade in app/db.py adds it to databases created before it; change both together.
-- high and low are the day's, adjusted on the close's basis; NULL in rows stored before them, until the next fetch
-- replaces those. The upgrade in app/db.py adds them to databases created before them; change both together.
-- open is the day's, adjusted as the close is, and NULL in rows stored before it, as the high and low were; the ninth
-- upgrade adds it.
CREATE TABLE IF NOT EXISTS price_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT,
       date TEXT NOT NULL CHECK (date(date) IS date),
       close REAL,
       volume INTEGER,
       high REAL,
       low REAL,
       open REAL,
       UNIQUE (ticker, date)
);

-- volume_ratio is the day's volume over avg_vol_20d. The upgrades in app/db.py bring databases created before them
-- to this table: the fourth drops the columns of the core rules the breakout strategy removed (max_vol_10d, ema_10,
-- dma_50_25d_ago, price_gt_10ema) and adds volume_ratio, the fifth renames rsi_50_75 and macd_bullish and adds the
-- trend and momentum indicators' columns after it, the sixth the price action indicators' after those, and the
-- seventh the relative strength columns after those, and the eighth secondary_passed and status after those, where
-- ALTER TABLE places them; change both together. status is the strategy's status (with_status in app/analyzer.py),
-- NULL for a ticker with none.
CREATE TABLE IF NOT EXISTS stock_criteria (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT NOT NULL,
       price REAL,
       core_criteria_met BOOLEAN,
       core_criteria_score INTEGER,
       high_20d REAL,
       high_60d REAL,
       volume INTEGER,
       avg_vol_20d INTEGER,
       dma_50 REAL,
       dma_50_prev_15d REAL,
       rsi REAL,
       macd REAL,
       signal REAL,
       breakout_20d INTEGER,
       breakout_60d INTEGER,
       breakout_confirmed INTEGER,
       price_gt_50dma INTEGER,
       dma_50_rising INTEGER,
       rsi_50_70 INTEGER,
       macd_above_signal INTEGER,
       date_created DATE NOT NULL DEFAULT (date('now')),
       volume_ratio REAL,
       macd_hist REAL,
       macd_hist_prev REAL,
       macd_days_above INTEGER,
       macd_crossover INTEGER,
       macd_hist_rising INTEGER,
       dma_20 REAL,
       dma_20_prev_15d REAL,
       dma_200 REAL,
       dma_200_prev_15d REAL,
       price_gt_20dma INTEGER,
       dma_20_rising INTEGER,
       price_gt_200dma INTEGER,
       dma_200_rising INTEGER,
       ema_21 REAL,
       ema_21_prev_5d REAL,
       price_gt_21ema INTEGER,
       ema_21_rising INTEGER,
       ema_21w REAL,
       ema_21w_prev_1w REAL,
       price_gt_21wema INTEGER,
       ema_21w_rising INTEGER,
       close_position REAL,
       close_upper_quarter INTEGER,
       consolidation_range_pct REAL,
       consolidation_tight INTEGER,
       consolidation_vol_ratio REAL,
       consolidation_vol_declining INTEGER,
       dist_high_20d_pct REAL,
       dist_high_60d_pct REAL,
       near_resistance INTEGER,
       return_1m_pct REAL,
       return_3m_pct REAL,
       rs_market_1m REAL,
       rs_market_3m REAL,
       rs_sector_1m REAL,
       rs_sector_3m REAL,
       outperforms_market_1m INTEGER,
       outperforms_market_3m INTEGER,
       outperforms_sector_1m INTEGER,
       outperforms_sector_3m INTEGER,
       secondary_passed INTEGER,
       status TEXT CHECK (status IN ('high_conviction', 'confirmed', 'pre_breakout')),
       UNIQUE (ticker)
);

-- The daily adjusted closes and opens of each market's benchmark (BENCHMARKS in app/markets.py) over the window
-- price_data keeps, replaced by every fetch that covers its market. open is NULL in rows stored before the ninth
-- upgrade added it, until the next fetch replaces them. Apart from price_data, so the dashboard's ticker removals,
-- which delete a ticker's price_data rows, never touch it, even for a benchmark also listed as a ticker. The upgrade
-- in app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS benchmark_price (
       ticker TEXT NOT NULL,
       date TEXT NOT NULL CHECK (date(date) IS date),
       close REAL NOT NULL,
       open REAL,
       PRIMARY KEY (ticker, date)
);

-- The sessions each market was scored on: one row per market (MARKETS in app/markets.py) and session date, written
-- by every analyze that scored at least one of that market's tickers on it, with the rules it scored by
-- (SCREENER_VERSION and rule_settings() in app/analyzer.py, the latter as JSON), the screener's git commit (NULL when
-- unknown) and the stock_analyzer screener_run the run was (run_id, NULL for a run its runner did not start; no
-- foreign key, since that table is the dashboard's). A rerun of a session replaces its row. Not market data: db-init
-- keeps it. The tenth upgrade in app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS screened_session (
       market TEXT NOT NULL CHECK (market IN ('US', 'Canada')),
       session_date TEXT NOT NULL CHECK (date(session_date) IS session_date),
       screener_version TEXT NOT NULL,
       rule_settings TEXT NOT NULL CHECK (json_valid(rule_settings)),
       git_commit TEXT,
       run_id INTEGER,
       tickers_scored INTEGER NOT NULL CHECK (tickers_scored > 0),
       screened_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       PRIMARY KEY (market, session_date)
);

-- The breakout archive: one row per ticker that met all four mandatory conditions (status confirmed or
-- high_conviction) on a session, append-only. A ticker's market, exchange (NULL when unknown) and market time zone
-- (MARKET_CLOSES in app/markets.py); the session it was scored on and that session's close as price_data holds it,
-- unrounded, where price is rounded; the sector and subsector names at signal time, not ids, since the dashboard
-- renames and deletes them; every value and flag column of stock_criteria under the same name and type
-- (tests/test_signal_schema.py fails on one it lacks), with secondary_passed and the status; the rules it was scored
-- by, the commit and the run, as in screened_session; and its source, live from a run or backfill from a replayed
-- session. A repeated signal points at the ticker's previous one and counts the trading days since it. A rerun of
-- a session adds nothing: one signal per ticker, session and screener version, whose unique index also serves the
-- lookups by ticker. Not market data: db-init and the dashboard's ticker removals never touch it. The eleventh
-- upgrade in app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS signal (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       ticker TEXT NOT NULL,
       market TEXT NOT NULL CHECK (market IN ('US', 'Canada')),
       exchange TEXT,
       time_zone TEXT NOT NULL,
       session_date TEXT NOT NULL CHECK (date(session_date) IS session_date),
       close REAL NOT NULL,
       sector TEXT,
       subsector TEXT,
       price REAL,
       core_criteria_met BOOLEAN,
       core_criteria_score INTEGER,
       high_20d REAL,
       high_60d REAL,
       volume INTEGER,
       avg_vol_20d INTEGER,
       dma_50 REAL,
       dma_50_prev_15d REAL,
       rsi REAL,
       macd REAL,
       signal REAL,
       breakout_20d INTEGER,
       breakout_60d INTEGER,
       breakout_confirmed INTEGER,
       price_gt_50dma INTEGER,
       dma_50_rising INTEGER,
       rsi_50_70 INTEGER,
       macd_above_signal INTEGER,
       volume_ratio REAL,
       macd_hist REAL,
       macd_hist_prev REAL,
       macd_days_above INTEGER,
       macd_crossover INTEGER,
       macd_hist_rising INTEGER,
       dma_20 REAL,
       dma_20_prev_15d REAL,
       dma_200 REAL,
       dma_200_prev_15d REAL,
       price_gt_20dma INTEGER,
       dma_20_rising INTEGER,
       price_gt_200dma INTEGER,
       dma_200_rising INTEGER,
       ema_21 REAL,
       ema_21_prev_5d REAL,
       price_gt_21ema INTEGER,
       ema_21_rising INTEGER,
       ema_21w REAL,
       ema_21w_prev_1w REAL,
       price_gt_21wema INTEGER,
       ema_21w_rising INTEGER,
       close_position REAL,
       close_upper_quarter INTEGER,
       consolidation_range_pct REAL,
       consolidation_tight INTEGER,
       consolidation_vol_ratio REAL,
       consolidation_vol_declining INTEGER,
       dist_high_20d_pct REAL,
       dist_high_60d_pct REAL,
       near_resistance INTEGER,
       return_1m_pct REAL,
       return_3m_pct REAL,
       rs_market_1m REAL,
       rs_market_3m REAL,
       rs_sector_1m REAL,
       rs_sector_3m REAL,
       outperforms_market_1m INTEGER,
       outperforms_market_3m INTEGER,
       outperforms_sector_1m INTEGER,
       outperforms_sector_3m INTEGER,
       secondary_passed INTEGER,
       status TEXT NOT NULL CHECK (status IN ('high_conviction', 'confirmed')),
       screener_version TEXT NOT NULL,
       rule_settings TEXT NOT NULL CHECK (json_valid(rule_settings)),
       git_commit TEXT,
       run_id INTEGER,
       source TEXT NOT NULL CHECK (source IN ('live', 'backfill')),
       previous_signal_id INTEGER REFERENCES signal (id),
       trading_days_since_previous INTEGER CHECK (trading_days_since_previous > 0),
       archived_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       CHECK ((previous_signal_id IS NULL) = (trading_days_since_previous IS NULL)),
       UNIQUE (ticker, session_date, screener_version)
);

CREATE INDEX IF NOT EXISTS idx_signal_session_date ON signal(session_date);

-- Each signal's own price path, so its performance survives price_data being rewritten by every run and deleted
-- when the dashboard removes a ticker: one row per signal and session, session_index 0 the signal session and N the
-- Nth session after it, up to the 120-session horizon. The ticker's open, high, low, close and volume, and its
-- market's benchmark's (BENCHMARKS in app/markets.py) open and close, the two benchmark_price keeps, all as that
-- session's price_data and benchmark_price held them in the run that last wrote the path. A run rewrites a path that
-- is not frozen whole, so every value in it shares one adjustment basis. Not market data: db-init and the
-- dashboard's ticker removals never touch it. The twelfth upgrade in app/db.py creates it in databases from before
-- it; change both together.
CREATE TABLE IF NOT EXISTS signal_price (
       signal_id INTEGER NOT NULL REFERENCES signal (id),
       session_index INTEGER NOT NULL CHECK (session_index BETWEEN 0 AND 120),
       session_date TEXT NOT NULL CHECK (date(session_date) IS session_date),
       open REAL,
       high REAL,
       low REAL,
       close REAL NOT NULL,
       volume INTEGER,
       benchmark_open REAL,
       benchmark_close REAL,
       PRIMARY KEY (signal_id, session_index),
       UNIQUE (signal_id, session_date)
);

-- One row per signal and horizon (5, 10, 20, 40, 60 and 120 sessions after the signal session). Entry is the open
-- of the session after the signal session, exit the close of the horizon's session; the returns are percentages,
-- gross and net of the round-trip cost in basis points it was computed with, the benchmark's over the same entry
-- open to exit close, and the excess return the net return over the benchmark's. mfe_pct and mae_pct are the highest
-- high and lowest low up to the horizon against the entry. A horizon whose session is not in the path yet is pending
-- and has no row: the screener never writes status pending. It is complete once its session is in the path, and then
-- never recomputed, apart from benchmark_return_pct and excess_return_pct, which a complete row stored without them
-- gets once its path has the benchmark's entry and exit prices, or ended when the ticker stopped getting prices first,
-- with the reason and the last session it had prices for, a row deleted again when a later run has prices for the
-- ticker. Not market data: db-init and the dashboard's ticker removals never touch it. The twelfth upgrade in
-- app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS signal_outcome (
       signal_id INTEGER NOT NULL REFERENCES signal (id),
       horizon INTEGER NOT NULL CHECK (horizon IN (5, 10, 20, 40, 60, 120)),
       status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'ended')),
       entry_date TEXT CHECK (date(entry_date) IS entry_date),
       entry_price REAL,
       exit_date TEXT CHECK (date(exit_date) IS exit_date),
       exit_price REAL,
       cost_bps INTEGER CHECK (cost_bps >= 0),
       gross_return_pct REAL,
       net_return_pct REAL,
       benchmark_return_pct REAL,
       excess_return_pct REAL,
       mfe_pct REAL,
       mae_pct REAL,
       ended_reason TEXT,
       last_session_date TEXT CHECK (date(last_session_date) IS last_session_date),
       completed_at TIMESTAMP,
       updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       CHECK ((status = 'complete') = (completed_at IS NOT NULL)),
       CHECK (status <> 'complete' OR (entry_date IS NOT NULL AND entry_price IS NOT NULL
              AND exit_date IS NOT NULL AND exit_price IS NOT NULL AND cost_bps IS NOT NULL
              AND gross_return_pct IS NOT NULL AND net_return_pct IS NOT NULL
              AND mfe_pct IS NOT NULL AND mae_pct IS NOT NULL)),
       CHECK ((status = 'ended') = (ended_reason IS NOT NULL)),
       CHECK ((status = 'ended') = (last_session_date IS NOT NULL)),
       PRIMARY KEY (signal_id, horizon)
);

-- One row per tracked signal, what is decided once for the signal rather than per horizon. path_frozen_at is when
-- its path in signal_price reached the 120-session horizon, after which no run rewrites it; a path lacking a
-- benchmark price its outcomes need waits for the run whose benchmark covers it, unless the gap is inside the
-- benchmark's stored range. breakout_level is the level the breakout cleared: the prior 60-day high when the 60-day
-- breakout passed, the prior 20-day high otherwise. breakout_failed is NULL until decided, 1 once a close in the 10
-- sessions after the signal session is below that level, with the first such session's date, and 0 once all 10 have
-- closed at or above it. Not market data: db-init and the dashboard's ticker removals never touch it. The twelfth
-- upgrade in app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS signal_track (
       signal_id INTEGER NOT NULL PRIMARY KEY REFERENCES signal (id),
       path_frozen_at TIMESTAMP,
       breakout_level REAL,
       breakout_failed INTEGER CHECK (breakout_failed IN (0, 1)),
       failure_date TEXT CHECK (date(failure_date) IS failure_date),
       failure_decided_at TIMESTAMP,
       CHECK ((breakout_failed IS 1) = (failure_date IS NOT NULL)),
       CHECK ((breakout_failed IS NULL) = (failure_decided_at IS NULL))
);

-- One row per backfill-history run that finished, what it covered and the biases it carries, for the report on the
-- backfilled signals to state. universe_date is the day the sectors page was read: the run replays the tickers listed
-- that day, each in the subsector that lists it first, so delisted and removed tickers are missing and sector
-- membership is that day's. first_session and last_session are the range of sessions it replayed, NULL when there
-- were none. price_api is the price API the replayed tickers' histories were fetched from, the APIs comma separated
-- when they differ, NULL when no ticker had one. tickers_listed counts the tickers listed, tickers_replayed those
-- with a history and tickers_scored those scored on at least one session. left_out maps each listed ticker never
-- scored to why: no_history, unsupported (an exchange suffix no market has), short_history (a history that starts
-- late and leaves no session to score it on) or never_scored (a history that starts on time but leaves no session
-- to score it on, such as one stale or frozen throughout). The short_history column maps each replayed ticker whose
-- history starts late, more than two weeks after the 10 years the fetch asked for, scored ones included, to its first
-- day. signals and relinked count the signals it archived and relinked. Not market data: db-init never touches it.
-- The thirteenth upgrade in app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS backfill_run (
       id INTEGER PRIMARY KEY,
       started_at TIMESTAMP NOT NULL,
       finished_at TIMESTAMP NOT NULL,
       universe_date TEXT NOT NULL CHECK (date(universe_date) IS universe_date),
       first_session TEXT CHECK (date(first_session) IS first_session),
       last_session TEXT CHECK (date(last_session) IS last_session),
       sessions INTEGER NOT NULL CHECK (sessions >= 0),
       price_api TEXT,
       tickers_listed INTEGER NOT NULL CHECK (tickers_listed >= 0),
       tickers_replayed INTEGER NOT NULL CHECK (tickers_replayed >= 0),
       tickers_scored INTEGER NOT NULL CHECK (tickers_scored >= 0),
       left_out TEXT NOT NULL CHECK (json_valid(left_out)),
       short_history TEXT NOT NULL CHECK (json_valid(short_history)),
       signals INTEGER NOT NULL CHECK (signals >= 0),
       relinked INTEGER NOT NULL CHECK (relinked >= 0),
       screener_version TEXT NOT NULL,
       git_commit TEXT,
       CHECK ((first_session IS NULL) = (last_session IS NULL)),
       CHECK ((first_session IS NULL) = (sessions = 0))
);

-- Each ticker's company name, looked up from the price API named in price_api. Not market data: db-init keeps it,
-- so names survive a re-initialisation. The dashboard writes it too, when an admin adds a ticker. The upgrade in
-- app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS company_name (
       ticker TEXT NOT NULL PRIMARY KEY COLLATE NOCASE,
       name TEXT NOT NULL,
       price_api TEXT NOT NULL CHECK (price_api IN ('tiingo', 'eodhd')),
       stored_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);
