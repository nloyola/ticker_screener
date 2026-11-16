# CLAUDE.md

## Project Overview

The ticker screener: a **Python 3.12** command line program (managed with **uv**) that fetches stock price data from **Tiingo** or **EODHD** (the `PRICE_API` setting), computes the screening criteria (RSI, MACD and the like) and writes everything to a **SQLite** database (stdlib `sqlite3`). Its schema is `schema.sql`.

This repo is checked out as a git submodule at `screener/` in [nloyola/stock_analyzer](https://github.com/nloyola/stock_analyzer), a Go web app that reads the same database and renders the dashboard. The two programs share one `stock_data.db`:

- The screener owns the schema and fills the market data tables: `ticker`, `price_data`, `benchmark_price` (the markets' benchmark closes and opens, `BENCHMARKS` in `app/markets.py`), `stock_criteria` and `metadata` (`MARKET_DATA_TABLES` in `app/db.py`). It also owns `company_name`, each ticker's company name, `screened_session`, each market's sessions `analyze` scored with the rules it scored by, `signal`, the append-only archive of every confirmed breakout, and the signal tracking tables, `signal_price` (each signal's own price path), `signal_outcome` (its outcome per horizon) and `signal_track` (its frozen path and breakout failure), and `backfill_run`, each `backfill-history` run's coverage and biases, and `analysis_run` with `analysis_result`, BreakoutIQ's analysis runs and their findings, none of which is market data: `db-init` keeps them all. `ticker-fetcher` fills in the names missing for the tickers it fetched (README.md, "Refreshing the database").
- The Go app's admin pages are the source of truth for `sector` and `subsector`, the sectors, subsectors and tickers the screener fetches. The `import` command still deletes and rebuilds both tables from a spreadsheet, undoing every admin edit, until it is retired or made to merge.
- The Go app also keeps its own tables (`admin_user`, `admin_session`, `analyzer_schema_migration`) in the same file. The screener must never touch them, and `db-init` must never drop or empty them.
- The Go app's tests use a verbatim copy of `schema.sql` (`internal/repo/testdata/screener_schema.sql` in the parent, checked by its `make schema:check`), so a schema change here needs a matching change there.

In production the screener runs as the `ticker-screener` Docker image (`Dockerfile`), built and deployed by the parent repo's `make release:*` targets. It runs by hand on `tikitak-vps` in one-off containers, with `DB_NAME` and `TIINGO_API_KEY` in the environment and the host's data directory mounted at `/data`.

## Architecture

Under `app/`, wired together by a small dependency-injection container:

- **models.py** - Plain data types (`Ticker`, `PriceData`, `StockCriteria`, ...).
- **db.py** - `connect(db_path)`, the one way repositories open a connection (type converters, `sqlite3.Row` rows, foreign keys, commit or roll back, then close), the `SQLite` wrapper (one explicit transaction per `connect()`, for multi-statement work such as `import`), `upgrade_db` (the versioned upgrades, recorded in `PRAGMA user_version`, that change tables an older `schema.sql` created, or create tables added since, since only `db-init` runs `schema.sql`; the dashboard never reads `user_version`), `init_db` (upgrades, then runs `schema.sql`, which only creates missing tables) and `clear_market_data`.
- **repositories/** - One repository per table, hand-written SQL.
- **services/** - Logic over the repositories (`SectorService`).
- **analyzer.py**, **ticker_fetcher.py**, **ticker_sync_tiingo.py**, **backfill.py** - The work behind the `analyze`, `ticker-fetcher`, `ticker-sync` and `backfill-signals` commands (README.md, "Seeding the signal archive").
- **history_fetch.py** - `HistoryFetch`, the work behind `history-fetch`: years of each listed ticker's and benchmark's prices into the separate history database (`repositories/history_repo.py`, which creates its own tables), resumable the same day (README.md, "Fetching the price history").
- **history_backfill.py**, **replay.py** - `HistoryBackfill`, the work behind `backfill-history`: every session of the history replayed and its signals archived as backfill (README.md, "Replaying the price history"), with each ticker's indicators computed once over its whole history by `ticker_replay` (`replay.py`), giving the values `Analyzer.score_as_of` gives session by session.
- **history_paths.py** - `HistoryPaths`, which `backfill-history` runs after archiving: each backfill signal's path written from the history rather than `price_data`, frozen at 120 sessions, left for `run-refresh` to carry on when it reaches the history's last session, and its unreached horizons ended (`history_ended`) when its ticker's history stops early; `backfill-history` then stores the outcomes with `SignalOutcomeUpdates`, and records the run's coverage and biases in `backfill_run` (`repositories/backfill_run_repo.py`).
- **signal_paths.py** - `SignalPaths`, which `run-refresh` runs after `analyze`: each archived signal's own price path in `signal_price`, rewritten whole from the run's prices until it reaches 120 sessions and is frozen (later, while a benchmark price its outcomes need may still be downloaded), and the horizons it cannot reach marked ended when the ticker gets no prices (README.md, "Refreshing the database"). `HORIZONS` and `PATH_SESSIONS` are defined here.
- **signal_outcome_updates.py** - `SignalOutcomeUpdates`, which `run-refresh` runs after the paths and `compute-outcomes` runs on demand: each signal's newly complete horizons in `signal_outcome` and its breakout failure, once decided, in `signal_track`, computed from its stored path, never changing a complete row or a decided failure, except that a complete row stored without the benchmark's return gets it, with the excess return, once the path has the benchmark's prices.
- **signal_outcomes.py** - `compute_outcomes`, a pure function of a signal's path: per horizon the entry, exit, gross and net returns (`COST_BPS`), benchmark and excess returns, MFE and MAE, and the breakout failure (`FAILURE_SESSIONS`), with the breakout level rescaled to the path's adjustment basis.
- **breakoutiq/** - BreakoutIQ's statistical analysis of the signal archive (the parent's issue #51), with pandas, scipy and statsmodels: `dataset.py` builds the analysis dataset, one row per signal and resolved horizon (an ended horizon kept as censored), with the signal's raw values and flags, the derived mandatory strengths (`breakout_margin_pct`, `sma50_distance_pct`, `sma50_slope_pct`, `volume_ratio`), the market regime (the benchmark against its 200-day SMA, read from the history database too for backfilled signals) and the chronological period (development, validation, test, or gap: the `GAP_SESSIONS` sessions before each boundary). `baseline.py` measures the strategy at each horizon, over every signal and by status, source, market and regime: counts, completed and censored outcomes, positive-return, outperformance and failure rates, and the mean and median returns, MFE and MAE, each with a confidence interval from `stats.py`, which bootstraps blocks of consecutive signal dates as long as the horizon and holds the minimum sample a result needs to be shown (`MIN_SIGNAL_DATES`), the regression with errors clustered two-way by signal date and ticker (`clustered_ols`) and the Benjamini-Hochberg correction (`FDR`). `indicators.py` ranks the secondary indicators at each horizon: each raw value's highest quantile band against its lowest, and each flag, and each secondary indicator combining several flags, passing against failing, on the net excess return, the MAE and the breakout failure, each effect with its clustered interval and its raw and adjusted p-values, every comparison of a horizon one family. `mandatory.py` compares the mandatory strengths the same way, by quantile band, each result stating that it measures how strongly signals passed the rule, not whether the rule is needed, since every signal passed it. `confounders.py` regresses the net excess return on every indicator's raw value and every strength together, standardized, with sector and market fixed effects and the regime, errors clustered by date and ticker: each indicator's adjusted effect beside its unadjusted one, the correlations and variance inflation between indicators, each indicator named redundant when a correlated one carries the stronger evidence or its VIF is high and its effect not significant, and its effect fitted on each sector and regime alone. `candidates.py` searches development alone for candidate rule changes at each horizon, each raw value or strength above or at most a development quantile edge, each flag passing, and pairs of the strongest, each scored as the net excess return of the signals it keeps against those it drops, with clustered errors and the whole search one family; the best rule of each indicator or pair is checked on validation, and the survivors are scored once on test (`TestPeriod` refuses a second score), each candidate labelled preliminary until it holds on validation and test, and validated only if the signals it keeps also reach `MIN_NET_EXCESS_RETURN` on test. `run.py` is the work behind `breakoutiq-analysis`: `Analysis` builds the dataset once, runs every section of `SECTIONS` in order and stores one run with all their results, and `settings()` the constants and payload versions the run records. It only reads the archive (`repositories/analysis_data_repo.py`); runs and findings go to `analysis_run` and `analysis_result` (`repositories/analysis_repo.py`). The parent's `GLOSSARY.md` ("BreakoutIQ analysis") explains every method, constant and label, and must change with them.
- **price_apis/** - The price APIs behind one interface (`PriceApi`: daily prices, a company name, a quota check): `tiingo.py` (US tickers only) and `eodhd.py` (US and Canadian), with the HTTP retry and error helpers they share in `http.py`. Only `wiring.py` picks one, from `PRICE_API`, so either can be deleted cleanly.
- **markets.py** - `market_of()`, the market rule: a bare ticker is US, `.TO` and `.V` are Canada, any other suffix is refused; `market_sql()` is the same rule as an SQL expression, for the repositories. `BENCHMARKS` names each market's broad-market index fund, which every fetch downloads for `analyze`'s relative strength. `last_complete_session()` is the newest day whose daily candle is final in a market for a run started at a given time (its close in `MARKET_CLOSES` plus `PUBLISH_DELAY`); `analyze` drops the rows dated after it before scoring. `analyze` judges each ticker's freshness against its own market's newest price date, `run-refresh` scores and prunes only the markets its API fetched, and a standalone `analyze` scores every market (README.md, "Price APIs and markets").
- **provenance.py** - `git_commit()`, the screener's commit recorded with each screened session: `SCREENER_COMMIT`, which the image sets from its build's revision, or else the checkout's HEAD, marked `-dirty` when it has uncommitted changes.
- **redaction.py** - `redact()`, which masks the API keys and any `api_token=` value, applied to every log record by the `stdout` handler's filter and to a crash's traceback by `main.py`'s `sys.excepthook`.
- **commands/** - One `BaseCommand` subclass per CLI subcommand: a name, its arguments, the config keys it needs (`REQUIRED_CONFIG`, `DB_NAME` by default) and `handle(args, container)`, which takes its workers from the container. Registering a command builds nothing.
- **container.py**, **wiring.py** - The container (`Value`, `Singleton`, `Factory` providers) and `build_container(config)`, which registers every repository, service and worker as a lazy provider. Classes take the container and pull what they need from it, so a command builds only what it reads.
- **config.py** - `load_config(require=...)` reads `TIINGO_API_KEY`, `EODHD_API_KEY`, `DB_NAME`, `PRICE_API` and `HISTORY_DATABASE_FILE`, and raises `ConfigError` naming every required key that is missing. Requiring `PRICE_API` requires a known API (`tiingo` when unset; any other value is refused) and that API's key only. Importing it reads nothing, and it never changes the environment. Each key takes the first non-empty value from the environment, a `.env` piped on stdin (read only when stdin is not a terminal and the environment lacks a required key) and the `.env` file, which is always read when present. The container's key providers raise `ConfigError` if a command reads a key it did not list.

Entry point: `main.py`, which sets up logging from `config/0-stdout.json`, registers the commands, parses the command line, loads the config keys the named command needs (a missing one stops it with one line on stderr naming the key, exit 2), upgrades the database's schema if the command uses one, builds the container and runs the command. It finds `config/` and `schema.sql` relative to its own files, so it runs from any directory.

## Key Commands

```bash
uv sync                                        # install the dependencies into .venv
uv run python main.py --help                   # list the subcommands
uv run python main.py db-init                  # create missing tables, empty the market data tables
uv run python main.py db-upgrade               # upgrade the schema only (every command does first)
uv run python main.py import --excel FILE      # replace sectors, subsectors and tickers from a spreadsheet
uv run python main.py ticker-sync              # fetch every ticker symbol from Tiingo
uv run python main.py ticker-fetcher [--sector NAME] [--ticker SYMBOL]   # fetch price data from PRICE_API
uv run python main.py ticker-fetcher --strict [--max-failure-rate 0.5]   # exit 1 when too many tickers fail
uv run python main.py analyze                  # compute stock_criteria from price_data, every market
uv run python main.py run-refresh              # under screener.lock: ticker-fetcher --strict, analyze, signal paths and outcomes
uv run python main.py backfill-signals         # under screener.lock: archive the signals of every replayable stored session
uv run python main.py compute-outcomes         # under screener.lock: every signal's outcomes from its stored path
uv run python main.py history-fetch            # 10 years of prices into HISTORY_DATABASE_FILE, resumable
uv run python main.py backfill-history         # under screener.lock: archive the history's signals, their paths and outcomes
uv run python main.py breakoutiq-analysis      # run every BreakoutIQ section and store one analysis_run with its results
uv run pytest                                  # tests
uv run ruff check .                            # lint
uv run ruff format .                           # format
```

The tests in `tests/` cover what the parent's admin pages rely on, the `PROGRESS` and `FLAGGED` lines (`app/progress.py`) and `run-refresh` with its lock (`app/screener_lock.py`), both documented in `README.md`, the core criteria, secondary indicators and statuses `analyze` computes (`tests/test_analyzer.py`, made-up price series checked against hand-worked values, and `tests/test_analyzer_db.py`, which reads every secondary indicator and the status back from `stock_criteria`), which tickers `analyze` and the cleanup touch in each market (`tests/test_analyzer_db.py`, `tests/test_ticker_fetcher.py`), and that `analyze` scores only completed daily candles (`tests/test_markets.py`, `tests/test_analyzer_db.py`).

## Git Workflow

- `main` is the main branch
- Feature branches use the `dev-*` prefix
- After committing here, commit the new submodule commit in the parent repo

## Documentation

- **[AGENTS.md](AGENTS.md)** - Guide for coding agents: safe edit areas, coding standards, pre-merge checklist. Read this before making changes.
- The parent repo's `AGENTS.md`, `GLOSSARY.md` and `documentation/install.md` cover the dashboard, the shared database and production.

## Superpowers

- Plans go in `docs/superpowers/plans/`, specs in `docs/superpowers/specs/`.
- `docs/superpowers/` is gitignored; never commit it or `git add -f` it.
