# CLAUDE.md

## Project Overview

The ticker screener: a **Python 3.12** command line program (managed with **uv**) that fetches stock price data from **Tiingo**, computes the screening criteria (RSI, MACD and the like) and writes everything to a **SQLite** database (stdlib `sqlite3`). Its schema is `schema.sql`.

This repo is checked out as a git submodule at `screener/` in [nloyola/stock_analyzer](https://github.com/nloyola/stock_analyzer), a Go web app that reads the same database and renders the dashboard. The two programs share one `stock_data.db`:

- The screener owns the schema and fills the market data tables: `ticker`, `price_data`, `stock_criteria` and `metadata` (`MARKET_DATA_TABLES` in `app/db.py`).
- The Go app's admin pages are the source of truth for `sector` and `subsector`, the sectors, subsectors and tickers the screener fetches. The `import` command still deletes and rebuilds both tables from a spreadsheet, undoing every admin edit, until it is retired or made to merge.
- The Go app also keeps its own tables (`admin_user`, `admin_session`, `analyzer_schema_migration`) in the same file. The screener must never touch them, and `db-init` must never drop or empty them.
- The Go app's tests use a verbatim copy of `schema.sql` (`internal/repo/testdata/screener_schema.sql` in the parent, checked by its `make schema:check`), so a schema change here needs a matching change there.

In production the screener runs as the `ticker-screener` Docker image (`Dockerfile`), built and deployed by the parent repo's `make release:*` targets. It runs by hand on `tikitak-vps` in one-off containers, with `DB_NAME` and `TIINGO_API_KEY` in the environment and the host's data directory mounted at `/data`.

## Architecture

Under `app/`, wired together by a small dependency-injection container:

- **models.py** - Plain data types (`Ticker`, `PriceData`, `StockCriteria`, ...).
- **db.py** - The `SQLite` wrapper (one transaction per `connect()`), `init_db` (runs `schema.sql`) and `clear_market_data`.
- **repositories/** - One repository per table, hand-written SQL.
- **services/** - Logic over the repositories (`SectorService`).
- **analyzer.py**, **ticker_fetcher.py**, **ticker_sync_tiingo.py** - The work behind the `analyze`, `ticker-fetcher` and `ticker-sync` commands.
- **commands/** - One `BaseCommand` subclass per CLI subcommand: a name, its arguments, the config keys it needs (`REQUIRED_CONFIG`, `DB_NAME` by default) and `handle(args, container)`, which takes its workers from the container. Registering a command builds nothing.
- **container.py**, **wiring.py** - The container (`Value`, `Singleton`, `Factory` providers) and `build_container(config)`, which registers every repository, service and worker as a lazy provider. Classes take the container and pull what they need from it, so a command builds only what it reads.
- **config.py** - `load_config(require=...)` reads `TIINGO_API_KEY` and `DB_NAME`, and raises `ConfigError` naming every required key that is missing. Importing it reads nothing, and it never changes the environment. Each key takes the first non-empty value from the environment, a `.env` piped on stdin (read only when stdin is not a terminal) and the `.env` file; when the environment already holds every required key, neither of the others is read. The container's key providers raise `ConfigError` if a command reads a key it did not list.

Entry point: `main.py`, which sets up logging from `config/0-stdout.json`, registers the commands, parses the command line, loads the config keys the named command needs (a missing one stops it with one line on stderr naming the key, exit 2), builds the container and runs the command. It reads `config/` and `schema.sql` relative to the working directory, so run it from the repo root.

## Key Commands

```bash
uv sync                                        # install the dependencies into .venv
uv run python main.py --help                   # list the subcommands
uv run python main.py db-init                  # create missing tables, empty the market data tables
uv run python main.py import --excel FILE      # replace sectors, subsectors and tickers from a spreadsheet
uv run python main.py ticker-sync              # fetch every ticker symbol from Tiingo
uv run python main.py ticker-fetcher [--sector NAME] [--ticker SYMBOL]   # fetch price data
uv run python main.py ticker-fetcher --strict [--max-failure-rate 0.5]   # exit 1 when too many tickers fail
uv run python main.py analyze                  # compute stock_criteria from price_data
uv run python main.py run-refresh              # under screener.lock: ticker-fetcher --strict, then analyze
uv run pytest                                  # tests
uv run ruff check .                            # lint
uv run ruff format .                           # format
```

The tests in `tests/` cover what the parent's admin pages rely on, the `PROGRESS` lines (`app/progress.py`) and `run-refresh` with its lock (`app/screener_lock.py`), both documented in `README.md`, and the core criteria `analyze` computes (`tests/test_analyzer.py`, made-up price series checked against hand-worked values).

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
