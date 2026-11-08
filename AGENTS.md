# AGENTS: Working in this Repository

This guide is for human and AI coding agents contributing to the ticker screener. It summarizes the stack, safe edit areas, required tooling, and the commands you should run before merging a branch.


## Project Overview

- Stack: Python 3.12 (`.python-version`), uv (`pyproject.toml`, `uv.lock`), stdlib `sqlite3`, pandas, `ta` for the indicators, `requests` for the Tiingo and EODHD REST APIs, openpyxl for the `import` spreadsheet.
- Data: the SQLite database named by `DB_NAME`, shared with the Go dashboard in [nloyola/stock_analyzer](https://github.com/nloyola/stock_analyzer), where this repo is the `screener/` submodule. The screener owns `schema.sql` and fills `ticker`, `price_data`, `stock_criteria` and `metadata`, and `company_name`, each ticker's company name, which is not market data and survives `db-init`. The dashboard's admin pages own the contents of `sector` and `subsector`; `import` still replaces both from a spreadsheet. The dashboard's own tables (`admin_user`, `admin_session`, `analyzer_schema_migration`) live in the same file: never read, write, drop or empty them, and never add them to `schema.sql`. Only `clear_market_data` (`app/db.py`) empties tables, and only those in `MARKET_DATA_TABLES`.
- Structure: see [CLAUDE.md § Architecture](CLAUDE.md#architecture).
- Deployment: the `ticker-screener` image (`Dockerfile`, with `.dockerignore` as an allow-list), built and shipped by the parent repo's `make release:build` and `make release:push`, and run by hand on `tikitak-vps` in one-off containers (`docker compose -f compose.production.yaml run --rm screener analyze`). The runbook is the parent's `documentation/install.md`.


## Safe Areas To Edit

- Do edit: `app/**`, `tests/**`, `main.py`, `schema.sql`, `config/**`, `pyproject.toml`, `Dockerfile`, `.dockerignore`, `README.md`.
- Change with care:
  - `schema.sql` - the parent's tests use a verbatim copy (`internal/repo/testdata/screener_schema.sql`, checked by its `make schema:check`), and the dashboard's repositories read these tables. A schema change needs the matching change in the parent. `schema.sql` only creates missing tables, and only `db-init` runs it, so a change to an existing table also needs an upgrade appended to `UPGRADES` in `app/db.py`, leaving an old table exactly as `schema.sql` now creates it, and so does a new table, which the upgrade creates; never edit an upgrade that has shipped.
  - `uv.lock` - change it only through `uv add`, `uv remove` or `uv lock`, never by hand.
- Avoid editing (local data or build output, all gitignored):
  - `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
  - `data/`, `/*.json`, `/*.xlsx`


## Environment & Tooling

- Tools: `uv` and `ruff` (on the `PATH` via Nix; `uv run ruff` also works from the dev group). `uv sync` creates `.venv` from `uv.lock`.
- Environment file: `.env` at the root, read by `load_config` (`app/config.py`) after the environment and any `.env` piped on stdin (read only when the environment lacks a key the command needs), filling only keys they leave unset, so a `PRICE_API` set only here applies even when the environment holds every key. Keys: `DB_NAME` (the database path, needed by every command), `PRICE_API` (the price API `ticker-fetcher` and `run-refresh` fetch from: `tiingo`, the default, or `eodhd`), `TIINGO_API_KEY` (needed by `ticker-sync`, and by `ticker-fetcher` and `run-refresh` when `PRICE_API` is `tiingo`) and `EODHD_API_KEY` (needed by `ticker-fetcher` and `run-refresh` when `PRICE_API` is `eodhd`). A command missing a key it needs, or given an unknown `PRICE_API`, exits 2 with one line naming it. Never commit it, and never print it - list key names only (`grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env`).
- `main.py` runs from any directory: it finds `config/0-stdout.json` and `schema.sql` relative to its own files.


## Coding Standards

- `uv run ruff check .` and `uv run ruff format --check .` clean (`pyproject.toml`: 120 columns, single quotes, rules `E`, `F`, `I`, `UP`, `B`).
- Keep the layering: commands parse arguments and call a service or worker class; repositories own all SQL; `models.py` has no I/O. A new table gets a repository in `app/repositories/`, registered in `build_container` (`app/wiring.py`).
- A new subcommand subclasses `BaseCommand` in `app/commands/`, lists the config keys it needs in `REQUIRED_CONFIG`, takes its workers from the container in `handle` (never in `__init__`), and is registered in `main()`.
- Log through `logging.getLogger(__name__)`; `print` only a command's own summary or prompt.
- Repositories open every connection through `connect(db_path)` (`app/db.py`), never their own; it commits on success, rolls back on error and closes. Work spanning several statements outside a repository goes inside `SQLite.connect()`, which wraps it in one transaction.


## Testing

- `uv run pytest` runs the tests in `tests/`. They cover what the dashboard relies on: the progress and flagged-ticker lines (`app/progress.py`), `run-refresh` with its lock (`app/screener_lock.py`), both documented in `README.md`, the core criteria `Analyzer.analyze_ticker` computes (`tests/test_analyzer.py`), that `analyze` leaves stale and frozen tickers unscored, judging staleness against each ticker's own market, and that `run-refresh` scores and prunes only the markets it fetched (`tests/test_analyzer_db.py`, with the frozen rule itself in `tests/test_frozen.py`), and that `ticker-fetcher --ticker` finds a ticker's subsector by parsing the ticker lists (`tests/test_sector_service.py`). A change to the criteria or indicators must update those tests to match. Check any other change by running the affected commands against a scratch database (`DB_NAME=/path/to/scratch.db uv run python main.py db-init`), never against `data/stock_data.db` unless that is the point.
- `ticker-sync` calls Tiingo, and `ticker-fetcher` the API `PRICE_API` names, with the real API key, and count against its limits; run them on one sector or ticker (`--sector`, `--ticker`) when checking a change.


## Pre-Merge Checklist

Feature branches use the `dev-*` prefix and merge into `main`. Before merging:

- `uv run ruff check .` and `uv run ruff format --check .` are clean.
- `uv run pytest` passes.
- The changed commands run against a scratch database.
- Docs updated (`README.md`, this file, `CLAUDE.md`) when behavior or commands change.
- No `.env`, database or other gitignored files committed.
- After merging, commit the new submodule commit in the parent repo, with any matching schema snapshot change.


## Security & Secrets

- Never commit `.env`, databases or spreadsheets.
- Never log or print `TIINGO_API_KEY` or `EODHD_API_KEY`. `app/redaction.py` is the guard: `redact()` masks both keys' values and any `api_token=` value, the `stdout` handler's filter (`config/0-stdout.json`) applies it to every log record, tracebacks included, and `main.py`'s `sys.excepthook` applies it to a crash's traceback. Keep new output going through logging, and run any message holding a URL or an error through `error_detail()` or `redact()`.
