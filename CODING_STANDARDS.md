# Coding Standards

The rules code in this repository is held to. The screener is its own repository (`nloyola/ticker_screener`), but it lives inside [stock_analyzer](https://github.com/nloyola/stock_analyzer) as the `screener/` submodule and shares its database, so a change here is also checked against what the parent relies on. Code review checks changes against this file and reports a breach as a finding, citing the rule. [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) cover the stack, commands and workflow; this file covers what the code itself must look like.


## Python style

- `uv run ruff check .` and `uv run ruff format --check .` clean (`pyproject.toml`: Python 3.12, 120 columns, single quotes, rules `E`, `F`, `I`, `UP`, `B`).
- Type every function's parameters and return value, with the modern forms (`list[str]`, `str | None`, `collections.abc` for `Iterable`, `Sequence`, `Callable`).
- Plain values are dataclasses, `@dataclass(frozen=True)` unless they must change after construction (`models.py`, the BreakoutIQ results).
- Modules, classes and non-obvious functions have a docstring saying what the thing is or does, in full sentences, naming the README section or the parent's GLOSSARY entry that explains it at length.
- A threshold or window the rules use is a named module-level constant (`VOLUME_RATIO_MIN`, `FROZEN_DAYS`, `GAP_SESSIONS`), never an inline number.
- Change `uv.lock` only through `uv add`, `uv remove` or `uv lock`.


## Structure

- Commands parse arguments and call a service or worker class; repositories own all SQL; `models.py` has no I/O.
- A new subcommand subclasses `BaseCommand` in `app/commands/`, lists the config keys it needs in `REQUIRED_CONFIG`, takes its workers from the container in `handle` (never in `__init__`), and is registered in `main()`.
- A new repository, service or worker is registered in `build_container` (`app/wiring.py`) as a lazy provider. Classes take the container and pull what they need from it.
- Only `wiring.py` chooses a price API; everything else talks to the `PriceApi` interface (`app/price_apis/`), so either API can be deleted cleanly.
- The market of a ticker comes from `market_of()` (`app/markets.py`), and in SQL from `market_sql()`, never from a suffix test of its own.


## Database

- Repositories open every connection through `connect(db_path)` (`app/db.py`), never their own; work spanning several statements outside a repository goes inside `SQLite.connect()`, one transaction.
- Values are always bound as `?` parameters. An f-string in SQL may only insert a table or column name from a constant in the code.
- `schema.sql` only creates missing tables. A change to an existing table, or a new table, also appends an upgrade to `UPGRADES` in `app/db.py` that leaves an old database exactly as `schema.sql` now creates it. Never edit an upgrade that has shipped.
- A `schema.sql` change needs the matching change in the parent (its snapshot `internal/repo/testdata/screener_schema.sql` and its repositories), committed with the new submodule commit.
- Never read, write, drop or empty the dashboard's own tables, and never add them to `schema.sql`. The exceptions are reading `issuer` and `issuer_listing`, through `IssuerRepo` alone, which treats them missing as no issuers, and reading `security_master`'s states, through `SecurityStateRepo` alone, which treats it missing as no ticker held. Only `clear_market_data` empties tables, and only those in `MARKET_DATA_TABLES`. `signal` is append-only.


## Rules and versions

- Any change to what `analyze` scores bumps `SCREENER_VERSION` (`app/analyzer.py`, semver) in the same commit: major for a core condition or status rule, minor for a secondary indicator or a threshold or window, patch for a fix that keeps every rule's intent. A change that leaves every score as it was needs no bump.
- Every rule constant is listed in `rule_settings()`, or named in `NOT_RULE_SETTINGS` in `tests/test_rule_settings.py`.
- A new indicator column in `stock_criteria` is added to `signal` too, with its upgrade.
- A change to the shape of a BreakoutIQ section's payload bumps that section's `PAYLOAD_VERSION` and needs the parent's decoder (`internal/repo/analysis_payload.go`) to read it. A change to a BreakoutIQ method, constant or label updates the parent's `GLOSSARY.md` ("BreakoutIQ analysis") with it.


## Logging, output and secrets

- Log through `logging.getLogger(__name__)`; `print` only a command's own summary or prompt.
- The `PROGRESS` and `FLAGGED` lines the dashboard parses are written only through `app/progress.py`, which validates them.
- Never log or print `TIINGO_API_KEY` or `EODHD_API_KEY`. Keep output going through logging, whose `stdout` handler redacts every record, and run any message holding a URL or an error through `error_detail()` or `redact()`.
- A command missing a config key it needs stops with `ConfigError` naming every missing key (exit 2), not a traceback.


## Testing

- The test suite should not have tests that are tautological, or too structure-sensitive, or where stubs don't match reality.
- `uv run pytest` passes. Tests are plain pytest functions in `tests/test_*.py`, each module opening with a docstring saying what it covers, each test named for the behaviour it proves as a sentence (`test_ticker_with_fewer_rows_than_the_stretch_is_not_frozen`).
- Check computed values against hand-worked figures on made-up price series, not against the code's own output.
- Tests that touch the database use a real SQLite file under `tmp_path`, created from `schema.sql` or brought up through `UPGRADES` from an older version, never `data/stock_data.db`.
- Price APIs are faked from recorded answers or with a `Fake*` class implementing only what the code calls; tests never call Tiingo or EODHD. Config is set with `monkeypatch.setenv` and `monkeypatch.delenv`.
- A statistical method is tested on synthetic data from a seeded `np.random.default_rng`: it must find a planted effect, and find nothing in the large majority of simulations with no effect.
- A change to the criteria or indicators updates the tests that cover them in the same change. Check any other change by running the affected commands against a scratch database.
