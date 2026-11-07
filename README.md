# Ticker screener

Fetches stock price data from Tiingo or EODHD, computes the screening criteria and writes everything to the SQLite database that the [stock_analyzer](https://github.com/nloyola/stock_analyzer) dashboard reads. See [CLAUDE.md](CLAUDE.md) for the commands and the architecture.

## Refreshing the database

`run-refresh` is the one way to refresh the market data: the stock_analyzer admin pages run it, and so should an operator running the screener by hand. It runs `ticker-fetcher --strict`, deletes the price rows older than 120 days once that fetch has succeeded, and then runs `analyze`, while holding an exclusive lock on `screener.lock` in the directory of the database file (`DB_NAME`), so two refreshes never write the same database at once. `--max-failure-rate` works as it does for `ticker-fetcher`. Every fetch downloads each ticker's last 120 days and replaces its stored rows, since both APIs restate past adjusted closes after a split or dividend; a request that meets a 429, a 5xx or a failed connection is tried up to 3 times, with backoff and the API's `Retry-After`, while any other error (a 404 for an unknown ticker) fails at once; a ticker whose download still fails keeps its rows, and one whose download holds no day in the 120 is logged at WARNING as likely delisted. `analyze` then leaves that ticker unscored: any ticker whose last price is older than the newest one in its market keeps its previous `stock_criteria` row and its `date_created`, logged at WARNING, and the count is printed in the analyze summary.

## Price APIs and markets

The `PRICE_API` setting (`tiingo`, the default, or `eodhd`), read from the environment or `.env` like the keys, picks the API `ticker-fetcher` and `run-refresh` fetch from; only that API's key is needed. One run never mixes the two, and never falls back from one to the other.

A ticker's market is read from the ticker itself (`app/markets.py`): a bare ticker (`AAPL`, `BRK-B`) is US, one ending in `.TO` (Toronto Stock Exchange) or `.V` (TSX Venture) is Canada, and any other suffix is refused. Canadian prices are stored in CAD, unconverted. EODHD covers both markets, asked for a bare ticker as `<TICKER>.US`. Tiingo covers only the US: a Tiingo run skips Canadian tickers, which keep their stored rows and count as neither attempted nor failed, and its fetch summary says how many it skipped (`ticker-fetcher: 1228 attempted, 2 failed (0%, threshold 50%), 15 skipped (not covered by Tiingo)`).

The markets close on different holidays, so freshness is judged per market:

- `analyze` compares each ticker with the newest price date of its own market, so a US holiday on which Canada traded leaves no US ticker unscored, and the reverse.
- `run-refresh` scores only the markets its API fetched. The other market's tickers keep their previous `stock_criteria` rows and `date_created`, so their scores are never restamped as current. A standalone `analyze` scores every market.
- The 120-day cleanup after a strict fetch deletes old rows only in the markets the API fetched: a Tiingo run leaves Canadian prices alone.

`analyze` prints one summary line per market that has prices: `US: skipped 1 stale ticker (last price before 2026-11-06)` for a scored market, and `Canada: not scored, not fetched by this run (Tiingo)` for one a refresh left out.

Its exit code tells a caller what happened:

| Code | Meaning |
|---|---|
| `0` | The fetch and the analysis both ran. |
| `1` | The strict fetch failed (nothing fetched, every ticker failed, or the failure rate exceeded `--max-failure-rate`), so `analyze` did not run; or the price API stopped the fetch before any price request (EODHD with too few calls left today, or refusing the key), logged as an ERROR line ending in `ticker-fetcher stopped: <reason>`, which the stock_analyzer admin pages quote, so keep that wording; or a step raised an error. |
| `75` | Another refresh holds the lock. Nothing ran; try again later. |

The lock is a `flock`, which the kernel releases when its process exits or dies, so a crashed run never leaves it held. The `screener.lock` file stays on disk between runs and must not be deleted while a refresh may be running. Running `ticker-fetcher` or `analyze` on its own takes no lock.

## Progress lines

`ticker-fetcher` and `analyze`, and so `run-refresh`, print progress lines on stdout, one per line, flushed as soon as they are printed. The stock_analyzer admin pages parse them to show a run's stage and progress, so the format is a contract: change it only together with the parser there. `tests/test_progress.py` covers it.

Every progress line has this shape, and no other output line starts with `PROGRESS`:

```
PROGRESS stage=<stage> [<key>=<value> ...]
```

- The line starts with `PROGRESS`, then `stage=<stage>` as the first field. More fields may follow, separated by single spaces.
- Keys are lowercase letters and underscores. Values contain only letters, digits and `_ . / -`: never spaces, `=` or quotes.
- A reader should ignore fields and stages it does not know, so new ones can be added without breaking it.

The lines printed today:

| Command | Line | Meaning |
|---|---|---|
| `ticker-fetcher` | `PROGRESS stage=fetch subsector=<done>/<total>` | `<total>` is the number of subsectors this run fetches (after `--sector`; `1` with `--ticker`), and `<done>` how many are finished. Printed once with `0/<total>` before the first subsector, then after each one, so the last line is `<total>/<total>`. |
| `analyze` | `PROGRESS stage=analyze status=start` | Printed before the analysis starts. |
| `analyze` | `PROGRESS stage=analyze status=done` | Printed when the analysis finishes. |

A run that stops early - an unknown `--sector` or `--ticker`, or an error - may print no `fetch` line, or stop short of `<total>/<total>` or `status=done`.

For example, a fetch over three subsectors followed by an analysis prints, among its log lines:

```
PROGRESS stage=fetch subsector=0/3
PROGRESS stage=fetch subsector=1/3
PROGRESS stage=fetch subsector=2/3
PROGRESS stage=fetch subsector=3/3
PROGRESS stage=analyze status=start
PROGRESS stage=analyze status=done
```
