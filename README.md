# Ticker screener

Fetches stock price data from Tiingo or EODHD, computes the screening criteria and writes everything to the SQLite database that the [stock_analyzer](https://github.com/nloyola/stock_analyzer) dashboard reads. See [CLAUDE.md](CLAUDE.md) for the commands and the architecture.

## Refreshing the database

`run-refresh` is the one way to refresh the market data: the stock_analyzer admin pages run it, and so should an operator running the screener by hand. It runs `ticker-fetcher --strict`, deletes the price rows older than 400 days once that fetch has succeeded, and then runs `analyze`, while holding an exclusive lock on `screener.lock` in the directory of the database file (`DB_NAME`), so two refreshes never write the same database at once. `--max-failure-rate` works as it does for `ticker-fetcher`. Every fetch downloads each ticker's last 400 days (enough for a 200-day SMA and a 21-week EMA) and replaces its stored rows, since both APIs restate past adjusted prices after a split or dividend; each row holds the day's close, high, low and volume, the prices all adjusted on one basis (Tiingo's adjusted fields; EODHD's adjusted close, with its raw high and low scaled by the day's ratio of adjusted to raw close); a request that meets a 429, a 5xx or a failed connection is tried up to 3 times, with backoff and the API's `Retry-After`, while any other error (a 404 for an unknown ticker) fails at once; a ticker whose download still fails keeps its rows, and one whose download holds no day in the 400 is logged at WARNING as likely delisted. A ticker the API answers with a 404, one with no day in the 400, and every stale or frozen ticker below are also reported on a `FLAGGED` line (see [Flagged tickers](#flagged-tickers)). `analyze` then leaves that ticker unscored: any ticker whose last price is older than the newest one in its market keeps its previous `stock_criteria` row and its `date_created`, with its status cleared, since its stored prices no longer support it, logged at WARNING, and the count is printed in the analyze summary. Tiingo keeps repeating the last close of a ticker that stopped trading, at zero volume, so its rows look current while its price is not. A ticker that traded on each of the 30 days before such a flat stretch of 2 days or more is stale too, dated by its last trade, as is a ticker already stale whose rows end that way: EODHD's rows for it end on that last trade, so both price APIs reach the same verdict. Thinly traded tickers also have flat zero-volume days, so for them only a longer stretch counts: a frozen ticker, whose last 10 trading days all have one close and zero volume, also keeps its row, without its status.

After the tickers, every fetch downloads the broad-market benchmark of each market it requested tickers of (`BENCHMARKS` in `app/markets.py`: `SPY` for the US, `XIU.TO` for Canada), one request each over the same 400 days, and replaces its closes in `benchmark_price`, a table apart from `price_data`; `analyze` compares each ticker's 1- and 3-month returns with it. So a Tiingo run requests only `SPY`, and EODHD's quota check counts the benchmarks with the tickers. A benchmark whose download fails, after the same retries, or holds no day, is logged at WARNING and keeps its stored closes; it never counts as a failed ticker, and `analyze` leaves the market comparison empty wherever the benchmark lacks a date it needs.

After the prices, every fetch looks up the company name of each ticker whose prices it stored and that has no name in `company_name` yet: Tiingo's metadata (`/tiingo/daily/<TICKER>`), or EODHD's search on the ticker's own exchange (`US`, `TO` or `V`), keeping only the match with exactly its symbol. That is one extra request per such ticker, about one per listed ticker on the first run and then only for tickers added since; a name once stored is never looked up again. Each name is stored as soon as it is found, so an interrupted run keeps what it found. A lookup that fails, after the same retries as a price request, is logged at WARNING and skipped, and one that finds no name is logged at INFO: neither counts as a failed ticker, so names never fail a run or push it over `--max-failure-rate`.

## Upgrading the schema

Every command brings the database's schema up to date before it opens it (`upgrade_db` in `app/db.py`). `db-upgrade` does only that, and fails when the database file is missing. The stock_analyzer release runs it before the dashboard starts, since the dashboard reads the columns the upgrades add and never upgrades the database itself.

## Price APIs and markets

The `PRICE_API` setting (`tiingo`, the default, or `eodhd`), read from the environment or `.env` like the keys, picks the API `ticker-fetcher` and `run-refresh` fetch from; only that API's key is needed. One run never mixes the two, and never falls back from one to the other.

A ticker's market is read from the ticker itself (`app/markets.py`): a bare ticker (`AAPL`, `BRK-B`) is US, one ending in `.TO` (Toronto Stock Exchange) or `.V` (TSX Venture) is Canada, and any other suffix is refused. Canadian prices are stored in CAD, unconverted. EODHD covers both markets, asked for a bare ticker as `<TICKER>.US`. Tiingo covers only the US: a Tiingo run skips Canadian tickers, which keep their stored rows and count as neither attempted nor failed, and its fetch summary says how many it skipped (`ticker-fetcher: 1228 attempted, 2 failed (0%, threshold 50%), 15 skipped (not covered by Tiingo)`).

The markets close on different holidays, so freshness is judged per market:

- `analyze` scores only completed daily candles. A market's day is complete 3 hours after its 16:00 close (`MARKET_CLOSES` and `PUBLISH_DELAY` in `app/markets.py`), when both price APIs have published its final prices; a run started before then, in that market's time zone, first drops the rows dated that day, so a run during market hours scores the previous session. `run-refresh` judges this by when it started, not when its `analyze` begins.
- `analyze` compares each ticker with the newest price date of its own market, so a US holiday on which Canada traded leaves no US ticker unscored, and the reverse.
- `run-refresh` scores only the markets its API fetched. The other market's tickers keep their previous `stock_criteria` rows and `date_created`, so their scores are never restamped as current. A standalone `analyze` scores every market.
- The 400-day cleanup after a strict fetch deletes old rows only in the markets the API fetched: a Tiingo run leaves Canadian prices alone.

`analyze` prints one summary line per market that has prices: `US: skipped 1 stale ticker (last price before 2026-11-06) and 13 frozen tickers` for a scored market, and `Canada: not scored, not fetched by this run (Tiingo)` for one a refresh left out. Before the summary, a market whose day is not yet complete gets a line saying how many rows it dropped: `US: dropped 1202 rows dated after 2026-11-05, whose candles are final only after 19:00 America/New_York that day`.

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
| `ticker-fetcher` | `PROGRESS stage=names ticker=<done>/<total>` | After the fetch: `<total>` is the number of tickers whose company name it looks up (those it stored prices for that have no name yet, `0` when every one has a name), and `<done>` how many lookups are finished, found or not. Printed once with `0/<total>` before the first lookup, then after each one, so the last line is `<total>/<total>`. |
| `analyze` | `PROGRESS stage=analyze status=start` | Printed before the analysis starts. |
| `analyze` | `PROGRESS stage=analyze status=done` | Printed when the analysis finishes. |

A run that stops early - an unknown `--sector` or `--ticker`, or an error - may print no `fetch` or `names` line, or stop short of `<total>/<total>` or `status=done`.

For example, a fetch over three subsectors that stores prices for two tickers with no name yet, followed by an analysis, prints, among its log lines:

```
PROGRESS stage=fetch subsector=0/3
PROGRESS stage=fetch subsector=1/3
PROGRESS stage=fetch subsector=2/3
PROGRESS stage=fetch subsector=3/3
PROGRESS stage=names ticker=0/2
PROGRESS stage=names ticker=1/2
PROGRESS stage=names ticker=2/2
PROGRESS stage=analyze status=start
PROGRESS stage=analyze status=done
```

## Flagged tickers

`ticker-fetcher` and `analyze`, and so `run-refresh`, print one flagged line on stdout for each ticker that looks dead or behind, flushed as soon as it is printed, right after the log line that reports the same ticker (a warning, except for a `not_found` ticker without `--strict`, whose failed download is logged at DEBUG). The stock_analyzer admin pages collect them to list a run's problem tickers and offer to delete them, so the format is part of the same contract as the progress lines: change it only together with the parser there. `tests/test_progress.py` covers the format, and `tests/test_ticker_fetcher.py` and `tests/test_analyzer_db.py` when each line is printed.

Every flagged line has this shape, and no other output line starts with `FLAGGED`:

```
FLAGGED ticker=<ticker> reason=<reason> [last=<YYYY-MM-DD>]
```

- The fields follow the progress lines' rules: single spaces between them, lowercase keys, and values of only letters, digits and `_ . / -`. A ticker those values cannot carry, or one longer than 20 characters, is logged at WARNING instead of flagged.
- `last` is omitted when there is no date to give. A reader should ignore fields and reasons it does not know.

| Reason | Printed by | Meaning | `last` |
|---|---|---|---|
| `not_found` | `ticker-fetcher` | The price API answered the ticker's price request with a 404: it does not know the ticker. Printed with or without `--strict`. | The newest price date stored for the ticker before this run, if any. |
| `no_prices` | `ticker-fetcher` | The download holds no trading day in the last 400 days (the "likely delisted" warning). | The newest price date stored for the ticker before this run, if any. It is read during the fetch, since the prune that follows a strict fetch may delete every row the ticker had. |
| `stale` | `analyze` | Not scored, because its last price is older than the newest price date of its market, or because it traded on each of the 30 days before its rows began repeating its last close at zero volume, for 2 days or more: it stopped trading. | That last price date, or its last traded day when its rows end in that repeated close. |
| `frozen` | `analyze` | Not scored, because its last 10 trading days all have the same close and zero volume, while it traded too thinly before them to be `stale`: the API repeats the last close of a ticker that stopped trading (Tiingo does). | The first day of that flat stretch, which may be more than 10 trading days back. |

Each ticker is flagged at most once per run. A ticker listed in several subsectors is fetched, and so flagged, once. A ticker that fails its fetch keeps its stored rows, which `analyze` then finds stale, so `run-refresh` passes the tickers the fetch flagged to `analyze`, which still counts them in its stale or frozen summary but does not flag them again: the fetch's reason wins. A ticker both stale and frozen is counted and flagged as stale. A standalone `analyze` flags every stale and frozen ticker. Other failed downloads (a 401, a 5xx after its retries, a malformed answer) and tickers with an exchange suffix no market has are not flagged: they say nothing about whether the ticker still trades.

For example, a refresh where SMOP is unknown to the API, BLD has no recent prices, DINE is a day behind and ANSS has repeated one close with no volume since 2026-07-17 prints, among its log lines:

```
FLAGGED ticker=SMOP reason=not_found last=2026-10-31
FLAGGED ticker=BLD reason=no_prices last=2026-07-02
FLAGGED ticker=DINE reason=stale last=2026-11-05
FLAGGED ticker=ANSS reason=frozen last=2026-07-17
```
