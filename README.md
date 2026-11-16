# Ticker screener

Fetches stock price data from Tiingo or EODHD, computes the screening criteria and writes everything to the SQLite database that the [stock_analyzer](https://github.com/nloyola/stock_analyzer) dashboard reads. See [CLAUDE.md](CLAUDE.md) for the commands and the architecture.

## Refreshing the database

`run-refresh` is the one way to refresh the market data: the stock_analyzer admin pages run it, and so should an operator running the screener by hand. It runs `ticker-fetcher --strict`, deletes the price rows older than 400 days once that fetch has succeeded, and then runs `analyze`, while holding an exclusive lock on `screener.lock` in the directory of the database file (`DB_NAME`), so two refreshes never write the same database at once. `--max-failure-rate` works as it does for `ticker-fetcher`. Every fetch downloads each ticker's last 400 days (enough for a 200-day SMA and a 21-week EMA) and replaces its stored rows, since both APIs restate past adjusted prices after a split or dividend; each row holds the day's close, open, high, low and volume, the prices all adjusted on one basis (Tiingo's adjusted fields; EODHD's adjusted close, with its raw open, high and low scaled by the day's ratio of adjusted to raw close); a request that meets a 429, a 5xx or a failed connection is tried up to 3 times, with backoff and the API's `Retry-After`, while any other error (a 404 for an unknown ticker) fails at once; a ticker whose download still fails keeps its rows, and one whose download holds no day in the 400 is logged at WARNING as likely delisted. A ticker the API answers with a 404, one with no day in the 400, and every stale or frozen ticker below are also reported on a `FLAGGED` line (see [Flagged tickers](#flagged-tickers)). `analyze` then leaves that ticker unscored: any ticker whose last price is older than the newest one in its market keeps its previous `stock_criteria` row and its `date_created`, with its status cleared, since its stored prices no longer support it, logged at WARNING, and the count is printed in the analyze summary. Tiingo keeps repeating the last close of a ticker that stopped trading, at zero volume, so its rows look current while its price is not. A ticker that traded on each of the 30 days before such a flat stretch of 2 days or more is stale too, dated by its last trade, as is a ticker already stale whose rows end that way: EODHD's rows for it end on that last trade, so both price APIs reach the same verdict. Thinly traded tickers also have flat zero-volume days, so for them only a longer stretch counts: a frozen ticker, whose last 10 trading days all have one close and zero volume, also keeps its row, without its status. In every market it scores, `analyze` clears the status of each row it did not write, a ticker left with too few rows to score or with none included.

After the tickers, every fetch downloads the broad-market benchmark of each market it requested tickers of (`BENCHMARKS` in `app/markets.py`: `SPY` for the US, `XIU.TO` for Canada), one request each over the same 400 days, and replaces its closes and opens in `benchmark_price`, a table apart from `price_data`; `analyze` compares each ticker's 1- and 3-month returns with it. So a Tiingo run requests only `SPY`, and EODHD's quota check counts the benchmarks with the tickers. A benchmark whose download fails, after the same retries, or holds no day, is logged at WARNING and keeps its stored closes; it never counts as a failed ticker, and `analyze` leaves the market comparison empty wherever the benchmark lacks a date it needs.

After the prices, every fetch looks up the company name of each ticker whose prices it stored and that has no name in `company_name` yet: Tiingo's metadata (`/tiingo/daily/<TICKER>`), or EODHD's search on the ticker's own exchange (`US`, `TO` or `V`), keeping only the match with exactly its symbol. That is one extra request per such ticker, about one per listed ticker on the first run and then only for tickers added since; a name once stored is never looked up again. Each name is stored as soon as it is found, so an interrupted run keeps what it found. A lookup that fails, after the same retries as a price request, is logged at WARNING and skipped, and one that finds no name is logged at INFO: neither counts as a failed ticker, so names never fail a run or push it over `--max-failure-rate`.

Every `analyze` that scores at least one of a market's tickers records that market's session, its newest price date, in `screened_session`: the screener version and rule settings it scored by (`SCREENER_VERSION` and `rule_settings()` in `app/analyzer.py`, the settings as JSON), the screener's git commit, the stock_analyzer run it came from and how many of the market's tickers it scored. A run that scores a session again replaces its row, so each session carries the rules that last scored it. The commit is `SCREENER_COMMIT` when set, as the image sets it from its build's revision, or else the checkout's HEAD, ending in `-dirty` when the checkout has uncommitted changes, and is left empty when neither says. The run is `SCREENER_RUN_ID`, which the stock_analyzer screener runner sets to its run's id, and is left empty for a run started by hand. `db-init` keeps the table.

Last, `run-refresh` keeps each archived signal's own price path in `signal_price`, since `price_data` is rewritten by every run and deleted when the dashboard removes a ticker (`app/signal_paths.py`). For every signal of the markets it fetched whose path is not frozen, it rewrites the whole path from this run's `price_data` and `benchmark_price`, so every value shares one adjustment basis: the ticker's open, high, low, close and volume and the benchmark's open and close, from the signal session (index 0) to the market's last complete session, or to the 120th session after it, when the path is frozen (`signal_track.path_frozen_at`) and never rewritten again. A path that reaches the 120th session while it lacks a benchmark price its outcomes need, and the benchmark's stored closes end before its last session (the run's benchmark download failed and kept the closes it had), is not frozen until a later run's benchmark covers it; a gap inside the benchmark's stored range freezes as it is. A signal whose ticker the run flagged (`not_found`, `no_prices`, `stale` or `frozen`, the fetch's reason winning), or whose ticker has no `price_data` row left (`removed`), keeps its path as it is, and each horizon (5, 10, 20, 40, 60 and 120 sessions) beyond the path's last session is marked ended in `signal_outcome`, with that reason and that session; a later run that has prices for it again rewrites the path and clears those ended horizons. A complete horizon is never touched. A signal whose session is no longer in `price_data` keeps its path, logged at WARNING. It logs one summary line.

Then, from those paths, it stores each signal's outcomes (`app/signal_outcome_updates.py`, computed by `app/signal_outcomes.py`): every horizon whose exit session is now in the path becomes a complete row in `signal_outcome`, with the entry and exit, the gross and net returns (20 basis points of round-trip cost), the benchmark's and the excess return, and MFE and MAE; and once a close in the 10 sessions after the signal session falls below the breakout level, or all 10 have closed at or above it, the breakout failure is decided in `signal_track`. Until then `signal_track.breakout_level` holds the level rescaled to the path's current basis. A complete row and a decided failure are never computed or changed again, even when a later run restates the prices, with one exception: a horizon completed while its path lacked the benchmark's entry or exit price gets the benchmark's return, and the excess return over it, once a later run's path has them, and nothing else in its row changes; a horizon the path does not reach yet has no row, or stays ended. A signal archived without the high its breakout cleared gets its horizons but no failure, logged at WARNING. It logs one summary line.

`compute-outcomes` does only that last part on demand, from the paths already stored, and stores exactly the rows a run would; it holds `screener.lock` like `run-refresh`, exiting 75 while a refresh holds it. It never updates a path, so a signal with no path yet, such as one `backfill-signals` just archived, gets its outcomes after the next run.

## Seeding the signal archive

`backfill-signals` seeds `signal`, the archive of confirmed breakouts each `analyze` adds to, from the prices already stored. It replays every session the 400-day window can fully score, those with at least 215 sessions (the 200-day SMA and its rising check) and 51 calendar weeks (the 21-week EMA once its seed has faded, and its rising check) of its market's prices on or before them, oldest first, scoring each from only the rows dated on or before it, and archives its signals with source `backfill`, the current screener version, rule settings and commit, and no run. With a full window that is about the last seven weeks. A market closed on a session is not replayed on it, and a session whose candles are not final yet is left out. A signal already archived for the ticker, session and version, live or backfill, is kept as it was, so a second invocation adds nothing; a live signal archived before the backfill, on a later session than one the backfill archived, is relinked to the ticker's nearest earlier signal. It prints one line per session and a summary of the sessions replayed, signals archived and signals relinked, and holds `screener.lock` like `run-refresh`, exiting 75 while a refresh holds it. Run it once after the upgrade that adds `signal`.

## Fetching the price history

`history-fetch` fills the price history database, the SQLite file `HISTORY_DATABASE_FILE` names, apart from `DB_NAME`, with the last 10 years (`HISTORY_YEARS` in `app/history_fetch.py`) of daily prices of every ticker the sectors page lists and of the benchmark of every market the price API covers, from the API `PRICE_API` names, for replaying the screener over past sessions. It never writes `DB_NAME`, which it only reads for the tickers, so years of prices stay out of `price_data` and out of the live database's backups; the history can always be fetched again. Its tables, which it creates, are `history_price` and `history_benchmark_price`, each day's adjusted open, high, low, close and volume on one basis, as `price_data` stores them, so a split leaves no jump, and `history_fetch`, the day each ticker and benchmark was last fetched (in New York) and from which API. It stores only final candles: a day is left out until 3 hours after its market's close, since the API answers with an unfinished candle for it and a fetch the same day would not replace it.

It costs one request per ticker and benchmark on either API, and prints that count, and the tickers it leaves out, before the first request; EODHD's quota check stops it before any request when the day's calls cannot cover it. Each ticker's prices replace its stored ones and record it as fetched, in one transaction, and a fetch run again the same day on the same API skips those fetched already, so an interrupted fetch (it exits 130) resumes where it stopped and a finished one requests nothing. A request still failing after the usual retries keeps the stored prices and is requested again by the next run, and the fetch exits 1; a 404 or an empty answer keeps the stored prices and counts as fetched. It ends with the number of tickers and benchmarks with history, and lists those with none and those whose history starts more than 14 days after the requested start, with their first day. A ticker of a market the API does not cover (a Canadian one under Tiingo) is never requested.

## Replaying the price history

`backfill-history` replays the screener over the price history `history-fetch` stored, and archives the signals in `DB_NAME`'s `signal` with source `backfill`, as `backfill-signals` does from the 400-day window: every session the history can fully score (the same 215 sessions and 51 calendar weeks of its market's history on or before it, and only once its candles are final), for every ticker the sectors page lists now that has a history, each in the subsector that lists it first, scored against the history's benchmarks. The universe is today's, so a ticker no longer listed is left out, and a ticker with no history (a Canadian one fetched under Tiingo) is listed in the summary. The values are those `Analyzer.score_as_of` gives on the history for each session, but each indicator is computed once over a ticker's whole history (`app/replay.py`) rather than once per session, so the work grows with the length of the history rather than its square, and the tickers are replayed one subsector at a time, so only one subsector's histories are held in memory. A signal already archived for the ticker, session and version is kept as it was, so a second invocation adds nothing, and a later signal is relinked as `backfill-signals` relinks it. Then it measures the backfill signals from the history (`app/history_paths.py`): every backfill signal whose path is not frozen and whose ticker has a history, those `backfill-signals` archived too, gets its path in `signal_price` rewritten whole from the history and its benchmark's history, as `run-refresh` writes one from `price_data`, from the signal session to the 120th session after it, when it is frozen, or to the last session of the history. A path that reaches its market's last session in the history is left for the next `run-refresh` to carry on from `price_data`; one whose ticker's history stops earlier keeps that path, and each horizon it does not reach is marked ended with the reason `history_ended` and the path's last session. A live signal, and a backfill signal whose ticker has no history, is left to `run-refresh`. It then stores every signal's outcomes from the paths, as `compute-outcomes` does, so a backfill signal at least 120 sessions old has all six. It prints one line per subsector and a summary of the sessions replayed, with their range, the tickers, the signals archived and relinked, the time taken, the paths written, frozen and ended, and the horizons completed and breakout failures decided, and holds `screener.lock`, exiting 75 while a refresh holds it. It needs `DB_NAME` and `HISTORY_DATABASE_FILE`.

Each run that finishes is recorded in `DB_NAME`'s `backfill_run`, for the report on the backfilled signals to state its coverage and biases: the range of sessions replayed, the universe date (the day the sectors page was read, in New York, since the universe is that day's), how many tickers were listed, replayed and scored, the listed tickers never scored with why (`no_history`, `unsupported`, `short_history` for a history that starts late and leaves no session to score, or `never_scored`), every replayed ticker whose history starts late (more than two weeks after the 10 years `history-fetch` asked for on the day it fetched it) with its first day, the price API the histories came from, the signals archived and relinked, and the screener version and commit. The summary ends with the tickers scored on no session, those whose history starts late, and the run's id. A second run is recorded too, archiving nothing.

## Analysing the signal archive

`breakoutiq-analysis` runs BreakoutIQ's whole analysis of the signal archive (`app/breakoutiq/`): it builds the analysis dataset once, one row per signal and resolved horizon, runs every section on it (the baseline, the indicator ranking, the mandatory strengths, the confounders and the candidate rules) and stores one `analysis_run` row with every section's results in `analysis_result`, in one transaction. The run records the signal sessions and signals it read, the screener version and commit, and the settings it ran with, every constant the sections read and each section's payload version. It reads the regime of a backfilled signal's session from the benchmark's price history, so it needs `HISTORY_DATABASE_FILE` as well as `DB_NAME`. It only reads the archive and the history, takes no lock, and leaves the live screener as it is. It logs each section's result count and time as it finishes, and ends with a summary: the signals and resolved horizons read, how many were censored, each section's results and the run's id. An archive with no resolved horizon stores a run of 0 signals and no results. The stock_analyzer repo's `GLOSSARY.md` ("BreakoutIQ analysis") explains every method, constant and label.

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

`analyze` prints one summary line per market that has prices: `US: skipped 1 stale ticker (stopped trading, or last price before 2026-11-06) and 13 frozen tickers` for a scored market, and `Canada: not scored, not fetched by this run (Tiingo)` for one a refresh left out. Before the summary, a market whose day is not yet complete gets a line saying how many rows it dropped: `US: dropped 1202 rows dated after 2026-11-05, whose candles are final only after 19:00 America/New_York that day`.

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
