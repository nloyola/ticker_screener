# Ticker screener

Fetches stock price data from Tiingo, computes the screening criteria and writes everything to the SQLite database that the [stock_analyzer](https://github.com/nloyola/stock_analyzer) dashboard reads. See [CLAUDE.md](CLAUDE.md) for the commands and the architecture.

## Progress lines

`ticker-fetcher` and `analyze` print progress lines on stdout, one per line, flushed as soon as they are printed. The stock_analyzer admin pages parse them to show a run's stage and progress, so the format is a contract: change it only together with the parser there. `tests/test_progress.py` covers it.

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
