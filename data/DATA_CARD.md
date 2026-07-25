# Data card

## What this is

Metrics from 1,224 Claude Code session logs recorded on one person's machine
between 2026-06-01 and 2026-07-25, reduced to 1,213 sessions and 181,406 events.

Two tables:

| File | Rows | Grain |
| --- | --- | --- |
| `sessions.parquet` | 1,213 | one session |
| `events.parquet` | 181,406 | one line of a session log |

## Where it comes from

Claude Code writes one JSON Lines file per session under `~/.claude/projects/`.
Those files contain the full conversation: prompts, model output, tool inputs and
tool results. None of that is in this dataset.

The extraction reads a fixed allowlist of fields — categories, counters, token
usage — and discards everything else before any row is constructed. Text is
measured for its length and then dropped. The pipeline that produced this is in
`src/agent_telemetry/`; `agent-telemetry extract` reproduces it from your own
logs.

## What was done to the data

1. **Field allowlist.** Only fields named in `privacy/allowlist.py` are read.
   Unknown fields are counted and reported, never absorbed.
2. **Pseudonymization.** Session ids, working directories and branch names are
   replaced by keyed hashes. The key is local and is not published, so the labels
   are stable within the dataset and meaningless outside it.
3. **Timestamp truncation.** Timestamps are truncated to the hour.
   `offset_seconds` keeps finer resolution as a duration relative to the start of
   its session.
4. **k-anonymity, k=5.** Any category value occurring fewer than five times is
   folded into `other`. This affected 12 model names, 11 client versions, 40 tool
   names and 33 model combinations.
5. **Privacy scan.** A regex scan for API keys, tokens, private key blocks, email
   addresses, absolute paths, URLs, IP addresses and long free text runs twice:
   once before k-anonymity, so that a rare leak is not silently folded into
   `other`, and once on the exact frame that is written.

## Columns that are easy to misread

**`duration_minutes`** is the wall clock distance between the first and the last
event of a session. A session left open overnight reports the whole night. It is
elapsed time, not working time, and it is not a measure of effort. The p99 is
616 minutes while the median is 2.4.

**`total_tokens`** includes cache reads, and a cache read re-counts the entire
context on every turn. The dataset therefore reports 17.8 billion tokens across
1,213 sessions, which is arithmetically correct and not comparable to a token
count from one-shot API calls.

**`is_sidechain`** is a property of the whole session, not of individual events:
a subagent gets its own log file. 84 percent of session files in this dataset are
subagent runs. Counting them as "sessions that used a subagent" would confuse the
agent with its caller. See `sessions.delegating_sessions` for the other question.

**`primary_model`** is blank for sessions that produced no assistant message —
an aborted start, or a session that only changed a setting. Those rows are real
and are kept, but they have to be excluded from any model comparison.

**Empty strings, not nulls.** Missing categories are `""`. There are no null
values in the categorical columns.

## Known limitations

- **One user, one machine, two months.** Everything here reflects one working
  style. Nothing generalizes without a second dataset to compare against, and
  producing one is what the pipeline is for.
- **No outcome data.** The logs record what happened, never whether it worked. No
  success rate can be derived from this.
- **Tool errors are a floor.** Only results flagged `is_error` are counted. A tool
  that reports failure in its output text is invisible, because the output text is
  deliberately not read.
- **Mixed providers.** The models include hosted Anthropic models, a third party
  API and locally served models. They are not comparable on cost or capability;
  `analysis.models.provider_of` groups them for that reason.
- **Model choice is not random.** Models were selected per task. Any correlation
  between model and session size reflects that selection, not model behaviour.
- **Working hours are visible.** Timestamps are truncated to the hour but not
  removed, so the dataset shows when its author was working. This was a deliberate
  trade: the daily and weekly pattern is one of the few things worth analyzing in
  a single user dataset. If you publish your own extract and would rather not
  reveal that, drop `started_at`, `ended_at` and `timestamp` before publishing.
- **Rare categories are gone.** Anything below k=5 reads as `other`. Long tail
  questions about rarely used tools cannot be answered from this file.

## Reproducing it

```bash
agent-telemetry extract --source ~/.claude/projects --output data
agent-telemetry scan data/sessions.parquet
agent-telemetry summary
```

Your labels will differ from the ones here: the salt is per machine, so the same
session hashes to a different label for you than it does in this file.

## License

The dataset in this directory is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code that produced
it is MIT; see the repository root.
