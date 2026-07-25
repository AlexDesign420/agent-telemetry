# Contributing

## The most useful contribution is a second dataset

Every finding in this repository comes from one person's logs. That is enough to
describe a working style and not enough to conclude anything about agents in
general. The comparison between two datasets is where this gets interesting, and
it needs someone else to run the pipeline.

If you publish an extract of your own:

1. Run `agent-telemetry extract` and read the list of unrecognized log fields it
   prints. Your client version may write fields this allowlist has never seen.
2. Run `agent-telemetry scan` on both parquet files.
3. Read a sample of the rows yourself. The scan is a check, not a substitute for
   looking.
4. Consider whether the hour-resolution timestamps reveal more about your
   schedule than you want. Dropping `started_at`, `ended_at` and `timestamp` costs
   the time-of-day analysis and nothing else.
5. Write a data card. `data/DATA_CARD.md` is a template; the sections that matter
   are the columns that are easy to misread and the limitations.

## Changing the privacy code

Extra scrutiny applies to anything under `src/agent_telemetry/privacy/`.

- **New readable field**: add it to the allowlist in `allowlist.py` with a comment
  saying why it cannot carry text, and add it to the schema. Do not read a field
  that is not listed.
- **New scan rule**: add it to `RULES` in `scan.py` and add a test that fires it.
  Every existing rule has one.
- **Never turn a finding into a filter.** If the scan removed the offending value
  instead of raising, a hole in the allowlist would keep producing clean-looking
  output forever. Raising is the point.
- **Do not reorder the two scans in `write_dataset`.** The pass before
  k-anonymity exists so that a leak occurring once is not folded into `other`
  before anyone sees it; the pass after exists because that frame is what reaches
  disk. There is a test for each.

## Adding an analysis

Analyses live in `src/agent_telemetry/analysis/`, take a DataFrame and return a
DataFrame. Nothing in there reads from disk or draws a chart, which is what keeps
the notebooks thin and the results testable.

State the confounder in the docstring where one exists. `analysis/models.py` is
the example: the functions are useful, and the docstring says why the obvious
interpretation of them is wrong.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy
```

mypy runs in strict mode and the CI job is not advisory. New behaviour needs a
test; bug fixes need a test that fails before the fix.

## Regenerating the published dataset

```bash
agent-telemetry extract --source ~/.claude/projects --output data
agent-telemetry figures
cd notebooks && for f in *.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$f"; done
```

The numbers quoted in the README come from `notebooks/05-findings.ipynb`. If a
change moves them, update both.
