# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

First release.

### Added

- Extraction from Claude Code session logs behind a strict field allowlist, with
  unrecognized log fields counted and reported rather than absorbed
- Pseudonymization of session ids, working directories and branch names with a
  local keyed hash
- k-anonymity over categorical columns, reporting what it suppressed
- Privacy scan covering API keys, tokens, private keys, email addresses, absolute
  paths, URLs, IP addresses and long free text, run both before and after
  k-anonymity
- Session aggregation with cache hit rate, tool error rate and token totals
- Analyses of token economics, tool usage, model usage and session shape
- Command line interface: `extract`, `scan`, `summary`, `figures`
- Charts with one consistent style
- Anonymized dataset of 1,213 sessions and 181,406 events, with a data card
- Five notebooks covering data quality, token economics, tool usage, model
  comparison and the findings quoted in the README
