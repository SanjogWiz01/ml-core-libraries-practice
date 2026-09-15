# 04 — Reliability, Reproducibility and Workflows

## Exceptions
Use specific `try/except` blocks for expected failures. Do not hide bugs with
a broad `except:`.

## Logging
Use `logging` instead of scattered `print` statements in reusable scripts.
Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.

## Reproducibility
Seed Python's `random` module and separately seed NumPy/ML frameworks when used.

## hashlib
Useful for dataset/file fingerprints and cache keys.

## dataclasses
Excellent for experiment configuration and lightweight structured records.

## argparse
Turns a data-science script into a reusable command-line program.

## sqlite3
Useful for small local experiment/result stores.

## 80/20 workflow
config -> paths -> load -> validate -> transform -> experiment -> log ->
save artifact -> report metadata

These utilities complement, rather than replace, NumPy/Pandas/scikit-learn.
