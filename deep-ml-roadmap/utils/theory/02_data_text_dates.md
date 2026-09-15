# 02 — Data, Text and Dates

## json
Use `load/dump` for files and `loads/dumps` for strings.
Great for configs, metadata, API payloads, and experiment reports.

## csv
Know `reader`, `DictReader`, `writer`, and `DictWriter`.
Pandas remains the main tool for serious tabular analysis.

## pickle
Python-specific object serialization. Never load untrusted pickle data.

## datetime
Know `datetime.now`, `strptime`, `strftime`, and `timedelta`.
Prefer timezone-aware timestamps when the exact moment matters.

## re
Know `search`, `findall`, and `sub` for extracting and cleaning text.

## string
Useful constants include `digits`, `ascii_letters`, and `punctuation`.

### 80/20
Prioritize JSON, datetime, regex, and basic CSV handling.
