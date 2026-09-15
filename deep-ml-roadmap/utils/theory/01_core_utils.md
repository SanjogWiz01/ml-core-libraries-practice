# 01 — Core Utilities

## pathlib
Use `Path` for portable filesystem work:
`exists`, `is_file`, `is_dir`, `mkdir`, `glob`, `rglob`, `read_text`,
`write_text`, `name`, `stem`, `suffix`, `parent`.

## os
High-value functions: `getcwd`, `environ`, `getenv`.
Use environment variables for configuration and secrets rather than hardcoding.

## glob
Useful for patterns such as `*.csv`, `*.json`, and recursive file discovery.

## shutil
High-level file operations: `copy2`, `move`, `make_archive`.

## tempfile
Use temporary files/directories for intermediate ML/data artifacts.

### 80/20
Master `Path`, `glob/rglob`, `os.getenv`, `shutil.copy2`, and
`TemporaryDirectory`.
