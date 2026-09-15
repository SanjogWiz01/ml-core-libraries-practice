from pathlib import Path
for pattern in ("*.csv", "*.json", "*.py"):
    print(pattern, list(Path(".").rglob(pattern))[:10])
