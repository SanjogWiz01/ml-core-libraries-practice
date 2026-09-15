from pathlib import Path
p = Path(".")
print("Absolute:", p.resolve())
print("Python files:", list(p.glob("*.py"))[:10])
