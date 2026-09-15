from pathlib import Path
from tempfile import TemporaryDirectory
with TemporaryDirectory() as d:
    p = Path(d)/"intermediate.txt"
    p.write_text("temporary feature data")
    print(p.read_text())
