from pathlib import Path
import shutil, tempfile
with tempfile.TemporaryDirectory() as d:
    src, dst = Path(d)/"a.txt", Path(d)/"b.txt"
    src.write_text("data science")
    shutil.copy2(src, dst)
    print(dst.read_text())
