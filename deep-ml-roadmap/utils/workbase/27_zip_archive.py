from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
with TemporaryDirectory() as d:
    folder = Path(d)/"artifacts"; folder.mkdir()
    (folder/"metrics.txt").write_text("accuracy=0.91")
    archive = Path(d)/"artifacts.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        z.write(folder/"metrics.txt", arcname="metrics.txt")
    print("Created:", archive)
