import csv
from io import StringIO
raw = StringIO("name,score\nRam,82\nSita,91\nHari,76\n")
rows = list(csv.DictReader(raw))
print(rows)
print([int(r["score"]) for r in rows])
