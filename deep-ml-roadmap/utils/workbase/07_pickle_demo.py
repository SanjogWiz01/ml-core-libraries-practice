import pickle, tempfile
from pathlib import Path
obj = {"features": ["age", "income"], "version": 1}
with tempfile.TemporaryDirectory() as d:
    path = Path(d)/"meta.pkl"
    with path.open("wb") as f: pickle.dump(obj, f)
    with path.open("rb") as f: print(pickle.load(f))
# Never unpickle untrusted data.
