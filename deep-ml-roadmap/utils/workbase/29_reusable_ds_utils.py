from pathlib import Path
from collections import Counter
import json
def load_json(path): return json.loads(Path(path).read_text())
def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
def label_distribution(labels): return dict(Counter(labels))
if __name__ == "__main__":
    save_json("demo/config.json", {"seed":42})
    print(load_json("demo/config.json"))
    print(label_distribution(["A","B","A","A"]))
