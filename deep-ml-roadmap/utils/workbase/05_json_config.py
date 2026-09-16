import json
from pathlib import Path
config = {"seed": 42, "test_size": 0.2, "model": "random_forest"}
path = Path("config_demo.json")
path.write_text(json.dumps(config, indent=2))
print(json.loads(path.read_text()))
path.unlink(missing_ok=True)
