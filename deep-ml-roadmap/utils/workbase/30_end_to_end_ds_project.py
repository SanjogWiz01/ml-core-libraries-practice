from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, logging, random
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
@dataclass
class Config:
    seed: int = 42
    experiment: str = "utility_demo"
    output_dir: str = "artifacts"
def make_dataset(seed, n=100):
    random.seed(seed)
    return [{"id":i,"label":random.choice(["good","bad"]),"score":random.randint(50,100)} for i in range(n)]
def main():
    cfg = Config()
    out = Path(cfg.output_dir); out.mkdir(exist_ok=True)
    rows = make_dataset(cfg.seed)
    counts = Counter(r["label"] for r in rows)
    text = json.dumps(rows, sort_keys=True)
    report = {
        "config": asdict(cfg),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "label_counts": dict(counts),
        "mean_score": round(sum(r["score"] for r in rows)/len(rows), 3),
        "dataset_sha256": hashlib.sha256(text.encode()).hexdigest()
    }
    (out/"report.json").write_text(json.dumps(report, indent=2))
    logging.info("Experiment complete")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
