from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, logging, random
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
random.seed(42)
out = Path("ds_artifact_demo"); out.mkdir(exist_ok=True)
rows = [{"label":random.choice(["A","B"]), "score":random.randint(60,100)} for _ in range(20)]
report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "rows": len(rows),
    "label_counts": dict(Counter(r["label"] for r in rows)),
    "mean_score": sum(r["score"] for r in rows)/len(rows)
}
(out/"report.json").write_text(json.dumps(report, indent=2))
logging.info("Saved report")
print(report)
