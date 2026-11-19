import json
from pathlib import Path
import statistics

root = Path("measurements")
base = json.loads((root / "semantic-base.json").read_text())
fixed = json.loads((root / "semantic-fixed.json").read_text())
assert base["results"] == fixed["results"], "Public API outputs differ"
summary = {"comparisons": len(base["results"]), "workloads": [], "allocation": []}
for index in range(5):
    values = {}
    outputs = []
    for variant in ("base", "fixed"):
        rounds = [json.loads((root / f"timing-{variant}-{r}.json").read_text())["rows"][index]
                  for r in range(1, 6)]
        values[variant] = statistics.median(row["median_ns"] for row in rounds)
        outputs.extend(row["output"] for row in rounds)
    assert all(output == outputs[0] for output in outputs)
    row = rounds[0]
    summary["workloads"].append({"vertices": row["n"], "distinct": row["distinct"],
                                 "base_ms": values["base"] / 1e6,
                                 "fixed_ms": values["fixed"] / 1e6,
                                 "time_change_percent": 100 * (values["fixed"] / values["base"] - 1)})
    before = json.loads((root / "allocation-base.json").read_text())["rows"][index]
    after = json.loads((root / "allocation-fixed.json").read_text())["rows"][index]
    assert before["output"] == after["output"]
    summary["allocation"].append({"vertices": row["n"], "distinct": row["distinct"],
                                  "base_peak_B": before["traced_peak_B"],
                                  "fixed_peak_B": after["traced_peak_B"]})
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
