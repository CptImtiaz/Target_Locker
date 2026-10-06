from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


COLUMNS = [
    "timestamp",
    "condition",
    "thera_steps",
    "algorithm",
    "lock_retention_percent",
    "loss_events",
    "reacquisition_attempts",
    "successful_reacquisitions",
    "reacquisition_success_percent",
    "mean_reacquisition_latency_frames",
    "mean_sam_confidence",
    "mean_normalized_center_jump",
    "tracking_fps",
]


def append_comparison(metrics: Dict, out_dir, condition: str, thera_steps: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model3_comparison.csv"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": condition,
        "thera_steps": int(thera_steps),
        **{k: metrics.get(k, "") for k in COLUMNS if k not in {"timestamp", "condition", "thera_steps"}},
    }

    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    latest_json = out_dir / f"{metrics.get('algorithm_key','run')}_{row['timestamp'].replace(':','-')}.json"
    latest_json.write_text(json.dumps({**row, "evaluation_type": metrics.get("evaluation_type")}, indent=2))
    return csv_path


def rank_runs(csv_path) -> List[Dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in [
                "lock_retention_percent",
                "reacquisition_success_percent",
                "mean_reacquisition_latency_frames",
                "mean_sam_confidence",
                "mean_normalized_center_jump",
                "tracking_fps",
            ]:
                try:
                    row[key] = float(row[key])
                except Exception:
                    row[key] = 0.0

            # Proxy comparison score: continuity and successful recovery are rewarded;
            # latency and erratic center jumps are penalized.
            row["comparison_score"] = round(
                0.45 * row["lock_retention_percent"]
                + 0.30 * row["reacquisition_success_percent"]
                + 15.0 * row["mean_sam_confidence"]
                - 1.5 * row["mean_reacquisition_latency_frames"]
                - 100.0 * row["mean_normalized_center_jump"],
                3,
            )
            rows.append(row)

    return sorted(rows, key=lambda r: r["comparison_score"], reverse=True)
