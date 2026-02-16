import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence


class AnalyticsLogger:
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        write_every_seconds: float = 0.5,
        zone_names: Sequence[str] = (),
        output_dir: Optional[Path] = None,
    ) -> None:
        if output_dir is not None:
            base = Path(output_dir)
            self.csv_path = base / "logs" / "events.csv"
            self.summary_path = base / "logs" / "summary.json"
        else:
            self.csv_path = Path(csv_path or Path("logs/events.csv"))
            self.summary_path = Path(summary_path or Path("logs/summary.json"))

        self.write_every_seconds = float(write_every_seconds)
        self.zone_names = [str(name) for name in zone_names]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "timestamp",
            "row_type",
            "event_type",
            "message",
            "occupancy",
            "occupancy_value",
            "total_in",
            "total_out",
            *["zone_dwell_%s" % name for name in self.zone_names],
            "line_orientation",
            "capacity_level",
        ]

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
        self._writer.writeheader()

        self._last_write_time = -1e9
        self._closed = False

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def log_snapshot(self, t_now: float, snapshot: Dict[str, object]) -> None:
        if (float(t_now) - self._last_write_time) < self.write_every_seconds:
            return
        self._write_snapshot_row(snapshot)
        self._last_write_time = float(t_now)

    def _write_snapshot_row(self, snapshot: Dict[str, object]) -> None:
        dwell = snapshot.get("per_zone_dwell_seconds", {})
        if not isinstance(dwell, dict):
            dwell = {}

        row = {
            "timestamp": self._utc_now_iso(),
            "row_type": "snapshot",
            "event_type": "",
            "message": "",
            "occupancy": int(snapshot.get("occupancy_now", 0)),
            "occupancy_value": int(snapshot.get("occupancy_now", 0)),
            "total_in": int(snapshot.get("total_in", 0)),
            "total_out": int(snapshot.get("total_out", 0)),
            "line_orientation": str(snapshot.get("line_orientation", "pending")),
            "capacity_level": str(snapshot.get("capacity_level", "none")),
        }  # type: Dict[str, object]

        for name in self.zone_names:
            row["zone_dwell_%s" % name] = round(float(dwell.get(name, 0.0)), 3)

        self._writer.writerow(row)
        self._csv_file.flush()

    def log_event(self, event: Dict[str, object], snapshot: Optional[Dict[str, object]] = None) -> None:
        if self._closed:
            return

        if snapshot is None:
            snapshot = {}

        dwell = snapshot.get("per_zone_dwell_seconds", {})
        if not isinstance(dwell, dict):
            dwell = {}

        row = {
            "timestamp": self._utc_now_iso(),
            "row_type": "event",
            "event_type": str(event.get("event_type", "event")),
            "message": str(event.get("message", "")),
            "occupancy": int(snapshot.get("occupancy_now", event.get("occupancy_value", 0))),
            "occupancy_value": int(event.get("occupancy_value", snapshot.get("occupancy_now", 0))),
            "total_in": int(snapshot.get("total_in", 0)),
            "total_out": int(snapshot.get("total_out", 0)),
            "line_orientation": str(snapshot.get("line_orientation", "pending")),
            "capacity_level": str(event.get("capacity_level", snapshot.get("capacity_level", "none"))),
        }  # type: Dict[str, object]

        for name in self.zone_names:
            row["zone_dwell_%s" % name] = round(float(dwell.get(name, 0.0)), 3)

        self._writer.writerow(row)
        self._csv_file.flush()

    def finalize(self, snapshot: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
        if self._closed:
            return

        summary = {
            "occupancy_now": int(snapshot.get("occupancy_now", 0)),
            "final_occupancy": int(snapshot.get("occupancy_now", snapshot.get("final_occupancy", 0))),
            "peak_occupancy": int(snapshot.get("peak_occupancy", 0)),
            "total_in": int(snapshot.get("total_in", 0)),
            "total_out": int(snapshot.get("total_out", 0)),
            "per_zone_dwell_seconds": snapshot.get("per_zone_dwell_seconds", {}),
            "line_orientation": snapshot.get("line_orientation", "pending"),
            "line_position_px": snapshot.get("line_position_px"),
            "capacity_level": snapshot.get("capacity_level", "none"),
            "capacity_alert_counts": snapshot.get("capacity_alert_counts", {}),
        }  # type: Dict[str, object]

        if extra:
            summary.update(extra)

        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._csv_file.close()
        self._closed = True
