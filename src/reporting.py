import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_events(csv_path: Path) -> List[Dict[str, str]]:
    rows = []  # type: List[Dict[str, str]]
    if not csv_path.exists():
        return rows

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _snapshot_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []  # type: List[Dict[str, str]]
    for row in rows:
        row_type = str(row.get("row_type", "snapshot")).strip().lower()
        if row_type in {"", "snapshot"}:
            out.append(dict(row))
    return out


def make_occupancy_plot(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_snap = _snapshot_rows(rows)

    occupancy = []  # type: List[float]
    for row in rows_snap:
        try:
            occupancy.append(float(row.get("occupancy", 0.0)))
        except Exception:
            occupancy.append(0.0)

    x = np.arange(len(occupancy), dtype=np.int32)

    plt.figure(figsize=(10, 4))
    if len(occupancy) > 0:
        plt.plot(x, occupancy, color="#166534", linewidth=2)
        plt.title("Occupancy Over Time")
        plt.xlabel("Sample")
        plt.ylabel("People")
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "No occupancy rows available", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_heatmap(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_snap = _snapshot_rows(rows)

    zone_cols = []
    if rows_snap:
        zone_cols = [k for k in rows_snap[0].keys() if k.startswith("zone_dwell_")]

    plt.figure(figsize=(9, 5))

    if not rows_snap or not zone_cols:
        plt.text(0.5, 0.5, "No zone dwell columns available", ha="center", va="center")
        plt.axis("off")
    else:
        matrix = np.zeros((len(zone_cols), len(rows_snap)), dtype=np.float32)
        for c, col in enumerate(zone_cols):
            prev = 0.0
            for r, row in enumerate(rows_snap):
                try:
                    val = float(row.get(col, 0.0))
                except Exception:
                    val = prev
                matrix[c, r] = max(0.0, val - prev)
                prev = max(prev, val)

        plt.imshow(matrix, aspect="auto", cmap="hot", origin="lower")
        plt.colorbar(label="Dwell Increment (s)")
        plt.title("Zone Activity Heatmap")
        plt.xlabel("Sample")
        plt.ylabel("Zone")
        plt.yticks(range(len(zone_cols)), [col.replace("zone_dwell_", "") for col in zone_cols])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_report(events_csv_path: Path, occupancy_plot_path: Path, heatmap_path: Path) -> None:
    rows = read_events(Path(events_csv_path))
    make_occupancy_plot(rows, Path(occupancy_plot_path))
    make_heatmap(rows, Path(heatmap_path))
