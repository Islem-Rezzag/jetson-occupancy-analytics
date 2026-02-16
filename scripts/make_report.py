from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reporting import make_report as generate_report


def make_report(events_csv: Path, occupancy_png: Path, heatmap_png: Path) -> None:
    generate_report(Path(events_csv), Path(occupancy_png), Path(heatmap_png))


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python scripts/make_report.py <events_csv> <occupancy_png> <heatmap_png>")

    events_csv = Path(sys.argv[1])
    occupancy_png = Path(sys.argv[2])
    heatmap_png = Path(sys.argv[3])
    make_report(events_csv, occupancy_png, heatmap_png)


if __name__ == "__main__":
    main()
