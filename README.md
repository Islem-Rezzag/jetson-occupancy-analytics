# Smart Occupancy Analytics on Jetson Nano

Mini project deliverable for real-time people counting, occupancy analytics, and demo-ready reporting.

## Features

- Person detection (`jetson-inference detectNet` with HOG fallback)
- Track IDs + line crossing counts (`IN` / `OUT`)
- Zone dwell analytics + heatmap
- Capacity monitoring with warning/exceeded alerts
- Optional privacy mode (blur inside person boxes)
- Session-based demo pack outputs (video, logs, reports, run metadata)
- Optional local LAN dashboard (`http://JETSON_IP:8000`)

## Base Run

From project root (`occupancy_analytics/`):

```bash
python3 src/main.py --config config/config.yaml --mode camera --input-uri /dev/video0
```

Video mode:

```bash
python3 src/main.py --config config/config.yaml --mode video --input-uri data/inputs/test.mp4
```

No GUI mode:

```bash
python3 src/main.py --config config/config.yaml --mode camera --no-display
```

## Demo Pack Output

When `demo_pack.enable: true`, each run creates:

```text
data/sessions/YYYYMMDD_HHMMSS/
  raw_config.yaml
  run_info.json
  outputs/
    annotated.mp4 (or annotated.avi fallback)
    snapshots/
      snapshot_*.jpg
      capacity_alert_*.jpg
      manual_*.jpg
  logs/
    events.csv
    summary.json
  reports/
    occupancy_plot.png
    heatmap.png
```

Optional session zip archive is created when `demo_pack.zip_on_exit: true`.

## Dashboard (Optional)

Enable in `config/config.yaml`:

```yaml
dashboard:
  enable: true
  host: "0.0.0.0"
  port: 8000
  stream_fps: 10
```

Run the app, then open on the same LAN:

```text
http://JETSON_IP:8000
```

Dashboard endpoints:

- `GET /` live page (MJPEG stream + stats + download links)
- `GET /stream.mjpeg` live annotated stream
- `GET /stats.json` live JSON stats
- `GET /download?path=<relative_path>` safe artifact download (session-root only)

## Report Generation

Generate plots from any events CSV:

```bash
python3 scripts/make_report.py <events_csv> <occupancy_png> <heatmap_png>
```

Example:

```bash
python3 scripts/make_report.py logs/events.csv reports/occupancy_plot.png reports/heatmap.png
```

## Key Config Sections

- `demo_pack`: session folders, video recording, auto-report, zip on exit
- `capacity`: warning and max occupancy thresholds with cooldown
- `privacy`: blur person boxes
- `hud`: occupancy sparkline history
- `dashboard`: local web dashboard settings

## Notes

- First detectNet run on Jetson may be slower due to TensorRT engine build/cache.
- If GUI is unavailable (`DISPLAY` unset), app continues in headless mode.
- Press `s` while display window is active to save a manual snapshot.
