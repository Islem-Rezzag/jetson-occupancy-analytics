# Jetson Occupancy Analytics (Smart Occupancy Analytics)

Edge AI mini project on **NVIDIA Jetson Nano** that performs real-time people detection and occupancy analytics from a USB camera.

It uses:
- **NVIDIA jetson-inference detectNet** with **SSD-Mobilenet-v2 (COCO)** for person detection (Jetson GPU accelerated)
- A lightweight **IoU-based multi-object tracker** to keep stable person IDs over time
- Analytics on top of tracks:
  - occupancy now and peak occupancy
  - IN / OUT counting across a virtual line
  - dwell time inside user-defined zones
  - movement heatmap
- Outputs:
  - live annotated visualization
  - snapshots and heatmap images
  - CSV logs + JSON summary
  - report plots (occupancy over time + zone activity heatmap)

---

## Features

### Live (on-screen)
- Person bounding boxes + **ID per person**
- Occupancy (people currently in view)
- IN / OUT counters (line crossing)
- Zones (door, waiting, etc)
- Heatmap overlay + heatmap preview inset

### Saved outputs
- `data/outputs/snapshot_*.jpg` (annotated snapshots)
- `data/outputs/heatmap_*.png` (heatmap frames)
- `logs/events.csv` (time-series analytics log)
- `logs/summary.json` (final summary)
- `reports/occupancy_plot.png` (after running report script)
- `reports/heatmap.png` (note: see "Known notes" below)

---

## Project structure

```text
occupancy_analytics/
  config/
    config.yaml
  src/
    main.py          # frame capture loop + orchestration
    detector.py      # detectNet backend (Jetson) + HOG fallback (non-Jetson)
    tracker.py       # IoU tracker (assigns stable IDs)
    analytics.py     # occupancy + line crossing + zone dwell + movement heatmap
    visualizer.py    # draws overlays (boxes, IDs, zones, HUD, heatmap inset)
    logger.py        # writes events.csv and summary.json
    geometry.py      # bbox math (IoU, centers, point in rect)
    time_utils.py    # FPS and timestamps
  scripts/
    run_camera_demo.sh
    run_video_demo.sh
    make_report.sh
    make_report.py
  data/
    outputs/         # generated snapshots/heatmaps
  logs/              # generated logs
  reports/           # generated plots/images
```

---

## Hardware requirements
- Jetson Nano Developer Kit (4GB)
- microSD card (32GB minimum recommended)
- USB camera (UVC compatible)
- HDMI monitor/TV + cable
- USB keyboard + mouse
- Ethernet cable to router (recommended)
- Stable power:
  - 5V 4A barrel power supply
  - J48 jumper cap installed (for barrel jack power)

---

## Jetson OS setup (microSD)
1. Download the official Jetson Nano SD image (JetPack 4.6.x recommended).
2. Flash using balenaEtcher to microSD.
3. Boot Jetson and complete Ubuntu first-time setup (creates your username/password).

---

## Install dependencies on Jetson

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y git cmake build-essential
sudo apt install -y python3-pip python3-dev libpython3-dev
sudo apt install -y python3-numpy python3-yaml python3-matplotlib python3-opencv
sudo apt install -y v4l-utils dos2unix
```

### Install jetson-inference (detectNet)

```bash
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir -p build
cd build
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

### If build fails with: `cannot find -lnpymath`

Create a linker-visible symlink (Jetson Nano / Ubuntu 18.04 fix):

```bash
sudo ln -sf /usr/lib/python3/dist-packages/numpy/core/lib/libnpymath.a /usr/lib/aarch64-linux-gnu/libnpymath.a
```

Then rebuild from the build directory:

```bash
cd ~/jetson-inference/build
make -j$(nproc)
sudo make install
sudo ldconfig
```

Verify:

```bash
python3 -c "import jetson_inference, jetson_utils; print('jetson-inference OK')"
```

---

## Verify the camera is detected

Plug the USB camera into the Jetson, then run:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

Typical camera device: `/dev/video0`.

---

## Quick pretrained model test (detectNet demo)

```bash
cd ~/jetson-inference/python/examples
python3 detectnet.py /dev/video0
```

Press `q` to exit.

---

## Run this project on Jetson

### Camera mode

```bash
cd ~/occupancy_analytics
bash scripts/run_camera_demo.sh
```

Or directly:

```bash
python3 src/main.py --config config/config.yaml --mode camera --input-uri /dev/video0
```

Press `q` or `Esc` to exit the window.

### Video file mode

```bash
bash scripts/run_video_demo.sh data/inputs/test.mp4
```

---

## Generate report plots

After a run (so logs exist), generate plots:

```bash
bash scripts/make_report.sh
```

This creates:
- `reports/occupancy_plot.png`
- `reports/heatmap.png`

---

## Configuration (`config/config.yaml`)

Key settings:
- `input.uri`: `/dev/video0` or a video file
- `detector.network`: `ssd-mobilenet-v2`
- `detector.threshold`: detection threshold
- `line.orientation`: `auto`, `horizontal`, or `vertical`
- `line.position_ratio`: where the counting line is (`0.0` to `1.0`)
- `zones`: named rectangles (normalized coordinates `0.0` to `1.0`)
- `output.snapshot_every_seconds`: snapshot frequency
- `logging.write_every_seconds`: CSV sampling interval

---

## How IN / OUT counting works

- A counting line is placed on the frame.
- When a tracked person's center crosses the line, it counts as `IN` or `OUT` based on direction.
- There is a cooldown to reduce double counting.
- The line orientation can be auto-calibrated based on early motion, or set explicitly.

---

## Known notes

- `reports/heatmap.png` is used by two different outputs:
  - the live movement heatmap saved during runtime
  - the zone activity heatmap generated by `make_report.sh`
- Running `make_report.sh` will overwrite the runtime movement heatmap image.
- Recommendation: rename one of them (see "Recommended improvements").

- On Jetson Nano, VS Code Remote-SSH can fail due to OS runtime prerequisites.
- Recommended workflows:
  - edit on PC and copy via `scp`, or
  - use GitHub and `git pull` on Jetson.

---

## Recommended improvements (next steps)

- Save an annotated MP4 of the live run (camera mode) using `cv2.VideoWriter`
- Add dwell-time values to the on-screen HUD (not only in logs)
- Add capacity alerts + snapshots (useful real-world feature)
- Add a local web dashboard (LAN) for viewing stats and stream in a browser

---

## Troubleshooting

### No display window
If you see: `Display disabled: DISPLAY is not set`

Run with TV/monitor connected, or run headless:

```bash
python3 src/main.py --config config/config.yaml --mode camera --input-uri /dev/video0 --no-display
```

### Camera busy
If another app is using the camera, stop it first (only one process can use `/dev/video0`).

### Close the window
Press `q` inside the window, or use terminal `Ctrl+C`.
