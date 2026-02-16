import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from analytics import AnalyticsEngine
from dashboard_server import DashboardServer
from detector import PersonDetector
from logger import AnalyticsLogger
from time_utils import FPSMeter, now
from tracker import IoUTracker
from visualizer import Visualizer


class FramePacket:
    def __init__(self, frame_raw: np.ndarray, frame_bgr: np.ndarray, cuda_img: Optional[object]) -> None:
        self.frame_raw = frame_raw
        self.frame_bgr = frame_bgr
        self.cuda_img = cuda_img


class FrameSource:
    def __init__(self, uri: str, width: int, height: int, fps: int) -> None:
        self.uri = uri
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.backend = "opencv"
        self._cap = None
        self._jetson_utils = None
        self._jetson_source = None

        self._open_source()

    def _open_source(self) -> None:
        if self._try_open_jetson():
            return
        self._open_opencv()

    def _try_open_jetson(self) -> bool:
        try:
            import jetson_utils  # type: ignore

            self._jetson_utils = jetson_utils
            self._jetson_source = jetson_utils.videoSource(self.uri)
            self.backend = "jetson"
            return True
        except Exception:
            self._jetson_utils = None
            self._jetson_source = None
            return False

    def _open_opencv(self) -> None:
        source = self.uri  # type: object
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Could not open input URI: %s" % self.uri)

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._cap = cap
        self.backend = "opencv"

    def read(self) -> Optional[FramePacket]:
        if self.backend == "jetson" and self._jetson_source is not None:
            try:
                cuda_img = self._jetson_source.Capture()
            except Exception:
                return None

            if cuda_img is None:
                return None

            frame_raw = self._jetson_utils.cudaToNumpy(cuda_img)
            if frame_raw is None:
                return None

            if frame_raw.ndim == 3 and frame_raw.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGBA2BGR)
            elif frame_raw.ndim == 3 and frame_raw.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_raw.copy()

            return FramePacket(frame_raw=frame_raw, frame_bgr=frame_bgr, cuda_img=cuda_img)

        ok, frame_bgr = self._cap.read()
        if not ok:
            return None

        return FramePacket(frame_raw=frame_bgr, frame_bgr=frame_bgr, cuda_img=None)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Occupancy Analytics")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", default="camera", choices=["camera", "video"], help="Input mode")
    parser.add_argument("--input-uri", default=None, help="Override input URI from config")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def resolve_uri(project_root: Path, uri: str) -> str:
    if uri.startswith(("csi://", "v4l2://", "rtsp://", "webrtc://")):
        return uri
    if uri.startswith("/"):
        return uri
    if uri.isdigit():
        return uri

    p = Path(uri)
    if p.suffix:
        return str(resolve_path(project_root, uri))
    return uri


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_unique_session_dir(sessions_root: Path) -> Path:
    sessions_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = sessions_root / base
    idx = 1
    while candidate.exists():
        candidate = sessions_root / ("%s_%02d" % (base, idx))
        idx += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _is_subpath(target: Path, base: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def resolve_session_path(session_dir: Path, value: str, default_rel: str) -> Path:
    rel = str(value or default_rel).replace("\\", "/").lstrip("/")
    candidate = (session_dir / rel).resolve()
    if not _is_subpath(candidate, session_dir):
        candidate = (session_dir / default_rel).resolve()
    return candidate


def create_video_writer(path: Path, width: int, height: int, fps: float) -> Tuple[Optional[cv2.VideoWriter], Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps_safe = max(1.0, float(fps))
    size = (int(width), int(height))

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps_safe, size)
    if writer.isOpened():
        return writer, path, "mp4v"
    writer.release()

    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps_safe, size)
    if writer.isOpened():
        return writer, fallback, "MJPG"
    writer.release()
    return None, path, ""


def build_dashboard_stats(
    snapshot: Dict[str, object],
    frame_count: int,
    input_uri: str,
    detector_backend: str,
    session_dir: Optional[Path],
) -> Dict[str, object]:
    stats = {}  # type: Dict[str, object]
    src_stats = snapshot.get("stats", {})
    if isinstance(src_stats, dict):
        stats.update(src_stats)

    stats["occupancy"] = int(snapshot.get("occupancy_now", stats.get("occupancy", 0)))
    stats["total_in"] = int(snapshot.get("total_in", stats.get("in_count", 0)))
    stats["total_out"] = int(snapshot.get("total_out", stats.get("out_count", 0)))
    stats["fps"] = float(snapshot.get("fps", stats.get("fps", 0.0)))
    stats["capacity_level"] = str(snapshot.get("capacity_level", stats.get("capacity_level", "none")))
    stats["per_zone_dwell"] = snapshot.get("per_zone_dwell_seconds", {})
    stats["frames_processed"] = int(frame_count)
    stats["input_uri"] = input_uri
    stats["detector_backend"] = detector_backend
    stats["session_dir"] = str(session_dir) if session_dir is not None else ""
    return stats


def try_generate_reports(events_csv: Path, occupancy_plot: Path, heatmap_plot: Path) -> None:
    try:
        from reporting import make_report

        make_report(events_csv, occupancy_plot, heatmap_plot)
    except Exception as exc:
        print("Auto report generation failed: %s" % exc)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)

    input_cfg = config.get("input", {})
    detector_cfg = config.get("detector", {})
    tracker_cfg = config.get("tracker", {})
    line_cfg = config.get("line", {})
    zones_cfg = config.get("zones", [])
    heatmap_cfg = config.get("heatmap", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})
    demo_cfg = config.get("demo_pack", {})
    capacity_cfg = config.get("capacity", {})
    privacy_cfg = config.get("privacy", {})
    hud_cfg = config.get("hud", {})
    dashboard_cfg = config.get("dashboard", {})

    run_start_utc = utc_now_iso()
    t_start = now()

    demo_enabled = bool(demo_cfg.get("enable", True))
    session_dir = None  # type: Optional[Path]
    if demo_enabled:
        sessions_root = resolve_path(project_root, str(demo_cfg.get("sessions_root", "data/sessions")))
        session_dir = create_unique_session_dir(sessions_root)
        shutil.copyfile(str(config_path), str(session_dir / "raw_config.yaml"))

    input_uri = str(args.input_uri or input_cfg.get("uri", "/dev/video0"))
    input_uri = resolve_uri(project_root, input_uri)

    outputs_dir = resolve_path(project_root, "data/outputs")
    snapshot_dir = outputs_dir
    csv_path = resolve_path(project_root, str(logging_cfg.get("csv_path", "logs/events.csv")))
    summary_path = resolve_path(project_root, str(logging_cfg.get("summary_path", "logs/summary.json")))
    reports_dir = resolve_path(project_root, "reports")
    report_heatmap_path = reports_dir / "heatmap.png"
    report_occupancy_path = reports_dir / "occupancy_plot.png"

    if session_dir is not None:
        outputs_dir = session_dir / "outputs"
        snapshot_dir = resolve_session_path(
            session_dir, str(demo_cfg.get("snapshot_dir", "outputs/snapshots")), "outputs/snapshots"
        )
        csv_path = session_dir / "logs" / "events.csv"
        summary_path = session_dir / "logs" / "summary.json"
        reports_dir = session_dir / "reports"
        report_heatmap_path = reports_dir / "heatmap.png"
        report_occupancy_path = reports_dir / "occupancy_plot.png"

    source = None  # type: Optional[FrameSource]
    logger = None  # type: Optional[AnalyticsLogger]
    dashboard = None  # type: Optional[DashboardServer]
    detector = None  # type: Optional[PersonDetector]
    analytics = None  # type: Optional[AnalyticsEngine]
    video_writer = None  # type: Optional[cv2.VideoWriter]
    video_output_path = None  # type: Optional[Path]
    video_codec = ""
    frame_w = 0
    frame_h = 0
    frame_count = 0
    avg_fps = 0.0
    status = "ok"
    current_snapshot = {}  # type: Dict[str, object]

    fps_meter = FPSMeter(window_size=30)
    last_snapshot_save = -1e9
    last_heatmap_save = -1e9
    last_dashboard_push = -1e9

    save_snapshots = bool(output_cfg.get("save_snapshots", True))
    snapshot_every_seconds = float(output_cfg.get("snapshot_every_seconds", 3.0))
    heatmap_save_every_seconds = float(heatmap_cfg.get("save_every_seconds", 2.0))
    dashboard_stream_fps = max(1.0, float(dashboard_cfg.get("stream_fps", 10.0)))
    dashboard_push_interval = 1.0 / dashboard_stream_fps
    record_video = bool(demo_cfg.get("record_video", True)) and (session_dir is not None)
    auto_report = bool(demo_cfg.get("auto_report", True)) and (session_dir is not None)
    zip_on_exit = bool(demo_cfg.get("zip_on_exit", False)) and (session_dir is not None)
    run_duration_seconds = 0.0
    t_prev = t_start

    display_requested = bool(output_cfg.get("display", True)) and not args.no_display
    display = display_requested and bool(os.environ.get("DISPLAY"))
    if display_requested and not display:
        print("Display disabled: DISPLAY is not set. Running in headless mode.")

    window_name = "Smart Occupancy Analytics"
    pending_packet = None  # type: Optional[FramePacket]

    try:
        try:
            source = FrameSource(
                uri=input_uri,
                width=int(input_cfg.get("width", 640)),
                height=int(input_cfg.get("height", 480)),
                fps=int(input_cfg.get("fps", 30)),
            )
        except Exception as exc:
            status = "input_error"
            print("Input source error: %s" % exc)
            return

        first_packet = source.read()
        if first_packet is None:
            status = "input_error"
            print("Could not capture first frame from input source: %s" % input_uri)
            return

        frame_h, frame_w = first_packet.frame_bgr.shape[:2]

        detector = PersonDetector(
            network=str(detector_cfg.get("network", "ssd-mobilenet-v2")),
            threshold=float(detector_cfg.get("threshold", 0.5)),
        )

        tracker = IoUTracker(
            iou_threshold=float(tracker_cfg.get("iou_match_threshold", 0.30)),
            max_missed_frames=int(tracker_cfg.get("max_missed_frames", 10)),
            history_length=int(tracker_cfg.get("history_length", 30)),
        )

        analytics = AnalyticsEngine(
            frame_w=frame_w,
            frame_h=frame_h,
            line_cfg=line_cfg,
            zones_cfg=zones_cfg,
            heatmap_cfg=heatmap_cfg,
            capacity_cfg=capacity_cfg,
            hud_cfg=hud_cfg,
        )

        visualizer = Visualizer(heatmap_alpha=0.28, draw_history=True, privacy_cfg=privacy_cfg, hud_cfg=hud_cfg)

        logger = AnalyticsLogger(
            csv_path=csv_path,
            summary_path=summary_path,
            write_every_seconds=float(logging_cfg.get("write_every_seconds", 0.5)),
            zone_names=[str(zone["name"]) for zone in analytics.zones_px],
            output_dir=session_dir,
        )

        if record_video and session_dir is not None:
            video_rel = str(demo_cfg.get("video_filename", "outputs/annotated.mp4"))
            video_path = resolve_session_path(session_dir, video_rel, "outputs/annotated.mp4")
            requested_fps = float(demo_cfg.get("video_fps", 20))
            writer, path_used, codec = create_video_writer(video_path, frame_w, frame_h, requested_fps)
            if writer is None:
                print("Video recording disabled: could not create VideoWriter at %s" % video_path)
            else:
                video_writer = writer
                video_output_path = path_used
                video_codec = codec
                print("Recording annotated video to %s (%s)" % (video_output_path, video_codec))

        dashboard_enabled = bool(dashboard_cfg.get("enable", False))
        if dashboard_enabled:
            artifact_root = session_dir if session_dir is not None else project_root
            dashboard = DashboardServer(
                host=str(dashboard_cfg.get("host", "0.0.0.0")),
                port=int(dashboard_cfg.get("port", 8000)),
                stream_fps=dashboard_stream_fps,
                artifact_root=artifact_root,
            )
            if not dashboard.start():
                dashboard = None

        current_snapshot = analytics.final_summary()
        pending_packet = first_packet

        while True:
            t_now = now()
            dt = max(0.0, t_now - t_prev)

            packet = pending_packet if pending_packet is not None else source.read()
            pending_packet = None
            if packet is None:
                break

            detections = detector.detect(packet.cuda_img, frame_bgr=packet.frame_bgr)
            tracks = tracker.update(detections, t_now)
            current_snapshot = analytics.update(tracks, dt, t_now)

            current_snapshot["fps"] = fps_meter.update(t_now)
            stats = current_snapshot.get("stats", {})
            if isinstance(stats, dict):
                stats["fps"] = float(current_snapshot["fps"])

            current_snapshot["heatmap_overlay_bgr"] = analytics.render_heatmap_image(frame_w, frame_h)
            current_snapshot["heatmap_preview_bgr"] = analytics.render_heatmap_image(220, 140)

            annotated = visualizer.draw(packet.frame_raw, tracks, current_snapshot)

            if video_writer is not None:
                video_writer.write(annotated)

            if display:
                try:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        manual_path = snapshot_dir / ("manual_%06d.jpg" % frame_count)
                        save_image(manual_path, annotated)
                except cv2.error:
                    print("Display disabled: OpenCV GUI backend is unavailable. Continuing headless.")
                    display = False

            if logger is not None:
                logger.log_snapshot(t_now, current_snapshot)

                cap_event = current_snapshot.get("capacity_event")
                if isinstance(cap_event, dict):
                    logger.log_event(cap_event, current_snapshot)
                    if bool(current_snapshot.get("capacity_should_snapshot", False)):
                        alert_path = snapshot_dir / ("capacity_alert_%06d.jpg" % frame_count)
                        save_image(alert_path, annotated)

            if save_snapshots and (t_now - last_snapshot_save) >= snapshot_every_seconds:
                snapshot_path = snapshot_dir / ("snapshot_%06d.jpg" % frame_count)
                save_image(snapshot_path, annotated)
                last_snapshot_save = t_now

            if (t_now - last_heatmap_save) >= heatmap_save_every_seconds:
                heatmap_img = analytics.render_heatmap_image(frame_w, frame_h)
                save_image(report_heatmap_path, heatmap_img)
                heatmap_series_path = outputs_dir / ("heatmap_%06d.png" % frame_count)
                save_image(heatmap_series_path, heatmap_img)
                last_heatmap_save = t_now

            if dashboard is not None:
                dashboard_stats = build_dashboard_stats(
                    snapshot=current_snapshot,
                    frame_count=frame_count,
                    input_uri=input_uri,
                    detector_backend=detector.backend,
                    session_dir=session_dir,
                )
                dashboard.update_stats(dashboard_stats)
                if (t_now - last_dashboard_push) >= dashboard_push_interval:
                    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        dashboard.update_frame(encoded.tobytes())
                        last_dashboard_push = t_now

            frame_count += 1
            t_prev = t_now

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted by user, shutting down.")
    except Exception as exc:
        status = "runtime_error"
        print("Runtime error: %s" % exc)
    finally:
        run_duration_seconds = max(0.0, now() - t_start)
        avg_fps = float(frame_count) / run_duration_seconds if run_duration_seconds > 0 else 0.0

        if source is not None:
            source.release()

        if display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass

        if dashboard is not None:
            dashboard.stop()

        if analytics is not None and frame_w > 0 and frame_h > 0:
            final_heatmap = analytics.render_heatmap_image(frame_w, frame_h)
            save_image(report_heatmap_path, final_heatmap)
            if not current_snapshot:
                current_snapshot = analytics.final_summary()

        if logger is not None:
            detector_backend = detector.backend if detector is not None else "none"
            logger.finalize(
                current_snapshot,
                extra={
                    "project": str(config.get("project", {}).get("name", "smart_occupancy_analytics")),
                    "frames_processed": frame_count,
                    "input_uri": input_uri,
                    "mode": args.mode,
                    "detector_backend": detector_backend,
                    "session_duration_seconds": round(run_duration_seconds, 3),
                    "avg_fps": round(avg_fps, 3),
                    "status": status,
                },
            )

        if auto_report and csv_path.exists():
            try_generate_reports(csv_path, report_occupancy_path, report_heatmap_path)

        if session_dir is not None:
            detector_backend = detector.backend if detector is not None else "none"
            run_info = {
                "timestamp_start_utc": run_start_utc,
                "timestamp_end_utc": utc_now_iso(),
                "hostname": socket.gethostname(),
                "input_uri": input_uri,
                "mode": args.mode,
                "detector_network": str(detector_cfg.get("network", "ssd-mobilenet-v2")),
                "detector_backend": detector_backend,
                "source_backend": source.backend if source is not None else "unknown",
                "frames_processed": int(frame_count),
                "avg_fps": round(avg_fps, 3),
                "session_duration_seconds": round(run_duration_seconds, 3),
                "frame_size": {"width": int(frame_w), "height": int(frame_h)},
                "status": status,
                "artifacts": {
                    "events_csv": "logs/events.csv",
                    "summary_json": "logs/summary.json",
                    "occupancy_plot": "reports/occupancy_plot.png",
                    "heatmap_plot": "reports/heatmap.png",
                    "annotated_video": str(video_output_path.relative_to(session_dir)) if video_output_path else "",
                },
            }
            with (session_dir / "run_info.json").open("w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)

            if zip_on_exit:
                zip_path = shutil.make_archive(str(session_dir), "zip", root_dir=str(session_dir.parent), base_dir=session_dir.name)
                print("Session archive created: %s" % zip_path)


if __name__ == "__main__":
    main()
