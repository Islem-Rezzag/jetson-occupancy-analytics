from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from tracker import Track


class Visualizer:
    def __init__(
        self,
        heatmap_alpha: float = 0.30,
        draw_history: bool = True,
        privacy_cfg: Optional[Dict[str, object]] = None,
        hud_cfg: Optional[Dict[str, object]] = None,
    ) -> None:
        self.heatmap_alpha = float(heatmap_alpha)
        self.draw_history = bool(draw_history)

        privacy_cfg = privacy_cfg or {}
        self.privacy_enable = bool(privacy_cfg.get("enable", False))
        self.privacy_method = str(privacy_cfg.get("method", "blur_boxes")).strip().lower()
        self.blur_kernel = max(3, int(privacy_cfg.get("blur_kernel", 31)))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        hud_cfg = hud_cfg or {}
        self.sparkline_enable = bool(hud_cfg.get("sparkline", True))

    def draw(self, frame: np.ndarray, tracks: Iterable[Track], snapshot: dict) -> np.ndarray:
        canvas = self._ensure_bgr(frame)
        track_list = list(tracks)

        if self.privacy_enable and self.privacy_method == "blur_boxes":
            self._apply_privacy_blur(canvas, track_list)

        heatmap_overlay = snapshot.get("heatmap_overlay_bgr")
        if isinstance(heatmap_overlay, np.ndarray) and heatmap_overlay.size > 0:
            if heatmap_overlay.shape[:2] != canvas.shape[:2]:
                heatmap_overlay = cv2.resize(
                    heatmap_overlay, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            canvas = cv2.addWeighted(canvas, 1.0, heatmap_overlay, self.heatmap_alpha, 0.0)

        self._draw_capacity_banner(canvas, snapshot)
        self._draw_zones(canvas, snapshot)
        self._draw_counting_line(canvas, snapshot)
        self._draw_tracks(canvas, track_list)
        self._draw_hud(canvas, snapshot)

        heatmap_preview = snapshot.get("heatmap_preview_bgr")
        if isinstance(heatmap_preview, np.ndarray) and heatmap_preview.size > 0:
            self._draw_preview(canvas, heatmap_preview)

        return canvas

    def _ensure_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame.copy()

    def _apply_privacy_blur(self, image: np.ndarray, tracks: List[Track]) -> None:
        h, w = image.shape[:2]
        for track in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_px]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]
            if roi_h < 3 or roi_w < 3:
                continue

            k = min(self.blur_kernel, roi_w if roi_w % 2 == 1 else roi_w - 1, roi_h if roi_h % 2 == 1 else roi_h - 1)
            if k < 3:
                continue
            if k % 2 == 0:
                k -= 1
            if k < 3:
                continue

            image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    def _draw_capacity_banner(self, image: np.ndarray, snapshot: dict) -> None:
        level = str(snapshot.get("capacity_level", "none"))
        if level == "exceeded":
            text = "Capacity exceeded"
            bg = (0, 0, 200)
            fg = (255, 255, 255)
        elif level == "warn":
            text = "Approaching capacity"
            bg = (0, 200, 255)
            fg = (20, 20, 20)
        else:
            return

        h, w = image.shape[:2]
        margin = 12
        bar_h = 44
        cv2.rectangle(image, (margin, margin), (w - margin, margin + bar_h), bg, -1)
        cv2.rectangle(image, (margin, margin), (w - margin, margin + bar_h), (255, 255, 255), 1)
        cv2.putText(
            image,
            text,
            (margin + 14, margin + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            fg,
            2,
            cv2.LINE_AA,
        )

    def _draw_counting_line(self, image: np.ndarray, snapshot: dict) -> None:
        orientation = str(snapshot.get("line_orientation", "pending"))
        position_px = snapshot.get("line_position_px")

        if orientation not in {"horizontal", "vertical"} or position_px is None:
            cv2.putText(
                image,
                "Line: calibrating",
                (12, image.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            return

        color = (0, 220, 255)
        if orientation == "horizontal":
            y = int(round(float(position_px)))
            cv2.line(image, (0, y), (image.shape[1], y), color, 2)
        else:
            x = int(round(float(position_px)))
            cv2.line(image, (x, 0), (x, image.shape[0]), color, 2)

        cv2.putText(
            image,
            "Line: %s" % orientation,
            (12, image.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_zones(self, image: np.ndarray, snapshot: dict) -> None:
        zones = snapshot.get("zones_px", [])
        for zone in zones:
            name = str(zone.get("name", "zone"))
            rect = zone.get("rect_px", (0, 0, 0, 0))
            x1, y1, x2, y2 = [int(round(v)) for v in rect]
            cv2.rectangle(image, (x1, y1), (x2, y2), (70, 200, 90), 2)
            cv2.putText(
                image,
                name,
                (x1 + 4, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (70, 200, 90),
                2,
                cv2.LINE_AA,
            )

    def _draw_tracks(self, image: np.ndarray, tracks: Iterable[Track]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_px]
            cx, cy = [int(round(v)) for v in track.center_px]

            cv2.rectangle(image, (x1, y1), (x2, y2), (30, 130, 255), 2)
            cv2.circle(image, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(
                image,
                "ID: %d" % track.track_id,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (30, 130, 255),
                2,
                cv2.LINE_AA,
            )

            if self.draw_history and len(track.history_centers) > 1:
                points = np.array(track.history_centers, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], isClosed=False, color=(255, 190, 0), thickness=1)

    def _draw_hud(self, image: np.ndarray, snapshot: dict) -> None:
        occupancy_now = int(snapshot.get("occupancy_now", 0))
        total_in = int(snapshot.get("total_in", 0))
        total_out = int(snapshot.get("total_out", 0))
        fps = float(snapshot.get("fps", 0.0))

        panel_w = 360
        panel_h = 195
        cv2.rectangle(image, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 1)

        lines = [
            "Occupancy: %d" % occupancy_now,
            "IN: %d   OUT: %d" % (total_in, total_out),
            "FPS: %.1f" % fps,
        ]

        y = 34
        for line in lines:
            cv2.putText(
                image,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

        top_zones = snapshot.get("zone_dwell_top", [])
        if not top_zones:
            top_zones = self._zones_from_map(snapshot.get("per_zone_dwell_seconds", {}), 2)
        for idx, pair in enumerate(top_zones[:2]):
            zone_name = str(pair[0])
            dwell_sec = float(pair[1])
            cv2.putText(
                image,
                "%s: %.1fs" % (zone_name.title(), dwell_sec),
                (20, y + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (170, 255, 170),
                1,
                cv2.LINE_AA,
            )

        if self.sparkline_enable:
            history = snapshot.get("occupancy_history", [])
            if isinstance(history, list) and len(history) >= 2:
                self._draw_sparkline(
                    image=image,
                    values=history,
                    x=20,
                    y=150,
                    width=330,
                    height=42,
                    color=(80, 220, 255),
                )
                cv2.putText(
                    image,
                    "Occupancy last 60s",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )

    def _zones_from_map(self, dwell: object, limit: int) -> List[Tuple[str, float]]:
        if not isinstance(dwell, dict):
            return []
        pairs = []  # type: List[Tuple[str, float]]
        for k, v in dwell.items():
            try:
                pairs.append((str(k), float(v)))
            except Exception:
                continue
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        return pairs[: max(0, int(limit))]

    def _draw_sparkline(
        self,
        image: np.ndarray,
        values: List[object],
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ) -> None:
        if width <= 2 or height <= 2:
            return

        numeric = []
        for v in values:
            try:
                numeric.append(float(v))
            except Exception:
                numeric.append(0.0)

        if len(numeric) < 2:
            return

        vmax = max(1.0, max(numeric))
        pts = []  # type: List[List[int]]
        n = len(numeric)
        for i, val in enumerate(numeric):
            px = x + int(round((float(i) / float(max(1, n - 1))) * float(width - 1)))
            norm = float(val) / vmax
            py = y + int(round((1.0 - norm) * float(height - 1)))
            pts.append([px, py])

        cv2.rectangle(image, (x, y), (x + width, y + height), (90, 90, 90), 1)
        points = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False, color=color, thickness=1)

    def _draw_preview(self, image: np.ndarray, preview_bgr: np.ndarray) -> None:
        target_w = 220
        scale = target_w / max(1, preview_bgr.shape[1])
        target_h = int(preview_bgr.shape[0] * scale)
        inset = cv2.resize(preview_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        x2 = image.shape[1] - 12
        x1 = x2 - target_w
        y1 = 12
        y2 = y1 + target_h

        if x1 < 0 or y2 > image.shape[0]:
            return

        cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 22), (20, 20, 20), -1)
        image[y1:y2, x1:x2] = inset
        cv2.putText(
            image,
            "Heatmap",
            (x1, y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
