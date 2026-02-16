from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from geometry import PointPx, normalize_rect, point_in_rect
from tracker import Track


class AnalyticsEngine:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        line_cfg: Dict[str, object],
        zones_cfg: Sequence[Dict[str, object]],
        heatmap_cfg: Dict[str, object],
        capacity_cfg: Optional[Dict[str, object]] = None,
        hud_cfg: Optional[Dict[str, object]] = None,
    ) -> None:
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)

        self.occupancy_now = 0
        self.peak_occupancy = 0
        self.total_in = 0
        self.total_out = 0

        self._line_mode = str(line_cfg.get("orientation", "auto")).strip().lower()
        self._position_ratio = float(line_cfg.get("position_ratio", 0.55))
        self._calibration_seconds = float(line_cfg.get("calibration_seconds", 2.5))
        self._min_track_age_for_count = int(line_cfg.get("min_track_age_for_count", 5))
        self._cooldown_seconds = float(line_cfg.get("cooldown_seconds", 0.75))

        if self._line_mode in {"horizontal", "vertical"}:
            self.line_orientation = self._line_mode
        else:
            self.line_orientation = None
        self.line_position_px = self._compute_line_position(self.line_orientation)

        self._calibration_start_time = None  # type: Optional[float]
        self._calibration_dx_sum = 0.0
        self._calibration_dy_sum = 0.0
        self._calibration_samples = 0

        self.zones_px = []  # type: List[Dict[str, object]]
        self.per_zone_dwell_seconds = {}  # type: Dict[str, float]
        for zone in zones_cfg:
            name = str(zone.get("name", "zone"))
            rect_norm = (
                float(zone.get("x1", 0.0)),
                float(zone.get("y1", 0.0)),
                float(zone.get("x2", 0.0)),
                float(zone.get("y2", 0.0)),
            )
            rect_px = normalize_rect(rect_norm, self.frame_w, self.frame_h)
            self.zones_px.append({"name": name, "rect_px": rect_px})
            self.per_zone_dwell_seconds[name] = 0.0

        self._per_track_state = {}  # type: Dict[int, Dict[str, object]]

        self.grid_w = int(heatmap_cfg.get("grid_w", 160))
        self.grid_h = int(heatmap_cfg.get("grid_h", 90))
        self.heatmap_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        if capacity_cfg is None:
            capacity_cfg = {}
        self.capacity_enabled = bool(capacity_cfg.get("enable", False))
        self.warn_occupancy = int(capacity_cfg.get("warn_occupancy", 6))
        self.max_occupancy = int(capacity_cfg.get("max_occupancy", 8))
        if self.max_occupancy < self.warn_occupancy:
            self.max_occupancy = self.warn_occupancy
        self.capacity_cooldown_sec = float(capacity_cfg.get("cooldown_sec", 10.0))
        self.capacity_level = "none"
        self.capacity_alert_counts = {"warn": 0, "exceeded": 0}  # type: Dict[str, int]
        self._capacity_last_event_time = {"warn": -1e9, "exceeded": -1e9}  # type: Dict[str, float]

        if hud_cfg is None:
            hud_cfg = {}
        self._sparkline_enabled = bool(hud_cfg.get("sparkline", True))
        self._history_seconds = max(5.0, float(hud_cfg.get("history_seconds", 60.0)))
        self._occupancy_history = []  # type: List[Tuple[float, int]]

    def update(self, tracks: Sequence[Track], dt: float, t_now: float) -> Dict[str, object]:
        if self._calibration_start_time is None:
            self._calibration_start_time = float(t_now)

        self.occupancy_now = len(tracks)
        self.peak_occupancy = max(self.peak_occupancy, self.occupancy_now)

        self._update_heatmap(tracks)
        self._update_zones(tracks, dt)
        self._update_occupancy_history(t_now)

        if self.line_orientation is None:
            self._update_auto_line_calibration(tracks, t_now)

        crossing_events = []
        if self.line_orientation in {"horizontal", "vertical"}:
            crossing_events = self._update_counting(tracks, t_now)

        capacity_event = self._update_capacity(t_now)

        active_ids = {track.track_id for track in tracks}
        stale_ids = [track_id for track_id in self._per_track_state.keys() if track_id not in active_ids]
        for track_id in stale_ids:
            del self._per_track_state[track_id]

        per_zone_copy = dict(self.per_zone_dwell_seconds)
        stats = {
            "occupancy": self.occupancy_now,
            "in_count": self.total_in,
            "out_count": self.total_out,
            "per_zone_dwell": per_zone_copy,
            "capacity_level": self.capacity_level,
            "fps": 0.0,
        }

        return {
            "occupancy_now": self.occupancy_now,
            "peak_occupancy": self.peak_occupancy,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "per_zone_dwell_seconds": per_zone_copy,
            "line_orientation": self.line_orientation or "pending",
            "line_position_px": self.line_position_px,
            "line_position_ratio": self._position_ratio,
            "zones_px": [
                {"name": str(zone["name"]), "rect_px": tuple(zone["rect_px"])} for zone in self.zones_px
            ],
            "crossing_events": crossing_events,
            "capacity_level": self.capacity_level,
            "capacity_event": capacity_event,
            "capacity_should_snapshot": bool(capacity_event and capacity_event.get("event_type") == "capacity_exceeded"),
            "capacity_alert_counts": dict(self.capacity_alert_counts),
            "zone_dwell_top": self._top_zone_dwell(2),
            "occupancy_history": self._history_values(),
            "stats": stats,
        }

    def render_heatmap_image(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        if np.max(self.heatmap_grid) <= 0.0:
            out_w = self.frame_w if width is None else int(width)
            out_h = self.frame_h if height is None else int(height)
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

        normalized = cv2.normalize(self.heatmap_grid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heat_u8 = normalized.astype(np.uint8)
        heat_u8 = cv2.GaussianBlur(heat_u8, (11, 11), 0)
        color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
        heat_color = cv2.applyColorMap(heat_u8, color_map)

        out_w = self.frame_w if width is None else int(width)
        out_h = self.frame_h if height is None else int(height)
        return cv2.resize(heat_color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def final_summary(self) -> Dict[str, object]:
        return {
            "occupancy_now": self.occupancy_now,
            "final_occupancy": self.occupancy_now,
            "peak_occupancy": self.peak_occupancy,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "per_zone_dwell_seconds": dict(self.per_zone_dwell_seconds),
            "line_orientation": self.line_orientation or "pending",
            "line_position_px": self.line_position_px,
            "capacity_level": self.capacity_level,
            "capacity_alert_counts": dict(self.capacity_alert_counts),
        }

    def _compute_line_position(self, orientation: Optional[str]) -> Optional[float]:
        if orientation == "vertical":
            return self._position_ratio * self.frame_w
        if orientation == "horizontal":
            return self._position_ratio * self.frame_h
        return None

    def _update_auto_line_calibration(self, tracks: Sequence[Track], t_now: float) -> None:
        elapsed = float(t_now) - float(self._calibration_start_time)

        for track in tracks:
            if len(track.history_centers) < 2:
                continue
            p0 = track.history_centers[-2]
            p1 = track.history_centers[-1]
            self._calibration_dx_sum += abs(p1[0] - p0[0])
            self._calibration_dy_sum += abs(p1[1] - p0[1])
            self._calibration_samples += 1

        if elapsed < self._calibration_seconds:
            return

        if self._calibration_samples <= 0:
            self.line_orientation = "horizontal"
        else:
            avg_dx = self._calibration_dx_sum / self._calibration_samples
            avg_dy = self._calibration_dy_sum / self._calibration_samples
            self.line_orientation = "vertical" if avg_dx > avg_dy else "horizontal"

        self.line_position_px = self._compute_line_position(self.line_orientation)

    def _track_state(self, track: Track) -> Dict[str, object]:
        if track.track_id not in self._per_track_state:
            self._per_track_state[track.track_id] = {
                "last_center_px": track.center_px,
                "last_side": None,
                "last_crossing_time": -1e9,
                "zone_dwell_seconds": {name: 0.0 for name in self.per_zone_dwell_seconds.keys()},
            }
        return self._per_track_state[track.track_id]

    def _update_counting(self, tracks: Sequence[Track], t_now: float) -> List[Dict[str, object]]:
        events = []  # type: List[Dict[str, object]]
        if self.line_position_px is None:
            return events

        for track in tracks:
            state = self._track_state(track)
            prev_center = state["last_center_px"]
            prev_side = state["last_side"]

            center = track.center_px
            side = self._line_side(center)

            if prev_side is None:
                state["last_side"] = side
            else:
                crossed = (prev_side != 0) and (side != 0) and (prev_side != side)
                cooldown_ok = (float(t_now) - float(state["last_crossing_time"])) >= self._cooldown_seconds
                age_ok = track.age_frames >= self._min_track_age_for_count

                if crossed and cooldown_ok and age_ok:
                    direction = self._crossing_direction(prev_center, center)
                    if direction == "IN":
                        self.total_in += 1
                    elif direction == "OUT":
                        self.total_out += 1

                    if direction in {"IN", "OUT"}:
                        state["last_crossing_time"] = float(t_now)
                        events.append(
                            {
                                "track_id": track.track_id,
                                "direction": direction,
                                "center_px": center,
                            }
                        )

                if side != 0:
                    state["last_side"] = side

            state["last_center_px"] = center

        return events

    def _line_side(self, center: PointPx) -> int:
        if self.line_position_px is None:
            return 0

        eps = 1e-6
        if self.line_orientation == "vertical":
            delta = center[0] - self.line_position_px
        else:
            delta = center[1] - self.line_position_px

        if abs(delta) <= eps:
            return 0
        return 1 if delta > 0 else -1

    def _crossing_direction(self, prev_center: PointPx, center: PointPx) -> str:
        if self.line_position_px is None:
            return ""

        if self.line_orientation == "vertical":
            if prev_center[0] < self.line_position_px <= center[0]:
                return "IN"
            if prev_center[0] > self.line_position_px >= center[0]:
                return "OUT"
            return ""

        if prev_center[1] < self.line_position_px <= center[1]:
            return "IN"
        if prev_center[1] > self.line_position_px >= center[1]:
            return "OUT"
        return ""

    def _update_zones(self, tracks: Sequence[Track], dt: float) -> None:
        dt_safe = max(0.0, float(dt))
        for track in tracks:
            state = self._track_state(track)
            per_track_dwell = state["zone_dwell_seconds"]
            for zone in self.zones_px:
                zone_name = str(zone["name"])
                rect_px = zone["rect_px"]
                if point_in_rect(track.center_px, rect_px):
                    self.per_zone_dwell_seconds[zone_name] += dt_safe
                    per_track_dwell[zone_name] += dt_safe

    def _update_heatmap(self, tracks: Sequence[Track]) -> None:
        fw = max(1, self.frame_w)
        fh = max(1, self.frame_h)

        for track in tracks:
            cx, cy = track.center_px
            gx = int(np.clip((cx / fw) * self.grid_w, 0, self.grid_w - 1))
            gy = int(np.clip((cy / fh) * self.grid_h, 0, self.grid_h - 1))
            self.heatmap_grid[gy, gx] += 1.0

    def _update_capacity(self, t_now: float) -> Optional[Dict[str, object]]:
        if not self.capacity_enabled:
            self.capacity_level = "none"
            return None

        if self.occupancy_now >= self.max_occupancy:
            level = "exceeded"
        elif self.occupancy_now >= self.warn_occupancy:
            level = "warn"
        else:
            level = "none"

        self.capacity_level = level
        if level == "none":
            return None

        last_event_time = self._capacity_last_event_time[level]
        if (float(t_now) - float(last_event_time)) < self.capacity_cooldown_sec:
            return None

        self._capacity_last_event_time[level] = float(t_now)
        self.capacity_alert_counts[level] += 1

        if level == "exceeded":
            event_type = "capacity_exceeded"
            message = "Capacity exceeded"
        else:
            event_type = "capacity_warn"
            message = "Approaching capacity"

        return {
            "event_type": event_type,
            "capacity_level": level,
            "occupancy_value": self.occupancy_now,
            "message": message,
        }

    def _update_occupancy_history(self, t_now: float) -> None:
        if not self._sparkline_enabled:
            self._occupancy_history = []
            return

        self._occupancy_history.append((float(t_now), int(self.occupancy_now)))
        cutoff = float(t_now) - self._history_seconds
        while self._occupancy_history and self._occupancy_history[0][0] < cutoff:
            del self._occupancy_history[0]

    def _history_values(self) -> List[int]:
        if not self._sparkline_enabled:
            return []
        return [int(v) for _, v in self._occupancy_history]

    def _top_zone_dwell(self, limit: int) -> List[Tuple[str, float]]:
        pairs = sorted(self.per_zone_dwell_seconds.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for name, value in pairs[: max(0, int(limit))]:
            out.append((str(name), float(value)))
        return out
