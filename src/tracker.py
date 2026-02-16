from typing import Dict, List, Sequence, Set, Tuple

from detector import Detection
from geometry import BBoxPx, PointPx, bbox_center, bbox_iou


class Track:
    def __init__(
        self,
        track_id: int,
        bbox_px: BBoxPx,
        center_px: PointPx,
        missed_frames: int = 0,
        history_centers: List[PointPx] = None,
        age_frames: int = 1,
        last_update_time: float = 0.0,
    ) -> None:
        self.track_id = int(track_id)
        self.bbox_px = bbox_px
        self.center_px = center_px
        self.missed_frames = int(missed_frames)
        self.history_centers = history_centers if history_centers is not None else []
        self.age_frames = int(age_frames)
        self.last_update_time = float(last_update_time)

    def apply_detection(self, detection: Detection, t_now: float, history_length: int) -> None:
        self.bbox_px = detection.bbox_px
        self.center_px = bbox_center(detection.bbox_px)
        self.missed_frames = 0
        self.age_frames += 1
        self.last_update_time = float(t_now)
        self.history_centers.append(self.center_px)
        if len(self.history_centers) > history_length:
            self.history_centers = self.history_centers[-history_length:]


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 10,
        history_length: int = 30,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.history_length = int(history_length)

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection], t_now: float) -> List[Track]:
        if not self._tracks:
            for detection in detections:
                self._create_track(detection, t_now)
            return self.active_tracks()

        track_ids = list(self._tracks.keys())
        candidates: List[Tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self._tracks[track_id]
            for det_idx, detection in enumerate(detections):
                overlap = bbox_iou(track.bbox_px, detection.bbox_px)
                candidates.append((overlap, track_id, det_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: Set[int] = set()
        matched_detections: Set[int] = set()

        for overlap, track_id, det_idx in candidates:
            if overlap < self.iou_threshold:
                break
            if track_id in matched_tracks or det_idx in matched_detections:
                continue

            self._tracks[track_id].apply_detection(detections[det_idx], t_now, self.history_length)
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self._create_track(detection, t_now)

        stale_ids: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_tracks:
                continue
            track.missed_frames += 1
            track.age_frames += 1
            if track.missed_frames > self.max_missed_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._tracks[track_id]

        return self.active_tracks()

    def active_tracks(self) -> List[Track]:
        return [self._tracks[k] for k in sorted(self._tracks.keys())]

    def _create_track(self, detection: Detection, t_now: float) -> None:
        center = bbox_center(detection.bbox_px)
        track = Track(
            track_id=self._next_id,
            bbox_px=detection.bbox_px,
            center_px=center,
            missed_frames=0,
            history_centers=[center],
            age_frames=1,
            last_update_time=float(t_now),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
