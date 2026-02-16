from typing import List, Optional

import cv2
import numpy as np

from geometry import BBoxPx


class Detection:
    def __init__(self, bbox_px: BBoxPx, confidence: float, class_name: str = "person") -> None:
        self.bbox_px = bbox_px
        self.confidence = float(confidence)
        self.class_name = str(class_name)


class PersonDetector:
    def __init__(self, network: str = "ssd-mobilenet-v2", threshold: float = 0.5) -> None:
        self.network = str(network)
        self.threshold = float(threshold)
        self.backend = "none"

        self._net = None
        self._jetson_utils = None
        self._hog = None

        self._init_detectnet()
        if self.backend == "none":
            self._init_hog()

    def _init_detectnet(self) -> None:
        try:
            import jetson_inference  # type: ignore
            import jetson_utils  # type: ignore

            self._net = jetson_inference.detectNet(self.network, threshold=self.threshold)
            self._jetson_utils = jetson_utils
            self.backend = "detectnet"
        except Exception:
            self._net = None
            self._jetson_utils = None

    def _init_hog(self) -> None:
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
            self.backend = "hog"
        except Exception:
            self._hog = None
            self.backend = "none"

    def to_bbox_px(self, det: object) -> BBoxPx:
        return (float(det.Left), float(det.Top), float(det.Right), float(det.Bottom))

    def detect(self, cuda_img: Optional[object], frame_bgr: Optional[np.ndarray] = None) -> List[Detection]:
        if self.backend == "detectnet" and self._net is not None:
            return self._detect_detectnet(cuda_img, frame_bgr)

        if self.backend == "hog" and self._hog is not None:
            return self._detect_hog(cuda_img, frame_bgr)

        return []

    def _detect_detectnet(self, cuda_img: Optional[object], frame_bgr: Optional[np.ndarray]) -> List[Detection]:
        if isinstance(cuda_img, np.ndarray) and frame_bgr is None:
            if cuda_img.ndim == 3 and cuda_img.shape[2] == 4:
                frame_bgr = cv2.cvtColor(cuda_img, cv2.COLOR_RGBA2BGR)
            elif cuda_img.ndim == 3 and cuda_img.shape[2] == 3:
                frame_bgr = cuda_img
            cuda_img = None

        raw = None

        if cuda_img is not None:
            try:
                raw = self._net.Detect(cuda_img, overlay="none")
            except TypeError:
                raw = None
            except Exception:
                raw = None

        if raw is None and frame_bgr is not None and self._jetson_utils is not None:
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            rgba = np.ascontiguousarray(rgba)
            try:
                cuda_from_numpy = self._jetson_utils.cudaFromNumpy(rgba)
                raw = self._net.Detect(cuda_from_numpy, rgba.shape[1], rgba.shape[0], overlay="none")
            except TypeError:
                raw = self._net.Detect(cuda_from_numpy, overlay="none")
            except Exception:
                raw = []

        if raw is None:
            raw = []

        detections: List[Detection] = []
        for det in raw:
            confidence = float(det.Confidence)
            if confidence < self.threshold:
                continue

            class_id = int(det.ClassID)
            class_name = str(self._net.GetClassDesc(class_id)).strip().lower()
            if class_name != "person":
                continue

            detections.append(
                Detection(
                    bbox_px=self.to_bbox_px(det),
                    confidence=confidence,
                    class_name="person",
                )
            )

        return detections

    def _detect_hog(self, cuda_img: Optional[object], frame_bgr: Optional[np.ndarray]) -> List[Detection]:
        if frame_bgr is None and isinstance(cuda_img, np.ndarray):
            if cuda_img.ndim == 3 and cuda_img.shape[2] == 4:
                frame_bgr = cv2.cvtColor(cuda_img, cv2.COLOR_RGBA2BGR)
            elif cuda_img.ndim == 3 and cuda_img.shape[2] == 3:
                frame_bgr = cuda_img

        if frame_bgr is None:
            return []

        rects, weights = self._hog.detectMultiScale(
            frame_bgr,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
        )

        detections: List[Detection] = []
        for (x, y, w, h), score in zip(rects, weights):
            confidence = float(score)
            if confidence < max(0.0, self.threshold - 0.35):
                continue
            detections.append(
                Detection(
                    bbox_px=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=confidence,
                    class_name="person",
                )
            )

        return detections
