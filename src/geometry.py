from typing import Tuple

BBoxPx = Tuple[float, float, float, float]
PointPx = Tuple[float, float]
RectNorm = Tuple[float, float, float, float]


def bbox_center(bbox_px: BBoxPx) -> PointPx:
    x1, y1, x2, y2 = bbox_px
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_iou(bbox_a: BBoxPx, bbox_b: BBoxPx) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0

    return inter / union


def point_in_rect(px: PointPx, rect_px: BBoxPx) -> bool:
    x, y = px
    x1, y1, x2, y2 = rect_px
    return x1 <= x <= x2 and y1 <= y <= y2


def normalize_rect(rect_norm: RectNorm, frame_w: int, frame_h: int) -> BBoxPx:
    x1, y1, x2, y2 = rect_norm
    return (
        float(x1) * float(frame_w),
        float(y1) * float(frame_h),
        float(x2) * float(frame_w),
        float(y2) * float(frame_h),
    )
