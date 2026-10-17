import cv2
import numpy as np

from multicamera_ai.events import CountLine, RestrictedZone
from multicamera_ai.vision.geometry import normalized_to_pixel

COUNTING_LINE_COLOR = (80, 205, 40)


def project_regions(camera, width: int, height: int):
    """Convert validated YAML normalized coordinates to pixel-space regions."""
    lines = [
        CountLine(
            id=line.id,
            start=normalized_to_pixel((line.start.x, line.start.y), width, height),
            end=normalized_to_pixel((line.end.x, line.end.y), width, height),
            in_direction=line.in_direction,
        )
        for line in camera.lines
    ]
    zones = [
        RestrictedZone(
            id=zone.id,
            points=tuple(
                normalized_to_pixel((point.x, point.y), width, height) for point in zone.points
            ),
        )
        for zone in camera.zones
    ]
    return lines, zones


def draw_regions(image: np.ndarray, lines, zones) -> None:
    """Draw bidirectional counting lines and transparent zone polygons."""
    if zones:
        contours = [np.asarray(zone.points, dtype=np.int32) for zone in zones]
        overlay = image.copy()
        for contour in contours:
            cv2.fillPoly(overlay, [contour], (0, 0, 255))
        cv2.addWeighted(overlay, 0.12, image, 0.88, 0, dst=image)
        for contour in contours:
            cv2.polylines(image, [contour], True, (40, 40, 255), 2, cv2.LINE_AA)

    for line in lines:
        cv2.line(
            image,
            tuple(map(round, line.start)),
            tuple(map(round, line.end)),
            COUNTING_LINE_COLOR,
            3,
            cv2.LINE_AA,
        )


def draw_hud(
    image: np.ndarray, camera_id: str, counts: dict[str, int], inside: int, *, fps: float = 0.0
) -> None:
    """Draw session counters (accepted events) and visible zone occupancy."""
    height, width = image.shape[:2]
    x1, y1 = 14, 42
    x2, y2 = min(width - 4, 330), min(height - 4, 174)
    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2]
    cv2.addWeighted(np.zeros_like(roi), 0.70, roi, 0.30, 0, dst=roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), 1)
    for text, dy, font_scale, color in (
        (f"CAMERA {camera_id}", 25, 0.62, (255, 255, 255)),
        (f"FPS: {fps:.1f}", 114, 0.55, (255, 255, 255)),
        (f"IN {counts['IN']}     OUT {counts['OUT']}", 55, 0.70, (255, 255, 255)),
        (
            f"RESTRICTED: {inside} inside",
            84,
            0.52,
            (60, 60, 255) if inside else (190, 190, 190),
        ),
    ):
        cv2.putText(
            image,
            text,
            (x1 + 10, y1 + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
