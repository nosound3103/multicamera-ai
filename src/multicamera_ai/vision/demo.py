import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from multicamera_ai.config import load_config
from multicamera_ai.vision.geometry import (
    bottom_center,
    crossing_direction,
    normalized_to_pixel,
    point_in_polygon,
    side_of_line,
)

logger = logging.getLogger(__name__)

LINE_TO_COUNT = {
    "in_entrance": "IN",
    "out_entrance": "OUT",
}

LINE_COLORS = {
    "in_entrance": (80, 205, 40),
    "out_entrance": (0, 165, 255),
}

ZONE_CONFIRM_FRAMES = 3
MAX_TRACK_GAP_FRAMES = 10


@dataclass
class ZoneState:
    active: bool = False
    inside_streak: int = 0
    outside_streak: int = 0

    def update(self, inside: bool) -> bool:
        """Return True once when a new entry is confirmed."""
        if inside:
            self.inside_streak += 1
            self.outside_streak = 0

            if not self.active and self.inside_streak >= ZONE_CONFIRM_FRAMES:
                self.active = True
                return True
        else:
            self.outside_streak += 1
            self.inside_streak = 0

            if self.outside_streak >= ZONE_CONFIRM_FRAMES:
                self.active = False

        return False


def draw_regions(image, lines, zones) -> None:
    """Draw counting lines and translucent restricted areas."""
    overlay = image.copy()

    for _name, polygon in zones:
        contour = np.asarray(
            polygon,
            dtype=np.int32,
        )
        cv2.fillPoly(
            overlay,
            [contour],
            (0, 0, 255),
        )

    cv2.addWeighted(
        overlay,
        0.12,
        image,
        0.88,
        0,
        dst=image,
    )

    for _name, polygon in zones:
        contour = np.asarray(
            polygon,
            dtype=np.int32,
        )

        cv2.polylines(
            image,
            [contour],
            True,
            (40, 40, 255),
            2,
            cv2.LINE_AA,
        )

    for name, start, end in lines:
        color = LINE_COLORS[name]

        cv2.line(
            image,
            start,
            end,
            color,
            3,
            cv2.LINE_AA,
        )

        label = LINE_TO_COUNT[name]

        cv2.putText(
            image,
            label,
            (
                max(0, start[0]),
                max(20, start[1] - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_hud(
    image,
    camera_id: str,
    counts: dict[str, int],
    active: int,
) -> None:
    """Draw counters in a dedicated panel."""
    height, width = image.shape[:2]

    x1, y1 = 14, 42
    x2 = min(width - 4, 330)
    y2 = min(height - 4, 144)

    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2]
    black = np.zeros_like(roi)

    cv2.addWeighted(
        black,
        0.7,
        roi,
        0.3,
        0,
        dst=roi,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (180, 180, 180),
        1,
    )

    cv2.putText(
        image,
        f"CAMERA {camera_id}",
        (x1 + 10, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"IN {counts['IN']}     OUT {counts['OUT']}",
        (x1 + 10, y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"RESTRICTED: {active} inside",
        (x1 + 10, y1 + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (60, 60, 255) if active else (190, 190, 190),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-lane YOLO tracking demo")

    parser.add_argument(
        "--config",
        default="config/cameras.yaml",
    )
    parser.add_argument(
        "--camera",
        default="cam1",
    )
    parser.add_argument(
        "--input",
        help="Video path or RTSP URL overriding YAML",
    )
    parser.add_argument(
        "--output",
        default="artifacts/geometry_demo.mp4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # -------------------------------------
    # 1. Load YAML
    # -------------------------------------
    config = load_config(args.config)

    camera = next(
        (cam for cam in config.cameras if cam.id == args.camera),
        None,
    )

    if camera is None:
        parser.error(f"Unknown camera: {args.camera}")

    if {line.id for line in camera.lines} != set(LINE_TO_COUNT):
        parser.error("Camera must define exactly two lines: in_entrance and out_entrance")

    if {zone.id for zone in camera.zones} != {"restricted_area"}:
        parser.error("Camera must define one zone: restricted_area")

    # -------------------------------------
    # 2. Resolve source
    # -------------------------------------
    if args.input is not None:
        source: str | int = args.input

    elif camera.source.type == "webcam":
        assert camera.source.index is not None
        source = camera.source.index

    else:
        assert camera.source.uri is not None
        source = camera.source.uri

    output_path = Path(args.output)

    if isinstance(source, str) and "://" not in source:
        if Path(source).resolve() == output_path.resolve():
            parser.error("Input and output must be different files")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------
    # 3. Open video
    # -------------------------------------
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video for camera {camera.id}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    writer = None
    frame_count = 0
    started = time.perf_counter()

    counts = {
        "IN": 0,
        "OUT": 0,
    }

    # -------------------------------------
    # 4. Tracking / counting state
    # -------------------------------------
    previous_positions: dict[
        int,
        tuple[float, float],
    ] = {}

    last_seen: dict[int, int] = {}

    # Prevent counting the same track again.
    counted_tracks: set[int] = set()

    zone_states: dict[
        tuple[int, str],
        ZoneState,
    ] = {}

    try:
        # ---------------------------------
        # 5. Load YOLO
        # ---------------------------------
        model = YOLO(config.model.weights)

        # ---------------------------------
        # 6. Frame loop
        # ---------------------------------
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=config.model.confidence,
                imgsz=config.model.imgsz,
                classes=config.model.class_ids,
                device=config.model.device,
                verbose=False,
            )[0]

            annotated = result.plot()
            height, width = annotated.shape[:2]

            # ---------------------------------
            # 7. YAML regions -> pixel
            # ---------------------------------
            pixel_lines = [
                (
                    line.id,
                    normalized_to_pixel(
                        (line.start.x, line.start.y),
                        width,
                        height,
                    ),
                    normalized_to_pixel(
                        (line.end.x, line.end.y),
                        width,
                        height,
                    ),
                )
                for line in camera.lines
            ]

            pixel_zones = [
                (
                    zone.id,
                    [
                        normalized_to_pixel(
                            (point.x, point.y),
                            width,
                            height,
                        )
                        for point in zone.points
                    ],
                )
                for zone in camera.zones
            ]

            draw_regions(
                annotated,
                pixel_lines,
                pixel_zones,
            )

            active_inside: set[int] = set()

            # ---------------------------------
            # 8. Process tracked detections
            # ---------------------------------
            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    anchor = bottom_center(x1, y1, x2, y2)

                    cv2.circle(
                        annotated,
                        (
                            round(anchor[0]),
                            round(anchor[1]),
                        ),
                        4,
                        (255, 90, 0),
                        -1,
                        cv2.LINE_AA,
                    )

                    previous = previous_positions.get(track_id)
                    last_frame = last_seen.get(track_id)

                    # Avoid counting a large jump after
                    # the tracker temporarily disappears.
                    if last_frame is not None and frame_count - last_frame > MAX_TRACK_GAP_FRAMES:
                        previous = None

                        for key in tuple(zone_states):
                            if key[0] == track_id:
                                del zone_states[key]

                    # -----------------------------
                    # 9. Lane-based IN / OUT
                    # -----------------------------
                    if track_id not in counted_tracks and previous is not None:
                        for line_id, start, end in pixel_lines:
                            crossed = crossing_direction(
                                previous,
                                anchor,
                                start,
                                end,
                            )

                            if crossed is not None:
                                category = LINE_TO_COUNT[line_id]

                                counts[category] += 1
                                counted_tracks.add(track_id)

                                logger.info(
                                    "COUNT camera=%s track=%s line=%s category=%s total=%s",
                                    camera.id,
                                    track_id,
                                    line_id,
                                    category,
                                    counts[category],
                                )

                                break

                    # Keep previous point when current
                    # anchor lies exactly on a line.
                    on_line = any(
                        side_of_line(start, end, anchor) == 0 for _, start, end in pixel_lines
                    )

                    if not on_line:
                        previous_positions[track_id] = anchor

                    last_seen[track_id] = frame_count

                    # -----------------------------
                    # 10. Restricted area
                    # -----------------------------
                    for zone_id, polygon in pixel_zones:
                        inside = point_in_polygon(
                            anchor,
                            polygon,
                        )

                        key = (track_id, zone_id)

                        state = zone_states.setdefault(
                            key,
                            ZoneState(),
                        )

                        if state.update(inside):
                            logger.warning(
                                "ZONE ENTRY camera=%s track=%s zone=%s frame=%s",
                                camera.id,
                                track_id,
                                zone_id,
                                frame_count,
                            )

                        if state.active:
                            active_inside.add(track_id)

                            # Red bounding box:
                            # no overlapping violation text.
                            cv2.rectangle(
                                annotated,
                                (
                                    round(x1),
                                    round(y1),
                                ),
                                (
                                    round(x2),
                                    round(y2),
                                ),
                                (0, 0, 255),
                                3,
                                cv2.LINE_AA,
                            )

            # ---------------------------------
            # 11. Cleanup stale tracking state
            # ---------------------------------
            stale = [track_id for track_id, seen in last_seen.items() if frame_count - seen > 150]

            for track_id in stale:
                last_seen.pop(track_id, None)
                previous_positions.pop(track_id, None)

                for key in tuple(zone_states):
                    if key[0] == track_id:
                        del zone_states[key]

            # ---------------------------------
            # 12. Draw HUD
            # ---------------------------------
            draw_hud(
                annotated,
                camera.id,
                counts,
                len(active_inside),
            )

            # ---------------------------------
            # 13. Video output
            # ---------------------------------
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {output_path}")

            writer.write(annotated)

            if frame_count % 100 == 0:
                logger.info(
                    "Processed %s frames",
                    frame_count,
                )

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError("No frames processed")

    elapsed = time.perf_counter() - started

    logger.info(
        "Finished: %s frames, %.2f FPS, output=%s",
        frame_count,
        frame_count / elapsed if elapsed else 0.0,
        output_path,
    )


if __name__ == "__main__":
    main()
