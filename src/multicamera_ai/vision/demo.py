import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from multicamera_ai.config import load_config
from multicamera_ai.events import (
    CountLine,
    EventEngine,
    EventRecorder,
    EventType,
    RestrictedZone,
    SnapshotWriter,
    SQLiteEventStore,
    TrackedPoint,
)
from multicamera_ai.vision.geometry import (
    bottom_center,
    normalized_to_pixel,
    point_in_polygon,
)

logger = logging.getLogger(__name__)

LANE_DIRECTIONS = {"in_entrance": "IN", "out_entrance": "OUT"}
LINE_COLORS = {"IN": (80, 205, 40), "OUT": (0, 165, 255)}


def project_regions(camera, width: int, height: int):
    """Convert validated YAML normalized coordinates to pixel-space regions."""
    lines = [
        CountLine(
            id=line.id,
            start=normalized_to_pixel((line.start.x, line.start.y), width, height),
            end=normalized_to_pixel((line.end.x, line.end.y), width, height),
            count_as=LANE_DIRECTIONS.get(line.id, "IN"),
            in_direction=getattr(line, "in_direction", "negative_to_positive"),
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
    """Draw two labeled lane segments and transparent zone polygons."""
    overlay = image.copy()
    for zone in zones:
        cv2.fillPoly(
            overlay,
            [np.asarray(zone.points, dtype=np.int32)],
            (0, 0, 255),
        )
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, dst=image)

    for zone in zones:
        cv2.polylines(
            image,
            [np.asarray(zone.points, dtype=np.int32)],
            True,
            (40, 40, 255),
            2,
            cv2.LINE_AA,
        )

    for line in lines:
        color = LINE_COLORS[line.count_as]
        cv2.line(
            image,
            tuple(map(round, line.start)),
            tuple(map(round, line.end)),
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line.count_as,
            (max(0, round(line.start[0])), max(20, round(line.start[1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_hud(image: np.ndarray, camera_id: str, counts: dict[str, int], inside: int) -> None:
    """Draw session counters (persisted crossing events) and visible zone occupancy."""
    height, width = image.shape[:2]
    x1, y1 = 14, 42
    x2, y2 = min(width - 4, 330), min(height - 4, 144)
    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2]
    cv2.addWeighted(np.zeros_like(roi), 0.70, roi, 0.30, 0, dst=roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), 1)
    for text, dy, font_scale, color in (
        (f"CAMERA {camera_id}", 25, 0.62, (255, 255, 255)),
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


def run_demo(args: argparse.Namespace) -> None:
    # Import YOLO on actual execution; storage/unit tests need not install it.
    from ultralytics import YOLO

    config = load_config(args.config)
    camera = next((c for c in config.cameras if c.id == args.camera), None)
    if camera is None:
        raise ValueError(f"Unknown camera ID: {args.camera}")

    if {line.id for line in camera.lines} != set(LANE_DIRECTIONS):
        raise ValueError("Demo camera needs exactly two lines: in_entrance, out_entrance")
    if {zone.id for zone in camera.zones} != {"restricted_area"}:
        raise ValueError("Demo camera needs one zone: restricted_area")

    if args.input is not None:
        source: str | int = args.input
    elif camera.source.type == "webcam":
        if camera.source.index is None:
            raise ValueError("Webcam source requires index")
        source = camera.source.index
    else:
        if not camera.source.uri:
            raise ValueError("Video source requires uri")
        source = camera.source.uri

    if isinstance(source, str) and "://" not in source:
        source = str(Path(source).expanduser().resolve())
        if not Path(source).is_file():
            raise FileNotFoundError(f"Input video not found: {source}")

    output_path = Path(args.output)
    if isinstance(source, str) and "://" not in source:
        if Path(source).resolve() == output_path.resolve():
            raise ValueError("Input and output must be different paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteEventStore(args.db)
    recorder = EventRecorder(storage, SnapshotWriter(Path(args.snapshots)))
    engine = EventEngine(camera_id=camera.id)
    counts = {"IN": 0, "OUT": 0}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video source for {camera.id}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    writer = None
    frame_index = 0
    started = time.perf_counter()
    cached_size: tuple[int, int] | None = None
    lines: list[CountLine] = []
    zones: list[RestrictedZone] = []

    try:
        model = YOLO(config.model.weights)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

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
            if cached_size != (width, height):
                lines, zones = project_regions(camera, width, height)
                cached_size = (width, height)

            detections: list[TrackedPoint] = []
            visible_objects: list[tuple[int, tuple[float, float], tuple[float, ...]]] = []
            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                    anchor = bottom_center(*xyxy)
                    detections.append(TrackedPoint(track_id=track_id, anchor=anchor))
                    visible_objects.append((track_id, anchor, xyxy))

            # The ONLY place that decides whether a new event happened.
            events = engine.process_frame(
                frame_index=frame_index,
                detections=detections,
                lines=lines,
                zones=zones,
            )

            draw_regions(annotated, lines, zones)
            inside_tracks: set[int] = set()
            for track_id, anchor, (x1, y1, x2, y2) in visible_objects:
                cv2.circle(
                    annotated,
                    (round(anchor[0]), round(anchor[1])),
                    4,
                    (255, 90, 0),
                    -1,
                    cv2.LINE_AA,
                )
                if any(point_in_polygon(anchor, z.points) for z in zones):
                    inside_tracks.add(track_id)
                    cv2.rectangle(
                        annotated,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

            # Draw the updated counts on snapshots from this frame.
            next_counts = counts.copy()
            for event in events:
                if event.event_type == EventType.LINE_CROSSING:
                    assert event.direction is not None
                    next_counts[event.direction] += 1
            draw_hud(annotated, camera.id, next_counts, len(inside_tracks))

            for event in events:
                recorded = recorder.record(event, annotated)
                logger.info(
                    "SAVED event=%s camera=%s track=%s type=%s region=%s direction=%s snapshot=%s",
                    recorded.event_id,
                    recorded.camera_id,
                    recorded.track_id,
                    recorded.event_type.value,
                    recorded.region_id,
                    recorded.direction,
                    recorded.snapshot_path,
                )
            counts = next_counts

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video output: {output_path}")
            writer.write(annotated)
            if frame_index % 100 == 0:
                logger.info(
                    "Processed %s frames | IN=%s OUT=%s",
                    frame_index,
                    counts["IN"],
                    counts["OUT"],
                )
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_index == 0:
        raise RuntimeError("No video frames were processed")
    elapsed = time.perf_counter() - started
    logger.info(
        "Done: camera=%s frames=%s processing_FPS=%.2f IN=%s OUT=%s video=%s database=%s",
        camera.id,
        frame_index,
        frame_index / elapsed if elapsed else 0.0,
        counts["IN"],
        counts["OUT"],
        output_path,
        storage.db_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO/ByteTrack demo + SQLite events")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--input", help="Override YAML with a file or RTSP URL")
    parser.add_argument("--output", default="artifacts/events_demo.mp4")
    parser.add_argument("--db", default="storage/events.db")
    parser.add_argument("--snapshots", default="storage/snapshots")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_demo(args)


if __name__ == "__main__":
    main()
