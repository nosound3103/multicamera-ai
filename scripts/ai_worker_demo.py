from __future__ import annotations

import argparse
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from multicamera_ai.cameras import CameraState, CameraWorker
from multicamera_ai.cameras.worker import FramePacket
from multicamera_ai.config import load_config
from multicamera_ai.events import EventRecorder, SnapshotWriter, SQLiteEventStore
from multicamera_ai.pipeline import AIState, AIWorker

logger = logging.getLogger(__name__)


def resolve_source(camera: Any, input_override: str | None) -> tuple[str | int, bool]:
    """Resolve one source, leaving deployment YAML unchanged."""
    if input_override:
        if input_override.lower().startswith(("rtsp://", "rtsps://")):
            return input_override, False
        return input_override, True

    if camera.source.type == "webcam":
        if camera.source.index is None:
            raise ValueError("Webcam requires source.index")
        return camera.source.index, False

    if not camera.source.uri:
        raise ValueError(f"{camera.id}: source.uri is missing")
    return camera.source.uri, camera.source.type == "file"


class _MP4Writer:
    def __init__(self, path: Path, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Output FPS must be positive and finite")
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.frames_written = 0

    def write(self, image: Any) -> None:
        height, width = image.shape[:2]
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Cannot open video output {self.path}")
        self.writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def run_file(
    *,
    source: str,
    camera: Any,
    model_config: Any,
    recorder: Any,
    output_path: Path,
    max_frames: int,
    output_fps: float | None,
) -> int:
    """Offline file: sequential read -> process -> write, NO latest-frame drops.

    Deliberately calls AIWorker.process_packet() without starting camera or AI
    threads. The Event Engine and tracker implementation stay exactly the same.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video file: {source}")

    writer: _MP4Writer | None = None
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            logger.warning("Source FPS unavailable, assuming 25 FPS for output")
            source_fps = 25.0

        # CameraWorker is a source adapter here; it is NOT started for MP4.
        # This avoids a background capture thread advancing ahead of YOLO.
        source_adapter = CameraWorker(camera.id, source, is_file=True)
        worker = AIWorker(
            camera=camera,
            model_config=model_config,
            source=source_adapter,
            recorder=recorder,
        )
        writer = _MP4Writer(output_path, output_fps or source_fps)

        # File events have a synthetic time axis anchored to processing start.
        # This is not the original camera recording timestamp.
        timeline_start = datetime.now(UTC)
        sequence = 0
        while max_frames == 0 or sequence < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            sequence += 1
            packet = FramePacket(
                sequence=sequence,
                session_id=1,
                captured_at=timeline_start + timedelta(seconds=(sequence - 1) / source_fps),
                frame=frame,
            )
            annotated = worker.process_packet(packet)
            writer.write(annotated.annotated)

        if writer.frames_written == 0:
            raise RuntimeError("No frames processed from input MP4")
        status = worker.status()
        logger.info(
            "Offline MP4: read=%d written=%d skipped=%d input_fps=%.3f output_fps=%.3f "
            "events=%d -> %s",
            sequence,
            writer.frames_written,
            status.skipped_capture_frames,
            source_fps,
            writer.fps,
            status.saved_events,
            output_path,
        )
        return writer.frames_written
    finally:
        cap.release()
        if writer is not None:
            writer.close()


def run_live(
    *,
    source: str | int,
    camera: Any,
    model_config: Any,
    recorder: Any,
    output_path: Path,
    max_frames: int,
    output_fps: float | None,
) -> int:
    """RTSP/webcam: latest-frame semantics, as required by F02."""
    capture = CameraWorker(camera.id, source, is_file=False)
    worker = AIWorker(camera=camera, model_config=model_config, source=capture, recorder=recorder)
    writer = _MP4Writer(output_path, output_fps or 15.0)
    last_sequence = 0
    try:
        capture.start()
        worker.start()
        while max_frames == 0 or writer.frames_written < max_frames:
            packet = worker.wait_for_frame(after_sequence=last_sequence, timeout=0.5)
            if packet is not None:
                last_sequence = packet.sequence
                writer.write(packet.annotated)
                continue
            if worker.status().state in (AIState.ERROR, AIState.ENDED, AIState.STOPPED):
                break

        ai_status = worker.status()
        capture_status = capture.status()
        logger.info("AI status: %s", ai_status)
        logger.info("Capture status: %s", capture_status)
        if ai_status.state == AIState.ERROR or capture_status.state == CameraState.ERROR:
            raise RuntimeError(f"Camera {camera.id} failed; check logged error")
        if writer.frames_written == 0:
            raise RuntimeError("No processed frames were available")
        return writer.frames_written
    finally:
        worker.stop(timeout=10.0)
        capture.stop(timeout=10.0)
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture + AI worker smoke test")
    parser.add_argument("--config", default="config/cameras.local.yaml")
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--input", help="MP4/RTSP override for one camera")
    parser.add_argument("--output", default="artifacts/ai_worker_demo.mp4")
    parser.add_argument("--db", default="storage/events.db")
    parser.add_argument("--snapshots", default="storage/snapshots")
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Override output FPS. Default: source FPS for MP4; 15 for live sources.",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="0 means unlimited")
    args = parser.parse_args()
    if args.max_frames < 0 or (
        args.output_fps is not None and (not math.isfinite(args.output_fps) or args.output_fps <= 0)
    ):
        parser.error("--max-frames must be >= 0 and --output-fps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    camera = next((item for item in config.cameras if item.id == args.camera), None)
    if camera is None:
        parser.error(f"Camera not found: {args.camera}")
    source, is_file = resolve_source(camera, args.input)
    output_path = Path(args.output)
    if is_file and isinstance(source, str) and Path(source).resolve() == output_path.resolve():
        parser.error("Input and output video cannot be the same file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = SQLiteEventStore(args.db)
    recorder = EventRecorder(store, SnapshotWriter(Path(args.snapshots)))
    if is_file:
        assert isinstance(source, str)
        written = run_file(
            source=source,
            camera=camera,
            model_config=config.model,
            recorder=recorder,
            output_path=output_path,
            max_frames=args.max_frames,
            output_fps=args.output_fps,
        )
    else:
        written = run_live(
            source=source,
            camera=camera,
            model_config=config.model,
            recorder=recorder,
            output_path=output_path,
            max_frames=args.max_frames,
            output_fps=args.output_fps,
        )
    logger.info("Saved %d annotated frames to %s", written, output_path)


if __name__ == "__main__":
    main()
