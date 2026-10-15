import argparse
import logging
import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from multicamera_ai.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Command-line arguments
    parser = argparse.ArgumentParser(description="YOLO + OpenCV video demo")

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
        default=None,
    )
    parser.add_argument(
        "--output",
        default="artifacts/demo_output.mp4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = unlimited)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 2. Load configuration
    config = load_config(args.config)

    camera = next(
        (cam for cam in config.cameras if cam.id == args.camera),
        None,
    )

    if camera is None:
        parser.error(f"Camera not found: {args.camera}")

    # 3. Resolve video source
    if args.input is not None:
        source = args.input
    elif camera.source.type == "webcam":
        source = camera.source.index
    else:
        source = camera.source.uri

    logger.info("Opening video source: %s", source)

    # 4. Open video with OpenCV
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    frame_count = 0
    started_at = time.perf_counter()

    try:
        # 5. Load YOLO model
        model = YOLO(config.model.weights)

        # 6. Get source FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        output = Path(args.output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 7. Process frames
        while True:
            success, frame = cap.read()

            if not success:
                logger.info("End of video or failed to read frame")
                break

            # 8. Detection + tracking
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=config.model.confidence,
                imgsz=config.model.imgsz,
                classes=config.model.class_ids,
                device=config.model.device,
                verbose=False,
            )

            result = results[0]

            # 9. Draw boxes, classes and track IDs
            annotated = result.plot()

            # 10. Initialize output video writer
            if writer is None:
                height, width = annotated.shape[:2]

                fourcc = cv2.VideoWriter.fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output),
                    fourcc,
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output}")

            # 11. Write annotated frame
            writer.write(annotated)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(
                    "Processed %d frames",
                    frame_count,
                )

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    finally:
        # 12. Clean up resources
        cap.release()

        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started_at
    processing_fps = frame_count / elapsed if elapsed > 0 else 0.0

    logger.info(
        "Finished: %d frames, processing FPS: %.2f",
        frame_count,
        processing_fps,
    )
    logger.info("Output saved to: %s", args.output)


if __name__ == "__main__":
    main()
