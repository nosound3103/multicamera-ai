import argparse
import logging
import math
import time
from pathlib import Path

from multicamera_ai.cameras import CameraManager, CameraState
from multicamera_ai.config import load_config
from multicamera_ai.config.schemas import VideoSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--reconnect-delay", type=float, default=1.0)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Optionally save most recent frame from each camera as JPEG",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        metavar="CAMERA_ID=FILE",
        help="Override a camera with a local file (relative to current directory); repeat as needed",
    )
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error("--seconds must be in (0, 3600]")
    if not math.isfinite(args.reconnect_delay) or args.reconnect_delay <= 0:
        parser.error("--reconnect-delay must be positive and finite")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    cameras = {camera.id: camera for camera in config.cameras}
    for override in args.source_override:
        camera_id, separator, filename = override.partition("=")
        if not separator or camera_id not in cameras or not filename.strip():
            parser.error("--source-override expects a configured CAMERA_ID=FILE")
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            parser.error(f"Override file not found: {path}")
        cameras[camera_id].source = VideoSource(type="file", uri=str(path))
    if not any(camera.enabled for camera in config.cameras):
        parser.error("YAML has no enabled cameras")
    manager = CameraManager.from_app_config(config, reconnect_delay=args.reconnect_delay)
    if not manager.camera_ids:
        parser.error("YAML has no configured cameras")

    previews = {}
    exit_code = 0
    manager.start_all()
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            for camera_id, status in manager.status().items():
                print(
                    f"{camera_id}: state={status.state} frames={status.frames_received} "
                    f"capture_fps={status.capture_fps:.1f} reconnects={status.reconnect_attempts} "
                    f"session={status.session_id} error={status.last_error}"
                )
                if args.preview_dir and status.state in (CameraState.RUNNING, CameraState.ENDED):
                    packet = manager.latest(camera_id)
                    if packet is not None:
                        previews[camera_id] = packet.frame.copy()
            if all(
                s.state in (CameraState.STOPPED, CameraState.ENDED, CameraState.ERROR)
                for s in manager.status().values()
            ):
                break
            print("-")
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("Stopping workers (Ctrl+C)")
        exit_code = 130
    finally:
        # Include short files that reached EOF between polling iterations.
        if args.preview_dir:
            for camera_id in manager.camera_ids:
                packet = manager.latest(camera_id)
                if packet is not None:
                    previews[camera_id] = packet.frame.copy()
        capture_statuses = manager.status()
        failed_cameras = [
            camera.id
            for camera in config.cameras
            if camera.enabled
            and (
                capture_statuses[camera.id].frames_received == 0
                or capture_statuses[camera.id].state
                in (CameraState.ERROR, CameraState.CONNECTING, CameraState.RECONNECTING)
            )
        ]
        if failed_cameras:
            print(f"ERROR: capture incomplete or failed for: {', '.join(failed_cameras)}")
            exit_code = exit_code or 1
        stopped = manager.stop_all(timeout=5.0)
        for camera_id, joined in stopped.items():
            if not joined:
                print(f"WARNING: {camera_id} capture did not stop within 5 s")
                exit_code = exit_code or 1
        for camera_id, status in manager.status().items():
            print(
                f"Final {camera_id}: state={status.state} frames={status.frames_received} "
                f"session={status.session_id} stopped={stopped[camera_id]}"
            )
        if args.preview_dir:
            import cv2

            args.preview_dir.mkdir(parents=True, exist_ok=True)
            for camera_id, frame in previews.items():
                output = args.preview_dir / f"{camera_id}.jpg"
                if not cv2.imwrite(str(output), frame):
                    print(f"WARNING: failed to save {output}")
                    exit_code = exit_code or 1
                else:
                    print(f"Saved {output}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
