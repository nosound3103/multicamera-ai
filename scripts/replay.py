"""Replay configured files or one file as real-time RTSP inputs, restarting failed publishers."""

import argparse
import logging
import subprocess
import time
from pathlib import Path


def replay_watermarked(args):
    from multicamera_ai.cameras import CameraState, CameraWorker
    from multicamera_ai.config.schemas import StreamingConfig
    from multicamera_ai.pipeline.ai_worker import AIFrame
    from multicamera_ai.pipeline.publisher import RTSPPublisher

    class Adapter:
        def __init__(self, capture):
            self.capture = capture

        def convert(self, packet):
            if packet is None:
                return None
            return AIFrame(
                self.capture.camera_id,
                packet.sequence,
                packet.session_id,
                packet.captured_at,
                packet.frame,
                0,
            )

        def wait_for_frame(self, **kwargs):
            return self.convert(self.capture.wait_for_frame(**kwargs))

        def latest(self):
            return self.convert(self.capture.get_latest())

    captures, publishers = [], []
    try:
        for camera_id, source in args.inputs:
            capture = CameraWorker(
                f"replay_{camera_id}", source, is_file=True, file_max_fps=args.fps or None
            )
            captures.append(capture)
            capture.start()
            publisher = RTSPPublisher(
                f"replay_{camera_id}",
                f"{args.server}/input_{camera_id}",
                Adapter(capture),
                StreamingConfig(watermark=True, fps=args.fps),
            )
            publishers.append(publisher)
            publisher.start()
        while True:
            for capture in captures:
                if capture.status().state == CameraState.ENDED:
                    capture.start()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for publisher in publishers:
            publisher.stop()
        for capture in captures:
            capture.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input")
    inputs.add_argument("--config", help="File-source camera YAML; replays each enabled camera")
    parser.add_argument(
        "--fps", type=int, default=0, help="Output FPS cap; 0 preserves full source FPS (default)"
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--server", default="rtsp://127.0.0.1:8554")
    parser.add_argument(
        "--watermark", action="store_true", help="Embed capture timestamps for end-to-end probes"
    )
    args = parser.parse_args()
    if args.fps < 0 or args.start_index < 1 or (args.count is not None and args.count < 1):
        parser.error("Count/start-index must be positive and FPS nonnegative")
    if args.config:
        from multicamera_ai.config import ConfigError, load_config

        try:
            config = load_config(args.config)
        except ConfigError as exc:
            parser.error(str(exc))
        cameras = [c for c in config.cameras if c.enabled]
        if args.count is not None:
            cameras = cameras[: args.count]
        if not cameras or any(c.source.type != "file" for c in cameras):
            parser.error("Replay config requires enabled file-source cameras")
        args.inputs = [(c.id, c.source.uri) for c in cameras]
    else:
        args.inputs = [
            (f"cam{i}", args.input)
            for i in range(args.start_index, args.start_index + (args.count or 4))
        ]
    if any(not Path(source).is_file() for _, source in args.inputs):
        parser.error("Every input video must exist")
    logging.basicConfig(level=logging.INFO)
    if args.watermark:
        return replay_watermarked(args)
    children = {}
    try:
        while True:
            for camera_id, source in args.inputs:
                process = children.get(camera_id)
                if process and process.poll() is None:
                    continue
                if process:
                    process.wait()
                children[camera_id] = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-re",
                        "-stream_loop",
                        "-1",
                        "-i",
                        source,
                        *(["-vf", f"fps={args.fps}"] if args.fps else []),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-tune",
                        "zerolatency",
                        "-threads",
                        "1",
                        "-pix_fmt",
                        "yuv420p",
                        "-g",
                        "15",
                        "-rtsp_transport",
                        "tcp",
                        "-f",
                        "rtsp",
                        f"{args.server}/input_{camera_id}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logging.info("Replay started camera=%s pid=%d", camera_id, children[camera_id].pid)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for process in children.values():
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
