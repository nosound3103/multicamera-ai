"""Replay one file as N real-time RTSP inputs, restarting failed publishers."""

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
        for i in range(args.start_index, args.start_index + args.count):
            capture = CameraWorker(f"replay{i}", args.input, is_file=True)
            captures.append(capture)
            capture.start()
            publisher = RTSPPublisher(
                f"replay{i}",
                f"{args.server}/input_cam{i}",
                Adapter(capture),
                StreamingConfig(watermark=True, fps=25),
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--server", default="rtsp://127.0.0.1:8554")
    parser.add_argument(
        "--watermark", action="store_true", help="Embed capture timestamps for end-to-end probes"
    )
    args = parser.parse_args()
    if not Path(args.input).is_file() or min(args.count, args.start_index) < 1:
        parser.error("Input must exist and count must be positive")
    logging.basicConfig(level=logging.INFO)
    if args.watermark:
        return replay_watermarked(args)
    children = {}
    try:
        while True:
            for i in range(args.start_index, args.start_index + args.count):
                process = children.get(i)
                if process and process.poll() is None:
                    continue
                if process:
                    process.wait()
                children[i] = subprocess.Popen(
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
                        args.input,
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
                        f"{args.server}/input_cam{i}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logging.info("Replay started camera=cam%d pid=%d", i, children[i].pid)
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
