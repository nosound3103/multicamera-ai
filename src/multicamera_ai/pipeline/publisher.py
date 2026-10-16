from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from datetime import UTC, datetime

import cv2

from multicamera_ai.telemetry import stamp_frame

logger = logging.getLogger(__name__)


class RTSPPublisher:
    """Latest-only encoder input with a watchdog that kills blocked FFmpeg writes."""

    def __init__(self, camera_id, url, source, config):
        self.camera_id, self.url, self.source, self.config = camera_id, url, source, config
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process = None
        self._writing_since = None
        self._times = deque(maxlen=100)
        self._frames = 0
        self._restarts = 0
        self._latency_ms = 0.0
        self._last_error = None
        self._thread = threading.Thread(target=self._run, name=f"output-{camera_id}", daemon=True)
        self._watchdog = threading.Thread(
            target=self._watch, name=f"watch-{camera_id}", daemon=True
        )

    def start(self):
        self._watchdog.start()
        self._thread.start()

    def status(self):
        with self._lock:
            fps = 0.0
            if len(self._times) > 1 and time.monotonic() - self._times[-1] < 2:
                fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
            return {
                "frames": self._frames,
                "input_fps": fps,
                "capture_to_pipe_ms": self._latency_ms,
                "restarts": self._restarts,
                "last_error": self._last_error,
                "ffmpeg_pid": self._process.pid if self._process else None,
            }

    def _spawn(self):
        c = self.config
        command = [
            c.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{c.width}x{c.height}",
            "-framerate",
            str(c.fps),
            "-i",
            "pipe:0",
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
            str(c.fps),
            "-bf",
            "0",
            "-rtsp_transport",
            "tcp",
            "-f",
            "rtsp",
            self.url,
        ]
        # URLs can contain credentials. Do not forward FFmpeg stderr to logs.
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _kill(self):
        with self._lock:
            process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass

    def _watch(self):
        while not self._stop.wait(0.1):
            with self._lock:
                started = self._writing_since
            if started is not None and time.monotonic() - started > self.config.write_timeout:
                logger.error("FFmpeg write timeout camera=%s", self.camera_id)
                self._kill()

    def stop(self, timeout=5.0):
        self._stop.set()
        self._kill()
        self._thread.join(timeout)
        self._watchdog.join(timeout)
        return not self._thread.is_alive() and not self._watchdog.is_alive()

    def _dispose(self):
        self._kill()
        with self._lock:
            process, self._process = self._process, None
            self._writing_since = None
        if process is not None:
            process.wait(timeout=3)
            if process.stdin:
                process.stdin.close()

    def _run(self):
        sequence = 0
        deadline = 0.0
        try:
            while not self._stop.is_set():
                packet = self.source.wait_for_frame(after_sequence=sequence, timeout=0.2)
                if packet is None:
                    continue
                sequence = packet.sequence
                if (
                    datetime.now(UTC) - packet.captured_at
                ).total_seconds() > self.config.max_frame_age:
                    continue
                if self._stop.wait(max(0, deadline - time.monotonic())):
                    break
                # Refresh after pacing, so a slower output never builds a FIFO.
                packet = self.source.latest() or packet
                sequence = packet.sequence
                frame = cv2.resize(packet.annotated, (self.config.width, self.config.height))
                if self.config.watermark:
                    stamp_frame(frame, packet.captured_at, sequence)
                try:
                    with self._lock:
                        process = self._process
                    if process is None or process.poll() is not None:
                        self._dispose()
                        process = self._spawn()
                        with self._lock:
                            self._process = process
                    with self._lock:
                        self._writing_since = time.monotonic()
                    data = memoryview(frame).cast("B")
                    while data and not self._stop.is_set():
                        written = process.stdin.write(data)
                        if not written:
                            raise BrokenPipeError
                        data = data[written:]
                    with self._lock:
                        self._writing_since = None
                        self._frames += 1
                        self._times.append(time.monotonic())
                        self._latency_ms = (
                            datetime.now(UTC) - packet.captured_at
                        ).total_seconds() * 1000
                        self._last_error = None
                    deadline = time.monotonic() + 1 / self.config.fps
                except (OSError, ValueError) as exc:
                    with self._lock:
                        self._last_error = type(exc).__name__
                        self._restarts += 1
                    logger.warning(
                        "FFmpeg restarting camera=%s error=%s", self.camera_id, type(exc).__name__
                    )
                    self._dispose()
                    self._stop.wait(0.5)
        finally:
            self._dispose()
