from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Capture(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV API

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def get(self, property_id: int) -> float: ...


CaptureFactory = Callable[[str | int], Capture]


class CameraState(StrEnum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    session_id: int
    captured_at: datetime
    frame: Any  # OpenCV numpy.ndarray; optional vision dependency


@dataclass(frozen=True, slots=True)
class CameraStatus:
    camera_id: str
    state: CameraState
    frames_received: int
    capture_fps: float
    reconnect_attempts: int
    session_id: int
    last_frame_at: datetime | None
    last_error: str | None


def opencv_capture_factory(source: str | int) -> Capture:
    """Construct VideoCapture with bounded FFmpeg RTSP open/read operations.

    OpenCV must have an FFmpeg-enabled build for its RTSP timeout properties.
    Some backends may not honor the timeouts; stop() reports a timed-out join.
    """
    import cv2

    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                3000,
            ],
        )
    return cv2.VideoCapture(source)


class CameraWorker:
    """One capture thread; overwrites a single latest-frame slot.

    Live sources (RTSP/webcam) reconnect on failures. Video files end on EOF
    without silently looping. A successful reopen increments session_id, so
    downstream ByteTrack/EventEngine can reset their per-session state.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        *,
        is_file: bool = False,
        timestamp_watermark: bool = False,
        reconnect_delay: float = 1.0,
        file_realtime: bool = True,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", camera_id):
            raise ValueError("camera_id must match [a-z][a-z0-9_-]*")
        if isinstance(source, bool) or not isinstance(source, (str, int)):
            raise ValueError("source must be a URL/path or webcam index")
        if isinstance(source, str) and not source.strip():
            raise ValueError("source must not be empty")
        if not math.isfinite(reconnect_delay) or reconnect_delay <= 0:
            raise ValueError("reconnect_delay must be positive and finite")
        if isinstance(source, int) and source < 0:
            raise ValueError("webcam index must be non-negative")
        if is_file and isinstance(source, int):
            raise ValueError("File source must be a path")

        self.camera_id = camera_id
        self.source = source
        self.timestamp_watermark = timestamp_watermark
        self.is_file = is_file
        self.reconnect_delay = reconnect_delay
        self.file_realtime = file_realtime
        self._validate_file = capture_factory is None
        self._capture_factory = capture_factory or opencv_capture_factory
        self._lifecycle_lock = threading.Lock()

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = CameraState.STOPPED
        self._latest: FramePacket | None = None
        self._sequence = 0
        self._session_id = 0
        self._reconnect_attempts = 0
        self._last_frame_at: datetime | None = None
        self._recent_frame_times: deque[float] = deque(maxlen=60)
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            # Terminal status can be published just before the thread releases capture.
            with self._condition:
                previous = self._thread
                terminal = self._state in (CameraState.ENDED, CameraState.ERROR)
            if terminal and previous is not None:
                previous.join()
            self._start()

    def _start(self) -> None:
        """Start/restart worker. No-op if already active."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._latest = None
            self._last_error = None
            self._last_frame_at = None
            self._recent_frame_times.clear()
            self._state = CameraState.CONNECTING
            self._thread = threading.Thread(
                target=self._run,
                name=f"capture-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lifecycle_lock:
            return self._stop(timeout)

    def _stop(self, timeout: float) -> bool:
        """Signal stop; return False if backend is still blocked after timeout."""
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and >= 0")
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def get_latest(self) -> FramePacket | None:
        """Return latest packet instantly, not a FIFO queue (no deep copy)."""
        with self._condition:
            return self._latest

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        timeout: float | None = 1.0,
    ) -> FramePacket | None:
        """Wait for sequence > after_sequence; skip any intermediate frames."""
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and >= 0 or None")
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    (self._latest is not None and self._latest.sequence > after_sequence)
                    or self._state in (CameraState.STOPPED, CameraState.ENDED, CameraState.ERROR)
                ),
                timeout=timeout,
            )
            if not ready:
                return None
            packet = self._latest
            return packet if packet is not None and packet.sequence > after_sequence else None

    def status(self) -> CameraStatus:
        with self._condition:
            fps = 0.0
            if self._state == CameraState.RUNNING and len(self._recent_frame_times) >= 2:
                span = self._recent_frame_times[-1] - self._recent_frame_times[0]
                if span > 0 and time.monotonic() - self._recent_frame_times[-1] < 2.0:
                    fps = (len(self._recent_frame_times) - 1) / span
            return CameraStatus(
                camera_id=self.camera_id,
                state=self._state,
                frames_received=self._sequence,
                capture_fps=fps,
                reconnect_attempts=self._reconnect_attempts,
                session_id=self._session_id,
                last_frame_at=self._last_frame_at,
                last_error=self._last_error,
            )

    def _set_state(self, state: CameraState, error: str | None = None) -> None:
        with self._condition:
            self._state = state
            self._last_error = error
            if state in (CameraState.RECONNECTING, CameraState.ERROR):
                self._latest = None  # Prevent using a stale live frame
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            attempt = 0
            while not self._stop_event.is_set():
                self._set_state(
                    CameraState.CONNECTING if attempt == 0 else CameraState.RECONNECTING
                )
                capture: Capture | None = None
                opened = False
                try:
                    if self.is_file and self._validate_file and not Path(self.source).is_file():
                        self._set_state(CameraState.ERROR, f"File not found: {self.source}")
                        return
                    capture = self._capture_factory(self.source)
                    opened = capture.isOpened()
                    if not opened:
                        raise OSError("capture could not be opened")

                    with self._condition:
                        self._session_id += 1
                        session_id = self._session_id
                        self._recent_frame_times.clear()
                    self._set_state(CameraState.RUNNING)
                    logger.info("Camera %s connected (session=%d)", self.camera_id, session_id)
                    attempt = 0
                    frame_number = 0
                    frame_period = self._file_frame_period(capture) if self.is_file else None
                    deadline = time.monotonic()

                    while not self._stop_event.is_set():
                        if frame_number and frame_period is not None:
                            deadline += frame_period
                            if self._stop_event.wait(max(0.0, deadline - time.monotonic())):
                                break
                            # Prevent unbounded catch-up after a scheduling pause.
                            deadline = max(deadline, time.monotonic() - frame_period)

                        ok, frame = capture.read()
                        if not ok or frame is None:
                            if self.is_file:
                                self._set_state(CameraState.ENDED)
                                return
                            raise OSError("capture read failed")

                        captured_at = datetime.now(UTC)
                        if self.timestamp_watermark:
                            from multicamera_ai.telemetry import read_stamp

                            stamp = read_stamp(frame)
                            if stamp is None:
                                raise ValueError("Source timestamp watermark missing or corrupt")
                            captured_at = stamp[0]
                        with self._condition:
                            self._sequence += 1
                            self._recent_frame_times.append(time.monotonic())
                            self._last_frame_at = captured_at
                            self._latest = FramePacket(
                                sequence=self._sequence,
                                session_id=session_id,
                                captured_at=captured_at,
                                frame=frame,
                            )
                            self._condition.notify_all()
                        frame_number += 1

                except Exception as exc:
                    # Avoid exposing RTSP URLs (which may contain credentials).
                    reason = f"{type(exc).__name__} during {'read' if opened else 'open'}"
                    logger.warning("Camera %s: %s", self.camera_id, reason)
                    if self.is_file:
                        self._set_state(CameraState.ERROR, reason)
                        return
                    self._set_state(CameraState.RECONNECTING, reason)
                    with self._condition:
                        self._reconnect_attempts += 1
                    attempt += 1
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            logger.warning("Camera %s: capture release failed", self.camera_id)
                if self._stop_event.wait(self.reconnect_delay):
                    break
        finally:
            if self._stop_event.is_set():
                self._set_state(CameraState.STOPPED)
            with self._condition:
                self._condition.notify_all()

    def _file_frame_period(self, capture: Capture) -> float | None:
        if not self.file_realtime:
            return None
        try:
            # cv2.CAP_PROP_FPS; constant 5 even without cv2 installed
            fps = capture.get(5)
        except (AttributeError, TypeError, ValueError):
            return 1.0 / 25.0
        if not math.isfinite(fps) or fps <= 0:
            return 1.0 / 25.0
        return 1.0 / fps
