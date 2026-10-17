from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from multicamera_ai.cameras.worker import CameraState, FramePacket
from multicamera_ai.events import (
    EventEngine,
    EventType,
    TrackedPoint,
)
from multicamera_ai.vision.geometry import bottom_center

if TYPE_CHECKING:
    from multicamera_ai.events.recorder import EventRecorder

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    """Implemented by CameraWorker; fakeable without RTSP or webcams."""

    def wait_for_frame(
        self, *, after_sequence: int = 0, timeout: float | None = 1.0
    ) -> FramePacket | None: ...

    def status(self) -> Any: ...


class TrackingModel(Protocol):
    def track(self, frame: Any, **kwargs: Any) -> Any: ...


ModelFactory = Callable[[str], TrackingModel]


def _make_yolo(weights: str, cpu_threads: int = 1) -> TrackingModel:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(cpu_threads)
    model = YOLO(weights)

    def limit_cpu_threads(predictor):
        # CUDA inference still has CPU tensor preprocessing. Bound both backends
        # so concurrent cameras do not each saturate the host's CPU cores.
        if torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)

    model.add_callback("on_predict_start", limit_cpu_threads)
    return model


class AIState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ENDED = "ended"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AIFrame:
    camera_id: str
    sequence: int
    session_id: int
    captured_at: datetime
    annotated: Any  # ndarray, owned by AIWorker; consumers should copy before mutating
    event_count: int


@dataclass(frozen=True, slots=True)
class AIStatus:
    camera_id: str
    state: AIState
    processed_frames: int
    skipped_capture_frames: int
    processing_fps: float
    inference_ms: float
    capture_to_ai_ms: float
    session_id: int | None
    last_processed_at: datetime | None
    counts: dict[str, int]
    saved_events: int
    last_error: str | None


class AIWorker:
    """Consume newest frames and persist geometry events for one camera.

    The capture worker must already be running. This worker does not own or
    stop its capture source; a later PipelineManager will coordinate both.
    """

    def __init__(
        self,
        *,
        camera: Any,
        model_config: Any,
        source: FrameSource,
        recorder: EventRecorder,
        model_factory: ModelFactory = _make_yolo,
        poll_timeout: float = 0.2,
    ) -> None:
        if not camera.id:
            raise ValueError("camera.id cannot be empty")
        if not math.isfinite(poll_timeout) or poll_timeout <= 0:
            raise ValueError("poll_timeout must be positive and finite")

        self.camera = camera
        self.model_config = model_config
        self.source = source
        self.recorder = recorder
        self.model_factory = model_factory
        self.poll_timeout = poll_timeout

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = AIState.STOPPED
        self._latest: AIFrame | None = None
        self._last_error: str | None = None
        self._last_sequence: int | None = None
        self._session_id: int | None = None
        self._session_frame_index = 0
        self._model: TrackingModel | None = None
        self._engine: EventEngine | None = None
        self._size: tuple[int, int] | None = None
        self._lines: Any = ()
        self._zones: Any = ()
        self._processed_frames = 0
        self._skipped_frames = 0
        self._saved_events = 0
        self._counts = {"IN": 0, "OUT": 0}
        self._capture_to_ai_ms = 0.0
        self._inference_ms = 0.0
        self._last_processed_at: datetime | None = None
        self._frame_times: deque[float] = deque(maxlen=50)

    def start(self) -> None:
        """Start this AI consumer; capture source must be started separately."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._state = AIState.STARTING
            self._last_error = None
            self._latest = None
            self._last_sequence = None
            self._session_id = None
            self._session_frame_index = 0
            self._model = None
            self._engine = None
            self._size = None
            self._lines = ()
            self._zones = ()
            self._frame_times.clear()
            self._counts = {"IN": 0, "OUT": 0}
            self._processed_frames = 0
            self._skipped_frames = 0
            self._saved_events = 0
            self._last_processed_at = None
            self._inference_ms = 0.0
            self._thread = threading.Thread(
                target=self._run,
                name=f"ai-{self.camera.id}",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()

    def stop(self, timeout: float = 5.0) -> bool:
        """Request stop; ongoing YOLO inference cannot be interrupted midway."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and >= 0")
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def latest(self) -> AIFrame | None:
        with self._condition:
            return self._latest

    def wait_for_frame(self, *, after_sequence: int = 0, timeout: float = 1.0) -> AIFrame | None:
        """Return the newest annotated frame; never queue old outputs."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    (self._latest is not None and self._latest.sequence > after_sequence)
                    or self._state in (AIState.STOPPED, AIState.ENDED, AIState.ERROR)
                ),
                timeout=timeout,
            )
            packet = self._latest
            if packet is not None and packet.sequence > after_sequence:
                return packet
            return None

    def status(self) -> AIStatus:
        with self._condition:
            fps = 0.0
            if (
                self._state == AIState.RUNNING
                and len(self._frame_times) >= 2
                and time.perf_counter() - self._frame_times[-1] < 2.0
            ):
                elapsed = self._frame_times[-1] - self._frame_times[0]
                if elapsed > 0:
                    fps = (len(self._frame_times) - 1) / elapsed
            return AIStatus(
                camera_id=self.camera.id,
                state=self._state,
                processed_frames=self._processed_frames,
                skipped_capture_frames=self._skipped_frames,
                processing_fps=fps,
                inference_ms=self._inference_ms,
                capture_to_ai_ms=self._capture_to_ai_ms,
                session_id=self._session_id,
                last_processed_at=self._last_processed_at,
                counts=self._counts.copy(),
                saved_events=self._saved_events,
                last_error=self._last_error,
            )

    def process_packet(self, packet: FramePacket) -> AIFrame:
        """Synchronous adapter for one packet; useful for deterministic tests.

        Must only be called by one processing thread at a time. The consumer
        uses capture sequence gaps to report how many frames it skipped.
        """
        from multicamera_ai.vision.overlay import draw_hud, draw_regions, project_regions

        if self._last_sequence is not None and packet.sequence <= self._last_sequence:
            raise ValueError("Capture sequence must strictly increase")
        if self._last_sequence is not None:
            self._skipped_frames += packet.sequence - self._last_sequence - 1
        self._last_sequence = packet.sequence

        # A new session means new ByteTrack ID namespace. Rebuild YOLO rather
        # than reusing an Ultralytics predictor/tracker from the old stream.
        if packet.session_id != self._session_id:
            if self.model_factory is _make_yolo:
                self._model = _make_yolo(self.model_config.weights, self.model_config.cpu_threads)
                from multicamera_ai.config.models import validate_class_ids

                validate_class_ids(self.model_config, self._model.names)
            else:
                self._model = self.model_factory(self.model_config.weights)
            self._engine = EventEngine(self.camera.id)
            self._session_id = packet.session_id
            self._session_frame_index = 0
            self._size = None
            self._lines, self._zones = (), ()
            logger.info("AI camera %s session=%d", self.camera.id, packet.session_id)

        assert self._model is not None
        assert self._engine is not None
        self._session_frame_index += 1

        # Capture buffers are shared. Tracking/drawing must not mutate them.
        frame = packet.frame.copy()
        started = time.perf_counter()
        result = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.model_config.confidence,
            imgsz=self.model_config.imgsz,
            classes=self.model_config.class_ids,
            device=self.model_config.device,
            verbose=False,
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000
        annotated = result.plot()
        height, width = annotated.shape[:2]
        if self._size != (width, height):
            self._lines, self._zones = project_regions(self.camera, width, height)
            self._size = (width, height)

        tracked = []
        visible = []
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                anchor = bottom_center(*xyxy)
                tracked.append(TrackedPoint(track_id=track_id, anchor=anchor))
                visible.append((track_id, anchor, xyxy))

        events = self._engine.process_frame(
            frame_index=self._session_frame_index,
            detections=tracked,
            lines=self._lines,
            zones=self._zones,
        )

        draw_regions(annotated, self._lines, self._zones)
        inside_ids = self._engine.inside_track_ids
        import cv2

        for track_id, anchor, (x1, y1, x2, y2) in visible:
            cv2.circle(
                annotated,
                (round(anchor[0]), round(anchor[1])),
                4,
                (255, 90, 0),
                -1,
                cv2.LINE_AA,
            )
            if track_id in inside_ids:
                cv2.rectangle(
                    annotated,
                    (round(x1), round(y1)),
                    (round(x2), round(y2)),
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        # Snapshots show regions and bounding boxes. Counters reflect only
        # accepted events (async persistence status is reported separately).
        for event in events:
            # Capture timestamp, not time when inference happened to complete.
            event = event.model_copy(update={"occurred_at": packet.captured_at})
            recorded = self.recorder.record(event, annotated)
            if recorded is False:
                continue
            with self._condition:
                self._saved_events += 1
                if event.event_type == EventType.LINE_CROSSING:
                    assert event.direction is not None
                    self._counts[event.direction] += 1

        draw_hud(
            annotated,
            self.camera.id,
            self._counts,
            len(inside_ids),
            fps=self.status().processing_fps,
        )
        output = AIFrame(
            camera_id=self.camera.id,
            sequence=packet.sequence,
            session_id=packet.session_id,
            captured_at=packet.captured_at,
            annotated=annotated,
            event_count=len(events),
        )
        with self._condition:
            self._state = AIState.RUNNING
            self._processed_frames += 1
            self._capture_to_ai_ms = (datetime.now(UTC) - packet.captured_at).total_seconds() * 1000
            self._inference_ms = inference_ms
            self._last_processed_at = datetime.now(UTC)
            self._frame_times.append(time.perf_counter())
            self._latest = output
            self._condition.notify_all()
        return output

    def _run(self) -> None:
        last_sequence = 0
        try:
            while not self._stop_event.is_set():
                packet = self.source.wait_for_frame(
                    after_sequence=last_sequence, timeout=self.poll_timeout
                )
                if packet is not None:
                    last_sequence = packet.sequence
                    self.process_packet(packet)
                    continue

                state = self.source.status().state
                if state == CameraState.ENDED:
                    with self._condition:
                        self._state = AIState.ENDED
                        self._condition.notify_all()
                    return
                if state == CameraState.ERROR:
                    with self._condition:
                        self._state = AIState.ERROR
                        self._last_error = "Capture worker entered error state"
                        self._condition.notify_all()
                    return
                if state == CameraState.STOPPED:
                    with self._condition:
                        self._state = AIState.STOPPED
                        self._condition.notify_all()
                    return
                # CONNECTING/RECONNECTING: keep waiting for a fresh packet.
        except Exception as exc:
            # Exception strings may embed private RTSP URLs; log type only.
            logger.error(
                "AI processing failed for camera %s (%s)",
                self.camera.id,
                type(exc).__name__,
            )
            with self._condition:
                self._state = AIState.ERROR
                self._last_error = type(exc).__name__
                self._condition.notify_all()
        finally:
            if self._stop_event.is_set():
                with self._condition:
                    self._state = AIState.STOPPED
                    self._condition.notify_all()
