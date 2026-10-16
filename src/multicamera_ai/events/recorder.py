from typing import Any

from multicamera_ai.events.models import Event, RecordedEvent
from multicamera_ai.events.snapshot import SnapshotWriter
from multicamera_ai.events.storage import SQLiteEventStore


class EventRecorder:
    def __init__(self, storage: SQLiteEventStore, snapshots: SnapshotWriter) -> None:
        self.storage = storage
        self.snapshots = snapshots

    def record(self, event: Event, annotated_frame: Any) -> RecordedEvent:
        """Save JPEG then SQLite row; remove JPEG if database insert fails.

        Filesystem and SQLite cannot share a single atomic transaction. This
        compensates for ordinary insert failures, but an abrupt process crash
        may still leave an orphan JPEG for later cleanup.
        """
        path = self.snapshots.save(event, annotated_frame)
        try:
            recorded = RecordedEvent.model_validate(
                {**event.model_dump(), "snapshot_path": str(path)}
            )
            self.storage.insert(recorded)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return recorded


class AsyncEventRecorder:
    """Bounded disk writer. Overflow/failures are visible; inference never waits on disk."""

    def __init__(self, recorder: EventRecorder, capacity: int = 8) -> None:
        import queue
        import threading

        self.recorder = recorder
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.saved = 0
        self.failed = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self._thread.start()

    def record(self, event: Event, annotated_frame: Any) -> bool:
        import logging
        import queue

        try:
            self._queue.put_nowait((event, annotated_frame.copy()))
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            logging.getLogger(__name__).error("Event queue full camera=%s", event.camera_id)
            return False

    def status(self) -> dict:
        with self._lock:
            return {
                "saved": self.saved,
                "failed": self.failed,
                "dropped": self.dropped,
                "pending": self._queue.unfinished_tasks,
            }

    def stop(self, timeout: float = 12.0) -> bool:
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        import logging
        import queue

        while not self._stop.is_set() or not self._queue.empty():
            try:
                event, frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.recorder.record(event, frame)
                with self._lock:
                    self.saved += 1
            except Exception as exc:
                with self._lock:
                    self.failed += 1
                logging.getLogger(__name__).error(
                    "Event persistence failed camera=%s error=%s",
                    event.camera_id,
                    type(exc).__name__,
                )
            finally:
                self._queue.task_done()
