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
