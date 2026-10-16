import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from multicamera_ai.events.models import RecordedEvent


def _utc_text(value: datetime) -> str:
    """Stable lexicographically sortable UTC time for SQLite TEXT columns."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must have timezone information")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


class SQLiteEventStore:
    """Store and query events; use one database for all cameras."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        occurred_at TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        region_id TEXT NOT NULL,
                        direction TEXT,
                        snapshot_path TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_camera_time
                    ON events (camera_id, occurred_at)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_time
                    ON events (occurred_at)
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def insert(self, event: RecordedEvent) -> None:
        """Insert one event transactionally; duplicate event_id raises IntegrityError."""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO events (
                        event_id, occurred_at, camera_id, track_id,
                        event_type, region_id, direction, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.event_id),
                        _utc_text(event.occurred_at),
                        event.camera_id,
                        event.track_id,
                        event.event_type.value,
                        event.region_id,
                        event.direction,
                        event.snapshot_path,
                    ),
                )

    def list_events(
        self,
        *,
        camera_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[RecordedEvent]:
        """Return newest first, optionally filtered by camera and UTC time range."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

        start_text = _utc_text(start) if start is not None else None
        end_text = _utc_text(end) if end is not None else None
        if start_text is not None and end_text is not None and start_text > end_text:
            raise ValueError("start must be <= end")

        conditions: list[str] = []
        params: list[object] = []
        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if start_text is not None:
            conditions.append("occurred_at >= ?")
            params.append(start_text)
        if end_text is not None:
            conditions.append("occurred_at <= ?")
            params.append(end_text)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT event_id, occurred_at, camera_id, track_id,
                   event_type, region_id, direction, snapshot_path
            FROM events {where_clause}
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT ?
        """
        params.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [RecordedEvent.model_validate(dict(row)) for row in rows]
