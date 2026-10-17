"""SQLite event storage tests (no OpenCV or YOLO needed)."""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from multicamera_ai.events import Event, EventType, RecordedEvent, SQLiteEventStore


def _record(*, camera="cam1", minute=0, track_id=1):
    event = Event(
        camera_id=camera,
        track_id=track_id,
        event_type=EventType.LINE_CROSSING,
        region_id="counting_line",
        direction="IN",
        occurred_at=datetime(2026, 10, 15, 8, minute, tzinfo=UTC),
    )
    return RecordedEvent.model_validate(
        {**event.model_dump(), "snapshot_path": f"storage/snapshots/{camera}/{event.event_id}.jpg"}
    )


def test_sqlite_insert_and_read_back(tmp_path):
    storage = SQLiteEventStore(tmp_path / "nested" / "events.db")
    record = _record()
    storage.insert(record)

    rows = storage.list_events()
    assert len(rows) == 1
    assert rows[0] == record
    assert (tmp_path / "nested" / "events.db").exists()


def test_duplicate_event_id_is_rejected(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    record = _record()
    storage.insert(record)
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert(record)
    assert len(storage.list_events()) == 1


def test_filter_camera_and_utc_time(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    records = [
        _record(camera="cam1", minute=0),
        _record(camera="cam1", minute=10),
        _record(camera="cam2", minute=15),
    ]
    for rec in records:
        storage.insert(rec)

    vietnam_tz = timezone(timedelta(hours=7))
    start = datetime(2026, 10, 15, 15, 5, tzinfo=vietnam_tz)
    end = datetime(2026, 10, 15, 15, 12, tzinfo=vietnam_tz)
    rows = storage.list_events(camera_id="cam1", start=start, end=end)
    assert [row.event_id for row in rows] == [records[1].event_id]
    assert [row.event_id for row in storage.list_events(camera_id="cam2")] == [records[2].event_id]


def test_newest_first_and_limit(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    for n in range(4):
        storage.insert(_record(minute=n))
    results = storage.list_events(limit=2)
    assert [rec.occurred_at.minute for rec in results] == [3, 2]


@pytest.mark.parametrize("invalid", [0, -1, 1001, True, 3.5, "10"])
def test_invalid_limit(tmp_path, invalid):
    storage = SQLiteEventStore(tmp_path / "events.db")
    with pytest.raises(ValueError, match="limit"):
        storage.list_events(limit=invalid)


def test_naive_filter_timestamp_is_rejected(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    with pytest.raises(ValueError, match="timezone"):
        storage.list_events(start=datetime(2026, 10, 15, 8))


def test_reversed_range_is_rejected(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    with pytest.raises(ValueError, match="start"):
        storage.list_events(
            start=datetime(2026, 10, 16, tzinfo=UTC),
            end=datetime(2026, 10, 15, tzinfo=UTC),
        )


def test_sql_injection_is_treated_as_camera_id(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    storage.insert(_record())
    assert storage.list_events(camera_id="cam1' OR 1=1 --") == []
    assert len(storage.list_events()) == 1
