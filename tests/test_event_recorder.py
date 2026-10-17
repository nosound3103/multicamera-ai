"""Snapshot, SQLite, and frame-level EventEngine persistence integration."""

import sqlite3

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from multicamera_ai.events import (  # noqa: E402
    CountLine,
    Event,
    EventEngine,
    EventRecorder,
    EventType,
    RestrictedZone,
    SnapshotWriter,
    SQLiteEventStore,
    TrackedPoint,
)


def _zone_event(camera="cam1"):
    return Event(
        camera_id=camera,
        track_id=7,
        event_type=EventType.ZONE_ENTRY,
        region_id="restricted_area",
    )


def _frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (80, 90), (200, 200, 200), -1)
    return frame


def test_record_creates_valid_jpeg_and_sqlite_row(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    event = _zone_event()
    recorded = recorder.record(event, _frame())

    assert recorded.event_id == event.event_id
    assert recorded.snapshot_path.endswith(".jpg")
    assert cv2.imread(recorded.snapshot_path).shape == (120, 160, 3)
    assert storage.list_events() == [recorded]
    assert len(list((tmp_path / "snapshots").rglob("*.jpg"))) == 1


def test_duplicate_event_does_not_overwrite_existing_jpeg(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    event = _zone_event()
    recorded = recorder.record(event, _frame())
    with pytest.raises(FileExistsError):
        recorder.record(event, _frame())
    assert len(storage.list_events()) == 1
    assert (tmp_path / "snapshots" / "cam1" / f"{event.event_id.hex}.jpg").exists()
    assert recorded.snapshot_path


def test_sqlite_failure_removes_newly_written_snapshot(tmp_path):
    class FailingStorage(SQLiteEventStore):
        def insert(self, event):
            raise sqlite3.OperationalError("simulated I/O failure")

    storage = FailingStorage(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        recorder.record(_zone_event(), _frame())
    assert list((tmp_path / "snapshots").rglob("*.jpg")) == []


def test_invalid_frame_is_not_inserted(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    with pytest.raises(ValueError, match="empty"):
        recorder.record(_zone_event(), None)
    assert storage.list_events() == []


def test_camera_id_cannot_traverse_snapshot_directory(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    recorded = recorder.record(_zone_event(camera="../../private"), _frame())
    path = __import__("pathlib").Path(recorded.snapshot_path)
    assert path.parent.parent == tmp_path / "snapshots"
    assert not (tmp_path / "private").exists()


def test_engine_events_persist_once_and_reentry_creates_new_event(tmp_path):
    storage = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(storage, SnapshotWriter(tmp_path / "snapshots"))
    engine = EventEngine("cam1", zone_confirm_frames=2, zone_release_frames=2)
    lines = [CountLine("counting_line", (0, 50), (60, 50))]
    zones = [RestrictedZone("restricted_area", ((85, 15), (130, 15), (130, 45), (85, 45)))]
    frames = [
        [(1, (20, 25)), (2, (100, 60))],
        [(1, (20, 75)), (2, (100, 30))],
        [(1, (20, 75)), (2, (100, 30))],
        [(2, (100, 60))],
        [(2, (100, 60))],
        [(2, (100, 30))],
        [(2, (100, 30))],
    ]
    all_events = []
    for i, items in enumerate(frames, start=1):
        events = engine.process_frame(
            frame_index=i,
            detections=[TrackedPoint(track_id=tid, anchor=anchor) for tid, anchor in items],
            lines=lines,
            zones=zones,
        )
        all_events.extend(recorder.record(event, _frame()) for event in events)

    assert [e.event_type for e in all_events] == [
        EventType.LINE_CROSSING,
        EventType.ZONE_ENTRY,
        EventType.ZONE_ENTRY,
    ]
    assert all_events[0].direction == "IN"
    assert len(storage.list_events()) == 3
    assert len(list((tmp_path / "snapshots").rglob("*.jpg"))) == 3


def test_snapshot_quality_validation():
    with pytest.raises(ValueError, match="quality"):
        SnapshotWriter(__import__("pathlib").Path("snapshots"), jpeg_quality=0)
