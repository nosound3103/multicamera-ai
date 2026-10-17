import threading
from types import SimpleNamespace

import numpy as np

from multicamera_ai.events.recorder import AsyncEventRecorder


def test_bounded_queue_waits_without_losing_events_and_drains_on_stop():
    entered, release = threading.Event(), threading.Event()
    saved = []

    def record(event, frame):
        entered.set()
        release.wait(3)
        saved.append(frame.copy())

    recorder = AsyncEventRecorder(SimpleNamespace(record=record), capacity=1)
    event = SimpleNamespace(camera_id="cam1")
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert recorder.record(event, image)
    assert entered.wait(1)
    assert recorder.record(event, image)
    producer = threading.Thread(target=recorder.record, args=(event, image))
    producer.start()
    producer.join(0.1)
    assert producer.is_alive()
    image[:] = 255
    release.set()
    producer.join(2)
    assert not producer.is_alive()
    assert recorder.stop(2)
    assert recorder.status() == {"saved": 3, "failed": 0, "dropped": 0, "pending": 0}
    assert all(not frame.any() for frame in saved)


def test_disk_failure_is_reported_without_killing_writer():
    def fails(event, frame):
        raise OSError("disk full")

    recorder = AsyncEventRecorder(SimpleNamespace(record=fails), capacity=2)
    assert recorder.record(SimpleNamespace(camera_id="cam1"), np.zeros((1, 1, 3)))
    assert recorder.stop(2)
    assert recorder.status()["failed"] == 1


def test_sqlite_busy_retries_same_event_without_duplicate():
    import sqlite3

    attempts = []

    def record(event, frame):
        attempts.append(event)
        if len(attempts) < 3:
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc

    recorder = AsyncEventRecorder(SimpleNamespace(record=record))
    event = SimpleNamespace(camera_id="cam1")
    assert recorder.record(event, np.zeros((1, 1, 3)))
    assert recorder.stop(2)
    assert attempts == [event] * 3
    assert recorder.status() == {"saved": 1, "failed": 0, "dropped": 0, "pending": 0}


def test_real_sqlite_lock_recovers_with_one_row_and_snapshot(tmp_path, monkeypatch):
    import sqlite3

    from multicamera_ai.events.models import Event, EventType
    from multicamera_ai.events.recorder import EventRecorder
    from multicamera_ai.events.snapshot import SnapshotWriter
    from multicamera_ai.events.storage import SQLiteEventStore

    store = SQLiteEventStore(tmp_path / "events.db")
    busy = threading.Event()
    insert = store.insert

    def observed_insert(event):
        try:
            insert(event)
        except sqlite3.OperationalError:
            busy.set()
            raise

    monkeypatch.setattr(store, "insert", observed_insert)
    event = Event(camera_id="cam1", track_id=1, event_type=EventType.ZONE_ENTRY, region_id="zone")
    blocker = sqlite3.connect(store.db_path)
    recorder = AsyncEventRecorder(EventRecorder(store, SnapshotWriter(tmp_path / "snapshots")))
    try:
        blocker.execute("BEGIN IMMEDIATE")
        assert recorder.record(event, np.zeros((10, 10, 3), dtype=np.uint8))
        assert busy.wait(3)
        blocker.rollback()
        assert recorder.stop(3)
        assert recorder.status() == {"saved": 1, "failed": 0, "dropped": 0, "pending": 0}
        assert [row.event_id for row in store.list_events()] == [event.event_id]
        assert len(list((tmp_path / "snapshots").rglob("*.jpg"))) == 1
        assert not list(tmp_path.rglob("*.tmp"))
    finally:
        blocker.close()
        recorder.stop(3)
