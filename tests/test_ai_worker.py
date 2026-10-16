"""Tests run without ultralytics or real RTSP by injecting a fake YOLO model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from multicamera_ai.cameras.worker import CameraState, FramePacket
from multicamera_ai.events import EventRecorder, SnapshotWriter, SQLiteEventStore
from multicamera_ai.pipeline import AIState, AIWorker


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == 0
        return SimpleNamespace(tolist=lambda: self.values)


class FakeBox:
    def __init__(self, track_id, anchor):
        x, y = anchor
        self.id = FakeValue(track_id)
        self.xyxy = FakeCoordinates([x - 5, y - 10, x + 5, y])


class FakeResult:
    def __init__(self, frame, boxes):
        self.frame = frame
        # Ultralytics result.boxes.id is non-None when tracks exist.
        self.boxes = FakeBoxCollection(boxes) if boxes else None

    def plot(self):
        return self.frame.copy()


class FakeBoxCollection:
    def __init__(self, boxes):
        self.id = object()
        self.boxes = boxes

    def __iter__(self):
        return iter(self.boxes)


class FakeYOLO:
    def __init__(self, weights, calls):
        self.weights = weights
        self.calls = calls

    def track(self, frame, **kwargs):
        self.calls.append((self, kwargs))
        code = int(frame[0, 0, 0])
        # 1: IN lane above line; 2: below line; 3: inside zone;
        # 4: OUT lane below line; 5: above OUT line; 0: no detections.
        anchors = {
            1: (20, 20),
            2: (20, 80),
            3: (28, 26),
            4: (75, 80),
            5: (75, 20),
        }
        boxes = [FakeBox(11, anchors[code])] if code in anchors else []
        return [FakeResult(frame, boxes)]


class ScriptedSource:
    def __init__(self, packets):
        self.packets = list(packets)

    def wait_for_frame(self, *, after_sequence=0, timeout=1.0):
        if self.packets:
            result = self.packets.pop(0)
            assert result.sequence > after_sequence
            return result
        return None

    def status(self):
        return SimpleNamespace(state=CameraState.ENDED)


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def make_camera():
    return SimpleNamespace(
        id="cam1",
        lines=[
            SimpleNamespace(id="in_entrance", start=point(0.1, 0.5), end=point(0.45, 0.5)),
            SimpleNamespace(id="out_entrance", start=point(0.55, 0.5), end=point(0.95, 0.5)),
        ],
        zones=[
            SimpleNamespace(
                id="restricted_area",
                points=[point(0.1, 0.1), point(0.4, 0.1), point(0.4, 0.4), point(0.1, 0.4)],
            )
        ],
    )


def packet(seq, code, session=1):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[0, 0, 0] = code
    return FramePacket(
        sequence=seq,
        session_id=session,
        captured_at=datetime(2026, 10, 16, 3, 0, seq % 60, tzinfo=UTC),
        frame=image,
    )


def make_worker(tmp_path: Path, packets=None):
    calls = []
    model_instances = []

    def factory(weights):
        model = FakeYOLO(weights, calls)
        model_instances.append(model)
        return model

    db = SQLiteEventStore(tmp_path / "events.db")
    recorder = EventRecorder(db, SnapshotWriter(tmp_path / "snapshots"))
    worker = AIWorker(
        camera=make_camera(),
        model_config=SimpleNamespace(
            weights="fake.pt", confidence=0.35, imgsz=640, class_ids=[0, 2], device="cpu"
        ),
        source=ScriptedSource(packets or []),
        recorder=recorder,
        model_factory=factory,
        poll_timeout=0.01,
    )
    return worker, db, calls, model_instances


def test_process_single_in_event_persists_sqlite_snapshot(tmp_path):
    worker, db, calls, models = make_worker(tmp_path)
    p1, p2 = packet(1, 1), packet(2, 2)
    worker.process_packet(p1)
    output = worker.process_packet(p2)

    rows = db.list_events(camera_id="cam1")
    assert len(rows) == 1
    assert (rows[0].region_id, rows[0].direction) == ("in_entrance", "IN")
    assert rows[0].occurred_at == p2.captured_at
    snapshot = Path(rows[0].snapshot_path)
    assert snapshot.exists()
    assert cv2.imread(str(snapshot)) is not None
    assert output.event_count == 1
    assert output.sequence == 2
    assert worker.status().counts == {"IN": 1, "OUT": 0}
    assert worker.status().saved_events == 1
    assert len(models) == 1
    assert calls[0][1]["persist"] is True
    assert calls[0][1]["tracker"] == "bytetrack.yaml"
    assert calls[0][1]["classes"] == [0, 2]
    assert np.array_equal(p1.frame, packet(1, 1).frame), "must not mutate capture buffer"


def test_out_lane_counts_as_out(tmp_path):
    worker, db, _, _ = make_worker(tmp_path)
    worker.process_packet(packet(1, 4))
    worker.process_packet(packet(2, 5))
    assert [event.direction for event in db.list_events()] == ["OUT"]
    assert worker.status().counts == {"IN": 0, "OUT": 1}


def test_zone_three_observations_record_once(tmp_path):
    worker, db, _, _ = make_worker(tmp_path)
    for i in range(1, 6):
        worker.process_packet(packet(i, 3))
    events = db.list_events()
    assert len(events) == 1
    assert events[0].event_type.value == "zone_entry"
    assert events[0].region_id == "restricted_area"
    assert worker.status().saved_events == 1


def test_capture_sequence_gap_is_counted_and_old_frame_rejected(tmp_path):
    worker, _, _, _ = make_worker(tmp_path)
    worker.process_packet(packet(10, 1))
    worker.process_packet(packet(15, 2))
    assert worker.status().skipped_capture_frames == 4
    with pytest.raises(ValueError, match="increase"):
        worker.process_packet(packet(15, 1))


def test_new_session_rebuilds_tracker_and_resets_event_engine(tmp_path):
    worker, db, _, models = make_worker(tmp_path)
    worker.process_packet(packet(1, 1, session=1))
    worker.process_packet(packet(2, 2, session=1))
    worker.process_packet(packet(3, 2, session=2))
    # First frame of a new session MUST NOT form a crossing from old session.
    assert len(db.list_events()) == 1
    worker.process_packet(packet(4, 1, session=2))
    assert len(db.list_events()) == 2
    assert len(models) == 2
    assert worker.status().session_id == 2


def test_output_slot_replaces_previous_frame(tmp_path):
    worker, _, _, _ = make_worker(tmp_path)
    worker.process_packet(packet(1, 0))
    worker.process_packet(packet(2, 0))
    assert worker.latest().sequence == 2
    assert worker.wait_for_frame(after_sequence=1, timeout=0).sequence == 2
    assert worker.wait_for_frame(after_sequence=2, timeout=0) is None


def test_threaded_worker_drains_finite_source_and_ends(tmp_path):
    worker, db, _, _ = make_worker(tmp_path, [packet(1, 1), packet(2, 2)])
    worker.start()
    result = worker.wait_for_frame(after_sequence=0, timeout=2)
    assert result is not None
    assert worker.stop(timeout=2)
    assert len(db.list_events()) == 1
    assert worker.status().processed_frames == 2


def test_threaded_model_error_reports_status_without_leaking_url(tmp_path):
    worker, _, _, _ = make_worker(tmp_path, [packet(1, 1)])

    def fails(_weights):
        raise RuntimeError("Secret RTSP URL should never enter last_error")

    worker.model_factory = fails
    worker.start()
    import time

    deadline = time.monotonic() + 2
    while worker.status().state != AIState.ERROR and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.status().state == AIState.ERROR
    assert worker.status().last_error == "RuntimeError"
    worker.stop()


def test_missing_region_labels_rejected(tmp_path):
    worker, _, _, _ = make_worker(tmp_path)
    bad = make_camera()
    bad.lines = bad.lines[:1]
    with pytest.raises(ValueError, match="in_entrance"):
        AIWorker(
            camera=bad,
            model_config=worker.model_config,
            source=worker.source,
            recorder=worker.recorder,
        )


def test_real_opencv_mp4_capture_to_ai_worker(tmp_path):
    """Exercise CameraWorker -> AIWorker using a generated real MP4."""
    from multicamera_ai.cameras.worker import CameraWorker

    path = tmp_path / "sample.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 15.0, (100, 100))
    if not video.isOpened():
        pytest.skip("OpenCV MP4 codec unavailable")
    for _ in range(18):
        video.write(np.zeros((100, 100, 3), dtype=np.uint8))
    video.release()

    worker, _db, _, _ = make_worker(tmp_path)
    capture = CameraWorker("cam1", str(path), is_file=True, file_realtime=True)
    worker.source = capture
    try:
        capture.start()
        worker.start()
        import time

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = worker.status()
            if status.state in (AIState.ENDED, AIState.ERROR):
                break
            time.sleep(0.01)
        assert worker.status().state == AIState.ENDED
        assert worker.status().processed_frames > 0
        assert worker.latest() is not None
        assert capture.status().state == CameraState.ENDED
    finally:
        assert worker.stop(timeout=2)
        assert capture.stop(timeout=2)
