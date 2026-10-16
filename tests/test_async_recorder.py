import threading
from types import SimpleNamespace

import numpy as np

from multicamera_ai.events.recorder import AsyncEventRecorder


def test_bounded_queue_reports_overflow_and_drains_on_stop():
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
    assert recorder.record(event, image) is False
    image[:] = 255
    release.set()
    assert recorder.stop(2)
    assert recorder.status() == {"saved": 2, "failed": 0, "dropped": 1, "pending": 0}
    assert all(not frame.any() for frame in saved)


def test_disk_failure_is_reported_without_killing_writer():
    def fails(event, frame):
        raise OSError("disk full")

    recorder = AsyncEventRecorder(SimpleNamespace(record=fails), capacity=2)
    assert recorder.record(SimpleNamespace(camera_id="cam1"), np.zeros((1, 1, 3)))
    assert recorder.stop(2)
    assert recorder.status()["failed"] == 1
