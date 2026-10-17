import threading
import time
from types import SimpleNamespace

import pytest

from multicamera_ai.cameras import (
    CameraDefinition,
    CameraManager,
    CameraState,
    CameraWorker,
)


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeCapture:
    def __init__(self, frames=(), *, looping=False, opened=True, delay=0.005, fps=25.0):
        self.frames = list(frames)
        self.looping = looping
        self.opened = opened
        self.delay = delay
        self.fps = fps
        self.read_count = 0
        self.released = False

    def isOpened(self):  # noqa: N802
        return self.opened

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        self.read_count += 1
        if self.frames:
            if self.looping:
                return True, self.frames[(self.read_count - 1) % len(self.frames)]
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True

    def grab(self):
        if not self.frames:
            return False
        self.frames.pop(0)
        return True

    def get(self, prop):
        assert prop == 5
        return self.fps


def test_file_reads_to_eof_without_reconnecting():
    captures = []

    def factory(_):
        capture = FakeCapture([1, 2, 3], delay=0)
        captures.append(capture)
        return capture

    worker = CameraWorker(
        "cam1", "video.mp4", is_file=True, file_realtime=False, capture_factory=factory
    )
    worker.start()
    assert wait_until(lambda: worker.status().state == CameraState.ENDED)
    assert worker.status().frames_received == 3
    assert worker.status().reconnect_attempts == 0
    assert worker.get_latest().frame == 3
    assert worker.stop()
    assert captures[0].released
    assert len(captures) == 1


@pytest.mark.parametrize("fps,total,expected", [(60, 12, 3), (30, 12, 6), (10, 4, 4)])
def test_file_rate_limit_skips_conversion_without_speeding_up_playback(fps, total, expected):
    capture = FakeCapture(range(total), delay=0, fps=fps)
    worker = CameraWorker(
        "cam1", "video.mp4", is_file=True, file_max_fps=15, capture_factory=lambda _: capture
    )
    started = time.monotonic()
    worker.start()
    assert wait_until(lambda: worker.status().state == CameraState.ENDED)
    assert worker.stop()
    assert worker.status().frames_received == expected
    assert capture.read_count <= expected + 1
    assert time.monotonic() - started >= (total - 1) / fps - 0.02


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_file_rate_limit(rate):
    with pytest.raises(ValueError, match="file_max_fps"):
        CameraWorker("cam1", "video.mp4", is_file=True, file_max_fps=rate)


def test_live_capture_reconnects_with_new_session_id():
    attempts = []

    def factory(_):
        attempts.append(1)
        if len(attempts) == 1:
            return FakeCapture(["old"], delay=0.001)
        return FakeCapture(["new"], looping=True)

    worker = CameraWorker(
        "cam1", "rtsp://example.invalid/test", reconnect_delay=0.01, capture_factory=factory
    )
    worker.start()
    try:
        assert wait_until(lambda: worker.status().session_id >= 2)
        assert wait_until(
            lambda: worker.get_latest() is not None and worker.get_latest().session_id == 2
        )
        assert worker.status().reconnect_attempts >= 1
        assert worker.get_latest().frame == "new"
        assert worker.status().frames_received >= 2
    finally:
        assert worker.stop()


def test_latest_frame_overwrites_old_frames():
    worker = CameraWorker(
        "cam1",
        "finite.mp4",
        is_file=True,
        file_realtime=False,
        capture_factory=lambda _: FakeCapture(list(range(30)), delay=0),
    )
    worker.start()
    try:
        assert wait_until(lambda: worker.status().state == CameraState.ENDED)
        assert worker.wait_for_frame(after_sequence=1, timeout=0).sequence == 30
        assert worker.get_latest().frame == 29
        assert worker.wait_for_frame(after_sequence=30, timeout=0) is None
    finally:
        assert worker.stop()


def test_wait_for_fresh_frame_advances_sequence():
    worker = CameraWorker(
        "cam1",
        "rtsp://test",
        capture_factory=lambda _: FakeCapture([1], looping=True),
        reconnect_delay=0.01,
    )
    worker.start()
    try:
        first = worker.wait_for_frame(timeout=2.0)
        assert first is not None
        second = worker.wait_for_frame(after_sequence=first.sequence, timeout=2.0)
        assert second is not None and second.sequence > first.sequence
        assert second.session_id == first.session_id
        assert second.captured_at.tzinfo is not None
    finally:
        assert worker.stop()


def test_one_failed_camera_does_not_stop_other_workers():
    def factory(source):
        if source == "bad":
            raise OSError("No connection")
        return FakeCapture([source], looping=True)

    manager = CameraManager(
        [
            CameraDefinition("cam1", "ok1"),
            CameraDefinition("cam2", "bad"),
            CameraDefinition("cam3", "ok3"),
            CameraDefinition("cam4", "ok4"),
        ],
        capture_factory=factory,
        reconnect_delay=0.01,
    )
    manager.start_all()
    try:
        assert wait_until(
            lambda: all(
                manager.status()[name].frames_received > 2 for name in ("cam1", "cam3", "cam4")
            )
        )
        assert wait_until(lambda: manager.status()["cam2"].reconnect_attempts > 0)
        assert all(
            manager.status()[name].state == CameraState.RUNNING for name in ("cam1", "cam3", "cam4")
        )
    finally:
        assert all(manager.stop_all().values())


def test_file_open_error_does_not_retry():
    attempts = []

    def factory(_):
        attempts.append(1)
        return FakeCapture(opened=False)

    worker = CameraWorker("cam1", "missing.mp4", is_file=True, capture_factory=factory)
    worker.start()
    assert wait_until(lambda: worker.status().state == CameraState.ERROR)
    assert worker.get_latest() is None
    assert len(attempts) == 1
    assert worker.stop()


def test_stop_interrupts_retry_wait():
    worker = CameraWorker(
        "cam1",
        "rtsp://down",
        reconnect_delay=5,
        capture_factory=lambda _: FakeCapture(opened=False),
    )
    worker.start()
    assert wait_until(lambda: worker.status().reconnect_attempts > 0)
    start = time.monotonic()
    assert worker.stop(timeout=1)
    assert time.monotonic() - start < 1.0
    assert worker.status().state == CameraState.STOPPED


def test_stale_live_packet_cleared_during_reconnect():
    entered_second = threading.Event()
    release_second = threading.Event()
    calls = []

    def factory(_):
        calls.append(1)
        if len(calls) == 1:
            return FakeCapture(["old"], delay=0)
        entered_second.set()
        release_second.wait(timeout=1)
        return FakeCapture(["new"], looping=True)

    worker = CameraWorker("cam1", "rtsp://test", capture_factory=factory, reconnect_delay=0.01)
    worker.start()
    try:
        assert entered_second.wait(timeout=2)
        assert worker.status().state == CameraState.RECONNECTING
        assert worker.get_latest() is None
    finally:
        release_second.set()
        assert worker.stop()


def test_manager_reads_webcam_file_rtsp_and_enabled():
    def cam(cid, typ, uri=None, index=None, enabled=True):
        return SimpleNamespace(
            id=cid, enabled=enabled, source=SimpleNamespace(type=typ, uri=uri, index=index)
        )

    config = SimpleNamespace(
        cameras=[
            cam("webcam", "webcam", index=0),
            cam("file", "file", uri="demo.mp4"),
            cam("rtsp", "rtsp", uri="rtsp://example/test"),
            cam("disabled", "webcam", index=1, enabled=False),
        ]
    )
    calls = []

    def factory(source):
        calls.append(source)
        return FakeCapture([source], looping=True)

    manager = CameraManager.from_app_config(config, capture_factory=factory)
    manager.start_all()
    try:
        assert wait_until(lambda: len(calls) >= 3)
        assert manager.status()["disabled"].state == CameraState.STOPPED
        assert {0, "demo.mp4", "rtsp://example/test"}.issubset(set(calls))
    finally:
        assert all(manager.stop_all().values())


def test_camera_can_be_restarted_after_ending():
    worker = CameraWorker(
        "cam1",
        "a.mp4",
        is_file=True,
        file_realtime=False,
        capture_factory=lambda _: FakeCapture(["frame"], delay=0),
    )
    worker.start()
    assert wait_until(lambda: worker.status().state == CameraState.ENDED)
    assert worker.status().session_id == 1
    worker.start()
    assert wait_until(
        lambda: worker.status().state == CameraState.ENDED and worker.status().session_id == 2
    )
    assert worker.status().frames_received == 2
    assert worker.stop()


def test_invalid_inputs():
    with pytest.raises(ValueError, match="camera_id"):
        CameraWorker("", "source")
    with pytest.raises(ValueError, match="reconnect_delay"):
        CameraWorker("cam1", "source", reconnect_delay=0)
    with pytest.raises(ValueError, match="File source"):
        CameraWorker("cam1", 0, is_file=True)
    with pytest.raises(ValueError, match="Duplicate camera"):
        CameraManager([CameraDefinition("cam1", "a"), CameraDefinition("cam1", "b")])


def test_stop_timeout_is_reported_for_blocked_backend():
    entered_read = threading.Event()
    unblock = threading.Event()

    class BlockedCapture(FakeCapture):
        def read(self):
            entered_read.set()
            unblock.wait(timeout=2)
            return False, None

    worker = CameraWorker(
        "cam1", "rtsp://blocking", capture_factory=lambda _: BlockedCapture(), reconnect_delay=0.01
    )
    worker.start()
    assert entered_read.wait(timeout=1)
    try:
        assert not worker.stop(timeout=0.005)
    finally:
        unblock.set()
        assert worker.stop(timeout=2)


def test_manager_start_stop_single_camera():
    manager = CameraManager(
        [CameraDefinition("cam1", "file.mp4", is_file=True)],
        capture_factory=lambda _: FakeCapture([1], delay=0),
        file_realtime=False,
    )
    manager.start_camera("cam1")
    assert wait_until(lambda: manager.status()["cam1"].state == CameraState.ENDED)
    assert manager.latest("cam1").frame == 1
    assert manager.stop_camera("cam1")


@pytest.mark.parametrize("source", ["rtsp://test", 0])
def test_failed_capture_released_before_retry_delay(source):
    capture = FakeCapture(opened=False)
    worker = CameraWorker("cam1", source, reconnect_delay=5, capture_factory=lambda _: capture)
    worker.start()
    try:
        assert wait_until(lambda: capture.released)
        assert worker.status().reconnect_attempts == 1
        assert worker.status().capture_fps == 0
    finally:
        assert worker.stop()


def test_stop_restart_and_running_fps():
    captures = []

    def factory(_):
        capture = FakeCapture([1], looping=True)
        captures.append(capture)
        return capture

    worker = CameraWorker("cam1", 0, capture_factory=factory)
    worker.start()
    try:
        assert wait_until(
            lambda: worker.status().frames_received >= 3 and 0 < worker.status().capture_fps < 1000
        )
        first_sequence = worker.get_latest().sequence
        worker.start()  # Active start is idempotent.
        assert len(captures) == 1
        assert worker.stop()
        assert captures[0].released
        assert worker.status().capture_fps == 0
        worker.start()
        packet = worker.wait_for_frame(after_sequence=first_sequence, timeout=2)
        assert packet is not None and packet.session_id == 2
    finally:
        assert worker.stop()
    assert all(capture.released for capture in captures)


def test_missing_real_file_reports_error(tmp_path):
    missing = tmp_path / "missing.mp4"
    worker = CameraWorker("cam1", str(missing), is_file=True)
    worker.start()
    try:
        assert wait_until(lambda: worker.status().state == CameraState.ERROR)
        assert str(missing) in worker.status().last_error
        assert worker.status().frames_received == 0
        assert worker.status().reconnect_attempts == 0
    finally:
        assert worker.stop()


@pytest.mark.parametrize("fps", [0, float("nan"), float("inf")])
def test_invalid_file_fps_uses_paced_fallback(fps):
    worker = CameraWorker("cam1", "file.mp4", is_file=True, capture_factory=lambda _: FakeCapture())
    assert worker._file_frame_period(FakeCapture(fps=fps)) == 1 / 25


@pytest.mark.parametrize("camera_id,source", [("../bad", 0), ("cam1", -1)])
def test_invalid_camera_id_and_webcam_index(camera_id, source):
    with pytest.raises(ValueError):
        CameraWorker(camera_id, source)
