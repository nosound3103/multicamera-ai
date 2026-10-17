import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from multicamera_ai.config.schemas import StreamingConfig
from multicamera_ai.pipeline.publisher import RTSPPublisher


class Pipe:
    def __init__(self, process, blocking):
        self.process = process
        self.blocking = blocking

    def write(self, data):
        if self.blocking:
            self.process.killed.wait(3)
            raise BrokenPipeError
        return min(100_000, len(data))  # Exercise partial raw pipe writes.

    def close(self):
        pass


class Process:
    def __init__(self, blocking):
        self.pid = 123
        self.killed = threading.Event()
        self.stdin = Pipe(self, blocking)

    def poll(self):
        return -9 if self.killed.is_set() else None

    def kill(self):
        self.killed.set()

    def wait(self, timeout):
        return -9


class Source:
    def __init__(self):
        self.sequence = 0

    def latest(self):
        return SimpleNamespace(
            sequence=self.sequence,
            captured_at=datetime.now(UTC),
            annotated=np.zeros((120, 640, 3), dtype=np.uint8),
        )

    def wait_for_frame(self, **kwargs):
        time.sleep(0.01)
        self.sequence += 1
        return self.latest()


def test_ffmpeg_crash_restarts_and_stop_reaps_children(monkeypatch):
    processes = []

    def spawn(command, **kwargs):
        assert "-tune" in command and "zerolatency" in command
        p = Process(blocking=not processes)
        processes.append(p)
        return p

    monkeypatch.setattr("multicamera_ai.pipeline.publisher.subprocess.Popen", spawn)
    publisher = RTSPPublisher(
        "cam1",
        "rtsp://localhost/out",
        Source(),
        StreamingConfig(width=640, height=120, write_timeout=0.1),
    )
    publisher.start()
    try:
        deadline = time.monotonic() + 3
        while publisher.status()["frames"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert publisher.status()["frames"] >= 1
        assert publisher.status()["restarts"] >= 1
        assert len(processes) >= 2
    finally:
        assert publisher.stop(timeout=2)
    assert all(p.killed.is_set() for p in processes)


def test_stale_frames_are_dropped_before_encoding(monkeypatch):
    source = Source()
    packet = source.latest()
    packet.captured_at = datetime(2000, 1, 1, tzinfo=UTC)
    source.wait_for_frame = lambda **kwargs: time.sleep(0.01) or packet
    publisher = RTSPPublisher("cam1", "rtsp://localhost/out", source, StreamingConfig())
    publisher.start()
    time.sleep(0.1)
    assert publisher.stop()
    assert publisher.status()["frames"] == 0


def test_equal_clock_samples_do_not_crash_status():
    publisher = RTSPPublisher("cam1", "rtsp://localhost/out", Source(), StreamingConfig())
    now = time.monotonic()
    publisher._times.extend([now, now])
    assert publisher.status()["input_fps"] == 0
