import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from multicamera_ai.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_eight_cameras_repeat_sources_and_regions_with_independent_ids():
    original = load_config(ROOT / "config/cameras.docker.gpu.yaml")
    args = SimpleNamespace(
        server="rtsp://localhost:8554",
        source=None,
        output_fps=0,
        source_watermark=True,
        output="artifacts/benchmark/results.json",
    )
    config = benchmark.prepare_config(original, 8, "processes", args)
    assert len(original.cameras) == 4
    assert [c.id for c in config.cameras] == [f"cam{i}" for i in range(1, 9)]
    assert len({c.output_url for c in config.cameras}) == 8
    for first, repeated in zip(config.cameras[:4], config.cameras[4:], strict=True):
        assert repeated.source == first.source
        assert repeated.lines == first.lines
        assert repeated.zones == first.zones
        assert repeated.model == first.model
        assert repeated is not first


def test_ai_measurement_requires_real_inference_progress():
    before = {
        "cam1": {
            "updated_at": 10,
            "status_age_seconds": 0.1,
            "ai": {"processed_frames": 100, "state": "running"},
        }
    }
    end = {
        "updated_at": 12,
        "worker_alive": True,
        "status_age_seconds": 0.1,
        "ai": {"processed_frames": 160, "state": "running", "last_error": None},
    }
    measured = benchmark.ai_measurements(before, {"cam1": end})["cam1"]
    assert measured["ai_processing_fps"] == 30
    assert measured["ai_healthy"]
    end["ai"]["processed_frames"] = 100
    assert not benchmark.ai_measurements(before, {"cam1": end})["cam1"]["ai_healthy"]
    end["ai"]["processed_frames"] = 160
    end["ai"]["state"] = "error"
    assert not benchmark.ai_measurements(before, {"cam1": end})["cam1"]["ai_healthy"]


def test_ai_measurement_rejects_missing_baseline_and_stale_status():
    end = {
        "updated_at": 12,
        "worker_alive": True,
        "status_age_seconds": 0.1,
        "ai": {"processed_frames": 160, "state": "running", "last_error": None},
    }
    assert not benchmark.ai_measurements({}, {"cam1": end})["cam1"]["ai_healthy"]
    before = {
        "cam1": {
            "updated_at": 10,
            "status_age_seconds": 0.1,
            "ai": {"processed_frames": 100, "state": "running"},
        }
    }
    end["status_age_seconds"] = 10
    assert not benchmark.ai_measurements(before, {"cam1": end})["cam1"]["ai_healthy"]


@pytest.mark.parametrize(
    "response,available",
    [
        (b"RTSP/1.0 404 Not Found\r\n", False),
        (b"RTSP/1.0 200 OK\r\n", True),
        (b"RTSP/1.0 401 Unauthorized\r\n", True),
        (b"", False),
    ],
)
def test_rtsp_preflight_checks_path_without_opening_video(monkeypatch, response, available):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sendall(self, request):
            assert request.startswith(b"DESCRIBE rtsp://localhost:8554/bench1 RTSP/1.0")

        def recv(self, size):
            return response

    monkeypatch.setattr(benchmark.socket, "create_connection", lambda *args: Connection())
    assert benchmark.rtsp_output_available("rtsp://localhost:8554/bench1") is available


def running_status(frames=0):
    return {
        "worker_alive": True,
        "status_age_seconds": 0,
        "capture": {"state": "running"},
        "ai": {"state": "running", "processed_frames": frames},
        "output": {"frames": frames, "ffmpeg_pid": 123 if frames else None},
    }


def test_probes_wait_for_publisher_progress_before_starting(monkeypatch):
    connected = {"cam1": threading.Event(), "cam2": threading.Event()}
    polls = []
    starts = []

    class Service:
        def status(self):
            polls.append(1)
            return {"cam1": running_status(1), "cam2": running_status(len(polls) - 1)}

    class Thread:
        ident = None

        def __init__(self, camera_id):
            self.camera_id = camera_id

        def start(self):
            starts.append((self.camera_id, len(polls)))
            self.ident = 1
            connected[self.camera_id].set()

    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    threads = {cid: Thread(cid) for cid in connected}
    assert benchmark.wait_for_outputs(Service(), threads, connected, 5) == (True, None)
    assert starts == [("cam1", 1), ("cam2", 2)]


def test_startup_timeout_skips_warmup_and_measurement(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        benchmark.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    stopped = []

    class Service:
        def __init__(self, config):
            pass

        def start(self):
            pass

        def status(self):
            return {"cam1": running_status(0)}

        def stop(self):
            stopped.append(True)
            return {"cam1": True}

    monkeypatch.setattr(benchmark, "PipelineService", Service)
    args = SimpleNamespace(
        server="rtsp://localhost:8554",
        source=None,
        output_fps=0,
        source_watermark=True,
        output=str(tmp_path / "results.json"),
        startup_timeout=1,
        warmup=15,
        seconds=60,
    )
    config = load_config(ROOT / "config/cameras.docker.gpu.yaml")
    result = benchmark.run(config, 1, "threads", args)
    assert result["failure_reason"] == "startup_timeout"
    assert not result["valid"]
    assert not result["measurement_started"]
    assert result["measured_seconds"] == 0
    assert result["actual_warmup_seconds"] == 0
    assert result["resources"] == []
    assert result["all_probes_stopped"]
    assert stopped == [True]


def test_failed_camera_aborts_startup(monkeypatch):
    class Service:
        def status(self):
            return {"cam1": {"worker_alive": False, "error": "worker exited"}}

    assert benchmark.wait_for_outputs(
        Service(), {"cam1": None}, {"cam1": threading.Event()}, 5
    ) == (False, "camera_failed_during_startup")


def test_missing_rtsp_path_does_not_open_opencv(monkeypatch):
    stop = threading.Event()
    monkeypatch.setattr(benchmark, "rtsp_output_available", lambda _: False)
    monkeypatch.setattr(stop, "wait", lambda _: stop.set())

    def unexpected_capture(url):
        pytest.fail("OpenCV must wait until the RTSP path exists")

    monkeypatch.setattr(benchmark, "opencv_capture_factory", unexpected_capture)
    benchmark.probe("rtsp://localhost:8554/bench1", stop, [], threading.Event(), threading.Event())


def test_measurement_clock_excludes_setup_and_warmup(monkeypatch, tmp_path):
    clock = [0.0]
    resource_times = []
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        benchmark.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    class Service:
        def __init__(self, config):
            pass

        def start(self):
            clock[0] += 5  # Camera setup is outside the measurement clock.

        def status(self):
            status = running_status(100 + int(clock[0] * 20))
            status["updated_at"] = clock[0]
            return {"cam1": status}

        def stop(self):
            return {"cam1": True}

    class ProbeThread:
        ident = None

        def __init__(self, target, args, daemon):
            self.connected = args[-1]

        def start(self):
            self.ident = 1
            self.connected.set()

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    class Process:
        def children(self, recursive):
            return []

        def memory_info(self):
            resource_times.append(clock[0])
            return SimpleNamespace(rss=1000)

        def cpu_times(self):
            return (clock[0], 0)

    monkeypatch.setattr(benchmark, "PipelineService", Service)
    monkeypatch.setattr(benchmark.threading, "Thread", ProbeThread)
    monkeypatch.setattr(benchmark.psutil, "Process", Process)
    args = SimpleNamespace(
        server="rtsp://localhost:8554",
        source=None,
        output_fps=0,
        source_watermark=True,
        output=str(tmp_path / "results.json"),
        startup_timeout=10,
        warmup=2,
        seconds=3,
    )
    config = load_config(ROOT / "config/cameras.docker.gpu.yaml")
    result = benchmark.run(config, 1, "threads", args)
    assert result["measurement_started"]
    assert result["failure_reason"] is None
    assert result["startup_seconds"] == 5
    assert result["actual_warmup_seconds"] == pytest.approx(2)
    assert result["measured_seconds"] == pytest.approx(3)
    assert len(resource_times) == 3
    assert resource_times[0] == pytest.approx(7)
    assert result["cameras"]["cam1"]["ai_processing_fps"] == pytest.approx(20)
