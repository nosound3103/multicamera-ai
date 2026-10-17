import functools
import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from multicamera_ai.api import create_app
from multicamera_ai.config.schemas import AppConfig
from multicamera_ai.pipeline import AIWorker
from multicamera_ai.service import PipelineService


def configuration(tmp_path):
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 15, (160, 120))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    return AppConfig.model_validate(
        {
            "model": {"weights": "fake.pt", "confidence": 0.5, "imgsz": 320, "class_ids": [0]},
            "runtime": {
                "database": str(tmp_path / "events.db"),
                "snapshots": str(tmp_path / "snapshots"),
                "shutdown_timeout": 3,
            },
            "streaming": {"enabled": False},
            "cameras": [
                {
                    "id": f"cam{i}",
                    "source": {"type": "file", "uri": str(video)},
                    "output_url": f"rtsp://localhost:8554/out{i}",
                }
                for i in range(1, 5)
            ],
        }
    )


class Model:
    def track(self, frame, **kwargs):
        result = SimpleNamespace(boxes=None, plot=lambda: frame.copy())
        result.cpu = lambda: result
        return [result]


def test_four_cameras_api_toggle_events_and_shutdown(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setattr(
        "multicamera_ai.service.AIWorker",
        functools.partial(AIWorker, model_factory=lambda _: Model()),
    )
    service = PipelineService(config)
    with TestClient(create_app(config, service)) as client:
        assert client.get("/health").status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            states = client.get("/cameras").json()
            if all(v.get("ai", {}).get("processed_frames", 0) for v in states.values()):
                break
            time.sleep(0.05)
        assert len(states) == 4
        assert all(v["ai"]["processed_frames"] > 0 for v in states.values())
        assert client.put("/cameras/cam1/enabled?enabled=false").status_code == 200
        assert client.get("/cameras").json()["cam1"]["enabled"] is False
        assert client.get("/cameras").json()["cam2"]["worker_alive"] is True
        assert client.put("/cameras/cam1/enabled?enabled=true").status_code == 200
        assert client.get("/events?camera_id=cam1").json() == []
        assert client.get("/events?start=2026-10-16T10:00:00").status_code == 422
        assert (
            client.get("/events?start=2026-10-17T00:00:00Z&end=2026-10-16T00:00:00Z").status_code
            == 422
        )
        assert client.get("/events?limit=0").status_code == 422
        assert client.get("/events?camera_id=missing").status_code == 404
        assert client.put("/cameras/missing/enabled?enabled=true").status_code == 404
    assert all(not v["worker_alive"] for v in service.status().values())
    assert all(service.stop().values())


def test_missing_ffmpeg_fails_before_starting_workers(tmp_path):
    config = configuration(tmp_path)
    config.streaming.enabled = True
    config.streaming.ffmpeg = "executable-that-does-not-exist"
    import pytest

    service = PipelineService(config)
    with pytest.raises(ValueError, match="FFmpeg"):
        service.start()
    assert all(not v["worker_alive"] for v in service.status().values())


def test_spawned_process_reports_capture_error_and_stops_cleanly(tmp_path):
    config = configuration(tmp_path)
    config.runtime.execution = "processes"
    config.runtime.log_dir = str(tmp_path / "logs")
    config.cameras = config.cameras[:1]
    config.cameras[0].source.uri = str(tmp_path / "missing.mp4")
    service = PipelineService(config)
    try:
        service.start()
        worker = service._handles["cam1"][0]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = service.status()["cam1"]
            if status.get("ai", {}).get("state") == "error":
                break
            time.sleep(0.1)
        assert status["capture"]["state"] == "error"
        assert status["ai"]["state"] == "error"
    finally:
        assert service.stop() == {"cam1": True}
    assert worker.exitcode == 0
    assert service.status()["cam1"]["worker_alive"] is False


def test_each_camera_uses_selected_model_profile(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config.cameras = config.cameras[:2]
    config.models = {
        "vehicle": config.model.model_copy(
            update={"weights": "vehicle.pt", "class_ids": [0, 1, 2, 3]}
        )
    }
    config.cameras[0].model = "vehicle"
    loaded = []

    def factory(weights):
        loaded.append(weights)
        return Model()

    monkeypatch.setattr(
        "multicamera_ai.service.AIWorker", functools.partial(AIWorker, model_factory=factory)
    )
    service = PipelineService(config)
    try:
        service.start()
        deadline = time.monotonic() + 5
        while len(loaded) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sorted(loaded) == ["fake.pt", "vehicle.pt"]
    finally:
        assert all(service.stop().values())


def test_killed_process_does_not_block_other_camera_cleanup(tmp_path):
    config = configuration(tmp_path)
    config.runtime.execution = "processes"
    config.runtime.log_dir = str(tmp_path / "logs")
    config.cameras = config.cameras[:2]
    for camera in config.cameras:
        camera.source.uri = str(tmp_path / "missing.mp4")
    service = PipelineService(config)
    service.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if all(s.get("ai", {}).get("state") == "error" for s in service.status().values()):
            break
        time.sleep(0.05)
    worker = service._handles["cam1"][0]
    worker.kill()
    worker.join(5)
    started = time.monotonic()
    assert service.stop() == {"cam1": False, "cam2": True}
    assert time.monotonic() - started < 5
