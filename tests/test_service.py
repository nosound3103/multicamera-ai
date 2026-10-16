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
        return [SimpleNamespace(boxes=None, plot=lambda: frame.copy())]


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
