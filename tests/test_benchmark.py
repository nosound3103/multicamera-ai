import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
