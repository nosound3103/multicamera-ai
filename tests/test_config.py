from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multicamera_ai.config import AppConfig, ConfigError, load_config


@pytest.fixture
def valid_data():
    return {
        "model": {
            "weights": "yolov8n.pt",
            "device": "cpu",
            "confidence": 0.4,
            "imgsz": 640,
            "class_ids": [0, 2, 3, 5, 7],
        },
        "cameras": [
            {
                "id": "cam1",
                "enabled": True,
                "source": {
                    "type": "rtsp",
                    "uri": "rtsp://localhost:8554/input_cam1",
                },
                "output_url": "rtsp://localhost:8554/output_cam1",
                "lines": [
                    {
                        "id": "gate",
                        "start": {"x": 0.1, "y": 0.5},
                        "end": {"x": 0.9, "y": 0.5},
                    }
                ],
                "zones": [
                    {
                        "id": "restricted",
                        "points": [
                            {"x": 0.2, "y": 0.2},
                            {"x": 0.8, "y": 0.2},
                            {"x": 0.5, "y": 0.8},
                        ],
                    }
                ],
            }
        ],
    }


def test_valid_configuration(valid_data):
    config = AppConfig.model_validate(valid_data)

    assert len(config.cameras) == 1
    assert config.cameras[0].id == "cam1"
    assert config.model.imgsz == 640


@pytest.mark.parametrize(
    "source",
    [
        {"type": "file", "uri": "assets/demo.mp4"},
        {"type": "webcam", "index": 0},
    ],
)
def test_other_source_types(valid_data, source):
    valid_data["cameras"][0]["source"] = source

    config = AppConfig.model_validate(valid_data)

    assert config.cameras[0].source.type == source["type"]


def test_invalid_confidence(valid_data):
    valid_data["model"]["confidence"] = 1.5

    with pytest.raises(ValidationError, match="confidence"):
        AppConfig.model_validate(valid_data)


def test_duplicate_camera_id(valid_data):
    duplicate = deepcopy(valid_data["cameras"][0])
    valid_data["cameras"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate camera"):
        AppConfig.model_validate(valid_data)


def test_invalid_rtsp_url(valid_data):
    valid_data["cameras"][0]["source"]["uri"] = "https://example.com"

    with pytest.raises(ValidationError, match="RTSP"):
        AppConfig.model_validate(valid_data)


def test_invalid_counting_line(valid_data):
    line = valid_data["cameras"][0]["lines"][0]
    line["end"] = line["start"]

    with pytest.raises(ValidationError, match="line endpoints"):
        AppConfig.model_validate(valid_data)


def test_polygon_needs_three_points(valid_data):
    points = valid_data["cameras"][0]["zones"][0]["points"]
    points.pop()

    with pytest.raises(ValidationError, match="at least 3"):
        AppConfig.model_validate(valid_data)


def test_point_must_be_normalized(valid_data):
    valid_data["cameras"][0]["lines"][0]["start"]["x"] = 1.5

    with pytest.raises(ValidationError, match="less than or equal"):
        AppConfig.model_validate(valid_data)


def test_load_yaml_file(valid_data, tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump(valid_data), encoding="utf-8")

    config = load_config(path)

    assert config.cameras[0].id == "cam1"


def test_invalid_yaml_syntax(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("cameras: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_missing_yaml_file(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "filename",
    [
        "cameras.yaml",
        "cameras.local.yaml",
        "cameras.cpu.yaml",
        "cameras.gpu.yaml",
        "cameras.docker.yaml",
        "cameras.docker.gpu.yaml",
    ],
)
def test_example_has_four_cameras(filename):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / filename)

    assert len(config.cameras) == 4
    assert [camera.id for camera in config.cameras] == ["cam1", "cam2", "cam3", "cam4"]
    assert all(camera.model == "vehicle" for camera in config.cameras)
    assert all(camera.lines and camera.zones for camera in config.cameras)


def test_local_configuration_preserves_deployment_settings(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    deployment = load_config(root / "config/cameras.yaml")
    local = load_config(root / "config/cameras.local.yaml")
    assert local.model == deployment.model
    assert len(local.cameras) >= 4
    for original, camera in zip(deployment.cameras, local.cameras, strict=True):
        assert camera.enabled
        assert camera.id == original.id
        assert camera.lines == original.lines
        assert camera.zones == original.zones
        assert camera.output_url == original.output_url
        assert camera.source.type == "file"
        assert Path(camera.source.uri).is_absolute()
        # Media is gitignored and may be absent on CI; opening validates existence.
        assert Path(camera.source.uri).parent == root / "assets"


def test_file_path_resolves_relative_to_yaml(valid_data, tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    valid_data["cameras"][0]["source"] = {"type": "file", "uri": "../video.mp4"}
    path = folder / "local.yaml"
    path.write_text(yaml.safe_dump(valid_data), encoding="utf-8")
    monkeypatch.chdir(folder)
    assert load_config(path).cameras[0].source.uri == str(tmp_path / "video.mp4")


def test_camera_model_profiles_and_legacy_fallback(valid_data):
    valid_data["models"] = {
        "vehicle": dict(valid_data["model"], weights="vehicle.pt", class_ids=[0, 1, 2, 3])
    }
    config = AppConfig.model_validate(valid_data)
    assert config.model_for(config.cameras[0]) is config.model
    config.cameras[0].model = "vehicle"
    assert config.model_for(config.cameras[0]).weights == "vehicle.pt"
    assert config.model_for(config.cameras[0]).class_ids == [0, 1, 2, 3]


def test_profiles_without_global_model_support_more_cameras(valid_data):
    valid_data["models"] = {"person": valid_data.pop("model")}
    template = valid_data["cameras"][0]
    valid_data["cameras"] = [
        dict(
            deepcopy(template), id=f"cam{i}", model="person", output_url=f"rtsp://localhost/out{i}"
        )
        for i in range(1, 10)
    ]
    config = AppConfig.model_validate(valid_data)
    assert len(config.cameras) == 9
    assert all(config.model_for(c) is config.models["person"] for c in config.cameras)


@pytest.mark.parametrize("profile", [None, "missing"])
def test_camera_requires_known_profile_without_default(valid_data, profile):
    valid_data.pop("model")
    valid_data["cameras"][0]["model"] = profile
    with pytest.raises(ValidationError, match="model profile"):
        AppConfig.model_validate(valid_data)


def test_model_paths_resolve_relative_to_yaml(valid_data, tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    valid_data["model"]["weights"] = "../models/default.pt"
    valid_data["models"] = {"vehicle": dict(valid_data["model"], weights="../models/vehicle.pt")}
    path = folder / "test.yaml"
    path.write_text(yaml.safe_dump(valid_data))
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.model.weights == str(tmp_path / "models/default.pt")
    assert config.models["vehicle"].weights == str(tmp_path / "models/vehicle.pt")


def test_runtime_model_validation_checks_each_profile(valid_data, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from multicamera_ai.config.models import validate_model_files

    weights = tmp_path / "model.pt"
    weights.touch()
    valid_data["model"]["weights"] = str(weights)
    valid_data["model"]["class_ids"] = [0]
    valid_data["models"] = {"vehicle": dict(valid_data["model"], class_ids=[7])}
    other = dict(
        deepcopy(valid_data["cameras"][0]),
        id="cam2",
        model="vehicle",
        output_url="rtsp://localhost/out2",
    )
    valid_data["cameras"].append(other)
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda _: SimpleNamespace(names={0: "bus", 1: "car"})),
    )
    with pytest.raises(ValueError, match="unknown class IDs.*7"):
        validate_model_files(AppConfig.model_validate(valid_data))


def test_runtime_missing_weights_fail_without_downloading(valid_data, tmp_path):
    from multicamera_ai.config.models import validate_model_files

    valid_data["model"]["weights"] = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="weights not found"):
        validate_model_files(AppConfig.model_validate(valid_data))
