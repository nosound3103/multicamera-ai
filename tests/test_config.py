
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


def test_example_has_four_cameras():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "cameras.yaml")

    assert len(config.cameras) == 4
