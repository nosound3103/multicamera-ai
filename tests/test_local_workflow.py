import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from multicamera_ai.cameras import CameraState, CameraWorker
from multicamera_ai.vision.geometry import normalized_to_pixel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "labelme_to_yaml", ROOT / "scripts/labelme_to_yaml.py"
)
labelme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labelme)
convert_labelme = labelme.convert_labelme
normalize_point = labelme.normalize_point


def test_labelme_boundary_coordinates_project_to_last_pixel():
    data = {
        "imageWidth": 1920,
        "imageHeight": 1080,
        "shapes": [
            {"label": "counting_line", "shape_type": "line", "points": [[0, 0], [1920, 1080]]},
            {
                "label": "restricted_area",
                "shape_type": "polygon",
                "points": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
            },
        ],
    }
    lines, zones = convert_labelme(data)
    assert lines[0]["end"] == {"x": 1, "y": 1}
    point = zones[0]["points"][2]
    assert normalized_to_pixel((point["x"], point["y"]), 1920, 1080) == (1919, 1079)
    assert set(lines[0]) == {"id", "start", "end"}
    interior = normalize_point([300, 400], 1920, 1080)
    assert normalized_to_pixel((interior["x"], interior["y"]), 1920, 1080) == (300, 400)


@pytest.mark.parametrize("point", [[1921, 0], [0, 1081], [-1, 0]])
def test_labelme_rejects_outside_boundary(point):
    with pytest.raises(ValueError, match="outside image"):
        normalize_point(point, 1920, 1080)


def test_real_repository_video_to_eof():
    pytest.importorskip("cv2")
    source = ROOT / "assets/people-walking.mp4"
    if not source.is_file():
        pytest.skip("Repository media is gitignored and unavailable")
    worker = CameraWorker("cam1", str(source), is_file=True, file_realtime=False)
    worker.start()
    try:
        sequence = 0
        while True:
            packet = worker.wait_for_frame(after_sequence=sequence, timeout=10)
            if packet is None:
                break
            assert packet.sequence > sequence
            sequence = packet.sequence
            assert packet.frame.size > 0
        status = worker.status()
        assert status.state == CameraState.ENDED
        assert status.frames_received > 0
        assert status.reconnect_attempts == 0
        assert worker.wait_for_frame(after_sequence=status.frames_received, timeout=0) is None
    finally:
        assert worker.stop()


@pytest.mark.parametrize(
    "entrypoint",
    [["-m", "multicamera_ai.vision.demo"], ["scripts/ai_worker_demo.py"]],
)
def test_demo_cli_keeps_existing_options(entrypoint):
    pytest.importorskip("cv2")
    result = subprocess.run(
        [sys.executable, *entrypoint, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--input", "--output", "--db", "--snapshots", "--max-frames"):
        assert option in result.stdout


def test_worker_cli_rejects_unknown_override():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/camera_workers_demo.py"),
            "--config",
            str(ROOT / "config/cameras.local.yaml"),
            "--source-override",
            "missing=anything.mp4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "configured CAMERA_ID=FILE" in result.stderr


@pytest.mark.parametrize("mode", ["all_failed", "partial_failure", "healthy"])
def test_worker_cli_reports_capture_failure_without_stopping_healthy_camera(tmp_path, mode):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    data = yaml.safe_load((ROOT / "config/cameras.local.yaml").read_text())
    for camera in data["cameras"]:
        camera["source"]["uri"] = str(tmp_path / "missing.mp4")
    if mode != "all_failed":
        source = tmp_path / "valid.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 25, (64, 48))
        assert writer.isOpened()
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        for camera in data["cameras"] if mode == "healthy" else data["cameras"][:1]:
            camera["source"]["uri"] = str(source)
    config = tmp_path / "cameras.yaml"
    config.write_text(yaml.safe_dump(data))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/camera_workers_demo.py"),
            "--config",
            str(config),
            "--seconds",
            "2",
            "--preview-dir",
            str(tmp_path / "previews"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if mode == "healthy" else 1), result.stdout + result.stderr
    assert ("Capture incomplete or failed for:" in result.stderr) == (mode != "healthy")
    assert "stopped=True" in result.stderr
    if mode != "all_failed":
        assert "Final cam1: state=ended frames=5" in result.stderr
        assert cv2.imread(str(tmp_path / "previews/cam1.jpg")) is not None


@pytest.mark.parametrize("delay", ["0", "-1", "nan", "inf"])
def test_worker_cli_rejects_invalid_reconnect_delay(delay):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/camera_workers_demo.py"), "--reconnect-delay", delay],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--reconnect-delay must be positive and finite" in result.stderr
    assert "Traceback" not in result.stderr
