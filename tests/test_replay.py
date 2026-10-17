"""Replay configuration must preserve source/camera identity."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "replay", Path(__file__).resolve().parents[1] / "scripts/replay.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_replay_config_selects_distinct_enabled_files(tmp_path, monkeypatch):
    videos = [tmp_path / f"{i}.mp4" for i in range(3)]
    for video in videos:
        video.touch()
    config = {
        "model": {"weights": "unused.pt", "confidence": 0.5, "imgsz": 320, "class_ids": [0]},
        "cameras": [
            {
                "id": f"gate{i}",
                "enabled": i != 1,
                "source": {"type": "file", "uri": video.name},
                "output_url": f"rtsp://localhost/out{i}",
            }
            for i, video in enumerate(videos)
        ],
    }
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump(config))
    selected = []
    monkeypatch.setattr(replay, "replay_watermarked", lambda args: selected.extend(args.inputs))
    monkeypatch.setattr(sys, "argv", ["replay", "--config", str(path), "--watermark"])
    replay.main()
    assert selected == [("gate0", str(videos[0])), ("gate2", str(videos[2]))]


def test_replay_single_input_preserves_legacy_numbering(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.touch()
    selected = []
    monkeypatch.setattr(replay, "replay_watermarked", lambda args: selected.extend(args.inputs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay", "--input", str(video), "--count", "2", "--start-index", "3", "--watermark"],
    )
    replay.main()
    assert selected == [("cam3", str(video)), ("cam4", str(video))]


def test_replay_missing_video_fails_before_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["replay", "--input", str(tmp_path / "missing.mp4")])
    with pytest.raises(SystemExit) as exc:
        replay.main()
    assert exc.value.code == 2
