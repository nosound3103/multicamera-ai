"""Exercise real YOLO, RTSP decode, FFmpeg crash recovery and API lifecycle.

Run with MediaMTX + replay.py already running. No model mocks are used.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import psutil
from benchmark import probe
from fastapi.testclient import TestClient

from multicamera_ai.api import create_app
from multicamera_ai.config import load_config
from multicamera_ai.service import PipelineService, configure_logging


def wait_for(predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("Integration condition timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--source", default="rtsp://127.0.0.1:8554/input_cam1")
    parser.add_argument("--server", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--mode", choices=["threads", "processes"], default="threads")
    parser.add_argument("--output", default="artifacts/integration-check.json")
    args = parser.parse_args()
    config = load_config(args.config)
    config.runtime.execution = args.mode
    config.model_for(config.cameras[0]).imgsz = 320
    config.streaming.watermark = True
    config.runtime.database = "storage/integration-check.db"
    config.runtime.snapshots = "storage/integration-check-snapshots"
    configure_logging(config.runtime.log_dir, "integration-check")
    healthy = config.cameras[0].model_copy(deep=True)
    healthy.id = "check1"
    healthy.source.type, healthy.source.uri = "rtsp", args.source
    healthy.source.index = None
    healthy.source.timestamp_watermark = True
    healthy.output_url = f"{args.server}/check1"
    broken = healthy.model_copy(deep=True)
    broken.id = "check2"
    broken.source.uri = f"{args.server}/intentionally_missing"
    broken.output_url = f"{args.server}/check2"
    config.cameras = [healthy, broken]
    service = PipelineService(config)
    encoder_pids = []
    report = {"execution": args.mode}
    probe_stop, probe_ready, connected = threading.Event(), threading.Event(), threading.Event()
    probe_ready.set()
    samples = []
    observer = threading.Thread(
        target=probe,
        args=(healthy.output_url, probe_stop, samples, probe_ready, connected),
        daemon=True,
    )
    try:
        with TestClient(create_app(config, service)) as client:
            observer.start()

            def statuses():
                response = client.get("/cameras")
                assert response.status_code == 200
                return response.json()

            status = wait_for(
                lambda: (
                    s
                    if (s := statuses())["check1"].get("output", {}).get("frames", 0) >= 15
                    else None
                )
            )
            report["healthy_with_broken_camera"] = status
            assert connected.wait(20), "Output RTSP was not decoded before kill"
            wait_for(lambda: len(samples) >= 10 and samples[-1][1] < 800)
            assert status["check2"]["capture"]["reconnect_attempts"] > 0
            encoder_pid = status["check1"]["output"]["ffmpeg_pid"]
            encoder_pids.append(encoder_pid)
            decoded_before_kill = len(samples)
            killed_at = time.monotonic()
            psutil.Process(encoder_pid).kill()
            status = wait_for(
                lambda: (
                    s
                    if (s := statuses())["check1"].get("output", {}).get("restarts", 0) > 0
                    and s["check1"]["output"]["frames"] > 20
                    else None
                )
            )
            report["after_encoder_restart"] = status["check1"]
            encoder_pids.append(status["check1"]["output"]["ffmpeg_pid"])
            assert encoder_pids[-1] != encoder_pids[0]
            wait_for(
                lambda: (
                    len(samples) >= decoded_before_kill + 5
                    and samples[-1][0] - samples[-1][1] / 1000 > killed_at
                )
            )
            report["kill_recovery_seconds"] = time.monotonic() - killed_at
            report["post_kill_capture_decoded"] = True
            report["decoded_frames_before_kill"] = decoded_before_kill
            report["decoded_frames_after_recovery"] = len(samples)
            response = client.put("/cameras/check1/enabled?enabled=false")
            assert response.status_code == 200, response.text
            assert response.json()["worker_alive"] is False
            assert statuses()["check2"]["worker_alive"] is True
            response = client.put("/cameras/check1/enabled?enabled=true")
            assert response.status_code == 200
            status = wait_for(
                lambda: (
                    s
                    if (s := statuses())["check1"].get("output", {}).get("frames", 0) >= 10
                    else None
                )
            )
            encoder_pids.append(status["check1"]["output"]["ffmpeg_pid"])
            report["after_enable"] = status["check1"]
            events = client.get("/events?camera_id=check1").json()
            assert all(Path(row["snapshot_path"]).is_file() for row in events)
            report["persisted_events"] = len(events)
        report["shutdown"] = service.stop()
        assert all(report["shutdown"].values())
        assert not any(psutil.pid_exists(pid) for pid in encoder_pids)
        report["all_encoders_reaped"] = True
        report["passed"] = True
    finally:
        probe_stop.set()
        service.stop()
        if observer.ident is not None:
            observer.join(7)
        report["probe_stopped"] = not observer.is_alive()
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
