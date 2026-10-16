"""Benchmark identical pipelines in thread/process modes with RTSP output probes.

Requires a running MediaMTX and live/replayed input(s). JSON includes raw per-camera
latency samples, status, hardware, memory and CPU. Capture-to-decode latency starts
AFTER OpenCV read, so it excludes source/decoder input buffering.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from multicamera_ai.cameras.worker import opencv_capture_factory
from multicamera_ai.config import load_config
from multicamera_ai.service import PipelineService, configure_logging
from multicamera_ai.telemetry import read_stamp


def probe(url, stop, samples, ready, connected):
    last = None
    while not stop.is_set():
        capture = opencv_capture_factory(url)
        try:
            if not capture.isOpened():
                stop.wait(0.5)
                continue
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                stamp = read_stamp(frame)
                if stamp is None or stamp == last:
                    continue
                last = stamp
                connected.set()
                if ready.is_set():
                    samples.append(
                        (time.monotonic(), (datetime.now(UTC) - stamp[0]).total_seconds() * 1000)
                    )
        finally:
            capture.release()


def run(config, count, mode, args):
    config = config.model_copy(deep=True)
    template = config.cameras[0]
    config.cameras = [
        config.cameras[i % len(config.cameras)].model_copy(deep=True) for i in range(count)
    ]
    for i, camera in enumerate(config.cameras, 1):
        camera.id = f"bench{i}"
        camera.enabled = True
        camera.source.timestamp_watermark = args.source_watermark
        camera.output_url = f"{args.server}/bench{i}"
        if args.source:
            camera.source = template.source.model_copy(
                update={
                    "type": "rtsp",
                    "uri": args.source,
                    "index": None,
                    "timestamp_watermark": args.source_watermark,
                }
            )
    config.runtime.execution = mode
    config.streaming.enabled = True
    config.streaming.watermark = True
    config.runtime.database = str(Path(args.output).parent / f"events-{mode}-{count}.db")
    service = PipelineService(config)
    stop, ready = threading.Event(), threading.Event()
    samples = {c.id: [] for c in config.cameras}
    connected = {c.id: threading.Event() for c in config.cameras}
    threads = [
        threading.Thread(
            target=probe,
            args=(c.output_url, stop, samples[c.id], ready, connected[c.id]),
            daemon=True,
        )
        for c in config.cameras
    ]
    resources = []
    parent = psutil.Process()
    service.start()
    try:
        for thread in threads:
            thread.start()
        startup_deadline = time.monotonic() + 30
        while time.monotonic() < startup_deadline:
            statuses = service.status()
            if all(event.is_set() for event in connected.values()):
                break
            if any(
                v.get("ai", {}).get("state") == "error" or v.get("error") for v in statuses.values()
            ):
                logging.error("Camera failed during benchmark startup; recording failed run")
                break
            time.sleep(0.2)
        else:
            logging.error("Not all camera outputs ready after 30s; recording degraded run")
        time.sleep(args.warmup)
        ready.set()
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            processes = [parent] + parent.children(recursive=True)
            rss, cpu = 0, 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    cpu += sum(process.cpu_times()[:2])
                except psutil.NoSuchProcess:
                    pass
            resources.append({"at": time.monotonic(), "rss_bytes": rss, "cpu_seconds": cpu})
            time.sleep(1)
        ready.clear()
        status = service.status()
    finally:
        stop.set()
        clean = service.stop()
        for thread in threads:
            thread.join(7)
    cameras = {}
    for camera_id, values in samples.items():
        latency = [v[1] for v in values]
        fps = len(values) / args.seconds
        cameras[camera_id] = {
            "decoded_unique_fps": fps,
            "frames": len(values),
            "capture_to_decode_p50_ms": float(np.percentile(latency, 50)) if latency else None,
            "capture_to_decode_p95_ms": float(np.percentile(latency, 95)) if latency else None,
            "capture_to_decode_max_ms": max(latency) if latency else None,
            "capture_to_decode_under_1s": bool(latency) and max(latency) < 1000,
            "meets_capture_to_decode_target": fps >= 10 and bool(latency) and max(latency) < 1000,
            "samples_ms": latency,
        }
    cpu_cores = None
    if len(resources) > 1:
        cpu_cores = (resources[-1]["cpu_seconds"] - resources[0]["cpu_seconds"]) / (
            resources[-1]["at"] - resources[0]["at"]
        )
    return {
        "execution": mode,
        "camera_count": count,
        "seconds": args.seconds,
        "warmup_seconds": args.warmup,
        "cameras": cameras,
        "status": status,
        "shutdown": clean,
        "peak_rss_bytes": max((v["rss_bytes"] for v in resources), default=0),
        "average_cpu_cores": cpu_cores,
        "resources": resources,
        "all_probes_stopped": all(not t.is_alive() for t in threads),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--modes", nargs="+", choices=["threads", "processes"], default=["threads", "processes"]
    )
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument(
        "--source-watermark",
        action="store_true",
        help="Input carries replay.py --watermark timestamps; measures end-to-end",
    )
    parser.add_argument("--source", help="Use the same looping RTSP source for every camera")
    parser.add_argument("--server", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--output", default="artifacts/benchmark/results.json")
    args = parser.parse_args()
    if min(args.counts) < 1 or args.seconds < 1 or args.warmup < 0:
        parser.error("Counts/seconds must be positive and warmup nonnegative")
    config = load_config(args.config)
    configure_logging(config.runtime.log_dir, "benchmark")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "model": config.model.model_dump(),
        "streaming": config.streaming.model_dump() | {"enabled": True, "watermark": True},
        "input_source_override": args.source,
        "source_watermark": args.source_watermark,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "ultralytics", "opencv-python-headless")
        },
        "code_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("src").rglob("*.py")))
            + Path(__file__).read_bytes()
        ).hexdigest(),
        "latency_scope": (
            "Source replay capture to RTSP output decode"
            if args.source_watermark
            else "OpenCV capture-read completion to RTSP output decode; excludes input buffering"
        ),
        "runs": [],
    }
    for count in args.counts:
        for mode in args.modes:
            logging.info("Benchmark count=%d mode=%s", count, mode)
            result["runs"].append(run(config, count, mode, args))
            path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
