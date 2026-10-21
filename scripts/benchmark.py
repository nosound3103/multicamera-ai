"""Benchmark identical pipelines in thread/process modes with RTSP output probes.

Requires a running MediaMTX and live/replayed input(s). JSON includes raw per-camera
latency samples, status, hardware, memory and CPU. With --source-watermark, latency
starts at replay capture; otherwise it starts AFTER the pipeline's OpenCV read.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import logging
import platform
import socket
import ssl
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import psutil

from multicamera_ai.cameras.worker import opencv_capture_factory
from multicamera_ai.config import load_config
from multicamera_ai.service import PipelineService, configure_logging
from multicamera_ai.telemetry import read_stamp


def rtsp_output_available(url, timeout=0.5):
    """Quiet DESCRIBE preflight: do not open noisy OpenCV probes on missing paths."""
    parsed = urlsplit(url)
    address = parsed.netloc.rsplit("@", 1)[-1]
    target = parsed._replace(netloc=address).geturl()
    headers = [f"DESCRIBE {target} RTSP/1.0", "CSeq: 1", "Accept: application/sdp"]
    if parsed.username is not None:
        credentials = f"{unquote(parsed.username)}:{unquote(parsed.password or '')}"
        headers.append("Authorization: Basic " + base64.b64encode(credentials.encode()).decode())
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 554), timeout) as connection:
            if parsed.scheme == "rtsps":
                connection = ssl.create_default_context().wrap_socket(
                    connection, server_hostname=parsed.hostname
                )
            try:
                connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())
                response = b""
                while b"\r\n" not in response and len(response) < 4096:
                    part = connection.recv(4096)
                    if not part:
                        return False
                    response += part
                status = response.split(b"\r\n", 1)[0].split()
                # Let OpenCV negotiate servers requiring Digest authentication.
                return len(status) >= 2 and status[1] in (b"200", b"401")
            finally:
                if parsed.scheme == "rtsps":
                    connection.close()
    except (OSError, ValueError):
        return False


def camera_failed(status):
    return (
        not status.get("worker_alive", False)
        or bool(status.get("error"))
        or status.get("ai", {}).get("state") in {"error", "ended", "stopped"}
        or status.get("capture", {}).get("state") in {"error", "ended", "stopped"}
    )


def publisher_ready(status):
    return (
        not camera_failed(status)
        and status.get("status_age_seconds", float("inf")) < 3
        and status.get("capture", {}).get("state") == "running"
        and status.get("ai", {}).get("state") == "running"
        and not status.get("ai", {}).get("last_error")
        and not (status.get("output") or {}).get("last_error")
        and status.get("ai", {}).get("processed_frames", 0) > 0
        and (status.get("output") or {}).get("frames", 0) > 0
        and (status.get("output") or {}).get("ffmpeg_pid") is not None
    )


def wait_for_outputs(service, threads, connected, timeout):
    """Start each probe only after its AI/publisher reports progress."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = service.status()
        for camera_id, thread in threads.items():
            status = statuses.get(camera_id, {})
            if camera_failed(status):
                return False, "camera_failed_during_startup"
            if thread.ident is None and publisher_ready(status):
                logging.info("Publisher ready camera=%s; waiting for RTSP decode", camera_id)
                thread.start()
        if all(event.is_set() for event in connected.values()):
            return True, None
        time.sleep(0.2)
    return False, "startup_timeout"


def probe(url, stop, samples, ready, connected):
    last = None
    while not stop.is_set():
        if not rtsp_output_available(url):
            stop.wait(0.5)
            continue
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


def prepare_config(config, count, mode, args):
    config = config.model_copy(deep=True)
    template = config.cameras[0]
    config.cameras = [
        config.cameras[i % len(config.cameras)].model_copy(deep=True) for i in range(count)
    ]
    for i, camera in enumerate(config.cameras, 1):
        camera.id = f"cam{i}"
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
    config.streaming.fps = args.output_fps
    config.streaming.enabled = True
    config.streaming.watermark = True
    config.runtime.database = str(Path(args.output).parent / f"events-{mode}-{count}.db")
    return config


def ai_measurements(before, after):
    """Use cumulative inference counters and worker report times, excluding warmup."""
    result = {}
    for camera_id, end in after.items():
        start = before.get(camera_id, {})
        frames = end.get("ai", {}).get("processed_frames", 0) - start.get("ai", {}).get(
            "processed_frames", 0
        )
        elapsed = end.get("updated_at", 0) - start.get("updated_at", 0)
        healthy = (
            start.get("ai", {}).get("state") == "running"
            and start.get("status_age_seconds", float("inf")) < 3
            and end.get("worker_alive", False)
            and not end.get("error")
            and end.get("ai", {}).get("state") == "running"
            and not end.get("ai", {}).get("last_error")
            and end.get("status_age_seconds", float("inf")) < 3
            and frames > 0
            and elapsed > 0
        )
        result[camera_id] = {
            "ai_processed_frames": max(0, frames),
            "ai_measured_seconds": max(0, elapsed),
            "ai_processing_fps": max(0, frames) / elapsed if elapsed > 0 else 0,
            "ai_healthy": healthy,
        }
    return result


def run(config, count, mode, args):
    config = prepare_config(config, count, mode, args)
    service = PipelineService(config)
    stop, ready = threading.Event(), threading.Event()
    samples = {c.id: [] for c in config.cameras}
    connected = {c.id: threading.Event() for c in config.cameras}
    threads = {
        c.id: threading.Thread(
            target=probe,
            args=(c.output_url, stop, samples[c.id], ready, connected[c.id]),
            daemon=True,
        )
        for c in config.cameras
    }
    resources = []
    parent = psutil.Process()
    started = ended = time.monotonic()
    startup_started = started
    startup_seconds = warmup_seconds = 0.0
    before = {}
    failure_reason = None
    measurement_started = False
    status = {}
    service.start()
    try:
        logging.info("Waiting up to %ss for AI, publishers and RTSP probes", args.startup_timeout)
        outputs_ready, failure_reason = wait_for_outputs(
            service, threads, connected, args.startup_timeout
        )
        startup_seconds = time.monotonic() - startup_started
        if outputs_ready:
            logging.info("All camera outputs decoded; warming up for %ss", args.warmup)
            warmup_started = time.monotonic()
            while time.monotonic() - warmup_started < args.warmup:
                status = service.status()
                if any(camera_failed(value) for value in status.values()):
                    failure_reason = "camera_failed_during_warmup"
                    break
                time.sleep(min(0.2, max(0, args.warmup - (time.monotonic() - warmup_started))))
            warmup_seconds = time.monotonic() - warmup_started
        if failure_reason is None:
            before = service.status()
            if not all(publisher_ready(before.get(c.id, {})) for c in config.cameras):
                failure_reason = "camera_not_ready_after_warmup"
        if failure_reason is None:
            logging.info("All cameras ready; starting %ss measurement", args.seconds)
            started = time.monotonic()
            measurement_started = True
            ready.set()
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
                time.sleep(min(1, max(0, args.seconds - (time.monotonic() - started))))
            ready.clear()
            ended = time.monotonic()
        else:
            logging.error("Benchmark measurement skipped: %s", failure_reason)
        status = service.status()
    finally:
        ready.clear()
        stop.set()
        clean = service.stop()
        for thread in threads.values():
            if thread.ident is not None:
                thread.join(7)
    ai = ai_measurements(before, status)
    cameras = {}
    elapsed = ended - started
    for camera_id, values in samples.items():
        values = [v for v in values if started <= v[0] <= ended]
        latency = [v[1] for v in values]
        fps = len(values) / elapsed if elapsed > 0 else 0
        cameras[camera_id] = {
            **ai[camera_id],
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
        "valid": (
            measurement_started
            and failure_reason is None
            and all(event.is_set() for event in connected.values())
            and set(status) == set(samples)
            and all(v["ai_healthy"] for v in ai.values())
            and all(c["frames"] > 0 for c in cameras.values())
            and all(clean.values())
            and all(not t.is_alive() for t in threads.values())
        ),
        "workload": "capture + YOLO tracking + events + overlay + RTSP encode/decode",
        "execution": mode,
        "camera_count": count,
        "seconds": args.seconds,
        "measured_seconds": elapsed,
        "warmup_seconds": args.warmup,
        "actual_warmup_seconds": warmup_seconds,
        "startup_seconds": startup_seconds,
        "measurement_started": measurement_started,
        "failure_reason": failure_reason,
        "all_outputs_ready": all(event.is_set() for event in connected.values()),
        "cameras": cameras,
        "status": status,
        "shutdown": clean,
        "peak_rss_bytes": max((v["rss_bytes"] for v in resources), default=0),
        "average_cpu_cores": cpu_cores,
        "resources": resources,
        "all_probes_stopped": all(not t.is_alive() for t in threads.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--modes", nargs="+", choices=["threads", "processes"], default=["threads", "processes"]
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=0,
        help="Publisher FPS cap; 0 runs without pacing (default)",
    )
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument(
        "--source-watermark",
        action="store_true",
        help="Input carries replay.py --watermark timestamps; measures end-to-end",
    )
    parser.add_argument("--source", help="Use the same looping RTSP source for every camera")
    parser.add_argument("--server", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--output", default="artifacts/benchmark/results.json")
    args = parser.parse_args()
    if (
        args.output_fps < 0
        or min(args.counts) < 1
        or args.seconds < 1
        or args.warmup < 0
        or args.startup_timeout < 1
    ):
        parser.error("Counts/seconds must be positive; warmup/output-fps nonnegative")
    config = load_config(args.config)
    from multicamera_ai.config.models import validate_model_files

    validate_model_files(config)
    import torch

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda_version": torch.version.cuda,
        }
    configure_logging(config.runtime.log_dir, "benchmark")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": gpu,
        "models": {c.id: config.model_for(c).model_dump() for c in config.cameras},
        "streaming": config.streaming.model_dump()
        | {"enabled": True, "watermark": True, "fps": args.output_fps},
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
    return 0 if all(run["valid"] for run in result["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
