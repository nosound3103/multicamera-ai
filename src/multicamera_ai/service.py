from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import threading
import time
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import psutil

from multicamera_ai.cameras import CameraWorker
from multicamera_ai.config.schemas import AppConfig
from multicamera_ai.events import EventRecorder, SnapshotWriter, SQLiteEventStore
from multicamera_ai.events.recorder import AsyncEventRecorder
from multicamera_ai.pipeline import AIWorker
from multicamera_ai.pipeline.publisher import RTSPPublisher

logger = logging.getLogger(__name__)


def configure_logging(folder: str, name: str = "service") -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        Path(folder) / f"{name}.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def _offer(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


class ProcessStop:
    """Lock-free stop flag: a killed child cannot leave an IPC lock held."""

    def __init__(self, context):
        self._flag = context.RawValue("b", False)

    def set(self):
        self._flag.value = True

    def is_set(self):
        return bool(self._flag.value)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        return self.is_set()


def run_camera(config, camera, stop_event, channel, child_logging=False):
    """Same camera pipeline in a thread or spawned process; tracker is never shared."""
    if child_logging:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        configure_logging(config.runtime.log_dir, camera.id)
    capture = ai = recorder = publisher = None
    clean = True
    failure = None
    try:
        source = camera.source.index if camera.source.type == "webcam" else camera.source.uri
        capture = CameraWorker(
            camera.id,
            source,
            is_file=camera.source.type == "file",
            timestamp_watermark=camera.source.timestamp_watermark,
            reconnect_delay=config.runtime.reconnect_delay,
        )
        store = SQLiteEventStore(config.runtime.database)
        recorder = AsyncEventRecorder(
            EventRecorder(store, SnapshotWriter(config.runtime.snapshots)),
            config.runtime.event_queue_size,
        )
        ai = AIWorker(
            camera=camera, model_config=config.model_for(camera), source=capture, recorder=recorder
        )
        if config.streaming.enabled:
            publisher = RTSPPublisher(camera.id, camera.output_url, ai, config.streaming)
        capture.start()
        ai.start()
        if publisher:
            publisher.start()
        while not stop_event.is_set():
            value = {
                "camera_id": camera.id,
                "capture": asdict(capture.status()),
                "ai": asdict(ai.status()),
                "events": recorder.status(),
                "output": publisher.status() if publisher else None,
                "updated_at": time.time(),
            }
            _offer(channel, value)
            logger.info(
                "metrics camera=%s capture_fps=%.2f ai_fps=%.2f inference_ms=%.1f "
                "capture_to_ai_ms=%.1f reconnects=%d output=%s events=%s",
                camera.id,
                value["capture"]["capture_fps"],
                value["ai"]["processing_fps"],
                value["ai"]["inference_ms"],
                value["ai"]["capture_to_ai_ms"],
                value["capture"]["reconnect_attempts"],
                value["output"],
                value["events"],
            )
            stop_event.wait(1.0)
    except Exception as exc:
        failure = type(exc).__name__
        logger.error("Camera pipeline failed camera=%s error=%s", camera.id, type(exc).__name__)
        _offer(
            channel,
            {"camera_id": camera.id, "error": type(exc).__name__, "updated_at": time.time()},
        )
    finally:
        deadline = time.monotonic() + config.runtime.shutdown_timeout
        # Stop output first to break a blocked pipe. Stop producers before draining disk writes.
        for component in (publisher, ai, capture, recorder):
            if component is not None:
                clean = component.stop(max(0, deadline - time.monotonic())) and clean
        _offer(
            channel,
            {
                "camera_id": camera.id,
                "stopped": True,
                "clean_shutdown": clean,
                "error": failure,
                "updated_at": time.time(),
            },
        )
        logger.info("Camera stopped camera=%s clean=%s", camera.id, clean)
        if not clean:
            logger.error("Camera shutdown timed out camera=%s", camera.id)
    return clean


def run_camera_process(config, camera, stop_event, channel, child_logging=True):
    """Exit the owned worker after explicit cleanup, without native atexit teardown.

    CUDA interpreter teardown can outlive the cleanup deadline on WSL. All video,
    event and FFmpeg resources are closed by run_camera; only plain status data
    crosses this process boundary. Flush that data before the terminal exit.
    """
    exit_code = 1
    try:
        exit_code = 0 if run_camera(config, camera, stop_event, channel, child_logging) else 1
    except Exception:
        logger.exception("Camera process failed camera=%s", camera.id)
    finally:
        try:
            channel.close()
            channel.join_thread()
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            exit_code = 1
        finally:
            os._exit(exit_code)


class PipelineService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.store = SQLiteEventStore(config.runtime.database)
        self._context = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._handles = {}
        self._status = {}
        self._enabled = {c.id: c.enabled for c in config.cameras}
        self._definitions = {c.id: c for c in config.cameras}

    def start(self):
        if self.config.streaming.enabled and not shutil.which(self.config.streaming.ffmpeg):
            raise ValueError("FFmpeg executable not found; install FFmpeg or set streaming.ffmpeg")
        try:
            for camera in self.config.cameras:
                if camera.enabled:
                    self.start_camera(camera.id)
        except Exception:
            self.stop()
            raise

    def start_camera(self, camera_id):
        with self._lock:
            camera = self._definitions[camera_id]
            old = self._handles.get(camera_id)
            if old and old[0].is_alive():
                return
            if old:
                old[0].join()
                if hasattr(old[2], "close"):
                    old[2].close()
                    old[2].join_thread()
            processes = self.config.runtime.execution == "processes"
            stop = ProcessStop(self._context) if processes else threading.Event()
            channel = self._context.Queue(maxsize=2) if processes else queue.Queue(maxsize=2)
            constructor = self._context.Process if processes else threading.Thread
            worker = constructor(
                target=run_camera_process if processes else run_camera,
                args=(self.config, camera, stop, channel, processes),
                name=f"pipeline-{camera_id}",
                daemon=False,
            )
            self._status[camera_id] = {"camera_id": camera_id, "state": "starting"}
            self._handles[camera_id] = (worker, stop, channel)
            self._enabled[camera_id] = True
            worker.start()

    def _read_status(self, camera_id):
        handle = self._handles.get(camera_id)
        if handle:
            while True:
                try:
                    self._status[camera_id] = handle[2].get_nowait()
                except queue.Empty:
                    break
        status = dict(self._status.get(camera_id, {"camera_id": camera_id, "state": "stopped"}))
        status["enabled"] = self._enabled[camera_id]
        status["worker_alive"] = bool(handle and handle[0].is_alive())
        if not status["worker_alive"] and status["enabled"]:
            status["state"] = "error"
            status.setdefault("error", "Pipeline worker exited")
        status["status_age_seconds"] = max(0, time.time() - status.get("updated_at", time.time()))
        return status

    def status(self):
        with self._lock:
            return {camera_id: self._read_status(camera_id) for camera_id in self._definitions}

    def stop_camera(self, camera_id):
        with self._lock:
            if camera_id not in self._definitions:
                raise KeyError(camera_id)
            self._enabled[camera_id] = False
            handle = self._handles.get(camera_id)
            if not handle:
                return True
            worker, stop, channel = handle
            stop.set()
            deadline = time.monotonic() + self.config.runtime.shutdown_timeout + 2
            while worker.is_alive() and time.monotonic() < deadline:
                self._read_status(camera_id)
                worker.join(0.05)
            forced = False
            if worker.is_alive() and self.config.runtime.execution == "processes":
                # Include FFmpeg descendants; a terminated parent cannot run its finally block.
                try:
                    descendants = psutil.Process(worker.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for child in reversed(descendants):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                worker.terminate()
                worker.join(3)
                psutil.wait_procs(descendants, timeout=3)
                forced = True
            result = self._read_status(camera_id)
            exit_ok = self.config.runtime.execution != "processes" or worker.exitcode == 0
            clean = (
                not worker.is_alive()
                and exit_ok
                and result.get("clean_shutdown", False)
                and not forced
            )
            if not clean:
                logger.error("Unclean shutdown camera=%s forced=%s", camera_id, forced)
            if not worker.is_alive() and hasattr(channel, "close"):
                channel.close()
                channel.join_thread()
                del self._handles[camera_id]
            return clean

    def stop(self):
        # Signal all cameras before joining any one of them.
        with self._lock:
            for _, stop, _ in self._handles.values():
                stop.set()
        results = {camera_id: self.stop_camera(camera_id) for camera_id in self._definitions}
        return results
