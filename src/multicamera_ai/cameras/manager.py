from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .worker import CameraStatus, CameraWorker, CaptureFactory, FramePacket

if TYPE_CHECKING:
    from multicamera_ai.config.schemas import AppConfig


@dataclass(frozen=True, slots=True)
class CameraDefinition:
    camera_id: str
    source: str | int
    is_file: bool = False
    enabled: bool = True


class CameraManager:
    def __init__(
        self,
        cameras: Iterable[CameraDefinition],
        *,
        reconnect_delay: float = 1.0,
        file_realtime: bool = True,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self._workers: dict[str, CameraWorker] = {}
        self._enabled: set[str] = set()
        for camera in cameras:
            if camera.camera_id in self._workers:
                raise ValueError(f"Duplicate camera ID: {camera.camera_id}")
            self._workers[camera.camera_id] = CameraWorker(
                camera.camera_id,
                camera.source,
                is_file=camera.is_file,
                reconnect_delay=reconnect_delay,
                file_realtime=file_realtime,
                capture_factory=capture_factory,
            )
            if camera.enabled:
                self._enabled.add(camera.camera_id)

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        **kwargs,
    ) -> CameraManager:
        definitions = []
        for camera in config.cameras:
            source_type = camera.source.type
            if source_type not in {"file", "rtsp", "webcam"}:
                raise ValueError(f"Camera {camera.id}: unsupported source type")
            if source_type == "webcam":
                if camera.source.index is None:
                    raise ValueError(f"Camera {camera.id}: webcam requires source.index")
                source: str | int = camera.source.index
            else:
                if not camera.source.uri:
                    raise ValueError(f"Camera {camera.id}: {source_type} requires source.uri")
                source = camera.source.uri
            definitions.append(
                CameraDefinition(
                    camera_id=camera.id,
                    source=source,
                    is_file=source_type == "file",
                    enabled=camera.enabled,
                )
            )
        return cls(definitions, **kwargs)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._workers)

    def start_all(self) -> None:
        for camera_id in self._enabled:
            self._workers[camera_id].start()

    def stop_all(self, timeout: float = 5.0) -> dict[str, bool]:
        return {camera_id: worker.stop(timeout) for camera_id, worker in self._workers.items()}

    def start_camera(self, camera_id: str) -> None:
        self._workers[camera_id].start()

    def stop_camera(self, camera_id: str, timeout: float = 5.0) -> bool:
        return self._workers[camera_id].stop(timeout)

    def latest(self, camera_id: str) -> FramePacket | None:
        return self._workers[camera_id].get_latest()

    def wait_for_frame(
        self, camera_id: str, *, after_sequence: int = 0, timeout: float = 1.0
    ) -> FramePacket | None:
        return self._workers[camera_id].wait_for_frame(
            after_sequence=after_sequence, timeout=timeout
        )

    def status(self) -> dict[str, CameraStatus]:
        return {camera_id: worker.status() for camera_id, worker in self._workers.items()}
