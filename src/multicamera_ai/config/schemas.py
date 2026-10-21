from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_rtsp_url(url: str) -> str:
    try:
        parsed = urlsplit(url)

        if parsed.scheme not in {"rtsp", "rtsps"}:
            raise ValueError

        if not parsed.hostname:
            raise ValueError

        # Accessing port also detects malformed port values.
        _ = parsed.port

    except ValueError:
        raise ValueError("RTSP URL must have a valid rtsp:// or rtsps:// address") from None

    return url


class Point(StrictModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class CountingLine(StrictModel):
    id: str = Field(min_length=1)
    start: Point
    end: Point

    in_direction: Literal[
        "negative_to_positive",
        "positive_to_negative",
    ] = "negative_to_positive"

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        if self.start == self.end:
            raise ValueError("line endpoints must differ")
        return self


class ForbiddenZone(StrictModel):
    id: str = Field(min_length=1)
    points: list[Point] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_polygon(self) -> Self:
        vertices = {(p.x, p.y) for p in self.points}

        if len(vertices) < 3:
            raise ValueError("polygon needs 3 distinct points")

        shifted = self.points[1:] + self.points[:1]

        twice_area = sum(p.x * q.y - q.x * p.y for p, q in zip(self.points, shifted, strict=True))

        if abs(twice_area) < 1e-9:
            raise ValueError("polygon must have non-zero area")

        return self


class VideoSource(StrictModel):
    timestamp_watermark: bool = False
    type: Literal["rtsp", "file", "webcam"]
    uri: str | None = None
    index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.type == "webcam":
            if self.index is None or self.uri is not None:
                raise ValueError("webcam requires index and forbids uri")
        else:
            if not self.uri or not self.uri.strip():
                raise ValueError("RTSP and file sources require uri")

            if self.index is not None:
                raise ValueError("RTSP and file sources forbid index")

            if self.type == "rtsp":
                validate_rtsp_url(self.uri)

        return self


class YOLOConfig(StrictModel):
    half: bool = False
    cpu_threads: int = Field(default=1, ge=1, le=64)
    weights: str = Field(min_length=1)
    device: str = Field(default="cpu", min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)
    imgsz: int = Field(gt=0)
    class_ids: list[int] = Field(min_length=1)

    @field_validator("class_ids")
    @classmethod
    def validate_classes(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("class IDs must be non-negative")

        if len(values) != len(set(values)):
            raise ValueError("duplicate class IDs")

        return values


class CameraConfig(StrictModel):
    model: str | None = Field(default=None, min_length=1)
    id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    enabled: bool = True
    source: VideoSource
    output_url: str
    lines: list[CountingLine] = Field(default_factory=list)
    zones: list[ForbiddenZone] = Field(default_factory=list)

    @field_validator("output_url")
    @classmethod
    def validate_output_url(cls, value: str) -> str:
        return validate_rtsp_url(value)

    @model_validator(mode="after")
    def validate_region_ids(self) -> Self:
        line_ids = [line.id for line in self.lines]
        zone_ids = [zone.id for zone in self.zones]

        if len(line_ids) != len(set(line_ids)):
            raise ValueError("duplicate counting line IDs")

        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("duplicate forbidden zone IDs")

        return self


class RuntimeConfig(StrictModel):
    execution: Literal["threads", "processes"] = "threads"
    database: str = "storage/events.db"
    snapshots: str = "storage/snapshots"
    log_dir: str = "logs"
    reconnect_delay: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    shutdown_timeout: float = Field(default=12.0, gt=0, allow_inf_nan=False)
    event_queue_size: int = Field(default=8, ge=1)
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class StreamingConfig(StrictModel):
    enabled: bool = True
    ffmpeg: str = "ffmpeg"
    # Zero removes publisher pacing; positive values explicitly cap output.
    fps: int = Field(default=0, ge=0)
    width: int = Field(default=960, ge=640, le=3840, multiple_of=2)
    height: int = Field(default=540, ge=120, le=2160, multiple_of=2)
    write_timeout: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    max_frame_age: float = Field(default=0.8, gt=0, allow_inf_nan=False)
    watermark: bool = False


class AppConfig(StrictModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    model: YOLOConfig | None = None
    models: dict[str, YOLOConfig] = Field(default_factory=dict)
    cameras: list[CameraConfig] = Field(min_length=1)

    def model_for(self, camera: CameraConfig) -> YOLOConfig:
        """Resolve a camera profile without sharing its model or tracker instance."""
        if camera.model is not None:
            return self.models[camera.model]
        if self.model is None:
            raise ValueError(f"Camera {camera.id} requires a model profile")
        return self.model

    @model_validator(mode="after")
    def validate_cameras(self) -> Self:
        for camera in self.cameras:
            if camera.model is not None and camera.model not in self.models:
                raise ValueError(f"Camera {camera.id}: unknown model profile '{camera.model}'")
            if camera.model is None and self.model is None:
                raise ValueError(f"Camera {camera.id} requires a model profile or default model")
        ids = [camera.id for camera in self.cameras]

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate camera IDs")

        outputs = [camera.output_url for camera in self.cameras]

        if len(outputs) != len(set(outputs)):
            raise ValueError("duplicate RTSP output URLs")

        return self
