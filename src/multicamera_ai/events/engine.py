from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Literal

from multicamera_ai.events.models import Event, EventType
from multicamera_ai.vision.geometry import (
    Point2D,
    cross_product,
    crossing_direction,
    point_in_polygon,
    side_of_line,
)


@dataclass(frozen=True, slots=True)
class TrackedPoint:
    track_id: int
    anchor: Point2D  # Bottom-center of YOLO bbox, in pixels


@dataclass(frozen=True, slots=True)
class CountLine:
    id: str
    start: Point2D
    end: Point2D
    count_as: Literal["IN", "OUT"]


@dataclass(frozen=True, slots=True)
class RestrictedZone:
    id: str
    points: tuple[Point2D, ...]


@dataclass(slots=True)
class _LineState:
    stable_point: Point2D | None = None


@dataclass(slots=True)
class _ZoneState:
    inside_streak: int = 0
    outside_streak: int = 0
    active: bool = False


@dataclass(slots=True)
class _TrackState:
    last_seen: int
    last_count_frame: int | None = None
    lines: dict[str, _LineState] = field(default_factory=dict)
    zones: dict[str, _ZoneState] = field(default_factory=dict)


class EventEngine:
    """Turn tracked positions into de-duplicated events.

    One engine belongs to one camera and one tracker session.
    Call process_frame once per video frame, even if there
    are no tracked detections.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        line_margin_px: float = 8.0,
        line_cooldown_frames: int = 20,
        zone_confirm_frames: int = 3,
        zone_release_frames: int = 3,
        max_observation_gap_frames: int = 5,
        max_missing_frames: int = 45,
    ) -> None:
        if not camera_id:
            raise ValueError("camera_id must not be empty")

        if not isfinite(line_margin_px) or line_margin_px < 0:
            raise ValueError("line_margin_px must be >= 0")

        if (
            min(
                line_cooldown_frames,
                zone_confirm_frames,
                zone_release_frames,
                max_observation_gap_frames,
                max_missing_frames,
            )
            < 1
        ):
            raise ValueError("Frame thresholds must be positive")

        if max_missing_frames < max_observation_gap_frames:
            raise ValueError("max_missing_frames must be >= max_observation_gap_frames")

        self.camera_id = camera_id
        self.line_margin_px = line_margin_px
        self.line_cooldown_frames = line_cooldown_frames
        self.zone_confirm_frames = zone_confirm_frames
        self.zone_release_frames = zone_release_frames
        self.max_observation_gap_frames = max_observation_gap_frames
        self.max_missing_frames = max_missing_frames

        self._tracks: dict[int, _TrackState] = {}
        self._last_frame: int | None = None

    @property
    def active_track_count(self) -> int:
        """Number of track IDs currently retained in memory."""
        return len(self._tracks)

    def reset(self) -> None:
        """Reset when a camera or tracker session restarts."""
        self._tracks.clear()
        self._last_frame = None

    def process_frame(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackedPoint],
        lines: Sequence[CountLine],
        zones: Sequence[RestrictedZone],
    ) -> list[Event]:
        """Process all tracked points from one video frame."""

        # ------------------------------------------
        # 1. Validate frame and track IDs
        # ------------------------------------------
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 1
            or (self._last_frame is not None and frame_index <= self._last_frame)
        ):
            raise ValueError("frame_index must increase and start from 1")

        track_ids = [item.track_id for item in detections]

        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Duplicate track_id within one frame")

        if any(
            isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0
            for track_id in track_ids
        ):
            raise ValueError("track_id must be a non-negative integer")

        if any(not all(isfinite(value) for value in item.anchor) for item in detections):
            raise ValueError("anchor coordinates must be finite")

        self._last_frame = frame_index
        events: list[Event] = []

        # ------------------------------------------
        # 2. Process each tracked object
        # ------------------------------------------
        for item in detections:
            state = self._tracks.get(item.track_id)

            if state is None or (frame_index - state.last_seen > self.max_observation_gap_frames):
                # Do not connect positions separated
                # by a large gap in tracking.
                state = _TrackState(last_seen=frame_index)
                self._tracks[item.track_id] = state

            # --------------------------------------
            # 3. Counting lines
            # --------------------------------------
            for line in lines:
                line_state = state.lines.setdefault(
                    line.id,
                    _LineState(),
                )

                dx = line.end[0] - line.start[0]
                dy = line.end[1] - line.start[1]
                line_length = hypot(dx, dy)

                if line_length == 0:
                    raise ValueError(f"Degenerate counting line: {line.id}")

                signed_distance = (
                    cross_product(
                        line.start,
                        line.end,
                        item.anchor,
                    )
                    / line_length
                )

                # Ignore positions close to the line.
                if abs(signed_distance) <= self.line_margin_px:
                    continue

                previous = line_state.stable_point

                if previous is not None:
                    before_side = side_of_line(
                        line.start,
                        line.end,
                        previous,
                    )
                    after_side = side_of_line(
                        line.start,
                        line.end,
                        item.anchor,
                    )

                    if before_side != after_side:
                        crossed = crossing_direction(
                            previous,
                            item.anchor,
                            line.start,
                            line.end,
                        )

                        cooldown_over = (
                            state.last_count_frame is None
                            or frame_index - state.last_count_frame >= self.line_cooldown_frames
                        )

                        if crossed is not None and cooldown_over:
                            events.append(
                                Event(
                                    camera_id=self.camera_id,
                                    track_id=item.track_id,
                                    event_type=EventType.LINE_CROSSING,
                                    region_id=line.id,
                                    direction=line.count_as,
                                )
                            )

                            # Shared cooldown across IN and OUT.
                            state.last_count_frame = frame_index

                # Update stable point even when cooldown
                # suppresses an otherwise valid crossing.
                line_state.stable_point = item.anchor

            # --------------------------------------
            # 4. Restricted zones
            # --------------------------------------
            for zone in zones:
                zone_state = state.zones.setdefault(
                    zone.id,
                    _ZoneState(),
                )

                inside = point_in_polygon(
                    item.anchor,
                    zone.points,
                )

                if inside:
                    zone_state.inside_streak += 1
                    zone_state.outside_streak = 0

                    if (
                        not zone_state.active
                        and zone_state.inside_streak >= self.zone_confirm_frames
                    ):
                        zone_state.active = True

                        events.append(
                            Event(
                                camera_id=self.camera_id,
                                track_id=item.track_id,
                                event_type=EventType.ZONE_ENTRY,
                                region_id=zone.id,
                            )
                        )

                else:
                    zone_state.inside_streak = 0
                    zone_state.outside_streak += 1

                    if zone_state.outside_streak >= self.zone_release_frames:
                        zone_state.active = False

            state.last_seen = frame_index

        # ------------------------------------------
        # 5. Clean up tracks that disappeared
        # ------------------------------------------
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if frame_index - state.last_seen > self.max_missing_frames
        ]

        for track_id in expired:
            del self._tracks[track_id]

        return events
