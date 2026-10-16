from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class EventType(StrEnum):
    LINE_CROSSING = "line_crossing"
    ZONE_ENTRY = "zone_entry"


class Event(BaseModel):
    """An event detected by the engine, before saving a snapshot."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    event_id: UUID = Field(default_factory=uuid4)

    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    camera_id: str = Field(min_length=1)

    track_id: int = Field(
        ge=0,
        strict=True,
    )

    event_type: EventType

    region_id: str = Field(min_length=1)

    direction: Literal["IN", "OUT"] | None = None

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        if self.event_type == EventType.LINE_CROSSING and self.direction is None:
            raise ValueError("line_crossing requires direction IN or OUT")

        if self.event_type == EventType.ZONE_ENTRY and self.direction is not None:
            raise ValueError("zone_entry must not have a direction")

        return self


class RecordedEvent(Event):
    """An event with a snapshot path, ready for SQLite."""

    snapshot_path: str = Field(min_length=1)
