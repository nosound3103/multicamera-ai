from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from multicamera_ai.events import (
    Event,
    EventType,
    RecordedEvent,
)


def test_create_line_crossing_event():
    event = Event(
        camera_id="cam1",
        track_id=12,
        event_type=EventType.LINE_CROSSING,
        region_id="counting_line",
        direction="IN",
    )

    assert event.direction == "IN"
    assert event.event_type == EventType.LINE_CROSSING
    assert event.occurred_at.tzinfo is not None
    assert event.event_id is not None


def test_create_zone_entry_event():
    event = Event(
        camera_id="cam1",
        track_id=7,
        event_type=EventType.ZONE_ENTRY,
        region_id="restricted_area",
    )

    assert event.direction is None


def test_each_event_has_unique_id():
    def create_event():
        return Event(
            camera_id="cam1",
            track_id=7,
            event_type=EventType.ZONE_ENTRY,
            region_id="restricted_area",
        )

    assert create_event().event_id != create_event().event_id


@pytest.mark.parametrize(
    ("event_type", "direction"),
    [
        (EventType.LINE_CROSSING, None),
        (EventType.ZONE_ENTRY, "IN"),
        (EventType.ZONE_ENTRY, "OUT"),
    ],
)
def test_reject_wrong_direction(event_type, direction):
    with pytest.raises(ValidationError):
        Event(
            camera_id="cam1",
            track_id=1,
            event_type=event_type,
            region_id="entrance",
            direction=direction,
        )


def test_reject_naive_timestamp():
    with pytest.raises(ValidationError, match="timezone"):
        Event(
            camera_id="cam1",
            track_id=1,
            event_type=EventType.ZONE_ENTRY,
            region_id="restricted_area",
            occurred_at=datetime(2026, 10, 15, 8),
        )


def test_accept_aware_timestamp():
    moment = datetime(
        2026,
        10,
        15,
        8,
        tzinfo=UTC,
    )

    event = Event(
        camera_id="cam1",
        track_id=1,
        event_type=EventType.ZONE_ENTRY,
        region_id="restricted_area",
        occurred_at=moment,
    )

    assert event.occurred_at == moment


@pytest.mark.parametrize(
    "bad_data",
    [
        {"track_id": -1},
        {"track_id": True},
        {"camera_id": ""},
        {"region_id": ""},
        {"unknown_field": "ignored?"},
    ],
)
def test_reject_invalid_fields(bad_data):
    payload = {
        "camera_id": "cam1",
        "track_id": 1,
        "event_type": "zone_entry",
        "region_id": "restricted_area",
    }

    payload.update(bad_data)

    with pytest.raises(ValidationError):
        Event.model_validate(payload)


def test_recorded_event_requires_snapshot():
    event = Event(
        camera_id="cam1",
        track_id=1,
        event_type=EventType.ZONE_ENTRY,
        region_id="restricted_area",
    )

    with pytest.raises(
        ValidationError,
        match="snapshot_path",
    ):
        RecordedEvent.model_validate(event.model_dump())

    record = RecordedEvent.model_validate(
        {
            **event.model_dump(),
            "snapshot_path": ("storage/snapshots/cam1/a.jpg"),
        }
    )

    assert record.event_id == event.event_id
    assert record.snapshot_path.endswith("a.jpg")

    serialized = record.model_dump(mode="json")
    assert serialized["event_type"] == "zone_entry"


def test_event_is_immutable():
    event = Event(
        camera_id="cam1",
        track_id=1,
        event_type=EventType.ZONE_ENTRY,
        region_id="restricted_area",
    )

    with pytest.raises(ValidationError, match="frozen"):
        event.track_id = 2
