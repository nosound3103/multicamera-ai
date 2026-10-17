import pytest

from multicamera_ai.events import (
    CountLine,
    EventEngine,
    EventType,
    RestrictedZone,
    TrackedPoint,
)

COUNTING_LINE = CountLine(
    id="counting_line",
    start=(0.0, 50.0),
    end=(100.0, 50.0),
)

LEFT_LINE = CountLine(
    id="line_a",
    start=(0.0, 50.0),
    end=(45.0, 50.0),
)

RIGHT_LINE = CountLine(
    id="line_b",
    start=(55.0, 50.0),
    end=(100.0, 50.0),
)

ZONE = RestrictedZone(
    id="restricted_area",
    points=(
        (10.0, 10.0),
        (40.0, 10.0),
        (40.0, 40.0),
        (10.0, 40.0),
    ),
)


def tick(
    engine,
    index,
    *points,
    lines=(LEFT_LINE, RIGHT_LINE),
    zones=(ZONE,),
):
    """Simulate one video frame."""
    return engine.process_frame(
        frame_index=index,
        detections=[TrackedPoint(track_id=track_id, anchor=point) for track_id, point in points],
        lines=lines,
        zones=zones,
    )


def test_first_frame_has_no_crossing():
    engine = EventEngine("cam1")

    assert tick(engine, 1, (12, (20, 30))) == []


def test_counting_line_generates_in_event():
    engine = EventEngine("cam1")

    tick(engine, 1, (12, (20, 30)), lines=(COUNTING_LINE,), zones=())
    events = tick(engine, 2, (12, (20, 70)), lines=(COUNTING_LINE,), zones=())

    assert len(events) == 1
    assert events[0].event_type == EventType.LINE_CROSSING
    assert events[0].direction == "IN"
    assert events[0].region_id == "counting_line"
    assert events[0].track_id == 12
    assert events[0].camera_id == "cam1"


def test_counting_line_generates_out_event():
    engine = EventEngine("cam2")

    tick(engine, 1, (20, (20, 70)), lines=(COUNTING_LINE,), zones=())
    events = tick(engine, 2, (20, (20, 30)), lines=(COUNTING_LINE,), zones=())

    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert events[0].region_id == "counting_line"


def test_crossing_outside_finite_segment():
    engine = EventEngine("cam1")

    tick(engine, 1, (1, (150, 25)), zones=())

    assert tick(engine, 2, (1, (150, 75)), zones=()) == []


def test_small_jitter_near_line():
    engine = EventEngine("cam1")

    tick(engine, 1, (2, (20, 60)), zones=())

    assert tick(engine, 2, (2, (20, 48)), zones=()) == []
    assert tick(engine, 3, (2, (20, 51)), zones=()) == []
    assert tick(engine, 4, (2, (20, 60)), zones=()) == []


def test_cooldown_prevents_duplicate_crossing():
    engine = EventEngine(
        "cam1",
        line_cooldown_frames=5,
    )

    tick(engine, 1, (3, (20, 30)), zones=())

    assert len(tick(engine, 2, (3, (20, 70)), zones=())) == 1

    # Return immediately: do not count again.
    assert tick(engine, 3, (3, (20, 30)), zones=()) == []

    assert tick(engine, 4, (3, (20, 70)), zones=()) == []

    # Later crossing can be counted as a new traversal.
    assert len(tick(engine, 7, (3, (20, 30)), zones=())) == 1


def test_lines_have_independent_cooldowns():
    engine = EventEngine("cam1")

    tick(engine, 1, (4, (20, 30)), zones=())
    assert len(tick(engine, 2, (4, (20, 70)), zones=())) == 1

    tick(engine, 3, (4, (80, 70)), zones=())

    # Separate physical lines must not suppress each other.
    events = tick(engine, 4, (4, (80, 30)), zones=())
    assert len(events) == 1
    assert events[0].direction == "OUT"


def test_tracks_have_independent_states():
    engine = EventEngine("cam1")

    tick(
        engine,
        1,
        (4, (20, 30)),
        (5, (80, 70)),
        zones=(),
    )

    events = tick(
        engine,
        2,
        (4, (20, 70)),
        (5, (80, 30)),
        zones=(),
    )

    assert {(event.track_id, event.direction) for event in events} == {
        (4, "IN"),
        (5, "OUT"),
    }


def test_zone_requires_three_observations():
    engine = EventEngine("cam1")

    assert tick(engine, 1, (6, (20, 20)), lines=()) == []
    assert tick(engine, 2, (6, (20, 20)), lines=()) == []

    events = tick(engine, 3, (6, (20, 20)), lines=())

    assert len(events) == 1
    assert events[0].event_type == EventType.ZONE_ENTRY
    assert events[0].region_id == "restricted_area"
    assert events[0].direction is None

    # Remaining inside does not create more events.
    assert tick(engine, 4, (6, (20, 20)), lines=()) == []


def test_zone_exit_and_reentry():
    engine = EventEngine(
        "cam1",
        zone_confirm_frames=2,
        zone_release_frames=2,
    )

    tick(engine, 1, (7, (50, 50)), lines=())

    assert tick(engine, 2, (7, (20, 20)), lines=()) == []

    assert len(tick(engine, 3, (7, (20, 20)), lines=())) == 1

    # Confirm leaving the polygon.
    tick(engine, 4, (7, (50, 50)), lines=())
    tick(engine, 5, (7, (50, 50)), lines=())

    # A new entry is allowed.
    assert tick(engine, 6, (7, (20, 20)), lines=()) == []

    assert len(tick(engine, 7, (7, (20, 20)), lines=())) == 1


def test_brief_exit_does_not_rearm():
    engine = EventEngine(
        "cam1",
        zone_confirm_frames=2,
        zone_release_frames=3,
    )

    tick(engine, 1, (7, (20, 20)), lines=())

    assert len(tick(engine, 2, (7, (20, 20)), lines=())) == 1

    tick(engine, 3, (7, (50, 50)), lines=())

    assert tick(engine, 4, (7, (20, 20)), lines=()) == []
    assert tick(engine, 5, (7, (20, 20)), lines=()) == []


def test_long_tracking_gap_prevents_false_crossing():
    engine = EventEngine(
        "cam1",
        max_observation_gap_frames=2,
    )

    tick(engine, 1, (8, (20, 30)), zones=())
    tick(engine, 2, zones=())
    tick(engine, 3, zones=())

    assert tick(engine, 4, (8, (20, 70)), zones=()) == []


def test_stale_track_is_removed():
    engine = EventEngine(
        "cam1",
        max_observation_gap_frames=1,
        max_missing_frames=2,
    )

    tick(engine, 1, (8, (20, 30)), zones=())

    assert engine.active_track_count == 1

    tick(engine, 2, zones=())
    tick(engine, 3, zones=())

    assert engine.active_track_count == 1

    tick(engine, 4, zones=())

    assert engine.active_track_count == 0


def test_reset_starts_new_session():
    engine = EventEngine("cam1")

    tick(engine, 1, (1, (20, 30)), zones=())

    engine.reset()

    assert tick(engine, 1, (1, (20, 70)), zones=()) == []


def test_duplicate_track_id_is_rejected():
    engine = EventEngine("cam1")

    with pytest.raises(ValueError, match="Duplicate"):
        tick(
            engine,
            1,
            (1, (20, 20)),
            (1, (30, 30)),
        )


def test_frame_index_must_increase():
    engine = EventEngine("cam1")

    tick(engine, 1)

    with pytest.raises(ValueError, match="frame_index"):
        tick(engine, 1)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        EventEngine("cam1", zone_confirm_frames=0)


def test_degenerate_line_is_rejected():
    engine = EventEngine("cam1")

    with pytest.raises(ValueError, match="Degenerate"):
        engine.process_frame(
            frame_index=1,
            detections=[
                TrackedPoint(1, (20, 30)),
            ],
            lines=[
                CountLine(
                    "counting_line",
                    (0, 0),
                    (0, 0),
                ),
            ],
            zones=[],
        )


def test_nonfinite_anchor_is_rejected():
    engine = EventEngine("cam1")

    with pytest.raises(ValueError, match="finite"):
        tick(
            engine,
            1,
            (1, (20, float("nan"))),
        )


def test_negative_track_id_is_rejected():
    engine = EventEngine("cam1")

    with pytest.raises(ValueError, match="non-negative"):
        tick(engine, 1, (-1, (20, 20)))


@pytest.mark.parametrize("ys,expected", [([30, 70], "IN"), ([70, 30], "OUT")])
def test_same_line_counts_both_directions(ys, expected):
    engine = EventEngine("cam1")
    tick(engine, 1, (1, (20, ys[0])), lines=(COUNTING_LINE,), zones=())
    events = tick(engine, 2, (1, (20, ys[1])), lines=(COUNTING_LINE,), zones=())
    assert [e.direction for e in events] == [expected]


def test_configured_in_direction_is_respected():
    line = CountLine("gate", (0, 50), (100, 50), in_direction="positive_to_negative")
    engine = EventEngine("cam1")
    tick(engine, 1, (1, (20, 70)), lines=(line,), zones=())
    assert tick(engine, 2, (1, (20, 30)), lines=(line,), zones=())[0].direction == "IN"


def test_zone_occlusion_does_not_duplicate_confirmed_entry():
    engine = EventEngine("cam1")
    for i in range(1, 4):
        tick(engine, i, (1, (20, 20)), lines=())
    for i in range(4, 10):
        tick(engine, i, lines=())
    for i in range(10, 14):
        assert tick(engine, i, (1, (20, 20)), lines=()) == []
