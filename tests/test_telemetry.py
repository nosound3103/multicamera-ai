from datetime import UTC, datetime

import pytest

np = pytest.importorskip("numpy")

from multicamera_ai.telemetry import read_stamp, stamp_frame  # noqa: E402


def test_timestamp_roundtrip_and_corruption_detection():
    image = np.zeros((120, 640, 3), dtype=np.uint8)
    stamp = datetime(2026, 10, 16, 4, 30, 20, 123000, UTC)
    stamp_frame(image, stamp, 42)
    assert read_stamp(image) == (stamp, 42)
    image[0:16, 0:8] = 255 - image[0:16, 0:8]
    assert read_stamp(image) is None


def test_timestamp_rejects_small_frame():
    assert read_stamp(np.zeros((100, 100, 3), dtype=np.uint8)) is None
