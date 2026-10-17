"""Exercise the real demo loop without downloading YOLO weights.

Needs the optional vision dependencies (cv2 + numpy). Fake Ultralytics is injected.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from multicamera_ai.events import EventType, SQLiteEventStore  # noqa: E402


def test_demo_video_writes_sqlite_events_and_snapshots(tmp_path, monkeypatch):
    from multicamera_ai.vision import demo

    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 15.0, (160, 120))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    def point(x, y):
        return SimpleNamespace(x=x, y=y)

    camera = SimpleNamespace(
        id="cam1",
        source=SimpleNamespace(type="file", uri=str(source)),
        lines=[
            SimpleNamespace(
                id="counting_line",
                start=point(0, 0.5),
                end=point(1, 0.5),
                in_direction="negative_to_positive",
            ),
        ],
        zones=[
            SimpleNamespace(
                id="restricted_area",
                points=[
                    point(0.57, 0.08),
                    point(0.87, 0.08),
                    point(0.87, 0.4),
                    point(0.57, 0.4),
                ],
            )
        ],
    )
    fake_config = SimpleNamespace(
        cameras=[camera],
        model=SimpleNamespace(
            weights="unused.pt", confidence=0.4, imgsz=640, class_ids=[0, 2], device="cpu"
        ),
    )
    fake_config.model_for = lambda camera: fake_config.model
    monkeypatch.setattr(demo, "load_config", lambda _: fake_config)

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Tensor:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class Box:
        def __init__(self, tid, coords):
            self.id = Scalar(tid)
            self.xyxy = [Tensor(coords)]

    class FakeResult:
        def __init__(self, frame, boxes):
            self.frame = frame
            self.boxes = Boxes(boxes)

        def plot(self):
            return self.frame.copy()

    # The demo iterates result.boxes; the mock implements both id and iteration.
    class Boxes:
        id = True

        def __init__(self, boxes):
            self.boxes = boxes

        def __iter__(self):
            return iter(self.boxes)

    class Model:
        def __init__(self, weights):
            self.n = 0

        def track(self, frame, **kwargs):
            self.n += 1
            y2 = 30 if self.n == 1 else 80
            boxes = [Box(1, (10, 4, 30, y2)), Box(2, (100, 2, 120, 25))]
            return [FakeResult(frame, boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))

    db_path = tmp_path / "db" / "events.db"
    output = tmp_path / "output.mp4"
    args = argparse.Namespace(
        config="unused.yaml",
        camera="cam1",
        input=None,
        output=str(output),
        db=str(db_path),
        snapshots=str(tmp_path / "snapshots"),
        max_frames=5,
    )
    demo.run_demo(args)

    records = SQLiteEventStore(db_path).list_events()
    assert sorted(e.event_type.value for e in records) == [
        EventType.LINE_CROSSING.value,
        EventType.ZONE_ENTRY.value,
    ]
    assert next(e.direction for e in records if e.event_type == EventType.LINE_CROSSING) == "IN"
    assert all(Path(e.snapshot_path).exists() for e in records)
    assert output.exists() and output.stat().st_size > 0
