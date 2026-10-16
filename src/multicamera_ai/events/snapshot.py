import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multicamera_ai.events.models import Event


def _camera_folder(camera_id: str) -> str:
    """Prevent path traversal without changing common IDs such as 'cam1'."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", camera_id):
        return camera_id
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", camera_id).strip("_") or "camera"
    suffix = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:8]
    return f"{prefix[:48]}_{suffix}"


@dataclass(slots=True)
class SnapshotWriter:
    root: Path
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be from 1 to 100")

    def save(self, event: Event, frame: Any) -> Path:
        """Encode and atomically publish a full annotated BGR frame as JPEG."""
        import cv2  # Optional vision dependency, needed only to save snapshots.

        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("Snapshot frame cannot be empty")

        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            raise OSError("OpenCV failed to encode JPEG snapshot")

        camera_dir = self.root / _camera_folder(event.camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / f"{event.event_id.hex}.jpg"
        temp_path = camera_dir / f"{event.event_id.hex}.jpg.tmp"
        if path.exists():
            raise FileExistsError(f"Snapshot already exists: {path}")
        try:
            # 'xb' ensures two attempts for the same UUID never silently overwrite.
            with temp_path.open("xb") as output:
                output.write(encoded.tobytes())
            temp_path.replace(path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return path
