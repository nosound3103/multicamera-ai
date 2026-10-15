import argparse
import json
import math
from pathlib import Path

import yaml

from multicamera_ai.config import AppConfig

EXPECTED_SHAPES = {
    "in_entrance": "line",
    "out_entrance": "line",
    "restricted_area": "polygon",
}

# Small allowance for floating-point arithmetic at an image boundary, in pixels.
BOUNDARY_TOLERANCE = 1e-6


def normalize_point(point: list, width: int, height: int) -> dict[str, float]:
    """Map a Labelme coordinate to [0, 1] for normalized_to_pixel().

    Labelme works in continuous image coordinates and can store x == width
    or y == height at the right/bottom boundary. OpenCV indexes raster pixels
    from 0 to width-1 and 0 to height-1, so border points are clamped to the
    final pixel before normalization. Interior coordinates are unchanged.
    """
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Invalid Labelme point: {point!r}")

    x, y = point
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (x, y)):
        raise ValueError(f"Non-numeric Labelme point: {point!r}")

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"Non-finite Labelme point: {point!r}")

    if not (
        -BOUNDARY_TOLERANCE <= x <= width + BOUNDARY_TOLERANCE
        and -BOUNDARY_TOLERANCE <= y <= height + BOUNDARY_TOLERANCE
    ):
        raise ValueError(f"Point outside image: {point!r} (size={width}x{height})")

    x = max(0.0, min(float(x), width - 1))
    y = max(0.0, min(float(y), height - 1))

    return {
        "x": round(x / (width - 1), 8),
        "y": round(y / (height - 1), 8),
    }


def convert_labelme(data: dict) -> tuple[list[dict], list[dict]]:
    """Extract two counting lines and one polygon from Labelme data."""
    width = data.get("imageWidth")
    height = data.get("imageHeight")

    if type(width) is not int or type(height) is not int or width < 2 or height < 2:
        raise ValueError("Labelme imageWidth/imageHeight must be integers >= 2")

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("Labelme JSON must contain a shapes list")

    found: dict[str, list[dict[str, float]]] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError("Each Labelme shape must be a JSON object")

        label = shape.get("label")
        if label not in EXPECTED_SHAPES:
            continue
        if label in found:
            raise ValueError(f"Duplicate Labelme label: {label}")

        expected_kind = EXPECTED_SHAPES[label]
        points = shape.get("points")
        if shape.get("shape_type") != expected_kind or not isinstance(points, list):
            raise ValueError(f"{label} must be a Labelme {expected_kind}")

        if expected_kind == "line" and len(points) != 2:
            raise ValueError(f"{label} must contain exactly 2 points")
        if expected_kind == "polygon" and len(points) < 3:
            raise ValueError(f"{label} must contain at least 3 points")

        try:
            normalized = [normalize_point(p, width, height) for p in points]
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        found[label] = normalized

    missing = set(EXPECTED_SHAPES) - set(found)
    if missing:
        raise ValueError(f"Missing Labelme labels: {', '.join(sorted(missing))}")

    lines = [
        {"id": name, "start": found[name][0], "end": found[name][1]}
        for name in ("in_entrance", "out_entrance")
    ]
    zones = [{"id": "restricted_area", "points": found["restricted_area"]}]
    return lines, zones


def update_yaml(labelme_path: Path, config_path: Path, camera_id: str) -> None:
    """Replace only the target camera's regions after validating the config."""
    data = json.loads(labelme_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Labelme JSON root must be an object")
    lines, zones = convert_labelme(data)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("cameras"), list):
        raise ValueError("Config YAML must contain a cameras list")

    matches = [
        cam for cam in config["cameras"] if isinstance(cam, dict) and cam.get("id") == camera_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one camera with id={camera_id!r}")

    matches[0]["lines"] = lines
    matches[0]["zones"] = zones

    AppConfig.model_validate(config)

    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Updated {camera_id} in {config_path}")
    print("Lines: in_entrance, out_entrance | Zone: restricted_area")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labelme_json", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--camera", default="cam1")
    args = parser.parse_args()
    update_yaml(args.labelme_json, args.config, args.camera)


if __name__ == "__main__":
    main()
