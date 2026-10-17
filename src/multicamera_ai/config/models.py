"""Runtime model checks, kept separate from YAML-only validation."""

from pathlib import Path

from .schemas import AppConfig, YOLOConfig


def validate_class_ids(config: YOLOConfig, names: dict[int, str]) -> None:
    missing = sorted(set(config.class_ids) - set(names))
    if missing:
        raise ValueError(
            f"Model '{config.weights}': unknown class IDs {missing}; available classes: {names}"
        )


def validate_model_files(config: AppConfig) -> None:
    """Fail before starting cameras; never download an accidentally misspelled weight."""
    selected = {
        config.model_for(c).weights: config.model_for(c) for c in config.cameras if c.enabled
    }
    for weights in selected:
        if not Path(weights).is_file():
            raise ValueError(f"Model weights not found: {weights}")
    if not selected:
        return
    from ultralytics import YOLO

    # Validate each profile even when two profiles use the same weight file.
    for weights in selected:
        names = YOLO(weights).names
        for camera in config.cameras:
            model = config.model_for(camera)
            if model.weights == weights:
                validate_class_ids(model, names)
