from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import AppConfig


class ConfigError(Exception):
    """Raised when application configuration is invalid."""


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)

    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Cannot read config file '{config_path}': {exc}") from exc

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{config_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config '{config_path}' must contain a YAML mapping")

    try:
        config = AppConfig.model_validate(data)
        # YAML file paths are relative to the YAML directory, never the process cwd.
        for camera in config.cameras:
            if camera.source.type == "file" and camera.source.uri:
                source_path = Path(camera.source.uri).expanduser()
                if not source_path.is_absolute():
                    source_path = config_path.resolve().parent / source_path
                camera.source.uri = str(source_path.resolve())
        return config
    except ValidationError as exc:
        details = []

        for error in exc.errors():
            location = ".".join(map(str, error["loc"]))
            location = location or "<root>"
            details.append(f"  - {location}: {error['msg']}")

        message = "\n".join(details)

        raise ConfigError(f"Invalid configuration in '{config_path}':\n{message}") from exc
