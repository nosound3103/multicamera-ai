import argparse
import logging

from multicamera_ai.config import ConfigError, load_config

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Camera AI configuration checker")

    parser.add_argument(
        "--config",
        default="config/cameras.yaml",
        help="Path to YAML configuration",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Configuration failed:\n%s", exc)
        return 2

    logger.info(
        "Configuration valid: %d cameras, model=%s",
        len(config.cameras),
        config.model.weights,
    )

    for camera in config.cameras:
        logger.info(
            "Camera %s: enabled=%s, source=%s",
            camera.id,
            camera.enabled,
            camera.source.type,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
