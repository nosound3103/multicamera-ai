import argparse
import logging

from multicamera_ai.config import ConfigError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-camera AI service")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument(
        "--check", action="store_true", help="Validate YAML without starting workers"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logging.getLogger(__name__).error("Configuration failed: %s", exc)
        return 2
    if args.check:
        logging.getLogger(__name__).info("Configuration valid: %d cameras", len(config.cameras))
        return 0
    from multicamera_ai.config.models import validate_model_files

    try:
        validate_model_files(config)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error("Model configuration failed: %s", exc)
        return 2
    import uvicorn

    from multicamera_ai.api import create_app
    from multicamera_ai.service import configure_logging

    configure_logging(config.runtime.log_dir)
    uvicorn.run(
        create_app(config),
        host=config.runtime.host,
        port=config.runtime.port,
        log_config=None,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
