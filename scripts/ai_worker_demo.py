"""Compatibility entry point for the shared capture/AI demo."""

from multicamera_ai.vision.demo import main

if __name__ == "__main__":
    main(
        default_config="config/cameras.local.yaml",
        default_output="artifacts/ai_worker_demo.mp4",
        default_max_frames=300,
    )
