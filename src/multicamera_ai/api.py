from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import AwareDatetime

from multicamera_ai.service import PipelineService


def create_app(config, service=None):
    service = service or PipelineService(config)

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="Multi-camera AI", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    def health():
        return {"status": "ok", "execution": config.runtime.execution}

    @app.get("/cameras")
    def cameras():
        return service.status()

    @app.put("/cameras/{camera_id}/enabled")
    def enable(camera_id: str, enabled: bool):
        try:
            if enabled:
                service.start_camera(camera_id)
            elif not service.stop_camera(camera_id):
                raise HTTPException(503, "Camera shutdown was not clean; inspect logs/status")
        except KeyError:
            raise HTTPException(404, "Unknown camera") from None
        return service.status()[camera_id]

    @app.get("/events")
    def events(
        camera_id: str | None = None,
        start: AwareDatetime | None = None,
        end: AwareDatetime | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        if camera_id is not None and camera_id not in service.status():
            raise HTTPException(404, "Unknown camera")
        try:
            return service.store.list_events(camera_id=camera_id, start=start, end=end, limit=limit)
        except ValueError:
            raise HTTPException(422, "Invalid time range; start must be <= end") from None

    return app
