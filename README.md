# Multi-Camera AI

Real-time multi-camera AI video processing system

## Docker

Docker Compose starts the Python API and MediaMTX RTSP server. The default image
uses CPU PyTorch; the GPU override uses PyTorch 2.10.0 with CUDA 12.8. On Windows,
GPU containers require Docker Desktop's WSL 2 backend and an NVIDIA driver with
WSL support.

Place `yolo26n.pt` in the repository root. For the replay demo, also place the
input video at `assets/people-walking.mp4`. Weights, videos, logs, snapshots,
databases and benchmark measurements are kept outside the image.

```sh
# CPU demo (four replay sources)
docker compose build ai
docker compose --profile demo up -d

# API status: http://localhost:8000/cameras
# RTSP outputs: rtsp://localhost:8554/output_cam1 through output_cam4
docker compose logs -f ai
docker compose down
```

```sh
# NVIDIA GPU demo
docker compose -f compose.yaml -f compose.gpu.yaml build ai
docker compose -f compose.yaml -f compose.gpu.yaml --profile demo up -d
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps --entrypoint python ai -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
docker compose -f compose.yaml -f compose.gpu.yaml down
```

Edit `config/cameras.docker.yaml` (CPU) or `config/cameras.docker.gpu.yaml` (GPU)
to choose cameras, model settings, counting lines and forbidden zones. Camera
URLs inside the Compose network use `mediamtx:8554`; use `localhost:8554` when
opening the published streams from the host. For cameras on the host itself,
Docker Desktop provides `host.docker.internal`. The API health check confirms
API availability; inspect `/cameras` for individual pipeline errors.

The bind mounts preserve `storage/` (SQLite and snapshots) and `logs/` across
container recreation. `docker compose down` sends SIGTERM to the API and allows
60 seconds for pipeline cleanup. The replay and benchmark tools receive SIGINT
so their cleanup blocks run too.

### Benchmark 1–4 cameras

Stop `ai` before benchmarking to avoid running two sets of inference workers.
Start the replay sources and run the benchmark container on the same network:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile demo up -d mediamtx replay
docker compose -f compose.yaml -f compose.gpu.yaml stop ai
docker compose -f compose.yaml -f compose.gpu.yaml --profile benchmark run --rm benchmark
docker compose -f compose.yaml -f compose.gpu.yaml --profile demo down
```

The default benchmark measures 1, 2, 3 and 4 process pipelines for 60 seconds
each after 15 seconds of warmup, with capture timestamps carried through RTSP.
Results are written to ignored `artifacts/docker-benchmark/results.json`.
Both replay and measurement consume resources on the test machine. A successful
command exit alone does not mean the performance target was met: inspect FPS,
latency, per-camera errors and shutdown flags in the result.
