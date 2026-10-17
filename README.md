# Multi-Camera AI

Real-time multi-camera AI video processing system

## Docker

Docker Compose starts the Python API and MediaMTX RTSP server. The default image
uses CPU PyTorch; the GPU override uses PyTorch 2.10.0 with CUDA 12.8. On Windows,
GPU containers require Docker Desktop's WSL 2 backend and an NVIDIA driver with
WSL support.

Place `yolo26n.pt` and `yolo11-newdata.pt` in `models/`, and the four input
videos at `assets/cam_1.mp4` through `assets/cam_4.mp4`. Weights, videos, logs, snapshots,
databases and benchmark measurements are kept outside the image.

```sh
# CPU demo (four distinct replay sources)
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

### Benchmark 1, 4 and 8 cameras

Stop `ai` before benchmarking to avoid running two sets of inference workers.
Start the replay sources and run the benchmark container on the same network:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile demo up -d mediamtx replay
docker compose -f compose.yaml -f compose.gpu.yaml stop ai
docker compose -f compose.yaml -f compose.gpu.yaml --profile benchmark run --rm benchmark
docker compose -f compose.yaml -f compose.gpu.yaml --profile demo down
```

The default benchmark compares threads and processes with 1, 4 and 8 pipelines
for 60 seconds each after 15 seconds of warmup. Startup allows up to 180 seconds.
Cameras 5-8 reuse sources/regions/models from 1-4 with separate trackers and
outputs; no additional videos or replay encoders are required. Capture timestamps
are carried through RTSP. Invalid runs (worker loss or failed cleanup) return a
nonzero exit code; a valid run can still fail the FPS/latency target.
Results are written to ignored `artifacts/docker-benchmark/results.json`.
Both replay and measurement consume resources on the test machine. A successful
command exit alone does not mean the performance target was met: inspect FPS,
latency, per-camera errors and shutdown flags in the result.

## Camera models and annotations

Each camera selects a named profile with `model: vehicle` or `model: person`.
Profiles live under `models` in YAML and contain weights, class IDs, confidence,
imgsz, device and CPU thread limits. The vehicle weight uses IDs 0 bus, 1 car,
2 motorbike, 3 truck; the person profile uses COCO class 0 (person). Other COCO vehicle IDs are
2 car, 3 motorcycle, 5 bus and 7 truck when adding a broader profile. A legacy top-level `model` remains an optional
fallback for cameras without a profile. Model and file-source paths resolve
relative to the YAML file. `--check` validates YAML only; normal startup also
checks selected weights and class IDs before starting enabled cameras.

Add a camera entry with a unique ID/output URL, select its profile, then set
its source and optional `lines`/`zones`. No camera-count limit is hard-coded.
One `counting_line` counts both IN and OUT. With left-to-right endpoints,
`in_direction: negative_to_positive` means motion toward the positive side
(roughly down the image) is IN; reverse the setting to swap the directions.
The overlay keeps the line clear; IN/OUT counters and AI processing FPS appear
in the camera panel. Import the corresponding
Labelme `counting_line` and `restricted_area` into an existing camera with:

```sh
python scripts/labelme_to_yaml.py assets/cam_1/cam_1.json --config config/cameras.local.yaml --camera cam1
```

The replay service reads enabled file sources from `cameras.local.yaml` and
publishes each as `input_<camera-id>`. Keep matching camera IDs in Docker's RTSP
config. Use `--count 4` to replay only the first four cameras for a 1-4 camera test.
Each camera owns its model instance and tracker even when sharing a profile;
selecting a profile does not implement shared or batched inference.

### Measured thread/process comparison

Tested 2026-10-17: i5-11400H, 16 GB RAM, RTX 3050 Laptop 4 GB;
Docker Desktop WSL2 with 8 logical CPUs / 7.65 GiB. CUDA, vehicle model,
imgsz 320, 15 FPS replay, 15 s warmup, 60 s measurement per run.
FPS counts unique decoded RTSP frames; latency is source replay capture to
output decode, excluding player rendering/cache. Replay shares the test machine.

| Cameras | Execution | FPS per camera | Worst latency | Result |
| --- | --- | --- | --- | --- |
| 1 | threads | 14.92 | 395 ms | Pass |
| 1 | processes | 14.96 | 338 ms | Pass |
| 4 | threads | 5.28-6.16 | 931 ms | FPS below target |
| 4 | processes | 10.87-11.99 | 681 ms | Pass |
| 8 | threads | 1.78-2.08 | 3807 ms | Below target |
| 8 | processes | Invalid | Not comparable | 2 workers OOM-killed |

Processes are selected for four cameras: about twice the decoded throughput
of threads with clean shutdown. A previous 25 FPS replay baseline achieved only
7.86-8.87 FPS with four processes. Bounding file decoding to one thread, using
15 FPS replay and limiting BLAS/OpenMP threads improved throughput without
changing detection weights or imgsz. A separate tuned four-camera run also
passed at 10.98-11.77 FPS, maximum latency 711 ms. These are 60 s measurements,
not a long-duration reliability claim.

Benchmark containers default to 7 GiB RAM and 8 GiB total RAM+swap, configurable
with BENCHMARK_MEMORY_LIMIT and BENCHMARK_MEMORY_SWAP_LIMIT. Eight processes did
not fit that budget: the surviving workers' FPS does not represent eight-camera
capacity. Remaining workers cleaned up after OOM; killed workers were reported
as unclean. Supporting eight process pipelines needs lower per-worker memory or
more memory capacity; it is not currently verified on this laptop.

SIGINT (the Ctrl+C signal) was tested with four real CUDA/RTSP pipelines in both
execution modes: all camera components stopped cleanly and containers exited 0.
Abrupt worker kill has a regression test; a lock-free stop flag prevents a killed
worker from blocking shutdown through an abandoned IPC event lock. A 15 s loss
of only input_cam1 triggered reconnect/session reset while cameras 2-4 continued
producing frames. Webcam verification was excluded at the owner's request.
