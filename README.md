# tamserve-boxes

The Rust port of [`tamserve.boxes`](../docs/cpu-boxes.md): find and follow boxes on many cameras with **YOLOX-Nano on gray frames, on the CPU through OpenVINO**, for edge boxes with no NVIDIA GPU such as an Intel N100. It's one binary: no Python, no PyTorch. It reads cameras in-process with GStreamer, keeps them connected, reports their health, and spreads them over several boxes with a coordinator.

| | |
|---|---|
| **Cameras** | RTSP, HTTP or files, decoded in this process by GStreamer (no ffmpeg per camera). The gray plane is taken as the decoder makes it. VA-API decodes on the integrated GPU where the box has one. |
| **Passwords** | in a config file or files only their owner can read, set on the RTSP source as properties. Never on a command line, in a URL or in a log. |
| **Reconnects** | a stream that fails or stalls is reopened, with backoff from 1 s doubling to 30 s, with jitter |
| **Health** | per camera: state, frames decoded and used, reconnects, frame age, last error. Served as Prometheus `/metrics`, JSON `/status`, and OpenAI-compatible `/v1/chat/completions` |
| **Throughput** | several OpenVINO requests in flight; each letterboxes its share of a look on its own thread |
| **Many boxes** | a coordinator assigns cameras to boxes by capacity, moves a dead box's cameras within seconds, evens out the load, and collects every result |
| **Same results as Python** | byte-identical folded model, letterbox bit-identical to OpenCV, detections within 5e-5, the same ids on every frame |

## Build

```bash
# Ubuntu / Debian: GStreamer and its plugins; the VA driver for decoding on an Intel integrated GPU
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav intel-media-va-driver
cargo build --release          # target/release/tamserve-boxes
cargo run --release --example gst_probe   # which decoders this box has (vah264dec: VA-API is usable)
```

OpenVINO's C library is loaded at run time. Use either a release from Intel, or the one inside the `openvino` pip wheel. The wheel names it with a version suffix the Rust bindings don't recognise, so link it under its plain name:

```bash
pip install openvino
OV=$(python3 -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs
mkdir -p ~/.local/lib/openvino && ln -sf "$OV"/libopenvino_c.so.* ~/.local/lib/openvino/libopenvino_c.so
export LD_LIBRARY_PATH=~/.local/lib/openvino:$OV
```

Features:
- `fold` (on by default): makes the 1-channel model. Needs `onnx-protobuf`, which is MPL-2.0.
- `cameras` (on by default): GStreamer, the runner and the coordinator.

`--no-default-features` builds the detector and tracker alone.

## One box

```bash
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
tamserve-boxes fold yolox_nano.onnx yolox_nano_gray.onnx      # once: exact on gray input
tamserve-boxes run --config cameras.toml
```

[`config/cameras.toml`](config/cameras.toml) is a complete example. Each camera has a `url` without credentials, a `user`, and one of:
- `password_file`: a file that must be mode 600 or the load refuses it;
- `password_env`: an environment variable;
- `password`: in the config itself, which then must be mode 600.

A URL with `user:pass@` in it is refused.

`run` prints one JSON line per camera per step and logs state changes by camera name and host:

```json
{"camera":"gate","frame":186,"time_ms":1795072982293,"objects":[{"id":1,"label":"person","score":0.81,"box":[410.2,120.5,560.8,480.1]}]}
```

### Health and metrics

With `[http] listen` set (127.0.0.1 by default; add `token_file` to require a bearer token), the box serves:

| endpoint | what |
|---|---|
| `GET /metrics` | Prometheus, per camera: `tamserve_camera_up`, `_state{state=…}`, `_frames_decoded_total`, `_frames_used_total`, `_reconnects_total`, `_frame_age_seconds`, `_objects`. Per loop: `tamserve_steps_total`, `tamserve_step_seconds_total`/`_last`, `tamserve_looks_total` |
| `GET /status` | the same as JSON, with each camera's last error and latest objects |
| `GET /v1/models` and `POST /v1/chat/completions` | OpenAI-compatible. `model` is a camera's name or `"all"`, and the reply's message is that camera's latest objects and health as JSON |

Alert on `tamserve_camera_frame_age_seconds > 10`, or on `rate(tamserve_camera_reconnects_total[5m]) > 0`.

### Reconnects

Each camera has a supervisor thread:
1. It opens the stream.
2. It watches for an error, an end of stream (on a live source), or a stall: no frame for `stall_seconds`.
3. On any of those, it closes the stream, waits, and opens it again.

The wait starts at `backoff_first` and doubles each failure up to `backoff_max`, scaled by a random 0.5–1, so cameras that failed together don't all reconnect at once. A connection that held for 30 s resets the backoff. When a camera reconnects, its tracks start afresh: the frames before the gap aren't its recent past.

### Throughput

`requests = N` puts N OpenVINO inference requests in flight, each in its own OpenVINO stream. A look's frames are split between them, and each request letterboxes its share on its own thread. Splitting changes no frame's finds (tested).

On a 2-core VM at saturation (8 cameras, a look every frame), 2 requests raised throughput from 13.5 to 15.0 frames/s per camera at the same CPU. A 4-core N100 should gain more; measure it with `tamserve-boxes bench --requests N`.

### VA-API

`decoder = "vaapi"` ranks GStreamer's VA-API decoders (`vah264dec`, `vah265dec`) first. `"software"` ranks them out, and `"auto"` leaves GStreamer's own ranking. Decoded frames come back to the CPU, gray, for detection. If `vaapi` is asked for and there are no VA decoders, `run` refuses to start and names the packages to install. **The VA-API path has not been tested on hardware**: the machine this was built on has no Intel GPU. Run `gst_probe` on the box first.

## Many boxes

Each box has its own `cameras.toml`, passwords and all. The coordinator's config ([`config/coordinator.toml`](config/coordinator.toml)) lists only camera names and their weights, so **no camera address or password crosses the network**.

```bash
tamserve-boxes coordinator --config coordinator.toml
tamserve-boxes box --config cameras.toml --coordinator 10.0.0.2:7400 --token-file /etc/tamserve/coordinator.token \
    --name box-north --capacity 30
```

- **Assignment.** Each camera goes to a box that has it in its config, preferring the one with the most free capacity. Weights let a camera that looks every frame count as more than one that looks every third. `boxes = [...]` pins a camera to certain boxes.
- **Failover.** A box whose connection drops, or that sends no heartbeat for `dead_seconds`, loses its cameras to the other boxes. In the test below, a box killed with SIGKILL had its cameras running on the other box within **1.4 s** (the longest gap in any camera's results).
- **Rebalancing.** While boxes differ in fullness by more than `rebalance_gap`, one camera at a time moves from the fullest to the emptiest. A camera that just moved stays put for 30 s, so cameras don't bounce between boxes.
- **Results.** Every box sends its results to the coordinator, which prints them as JSON lines with a `box` field. It also serves `/metrics` (cameras per box, box load and heartbeat age, unassigned cameras, moves), `/status`, and `/v1/chat/completions` for every camera, whichever box serves it.
- **When the coordinator is away**, boxes keep serving their cameras and keep trying to reconnect, so analytics don't stop. If cameras were moved in the meantime, two boxes may serve one camera until the coordinator is back. That's deliberate: for safety monitoring a duplicate is better than a gap.

**Security of the link.** A box and the coordinator each prove they know a shared token, at least 32 characters in a mode-600 file. The proof is HMAC-SHA256 over fresh nonces, under a 10 s deadline for the whole handshake, so a peer that stalls part way can't hold the door. Every message after that carries an HMAC over its direction, sequence number and bytes, so a message can't be forged, changed, replayed or reordered on the path (tested). Messages are **not encrypted**: results (labels and boxes) can be read by anyone on the path. Keep boxes on a private network or a VPN if that matters.

## Fine-tuning on your sites

[`notebooks/finetune_gray.ipynb`](../notebooks/finetune_gray.ipynb) is one cell. Point `DATA` at labelled frames from your cameras (COCO JSON or YOLO txt), and it:

1. folds YOLOX-Nano's released weights to 1 channel, checked exact against the 3-channel model;
2. trains on your frames, read as gray, with Sharpness-Aware Minimization around SGD;
3. scores the released and the fine-tuned model on held-out frames, per frame and class ("is there one?" at `DECISION_SCORE`), with MCC, and a bootstrap interval for the gain;
4. exports the fine-tuned model only if it passes the gate: MCC at least the released model's plus `MIN_GAIN`, no class worse by more than `MAX_CLASS_DROP`, and optionally a gain that's clearly above chance (`REQUIRE_SURE_GAIN`). The ONNX file is 1-channel with any batch, in the released model's format, so `tamserve-boxes` runs it as it is (with `names` set if the classes changed).

It also checks that the released weights, exported through the same path, reproduce YOLOX's released ONNX (to 5.5e-5).

What was learned building it, on a 2-core CPU with demo data:
- **Momentum 0.9 diverged.** YOLOX-Nano with frozen BatchNorm is sharp: momentum compounds the step about tenfold, and even learning rate 1e-4 blew up. Plain SGD at 1e-4 lowers the loss steadily, so that's the default (`MOMENTUM = 0`). The gate and the export use an average of the weights (EMA), as YOLOX's own training does.
- **Keep every class your labels mark.** Dropping labels turns those objects into "background" the model must unlearn, which cost the classes that were kept.
- **The demo shows the pipeline, not a gain.** Left with no data, it downloads coco128 (GPL-3.0 as Ultralytics packages it), makes it CCTV-like, and scores people. But those pictures are in COCO's training set, which the released model already learned from. There, fine-tuning held MCC level (0.553 → 0.553) and the gate passed as "no worse"; earlier settings that made it worse were refused. A real gain needs your own footage, and enough held-out frames for the bootstrap interval to be narrow: a few hundred per camera type, held out by camera rather than by frame.

## CPU and memory

`tamserve-boxes bench` runs N synthetic 720p cameras at a set frame rate in this one process, or with `--separate`, one process per camera. Measured on a 2-core cloud VM (Intel Xeon, OpenVINO 2026.4, float32, `look_every=3`), **not an N100**. Decoding is left out, and both benchmarks hold about 27 MB of frames per camera, which is included below.

| cameras × fps | setup | CPU (cores) | memory | fps each, kept |
|---|---|---|---|---|
| 8 × 25 | **Rust, one process** | **0.93** | **338 MB** | **25.0** |
| 8 × 25 | Python, one process | 0.98 | 433 MB | 25.0 |
| 8 × 25 | Rust, 8 processes | 1.02 | 880 MB | 23.6 |
| 8 × 25 | Python, 8 processes | 1.05 | 1,544 MB | 21.9 |

## Tests

```bash
cargo test --release                                   # 28 tests; the 4 that need the model run when these are set:
TAMSERVE_YOLOX_ONNX=yolox_nano.onnx TAMSERVE_DOG_JPG=dog.jpg cargo test --release
scripts/rtsp_test.sh yolox_nano_gray.onnx some_h264.mp4    # end to end, needs mediamtx and ffmpeg
```

- **Unit and integration tests:**
  - the config: password files, permissions, URLs with credentials, mistakes;
  - the coordinator over loopback: wrong token, a stalled handshake cut off, changed and replayed messages refused, spreading by capacity, failover, cameras a box doesn't know, a box reconnecting;
  - the tracker tests ported from Python;
  - the model tests, including that splitting a look across requests changes nothing.
- **`scripts/rtsp_test.sh`**, against a real RTSP server that needs a password (all 10 checks pass):
  - four cameras up;
  - a wrong password reported and retried;
  - no password on any command line;
  - Prometheus and OpenAI-compatible endpoints;
  - cameras down with the server and back by themselves;
  - a coordinator with two boxes, one killed with SIGKILL, its cameras moved in 1.4 s.

## Not yet tested

- VA-API decoding on an Intel GPU, and everything on an N100.
- Live cameras other than the local RTSP server above.
- More than 4 cameras through GStreamer at once.
- A long soak. Run `bench` and `rtsp_test.sh` on the target box first, then a day-long run with `/metrics` scraped.

## Licences

Apache-2.0, as tamserve. Dependencies:
- `openvino` and the GStreamer bindings: Apache-2.0 / MIT.
- GStreamer itself: LGPL, linked at run time.
- `onnx-protobuf`: MPL-2.0, only with `fold`.

YOLOX and its weights are Megvii's, under Apache-2.0.
