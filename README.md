# tamserve-boxes

The Rust port of [`tamserve.boxes`](../docs/cpu-boxes.md): find and follow boxes on many cameras with **YOLOX-Nano on gray frames, on the CPU through OpenVINO**. It's for edge boxes with no NVIDIA GPU, such as an Intel N100. There's no Python and no PyTorch: one binary, plus OpenVINO's C library.

It behaves exactly like the Python version, so the two give the same results on the same frames. Measured against `tamserve.boxes` on the same frames:

| | Rust against Python |
|---|---|
| the 1-channel model `fold` writes | the same weights, byte for byte |
| letterboxed input | bit-identical to OpenCV's `cv2.resize(..., INTER_LINEAR)`, on 64 frame sizes from 17 x 9000 to 5000 x 30 |
| detections (float32) | the same finds, scores within 5e-5, boxes within 0.05 px |
| tracking, 3 cameras x 60 frames, `look_every` 1, 3 and 5 | the same ids and classes on every frame, boxes within 0.005 px |

## Build

```bash
cargo build --release          # target/release/tamserve-boxes
```

OpenVINO's C library is loaded at run time. Use either a release from Intel, or the one inside the `openvino` pip wheel. The wheel names it with a version suffix the Rust bindings don't recognise, so link it under its plain name:

```bash
pip install openvino
OV=$(python3 -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs
mkdir -p ~/.local/lib/openvino && ln -sf "$OV"/libopenvino_c.so.* ~/.local/lib/openvino/libopenvino_c.so
export LD_LIBRARY_PATH=~/.local/lib/openvino:$OV
```

With an installed OpenVINO release, `source setupvars.sh` (or `OPENVINO_INSTALL_DIR=/opt/intel/openvino`) is enough. `run` also needs `ffmpeg` and `ffprobe` on the PATH.

## Use

```bash
# once: YOLOX's released model as one that takes 1 channel and any batch (exact on gray input)
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
tamserve-boxes fold yolox_nano.onnx yolox_nano_gray.onnx

# live cameras: one JSON line per camera per round
tamserve-boxes run --model yolox_nano_gray.onnx --threads 3 --classes person,car,truck \
    --camera gate=rtsp://user:pass@10.0.0.21/Streaming/Channels/102 \
    --camera yard=rtsp://user:pass@10.0.0.22/Streaming/Channels/102
```

```json
{"t":12.400,"camera":"gate","frame":186,"objects":[{"id":1,"label":"person","score":0.812,"box":[410.2,120.5,560.8,480.1]}]}
```

`run` reads each camera with ffmpeg, which hands over the gray picture (the Y plane) with no colour conversion. Each camera keeps only its newest frame, so a loop that falls behind skips frames instead of building up a delay. Every round, the cameras due a look go through YOLOX in one batch.

| option | what it does |
|---|---|
| `--threads 3` | CPU threads for OpenVINO. Leave a core for decoding: 3 on a 4-core N100. Below full load, fewer use less CPU. |
| `--look-every 3` | frames between a camera's looks; `--look gate=1` gives one camera its own |
| `--hold 30` | frames an object no look sees is still followed, so it gets its id back |
| `--classes person,car` | the classes to keep |
| `--width 640 --height 360` | have ffmpeg scale the stream first. Detecting on the camera's sub-stream (`/102` on Hikvision) is cheaper still. |
| `--hwaccel vaapi` | decode on the integrated GPU; detection stays on the CPU |
| `--realtime` | read recordings at their own frame rate, as cameras would send them |
| `--precision f32` | OpenVINO's inference precision; by default f32 on an N100 |

`tamserve-boxes detect --model M picture.pgm ...` prints what one look finds on each picture.

### As a library

```rust
use tamserve_boxes::{BoxTracker, DetectorOptions, GrayFrame, TrackerOptions};

let mut tracker: BoxTracker = BoxTracker::from_model("yolox_nano_gray.onnx",
    DetectorOptions { threads: Some(3), classes: Some(vec!["person".into()]), ..Default::default() },
    TrackerOptions::default())?;
let found = tracker.step(&[("gate".to_string(), GrayFrame::new(&y_plane, 1280, 720)?)])?;
```

`GrayFrame::with_stride` takes a decoder's padded rows as they are. Any type that implements `Detect` can replace YOLOX, for example your own detector, or a stand-in for tests (`tests/tracker.rs`).

## CPU and memory

`tamserve-boxes bench` runs N synthetic 720p cameras at a set frame rate: in this one process, or with `--separate`, one process per camera. It was run on the same 2-core cloud VM as the Python benchmark (Intel Xeon, OpenVINO 2026.4, float32, `look_every=3`), **not an N100**. Decoding is left out. Both benchmarks hold about 27 MB of frames per camera in memory, which is included below.

| cameras × fps | setup | CPU (cores) | memory | fps each, kept |
|---|---|---|---|---|
| 8 × 25 | **Rust, one process** | **0.93** | **338 MB** | **25.0** |
| 8 × 25 | Python, one process | 0.98 | 433 MB | 25.0 |
| 8 × 25 | Rust, 8 processes | 1.02 | 880 MB | 23.6 |
| 8 × 25 | Python, 8 processes | 1.05 | 1,544 MB | 21.9 |
| 4 × 10 | Rust, one process | 0.23 | 210 MB | 10.0 |
| 4 × 10 | Python, one process | 0.24 | 296 MB | 10.0 |

- **Memory:** take away the benchmark's own frames, and the Rust process carries about 115 MB to Python's 210 MB. Most of what's left is OpenVINO itself.
- **CPU:** about 5% less. Both spend their time in the same OpenVINO kernels; what Rust removes is the work around them.
- **Sharing one model matters more than the language.** Either way, one process for all cameras uses far less than one process per camera, and keeps every camera at its frame rate.

```bash
tamserve-boxes bench --model yolox_nano_gray.onnx --image picture.pgm --cameras 8 --fps 25 --threads 1
tamserve-boxes bench --model yolox_nano_gray.onnx --image picture.pgm --cameras 8 --fps 25 --separate
```

## Tests

```bash
cargo test --release
# with the model, too:
TAMSERVE_YOLOX_ONNX=yolox_nano.onnx TAMSERVE_DOG_JPG=dog.jpg cargo test --release
```

`tests/tracker.rs` ports the Python tracker tests, with the same scripts and the same answers. `tests/model.rs` runs YOLOX-Nano itself: finds on a gray picture, a batch equal to its frames run alone, a panning camera, and OpenCV's resize on a known case.

## Licences

Apache-2.0, as tamserve. Dependencies:
- `openvino` (Apache-2.0)
- `onnx-protobuf` (MPL-2.0), only with the default `fold` feature: `cargo build --no-default-features` leaves it out of a runtime-only build.

YOLOX and its weights are Megvii's, under Apache-2.0.
