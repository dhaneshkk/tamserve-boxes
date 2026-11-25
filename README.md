# tamserve-boxes

The Rust port of [`tamserve.boxes`](https://github.com/dhaneshkk/tamserve/blob/main/docs/cpu-boxes.md) (from [tamserve](https://github.com/dhaneshkk/tamserve/)): find and follow boxes on many cameras with **YOLOX-Nano on gray frames, on the CPU through OpenVINO**, for edge boxes with no NVIDIA GPU such as an Intel N100. It's one binary: no Python, no PyTorch. It reads cameras in-process with GStreamer, keeps them connected, reports their health, and spreads them over several boxes with a coordinator.

| | |
|---|---|
| **Cameras** | RTSP, HTTP or files, decoded in this process by GStreamer (no ffmpeg per camera). The gray plane is taken as the decoder makes it. VA-API decodes on the integrated GPU where the box has one. |
| **Passwords** | in a config file or files only their owner can read, set on the RTSP source as properties. Never on a command line, in a URL or in a log. |
| **Reconnects** | a stream that fails or stalls is reopened, with backoff from 1 s doubling to 30 s, with jitter |
| **Health** | per camera: state, frames decoded and used, reconnects, frame age, last error. Served as Prometheus `/metrics`, JSON `/status`, and OpenAI-compatible `/v1/chat/completions` |
| **Throughput** | several OpenVINO requests in flight; each letterboxes its share of a look on its own thread |
| **Motion gate** | motion is found on a 100-row copy of each frame; a camera where nothing moves and nothing is followed costs no looks (Frigate's idea) |
| **One connection** | a camera that records or is watched live is read once: the detector, the recorder and every live viewer share the stream, never decoded twice or encoded |
| **Recording** | MPEG-TS segments cut on keyframes, kept by what was in them: every segment, motion, objects, a label's own days, pins; a disk cap that frees the soonest-expiring first. The rules are proved in Lean 4 |
| **Many boxes** | a coordinator assigns cameras to boxes by capacity, moves a dead box's cameras within seconds, evens out the load, and collects every result |
| **Same results as Python** | byte-identical folded model, letterbox bit-identical to OpenCV, detections within 5e-5, the same ids on every frame |

## Build

```bash
# Ubuntu / Debian: GStreamer and its plugins; the VA driver for decoding on an Intel integrated GPU
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav intel-media-va-driver
# recording uses splitmuxsink (good) and live view mpegtsmux (bad); tests/recording.rs makes its clip with x264enc (ugly)
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
- `onvif` (on by default): finds cameras and their streams over ONVIF. Uses [dhaneshkk/onvif-rs](https://github.com/dhaneshkk/onvif-rs), pinned to a commit, and OpenSSL through `native-tls` (`libssl-dev`).

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

### ONVIF cameras

A camera can give its ONVIF device in place of `url`. The box then asks the camera for its stream (Profile T through Media2; Media1 for older cameras) **every time it opens the stream**. So a camera that was replaced, reset, or moved to another address by its own DHCP lease is followed without editing the config.

```toml
[onvif]
timeout_seconds = 5
use_device_host = true    # default: reach the streams at the address the camera was reached at
# allow_username_token_over_http = true   # only for cameras with no Digest login
# ca_file = "/etc/tamserve/site-ca.pem"   # for https:// devices

[[camera]]
name = "yard"
onvif = "10.0.0.22"       # or "http://10.0.0.22/onvif/device_service"
stream = "gray"           # "gray" (the default), "colour", or a profile's token or name
user = "viewer"
password_file = "/etc/tamserve/secrets/yard"
```

ONVIF doesn't mark which stream is monochrome, so `stream` picks a profile as follows:
- **`gray`:** a profile named gray, grey or mono. Otherwise, the smallest H.264/H.265 stream at least `[decode] width` wide (the sub-stream, usually). The detector reads the gray plane of any stream, so the cheapest one that's big enough is the right one.
- **`colour`:** a profile named colour or color. Otherwise, the biggest stream not named gray (the main stream).
- **Any other value:** that profile's token, or its name with letter case ignored.

The `tamserve-boxes onvif` command finds the cameras on the network. It prints each camera's streams, then a `[[camera]]` block to paste into `cameras.toml`. For example:

```bash
tamserve-boxes onvif --user viewer --password-file /etc/tamserve/secrets/site --min-width 640
tamserve-boxes onvif --subnet 10.0.0.0/24 ...   # networks that drop multicast: ask every host
tamserve-boxes onvif --device 10.0.0.21 ...     # skip discovery
```

```toml
# http://10.0.0.21/onvif/device_service  HIKVISION DS-2CD2143G2-I
#   Profile_101 "mainStream"  H265 2560x1440 25 fps  rtsp://10.0.0.21:554/Streaming/Channels/101?transportmode=unicast
#   Profile_102 "subStream"  H264 640x360 25 fps  rtsp://10.0.0.21:554/Streaming/Channels/102?transportmode=unicast
# colour: Profile_101 (2560x1440)
[[camera]]
name = "cam-10-0-0-21"
onvif = "http://10.0.0.21/onvif/device_service"
stream = "gray"   # Profile_102 "subStream", H264 640x360
user = "viewer"
password_file = "/etc/tamserve/secrets/site"
```

How credentials are handled:
- The ONVIF login uses the camera's `user` and password. HTTPS certificates are checked: trust a site CA, or pin the cameras' own certificates, with `ca_file`.
- The weaker UsernameToken login is not sent over plain HTTP unless `allow_username_token_over_http` is set.
- Stream URLs the camera reports lose any credentials in them; the RTSP login goes on the source element, as for `url`.
- With `use_device_host = false`, a camera that names a service on another host is refused, so the password never goes there.

`run` prints one JSON line per camera per step and logs state changes by camera name and host:

```json
{"camera":"gate","frame":186,"time_ms":1795072982293,"objects":[{"id":1,"label":"person","score":0.81,"box":[410.2,120.5,560.8,480.1]}],"light":"day","settling":false}
```

`light` and `settling` are explained under [Day, night and the IR switch](#day-night-and-the-ir-switch).

### Day, night and the IR switch

At dusk a camera takes its IR cut filter out of the light path and turns on its IR LEDs. From then on it sees only brightness: its colour stream carries a gray picture too, so anything that draws boxes on "the colour frame" has no colour to draw on. The switch itself is a jump in exposure that can make the detector see things for a second or two, and can lose what is followed.

Every camera is followed for this, with no setup:
- **`light`: `"day"`, `"night"` or `"unknown"`.** When the decoder gives the frame's colour planes (NV12, I420: an H.264/H.265 stream, the sub-stream included), how far its chroma is from neutral tells. A camera in IR mode sends neutral chroma. The IR cut filter serves the whole sensor, so a colour sub-stream tells for the main stream too. A stream with no colour planes (a GRAY8 or mono profile) reports `"unknown"`. **Draw on the gray frame whenever `light` isn't `"day"`.**
- **`settling`.** A switch is a change of mode, motion's lighting reset (most of the picture changed at once), or the mean brightness jumping by more than `jump` between frames. For `settle_seconds` after one the camera is not looked at, and its reports carry the objects from before the switch with the same ids, so a person in a hazard zone doesn't vanish and come back as someone new. Don't raise new alarms on a settling camera.

```toml
[light]                  # these are the defaults; the section can be left out
settle_seconds = 2       # 0: report day/night but never settle
night_below = 2.0        # chroma (mean distance from neutral, 0-128) that looks like IR
day_above = 4.0          # and like colour; between the two is no evidence either way
confirm = 5              # frames in a row before day/night changes
jump = 40                # a brightness change (0-255) between frames that counts as a switch; 0: off
```

The mode changes only after `confirm` frames in a row agree, and only past two thresholds with a gap between them, so a dull or dim colour scene doesn't flicker. [`proofs/Light.lean`](proofs/Light.lean) proves it: values between the thresholds never change the mode, fewer than `confirm` frames never do, and `confirm` agreeing frames always do. `tests/light.rs` runs a clip through the real decoder, colour bars and then the same bars with the colour taken out, and checks for one switch from day to night.

### Health and metrics

With `[http] listen` set (127.0.0.1 by default; add `token_file` to require a bearer token), the box serves:

| endpoint | what |
|---|---|
| `GET /metrics` | Prometheus, per camera: `tamserve_camera_up`, `_state{state=…}`, `_frames_decoded_total`, `_frames_used_total`, `_reconnects_total`, `_frame_age_seconds`, `_objects`, `_light_mode` (0 unknown, 1 day, 2 night), `_light_switches_total`, `_settling`; with `[motion]`, `_moving`, `_motion_steps_total`, `_looks_skipped_total`, `_lighting_resets_total`; with recording and live view, `_segments_total`, `_live_viewers`, `_live_bytes_total`. Per loop: `tamserve_steps_total`, `tamserve_step_seconds_total`/`_last`, `tamserve_looks_total`. With `[record]`: `tamserve_recordings_bytes`, `_segments`, `_pinned`, `_deleted_total{why=expired\|space}`, `_sweep_errors_total` |
| `GET /status` | the same as JSON, with each camera's last error and latest objects |
| `GET /live/<camera>.ts` | a camera with `live = true`, as MPEG-TS from the box's own connection: `ffplay`, VLC, mpv or mpegts.js. At most 8 viewers per camera and per box; a viewer who falls behind is dropped |
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

## Motion, recording and live view

Three things taken from [Frigate](https://github.com/blakeblackshear/frigate), done in this one process:

```toml
[motion]                 # look only where something moves
hold = 10                # frames a camera stays awake after it last moved
# gate = false           # find motion (for metrics and recording) but make every look
# regions = true         # crop looks to squares around motion and what is followed (main streams)
# most_regions = 2

[record]
dir = "/var/lib/tamserve/recordings"
segment_seconds = 10
all_days = 1             # every segment
motion_days = 3          # segments with motion (needs [motion])
objects_days = 14        # segments with any object
labels = { person = 30 } # a label's own days
pre_seconds = 5          # kept before and after an event, with it
post_seconds = 5
max_gb = 400             # over this, the segments that would expire first go first

[[camera]]
name = "gate"
url = "rtsp://10.0.0.21/Streaming/Channels/101"
record = true            # or [record] all_cameras = true
live = true              # GET /live/gate.ts
motion_mask = [[0.8, 0.0, 1.0, 0.1]]   # [left, top, right, bottom] shares: the clock overlay
```

**The motion gate.** Each frame's gray plane is shrunk to 100 rows by area averaging and compared with a running average of the frames before it. Changed pixels are grouped into regions, and specks are dropped. When most of the picture changes at once, it's taken as the light, not motion: the IR cut filter switching at dusk, floodlights, lightning. The average then starts over from the new frame. A camera that hasn't moved for `hold` frames, and follows nothing (not even an object held after it was last seen), is left out of the look. It looks again on the first frame that moves, and a person standing still is still followed, because a camera that follows something is always looked at. With `[motion]` absent, nothing changes: `step_gated` with no quiet cameras is `step`, tested look for look.

`motion_mask` leaves parts of a camera's picture out of motion: a clock overlay, trees, a road beyond the fence. The parts are given as shares of the width and height, so the same mask fits the sub-stream and the main stream. Masked pixels don't count towards the light-change share either. Masks only stop motion from waking the gate: an object in a masked part is still found and followed when the camera looks.

**Regions** (`regions = true`). On a 1920 x 1080 main stream, the whole frame goes into YOLOX-Nano's 416 x 416 at under a quarter of its size, so small things vanish. With regions, a look is cropped instead to squares of at least the model's size around what moved and what is followed. Overlapping squares are joined, and each square goes in at full size. The squares are views into the frame, not copies. A find cut by a square's edge inside the frame is left out, and that camera's next look is whole. A camera's first look is always whole, and so is a look with nothing to centre on, more than `most_regions` squares, or squares covering more than half the frame. Each square is a look of its own, so this is for finding small things on big frames, not for saving looks. On a 640 x 360 sub-stream no 416 square fits, and every look stays whole. Tested with YOLOX-Nano: the dog at a third of its size on a 1080p frame is not found by a whole-frame look, but is found and followed, in the right place, through a region. [`proofs/Regions.lean`](proofs/Regions.lean) proves that a square, and every find moved out of it, stays inside the frame.

Measure the gate on your own footage before relying on it:

```bash
cargo run --release --example gate_mcc -- yolox_nano_gray.onnx night_clip.mp4 10 30   # hold, threshold
# {"tp":298,"fp":16,"tn":436,"fn":0,"mcc":0.957,"recall":1.0,"looks_saved":0.58,...}   (a synthetic clip)
```

It runs the detector on every frame as the truth, and counts how often the gate would have slept through a frame with something on it (`fn`). Tune `hold` and `threshold` on night and day footage from each site until recall is where it must be. Then read MCC and the share of looks saved. `alpha` is per frame used: something bright that stood still and then left lingers in the average, so the gate stays awake longer. It errs towards looking, never towards missing.

**One connection.** A camera with `record` or `live` is opened as `urisourcebin ! parsebin ! h264parse|h265parse ! tee`. The tee feeds the decoder (gray frames for detection, as before), a `splitmuxsink` writing MPEG-TS segments, and an `mpegtsmux` publishing to live viewers. The camera serves one client however many use it. Its stream is decoded once and never encoded. Parameter sets go before every keyframe, so each segment decodes on its own. The live tap keeps the latest PAT, the latest PMT, and every packet since the latest keyframe (the one `mpegtsmux` marks random-access, up to 8 MB). A viewer who joins mid-stream is sent those first, so their player shows a picture at once. A camera that only follows keeps the old `uridecodebin` pipeline. Recording and live view need H.264 or H.265; an MJPEG camera is still followed, with the reason in its `last_error`.

**Retention.** Segments are `DIR/<camera>/<start ms>.ts`. MPEG-TS, not MP4, so a segment cut short by a power cut still plays up to its last flushed write. The recorder writes through a 16 KB buffer, so a crash loses at most that much of the open segment, plus whatever the kernel hadn't written to disk yet. On SIGINT or SIGTERM, `run` and `box` stop every camera at once and send an end of stream down each pipeline, so each open segment is closed whole (waiting at most 3 s); they also write the activity logs. Each step, what a recording camera showed (motion, labels) goes to `DIR/<camera>/activity/<day>.jsonl`, with how far the log covers. A segment's keep-until is its end plus the longest retention of any activity within `pre_seconds` before it to `post_seconds` after it. A segment past the end of the log (the box stopped before writing what it showed) is kept the longest, since what it holds is unknown. A sweep every `sweep_seconds`:

1. deletes every segment whose keep-until has passed;
2. while the recordings take more than `max_gb`, deletes the segments that would expire first, so a motion-only segment goes before one with a person.

A pinned segment is never deleted, nor each camera's newest. Pins are in `DIR/pins.json`, `[{"camera": "gate", "from": ms, "to": ms, "note": "incident 41"}]`; a pin without `camera` holds them all. A pins file that can't be read stops the sweep rather than deleting what it may pin.

```bash
tamserve-boxes sweep --config cameras.toml --dry-run   # what would go now, as JSON
```

[`proofs/Retention.lean`](proofs/Retention.lean) proves, with core Lean 4.31 and `omega` (no Mathlib), that:
- every segment is kept `all_days`, or the longest retention past the log;
- a segment that meets an event is kept that event's days, so no age sweep deletes it sooner;
- a held segment is never deleted, for age or for space;
- the space rule frees only what expires no later than everything it keeps.

```bash
lean proofs/Retention.lean && lean proofs/Regions.lean && lean proofs/Light.lean
```

[`notebooks/motion_record_live.ipynb`](notebooks/motion_record_live.ipynb) is one Colab cell that does all of this on a test clip. It measures the gate against the detector (MCC), runs a box that records and streams, reads it back through `/v1/chat/completions`, sweeps the recordings, and checks the proof. Set `TSB_BRANCH` to try a branch.

## Uplink copies: decode once, use twice

`record` (above) keeps the camera's own stream untouched, for evidence. `encode` adds a **smaller copy for the uplink**: AV1 at a few hundred kbit/s, made from the decode detection already does. A camera can have any of the three.

A camera with `encode = true` is encoded from **the same decode** detection uses. After the decoder, a `tee` hands each decoded frame to both branches, each behind a leaky queue of its own, so a slow encoder drops frames (counted in `tamserve_camera_encode_dropped_total`) instead of stalling decoding or detection. There's no second RTSP connection, no second decode, and no process per camera.

```
rtsp ─ parse ─ tee ┬─ record: the camera's own H.264/H.265 ─▶ MPEG-TS segments
                   ├─ live: the same, ─▶ /live/<camera>.ts
                   └─ decode ─ tee ┬─ gray plane, newest frame ─▶ YOLOX (batched over all cameras) ─▶ tracker
                                   └─ frame-rate cap ─ scale ─ SVT-AV1 (1 thread, low delay) ─▶ Matroska segments
```

- **One CPU budget.** Each encoder is held to `threads` (SVT-AV1's `logical-processors`), so 20 encoders on a 4-core box don't each start a thread per core and crowd out the detector. At start-up the box logs detection threads + encoders × threads against its cores, and warns past twice the core count.
- **Low delay, no lookahead.** SVT-AV1 runs with `pred-struct=1:lookahead=0`, so no frame waits for later ones, and constant bitrate (which SVT-AV1 only does in low delay) suits an uplink. This is also what keeps each encoder small: it holds 2 frames in flight instead of 42.
- **Segments.** Matroska segments of `segment_seconds`. Each frame keeps its real capture time, and after a reconnect the numbering continues rather than overwriting. `max_files` keeps a rolling window, and `output = "null"` encodes and discards, for measuring.
- **Encoders.** `svtav1` is AV1 on the CPU, the smallest uplink for its CPU cost. `vah265` is HEVC on the integrated GPU: almost no CPU, but larger, and **untested here** (no Intel GPU). `x265` is HEVC on the CPU.

Measured (before recording and live view were added) on a 2-core VM, 4 RTSP cameras at 15 fps, encoding to 480×320 at 10 fps, SVT-AV1 preset 12, 300 kbit/s, with YOLOX following on every camera:

| 4 cameras, 1080p | CPU (cores) | memory |
|---|---|---|
| **one process, decode once** (this) | **1.06** | **438 MB** |
| detection + an ffmpeg per camera, decoding again | 1.18 | 843 MB |
| a `tamserve-boxes` process per camera | 1.02 | 822 MB |

With 960×640 cameras: 0.70 / 0.71 / 0.72 cores, and 297 / 572 / 699 MB.

**Memory halves; CPU falls less.** Software decoding of these streams is cheap next to encoding, so decoding once saved about 10% of CPU at 1080p and about nothing at 960×640. The CPU levers that matter are each encoder's `threads`, `preset`, `fps` and size, and decoding on the integrated GPU (`decoder = "vaapi"`), which takes decoding off the CPU altogether.

What low delay is worth, measured alone (one SVT-AV1 encoder on 36 s of 1080p input, preset 12, 1 thread, 300 kbit/s):

| encoding to | default (random access, lookahead) | low delay, no lookahead |
|---|---|---|
| 480×320 | 145 MB, 5.4 s CPU | 119 MB, 4.5 s CPU |
| 1920×1080 | 428 MB, 12.6 s CPU | **216 MB, 8.7 s CPU** |


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

[`notebooks/finetune_gray.ipynb`](notebooks/finetune_gray.ipynb) is one cell. Point `DATA` at labelled frames from your cameras (COCO JSON or YOLO txt), and it:

1. folds YOLOX-Nano's released weights to 1 channel, checked exact against the 3-channel model;
2. trains on your frames, read as gray, with Sharpness-Aware Minimization around SGD;
3. scores the released and the fine-tuned model on held-out frames, per frame and class ("is there one?" at `DECISION_SCORE`), with MCC, and a bootstrap interval for the gain;
4. exports the fine-tuned model only if it passes the gate: MCC at least the released model's plus `MIN_GAIN`, no class worse by more than `MAX_CLASS_DROP`, and optionally a gain that's clearly above chance (`REQUIRE_SURE_GAIN`). The ONNX file is 1-channel with any batch, in the released model's format, so `tamserve-boxes` runs it as it is (with `names` set if the classes changed).

It also checks that the released weights, exported through the same path, reproduce YOLOX's released ONNX (to 5.5e-5).

What was learned building it, on a 2-core CPU with demo data:
- **Momentum 0.9 diverged.** YOLOX-Nano with frozen BatchNorm is sharp: momentum compounds the step about tenfold, and even learning rate 1e-4 blew up. Plain SGD at 1e-4 lowers the loss steadily, so that's the default (`MOMENTUM = 0`). The gate and the export use an average of the weights (EMA), as YOLOX's own training does.
- **Keep every class your labels mark.** Dropping labels turns those objects into "background" the model must unlearn, which cost the classes that were kept.
- **The demo shows the pipeline, not a gain.** Left with no data, it downloads coco128 (GPL-3.0 as Ultralytics packages it), makes it CCTV-like, and scores people. But those pictures are in COCO's training set, which the released model already learned from. There, fine-tuning held MCC level (0.553 → 0.553) and the gate passed as "no worse"; earlier settings that made it worse were refused. A real gain needs your own footage, and enough held-out frames for the bootstrap interval to be narrow: a few hundred per camera type, held out by camera rather than by frame.

## Safety rules on the floor

A camera with `safety = "gate.safety.json"` in its config also runs safety rules on its objects, in metres on the floor. The code is `src/lens.rs`, `src/drift.rs`, `src/grammar.rs`, `src/datalog.rs` and `src/safety.rs`: the Rust port of `notebooks/vg360_grammar.ipynb`, frame for frame (see Tests). Its verdicts come out in three places:
- the camera's report (`safety`), so they are on `/status` and `/v1/chat/completions`;
- the metrics: `tamserve_safety_line_of_fire`, `_red_zone`, `_perimeter_breach`, `_calibration{state}`, `_drift_px`, `_ppe{item,verdict}` and `_ppe_looks_total`;
- with `label_queue` set, a JSON-lines file of crops worth labelling.

**The spec** is a JSON file: site, calibration, PPE heads, rule backend, label map. The notebook writes one; `src/safety.rs` documents it. It is checked when the config loads: a lens model that folds, a fill box outside the grid, or `datalog` without clauses stops the start. Leave out `ppe` until the detector knows `helmet` and `vest`. With a stock COCO model nobody ever has a helmet box, so everyone in plain view would be a violation. Set `perimeter = false` on a camera with no barricade in view, or the closure test will always say "open".

**From pixels to the floor.**
- **Lens first.** A point is undistorted with the camera's lens model, then mapped to the floor by a homography fitted to floor markings: `floor = H · G · undistort(u, v)`.
  - On a 2.8 mm lens, skipping the undistortion put people 39 cm off on average and up to 1 m.
  - Undistortion is Newton's method on OpenCV's model, to 1e-9 px. OpenCV 4's `undistortPoints` does not converge near the corners of a strong barrel lens.
  - A fitted model that folds inside the picture is refused (`Lens::monotone`).
- **Calibrate without a checkerboard.** The notebook fits the lens from straight floor lines (paint, grating edges, rails): 5.4 cm average error in simulation. Or it fits the lens jointly with the clicked markers: 6.9 cm. Starting the fit from no distortion once gave a model that folds; the start now comes from a coarse grid, with folding models penalised.
- **Has the camera moved?** With a `drift` section in the spec, the box checks this itself every `every_frames` frames (`src/drift.rs`, no OpenCV).
  - **How it checks.** The frame is reduced to about 320 px and undistorted. A homography to a reference frame is fitted by maximising their correlation (ECC, as OpenCV's `findTransformECC`), coarse to fine, from the best whole-pixel shift. Tracked people and forklifts are masked out.
  - **References.** One per light mode, since an IR night picture does not look like the day one. Take each with `POST /safety/<camera>/reference` while the view is the calibrated one, or let `auto_reference` take them. They are kept in `reference_dir` and reloaded when the box starts.
  - **When the camera has moved.** A measured move is corrected exactly for the floor (`G`).
  - **When the view cannot be confirmed** (lens covered, steam, camera turned away), every floor rule is UNKNOWN, never "clear", and `calibration-lost` is raised.
  - **The contrast check.** "Not confirmed" also needs a contrast test, not only correlation: a nearly covered lens keeps a faint copy of the scene that still correlates at 0.93 to 0.98. It must keep a quarter of the reference's contrast.
  - **On the notebook's test frames** (`tests/drift.rs`), knocks of 1° to 8° in tilt and pan, including re-lit (IR-like) frames, are corrected from up to 2.2 m of floor error back to the calibration's own 5 cm, within 0.3 cm of OpenCV's ECC on the same frames. Steam, a covered lens and a 30° turn are refused. A check takes about 30 ms, or 100 ms when it fails.
  - **On `vtest.avi`** (a fixed camera, a crowd walking through), it measured under 0.1 px of drift, at correlations between 0.95 and 0.997. A crowded scene pulls the correlation toward `rho_min` (0.9): watch `tamserve_safety_drift_px` and the correlation on a real site before trusting the threshold.
  - **An outside checker** can still post to `POST /safety/<camera>/drift`; a posted result wins for that frame.
  - **In simulation:** a 3° knock without the check dropped red-zone MCC from 0.78 to 0.31; with it, red-zone MCC was 0.64. Line of fire went from 0.77 to 0.80 and the barricade from 0.80 to 0.88. The correction recovers most, not all: recalibrate after a knock.
  - **A covered lens without the check** gave 1,046 false barricade alarms in 2,025 frames; with it, none, and UNKNOWN after 5 frames.

**The rules.**
- Line of fire, unless a barrier lies on the 4-connected ray between person and forklift.
- Red zone.
- Barricade closure: posts that persist are chained, and a fill from the hazard tests whether the chain is closed.
- Helmet and vest, as a person's head and torso bands, judged by a small SAM-trained model with conformal thresholds. An occluded person is UNKNOWN, never compliant.
- Every alarm goes through a leaky bucket, so it takes `hi - lo` frames to go on or off.
- `Backend::Datalog` runs the same rules as Datalog clauses in yappu-fsm's schema, with a derivation for every alarm. For a breach, the derivation is the escape path through the gap, re-checked by `valid_path`. The native backend gives the same verdicts and is faster.

**PPE needs pixels.** On the 640 px grey substream a far worker's head is a few pixels. In simulation helmet MCC was 0.58 there, against 0.95 on a 4x crop of the same person from the main stream (vest: 0.55 against 0.97). So a camera on its main stream can take a second look (`ppe_look` in the spec):
- Each step, up to `most` people's head-and-torso crops are cut from the frame at full resolution. They are views into the frame, not copies, and go through a PPE model in one batch. The main detector sees the whole frame shrunk to 416 px.
- People never looked at go first, then whoever was looked at longest ago, smaller ones first.
- For a person looked at, the PPE model's finds replace the main detector's helmet and vest finds there; everyone else keeps the main detector's.
- Each look costs one inference per person; `tamserve_safety_ppe_looks_total` counts them.
- On a 640 px substream a crop has no more pixels than the frame, so leave it out there.

`tests/model.rs` checks the look with YOLOX-Nano standing in for a PPE model: its find in a crop lands back on the object in the whole frame (IoU above 0.5), and replaces the main detector's.

**Filling short gaps.** About a third of PPE verdicts stay UNKNOWN even on crops: the person is occluded or missed on that frame. The report's `ppe_fused` fills them from the camera's own last clear look, at most 20 frames old by default (`fusion.own_window_frames`), and says where each verdict came from and how old it is. In simulation, helmet UNKNOWN fell from 32 % to 1 %, and MCC from 0.953 to 0.942. Of the filled "compliant" verdicts, 2.4 % were people who had just taken the helmet off: keep the window short. An entry camera's verdicts can be fused too (`PpeFusion`, matched by floor position). In simulation it filled fewer gaps than the camera's own history, and adding it to the history gained nothing.

**Label queue.** It picks the crops worth labelling: people in plain view whose PPE verdict was ambiguous or near a coin flip, one per person per few seconds, capped per hour. In simulation a queued crop was about ten times likelier to be a model mistake than a random visible person (helmet 23 % against 2.3 %). Label the crops, fine-tune with `notebooks/finetune_gray.ipynb`, and hold out whole cameras.

**Formal checks.** `proofs/Grammar.lean` covers:
- the bucket;
- the PPE decision;
- the grid;
- the breach-path check;
- the calibration gate;
- the fusion rule: a current verdict is never hidden, and a filled "compliant" always has a fresh witness.

```text
cargo run --release --example grammar_replay      # both backends on the golden episodes
native   0.06 ms/frame    datalog 0.55 ms/frame (one 40 x 30 grid, 2-core sandbox CPU)
```

The detector has to know the classes: `person`, `forklift`, `helmet`, `vest`, `barrier_post` (fine-tune with `notebooks/finetune_gray.ipynb`, or map your labels with `labels` in the spec).

## CPU and memory

`tamserve-boxes bench` runs N synthetic 720p cameras at a set frame rate in this one process, or with `--separate`, one process per camera. Measured on a 2-core cloud VM (Intel Xeon, OpenVINO 2026.4, float32, `look_every=3`), **not an N100**. Decoding is left out, and both benchmarks hold about 27 MB of frames per camera, which is included below.

| cameras × fps | setup | CPU (cores) | memory | fps each, kept |
|---|---|---|---|---|
| 8 × 25 | **Rust, one process** | **0.93** | **338 MB** | **25.0** |
| 8 × 25 | Python, one process | 0.98 | 433 MB | 25.0 |
| 8 × 25 | Rust, 8 processes | 1.02 | 880 MB | 23.6 |
| 8 × 25 | Python, 8 processes | 1.05 | 1,544 MB | 21.9 |

### Compressed swap (zram) on Ubuntu

zram gives a box more room to spare, but it doesn't make room for more cameras. Detection memory is almost all in use every frame: weights of a few MB, per-frame buffers that are reused, and ring buffers of decoded frames. Pages in use don't get swapped, and the incoming H.264 barely compresses anyway. What zram does compress is the box's idle memory: other services, logs, the parts of a database not being queried. That's headroom against running out of memory. It costs CPU on the cores detection uses, so it's set up to cost little:

```bash
sudo deploy/setup-zram.sh                     # lz4 at 25% of RAM, swappiness 100, page-cluster 0, zswap off
sudo deploy/setup-zram.sh --remove-swapfile   # and remove Ubuntu's /swap.img (recommended on SD or eMMC)
deploy/setup-zram.sh --check                  # report: swap devices, zramctl, zswap, sysctls, memory pressure
```

It uses `systemd-zram-generator` where Ubuntu packages it, and `zram-tools` otherwise. It loads the zram module from `linux-modules-extra` if the kernel needs that. The files it installs are in [`deploy/zram/`](deploy/zram/), so you can install them by hand instead.

- **lz4, not zstd**: zstd compresses better but takes several times the CPU per page.
- **swappiness 100, not the desktop 180**: a page fault that has to decompress stalls the thread that hit it, so swap only what's really idle.
- **zswap off**: in front of zram it would compress the same pages twice.
- **The detector never swaps.** [`deploy/systemd/tamserve-boxes.service`](deploy/systemd/tamserve-boxes.service) sets `MemorySwapMax=0` (cgroup v2, the default since Ubuntu 21.10), so inference and decode threads never wait on decompression. It also sets `OOMScoreAdjust=-500`, so if memory runs out the kernel kills something else first. `CPUAffinity=` (commented out) can leave core 0 to the kernel's reclaim; lower `[detector] threads` to match.

```bash
sudo useradd --system --no-create-home tamserve
sudo install -m 755 target/release/tamserve-boxes /usr/local/bin/
sudo install -m 644 deploy/systemd/tamserve-boxes.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now tamserve-boxes
```

`/etc/tamserve/env` (mode 600, optional) holds `LD_LIBRARY_PATH` for OpenVINO and any `password_env` variables. `cameras.toml` and the password files must belong to the `tamserve` user, because they must be mode 600.

To check zram is working, run your normal cameras plus some extra memory pressure, with zram on and with it off, and compare:
- p99 step time (`tamserve_step_seconds_last`) and frame age;
- `/proc/pressure/memory`, which should stay near 0;
- `vmstat 1` `si`/`so`, which should not rise and fall with slow steps;
- the TOTAL column in `zramctl`, which is the RAM really used.

`--check` also shows the detector's `memory.swap.max`, which should be 0.

## Tests

```bash
cargo test --release                                   # the 6 tests that need the model run when these are set:
TAMSERVE_YOLOX_ONNX=yolox_nano.onnx TAMSERVE_DOG_JPG=dog.jpg cargo test --release
scripts/rtsp_test.sh yolox_nano_gray.onnx some_h264.mp4    # end to end, needs mediamtx and ffmpeg
```

- **Unit and integration tests:**
  - the config: password files, permissions, URLs with credentials, mistakes, `onvif` cameras;
  - ONVIF, against a mock camera on localhost:
    - Media2, and a Media1-only camera without GetServices;
    - picking the gray and colour streams;
    - addresses rewritten and credentials stripped;
    - services on another host refused;
    - the `onvif` command's output loading as a config;
    - a camera source that looks its stream up over ONVIF and decodes it;
  - the coordinator over loopback: wrong token, a stalled handshake cut off, changed and replayed messages refused, spreading by capacity, failover, cameras a box doesn't know, a box reconnecting;
  - the tracker tests ported from Python;
  - the model tests, including that splitting a look across requests changes nothing;
  - motion: still pictures, a moving square found where it is, specks, the light changing, masks, the gate holding open;
  - regions: squares around small things, inside the frame, joined when they overlap, the whole frame when they wouldn't help, finds moved back and cut ones left out, duplicates; in the tracker, cropped looks landing where the whole-frame ones do; with the model, the small dog found only through a region;
  - the gate in the tracker: never quiet changes nothing, quiet cameras cost no looks, what is followed is still followed;
  - retention: each kind of segment's days, the seconds before and after an event, segments past the log, the space order, pins, a sweep on disk, a bad pins file;
  - live fan-out: every viewer in order, a slow viewer dropped, the limits, a late viewer starting at the latest keyframe with the tables;
  - `tests/recording.rs`: one clip followed with the gate, recorded and watched at once (needs `x264enc` to make the clip).
  - the safety spec in the config: found beside it, checked, refused with its mistakes; the drift route over a socket (token, unknown camera, bad JSON); `SafetyCam` from objects to verdicts, the label queue's file, a reconnect starting over, no barricade reported as unknown;
  - the drift check (`tests/drift.rs`): the notebook's frames with known corrections. Knocks are corrected to within 0.3 cm of OpenCV's ECC; steam, a covered lens and a 30° turn are refused; a saved reference loads back the same. A `SafetyCam` takes a reference when asked, corrects a knock, goes UNKNOWN when blind, and finds its reference again after a restart;
  - the PPE look (`tests/model.rs`, with the model): a crop's find maps back onto the object in the whole frame and replaces the main detector's;
  - the safety rules (`tests/grammar.rs`): two simulated episodes replayed against the Python reference's every decision. One has a barricade breach and a camera knocked 3° part way (drift-corrected); in the other the lens is covered part way (floor rules UNKNOWN). Over 440 frames and 2,366 per-person results, both backends match exactly: alarms, PPE verdicts, posts, gaps and breach proofs, with floor positions within 1e-13 m. Undistortion is checked against the reference, and the grid against its cell map.
- **The safety rules end to end, by hand**: `run` with the real model on OpenCV's `vtest.avi` (pedestrians), with the simulated site's spec.
  - YOLOX's finds go into the Datalog engine.
  - Verdicts reach `/status`, `/metrics` and `/v1/chat/completions`.
  - A posted "lens covered" drift check turns every floor verdict UNKNOWN with `calibration-lost`; a confirmed one restores them; an unknown camera gets 404.
  - The box's own drift check took its references by itself and checked every 10 frames (under 0.1 px measured on a fixed camera). `POST /safety/<camera>/reference` took a new one.
  - The PPE look cut 3 crops a step from the frame and ran them through the model (YOLOX-Nano standing in).
  - The label queue writes its file.
  - The verdicts themselves mean nothing there: that scene has no calibration, no PPE classes and no barricade.
- **`scripts/rtsp_test.sh`**, against a real RTSP server that needs a password (all 10 checks pass):
  - four cameras up;
  - a wrong password reported and retried;
  - no password on any command line;
  - Prometheus and OpenAI-compatible endpoints;
  - cameras down with the server and back by themselves;
  - a coordinator with two boxes, one killed with SIGKILL, its cameras moved in 1.4 s;
  - one camera recorded and watched live over RTSP with its password: segments decode on their own, a viewer joining mid-stream decodes from its first picture, a camera without `live` has no stream, and SIGTERM closes the open segment whole.

## Not yet tested

- VA-API decoding on an Intel GPU, and everything on an N100.
- Live cameras other than the local RTSP server above. ONVIF is tested only against the mock camera; WS-Discovery has not been tried on a real camera network.
- More than 4 cameras through GStreamer at once.
- A long soak. Run `bench` and `rtsp_test.sh` on the target box first, then a day-long run with `/metrics` scraped.
- The motion gate, masks and regions on real night and day footage: tune them with `gate_mcc` per site before using them for safety monitoring.
- VA-API decoding in the shared pipeline (as with VA-API at all: no Intel GPU here).
- Recording many cameras' main streams to one disk at once, and the sweeper on a disk with months of segments.
- Live view in a browser: mpegts.js is expected to play `/live/<camera>.ts`, but that hasn't been tried.
- The PPE look with a real PPE model (none trained yet), and an entry camera's fusion across boxes.
- The drift check on a camera that really moves, on IR night pictures, and on repetitive deck grating, which may give ECC more than one plausible fit.
- The safety rules on real footage. Everything above is a simulated site with a known camera: lens, drift and rules need tuning and MCC checks on each real site (`gate_mcc`-style) before anyone relies on them. Feature matching for the drift check has not been tried on repetitive deck grating, which may confuse it.
- `deploy/setup-zram.sh` on real Ubuntu. It passes shellcheck, but the machine this was built on has no zram module. Run `--check` before and after on the target box.

## Licences

Apache-2.0 (see `LICENSE` and `NOTICE`). Dependencies:
- `openvino` and the GStreamer bindings: Apache-2.0 / MIT.
- GStreamer itself: LGPL, linked at run time.
- `onnx-protobuf`: MPL-2.0, only with `fold`.

YOLOX and its weights are Megvii's, under Apache-2.0.
