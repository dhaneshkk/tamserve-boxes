//! Cameras read in this process with GStreamer, each kept open by a supervisor that
//! reconnects with backoff, and counted for health.
//!
//! A camera's stream goes `uridecodebin ! videoconvert ! videoscale ! appsink`. The
//! appsink takes the gray plane as the decoder made it (GRAY8, or the Y plane of NV12 or
//! I420, so there is usually no conversion at all) and keeps only the newest frame. The
//! camera's user and password are set on the RTSP source element as properties, never
//! put into a URL, a command line or a log.
//!
//! `Decoder::Vaapi` ranks GStreamer's VA-API decoders first, so H.264 and H.265 are
//! decoded on the integrated GPU; `Decoder::Software` ranks them out. Either way the
//! frames come back to the CPU for detection.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;

use crate::config::{Camera, DecodeSection, Decoder, EncodeSection, Encoder};
use crate::{Error, Result};

/// Where a camera stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    /// Opening the stream; no frame yet.
    Connecting = 0,
    /// Frames are arriving.
    Up = 1,
    /// The stream failed or stalled; waiting to try again.
    Down = 2,
    /// A file that has been read to its end.
    Ended = 3,
    /// Removed from this box.
    Stopped = 4,
}

impl State {
    fn from_u8(v: u8) -> State {
        match v {
            0 => State::Connecting,
            1 => State::Up,
            2 => State::Down,
            3 => State::Ended,
            _ => State::Stopped,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            State::Connecting => "connecting",
            State::Up => "up",
            State::Down => "down",
            State::Ended => "ended",
            State::Stopped => "stopped",
        }
    }
}

pub(crate) use crate::now_ms;

/// A camera's health, updated by its supervisor and its frames, read by the metrics.
#[derive(Default)]
pub struct Health {
    state: AtomicU8,
    /// Frames the decoder handed over.
    pub decoded: AtomicU64,
    /// Frames a step used. The rest were replaced by a newer frame before a step came.
    pub used: AtomicU64,
    /// Times the stream was opened again after failing or stalling.
    pub reconnects: AtomicU64,
    /// When the newest frame arrived, in ms since 1970; 0 before the first.
    pub last_frame_ms: AtomicU64,
    /// When the current connection's first frame arrived; 0 while not up.
    pub up_since_ms: AtomicU64,
    /// Counts connections: a new one means the tracker starts this camera afresh.
    pub generation: AtomicU64,
    pub width: AtomicU64,
    pub height: AtomicU64,
    /// Frames and bytes the encoder has produced (cameras with `encode = true`).
    pub encoded: AtomicU64,
    pub encoded_bytes: AtomicU64,
    /// Times the encoder's backlog was full, so a frame was dropped rather than stall decoding.
    pub encode_dropped: AtomicU64,
    last_error: Mutex<String>,
}

impl Health {
    pub fn state(&self) -> State {
        State::from_u8(self.state.load(Ordering::Relaxed))
    }
    fn set(&self, s: State) {
        self.state.store(s as u8, Ordering::Relaxed);
    }
    pub fn last_error(&self) -> String {
        self.last_error.lock().unwrap().clone()
    }
    fn fail(&self, why: String) {
        *self.last_error.lock().unwrap() = why;
    }
    /// Seconds since the newest frame, if there has been one.
    pub fn frame_age(&self) -> Option<f64> {
        match self.last_frame_ms.load(Ordering::Relaxed) {
            0 => None,
            t => Some(now_ms().saturating_sub(t) as f64 / 1000.0),
        }
    }
}

/// A decoded frame, packed: width bytes per row.
pub struct Frame {
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// The frame's number on this connection, from 1.
    pub number: u64,
    pub generation: u64,
}

/// Wakes the step loop when any camera has a new frame.
#[derive(Default)]
pub struct Bell {
    count: Mutex<u64>,
    cv: Condvar,
}

impl Bell {
    fn ring(&self) {
        *self.count.lock().unwrap() += 1;
        self.cv.notify_all();
    }
    /// Wait until a ring after `seen`, or `timeout`. Returns the latest count.
    pub fn wait(&self, seen: u64, timeout: Duration) -> u64 {
        let g = self.count.lock().unwrap();
        let (g, _) = self.cv.wait_timeout_while(g, timeout, |c| *c == seen).unwrap();
        *g
    }
    pub fn count(&self) -> u64 {
        *self.count.lock().unwrap()
    }
}

/// Reconnect delays: `first`, doubling up to `max`, each scaled by a random 0.5 to 1 so
/// that cameras that failed together do not all knock at once.
#[derive(Clone, Copy, Debug)]
pub struct Backoff {
    pub first: f64,
    pub max: f64,
}

impl Backoff {
    pub fn delay(&self, failures: u32) -> Duration {
        let base = (self.first * 2f64.powi(failures.min(30) as i32)).min(self.max);
        let mut b = [0u8; 2];
        let _ = getrandom::fill(&mut b);
        let jitter = 0.5 + 0.5 * (u16::from_le_bytes(b) as f64 / u16::MAX as f64);
        Duration::from_secs_f64(base * jitter)
    }
}

/// How sources are opened, from the config's `[decode]`.
#[derive(Clone, Debug)]
pub struct Opening {
    pub decoder: Decoder,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub tcp: bool,
    pub stall: Duration,
    pub backoff: Backoff,
    pub realtime: bool,
    /// How cameras with `encode = true` are encoded, from the same decode.
    pub encode: Option<EncodeSection>,
}

impl Opening {
    pub fn with_encode(mut self, e: Option<EncodeSection>) -> Self {
        self.encode = e;
        self
    }
}

impl From<&DecodeSection> for Opening {
    fn from(d: &DecodeSection) -> Self {
        Opening {
            encode: None,
            decoder: d.decoder,
            width: d.width,
            height: d.height,
            tcp: d.tcp.unwrap_or(true),
            stall: Duration::from_secs_f64(d.stall_seconds.unwrap_or(10.0)),
            backoff: Backoff { first: d.backoff_first.unwrap_or(1.0), max: d.backoff_max.unwrap_or(30.0) },
            realtime: d.realtime,
        }
    }
}

static INIT: std::sync::Once = std::sync::Once::new();

/// Start GStreamer and rank its decoders for `decoder`. Call once, before any source.
pub fn init(decoder: Decoder) -> Result<()> {
    let mut failed = None;
    INIT.call_once(|| {
        if let Err(e) = gst::init() {
            failed = Some(e.to_string());
        }
    });
    if let Some(e) = failed {
        return Err(Error::Invalid(format!("GStreamer could not start: {e}")));
    }
    let registry = gst::Registry::get();
    let va: Vec<gst::PluginFeature> = registry.features(gst::ElementFactory::static_type()).into_iter()
        .filter(|f| { let n = f.name(); n.starts_with("va") && n.ends_with("dec") })
        .collect();
    match decoder {
        Decoder::Auto => {}
        Decoder::Software => va.iter().for_each(|f| f.set_rank(gst::Rank::NONE)),
        Decoder::Vaapi => {
            if va.is_empty() {
                return Err(Error::Invalid("decoder = \"vaapi\", but GStreamer has no VA-API decoders here: install \
                    gstreamer1.0-plugins-bad and the GPU's VA driver (intel-media-va-driver), and check /dev/dri".into()));
            }
            va.iter().for_each(|f| f.set_rank(gst::Rank::PRIMARY + 100));
        }
    }
    Ok(())
}

/// A URI GStreamer can open: a path becomes file://.
fn uri(url: &str) -> Result<String> {
    if url.contains("://") {
        return Ok(url.to_string());
    }
    let p = std::fs::canonicalize(url).map_err(|e| Error::Invalid(format!("{url}: {e}")))?;
    Ok(format!("file://{}", p.display()))
}

fn is_live(uri: &str) -> bool {
    !uri.starts_with("file://")
}

/// One camera: its newest frame, its health, and the thread that keeps it open.
pub struct Source {
    pub camera: Camera,
    pub health: Arc<Health>,
    newest: Arc<Mutex<Option<Frame>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Source {
    /// Start reading `camera`; frames ring `bell`.
    pub fn open(camera: Camera, how: &Opening, bell: Arc<Bell>) -> Result<Source> {
        let target = uri(&camera.url)?;
        let health = Arc::new(Health::default());
        let newest = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let (h, n, s, cam, how) = (health.clone(), newest.clone(), stop.clone(), camera.clone(), how.clone());
        let thread = std::thread::Builder::new()
            .name(format!("camera {}", camera.name))
            .spawn(move || supervise(&cam, &target, &how, &h, &n, &s, &bell))
            .map_err(Error::Io)?;
        Ok(Source { camera, health, newest, stop, thread: Some(thread) })
    }

    /// The newest frame not yet taken, if any.
    pub fn take(&self) -> Option<Frame> {
        self.newest.lock().unwrap().take()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.health.set(State::Stopped);
    }
}

/// Keep `camera` open until told to stop: open, read, and on failure or a stall wait
/// the backoff and open again. A file is read once.
fn supervise(camera: &Camera, target: &str, how: &Opening, health: &Arc<Health>, newest: &Arc<Mutex<Option<Frame>>>,
             stop: &AtomicBool, bell: &Arc<Bell>) {
    let mut failures = 0u32;
    while !stop.load(Ordering::Relaxed) {
        health.set(State::Connecting);
        health.generation.fetch_add(1, Ordering::Relaxed);
        let opened = Instant::now();
        let outcome = stream(camera, target, how, health, newest, stop, bell);
        health.up_since_ms.store(0, Ordering::Relaxed);
        match outcome {
            Outcome::Stopped => break,
            Outcome::Ended => {
                health.set(State::Ended);
                bell.ring();
                break;
            }
            Outcome::Failed(why) => {
                health.fail(why);
                health.set(State::Down);
                // A connection that held for a while starts the backoff over.
                if opened.elapsed() > Duration::from_secs(30) {
                    failures = 0;
                }
                let wait = how.backoff.delay(failures);
                failures = failures.saturating_add(1);
                let until = Instant::now() + wait;
                while Instant::now() < until && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50).min(until - Instant::now()));
                }
                health.reconnects.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

enum Outcome {
    Stopped,
    Ended,
    Failed(String),
}

/// A name safe in a file path: letters, digits, '-', '_' and '.'; anything else becomes '_'.
fn safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect()
}

/// The segment number to start at, so a reconnect continues a camera's numbering and never
/// overwrites the segments it already wrote. `pattern` has one `%0Nd`.
fn next_index(pattern: &str) -> Result<u32> {
    let p = std::path::Path::new(pattern);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(dir)?;
    let file = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let (Some(at), Some(end)) = (file.find('%'), file.find('d')) else { return Ok(0) };
    let (head, tail) = (&file[..at], &file[end + 1..]);
    let mut next = 0;
    for e in std::fs::read_dir(dir)?.flatten() {
        let n = e.file_name();
        let n = n.to_string_lossy();
        if let Some(num) = n.strip_prefix(head).and_then(|r| r.strip_suffix(tail)) {
            if let Ok(i) = num.parse::<u32>() {
                next = next.max(i + 1);
            }
        }
    }
    Ok(next)
}

/// The encoder branch of a camera's pipeline, after the tee: a frame-rate cap, scaling, the
/// encoder at its thread budget and low-delay settings, and the segments (or nothing).
fn encode_branch(camera: &Camera, e: &EncodeSection) -> Result<String> {
    let fps = e.fps.unwrap_or(25);
    let key = ((e.keyframe_seconds * fps as f64).round() as u32).max(1);
    let mut b = format!("t. ! queue name=encq leaky=downstream max-size-buffers={} max-size-bytes=0 max-size-time=0 ! ", e.backlog);
    if let Some(f) = e.fps {
        b += &format!("videorate drop-only=true ! video/x-raw,framerate={f}/1 ! ");
    }
    let format = if e.encoder == Encoder::Vah265 { "NV12" } else { "I420" };
    b += "videoconvert ! ";
    let mut caps = format!("video/x-raw,format={format}");
    if let (Some(w), Some(h)) = (e.width, e.height) {
        b += "videoscale ! ";
        caps += &format!(",width={w},height={h}");
    }
    b += &caps;
    b += " ! ";
    b += &match e.encoder {
        // Low delay (pred-struct=1): no frame waits for later ones, and SVT-AV1's CBR needs it.
        // No lookahead: what keeps each encoder's memory small. Threads capped per encoder.
        Encoder::Svtav1 => {
            let rate = match e.bitrate_kbps { Some(k) => format!("target-bitrate={k} crf=-1"), None => format!("crf={}", e.crf) };
            format!("svtav1enc name=enc preset={} logical-processors={} intra-period-length={} {rate} \
                     parameters-string=\"pred-struct=1:lookahead=0\" ! av1parse", e.preset, e.threads, key - 1)
        }
        Encoder::Vah265 => {
            let rate = match e.bitrate_kbps { Some(k) => format!("rate-control=cbr bitrate={k}"), None => format!("rate-control=cqp qpi={}", e.crf.clamp(0, 51)) };
            format!("vah265enc name=enc key-int-max={key} b-frames=0 target-usage=7 {rate} ! h265parse")
        }
        Encoder::X265 => {
            let (rate, crf) = match e.bitrate_kbps { Some(k) => (format!("bitrate={k}"), String::new()), None => (String::new(), format!(":crf={}", e.crf)) };
            format!("x265enc name=enc speed-preset=ultrafast tune=zerolatency key-int-max={key} {rate} \
                     option-string=\"pools={}:frame-threads=1{crf}\" ! h265parse", e.threads)
        }
    };
    if e.output == "null" {
        b += " ! fakesink sync=false";
    } else {
        let location = e.output.replace("{camera}", &safe(&camera.name));
        let start = next_index(&location)?;
        b += &format!(" ! splitmuxsink async-finalize=true muxer-factory=matroskamux location={} start-index={start} \
                       max-size-time={} max-files={} send-keyframe-requests=true",
                      quote(&location), (e.segment_seconds * 1e9) as u64, e.max_files);
    }
    Ok(b)
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// One connection, until it ends, fails, stalls or is stopped.
fn stream(camera: &Camera, target: &str, how: &Opening, health: &Arc<Health>, newest: &Arc<Mutex<Option<Frame>>>,
          stop: &AtomicBool, bell: &Arc<Bell>) -> Outcome {
    let mut caps = "video/x-raw,format=(string){GRAY8,NV12,I420,YV12,NV21}".to_string();
    if let (Some(w), Some(h)) = (how.width, how.height) {
        caps += &format!(",width={w},height={h}");
    }
    let sync = how.realtime || !is_live(target);
    let detect = format!("videoconvert ! videoscale ! {caps} ! appsink name=sink max-buffers=1 drop=true sync={sync} \
                          enable-last-sample=false");
    // Decode once, use twice: a tee hands the decoder's frames to detection and to the encoder, each
    // behind a leaky queue of its own, so neither can stall the decoder or the other.
    let encoding = match (&how.encode, camera.encode) {
        (Some(e), true) => match encode_branch(camera, e) {
            Ok(b) => Some(b),
            Err(err) => return Outcome::Failed(format!("cannot set up the encoder: {err}")),
        },
        _ => None,
    };
    let desc = match &encoding {
        None => format!("uridecodebin name=src uri={} ! {detect}", quote(target)),
        Some(enc) => format!("uridecodebin name=src uri={} ! tee name=t \
                              t. ! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! {detect} \
                              {enc}", quote(target)),
    };
    let pipeline = match gst::parse::launch(&desc) {
        Ok(p) => match p.downcast::<gst::Pipeline>() {
            Ok(p) => p,
            Err(_) => return Outcome::Failed("not a pipeline".into()),
        },
        Err(e) => return Outcome::Failed(format!("could not build the pipeline: {e}")),
    };
    // Credentials and RTSP settings go onto the source element as properties.
    if let Some(src) = pipeline.by_name("src") {
        let (user, pw, tcp) = (camera.user.clone(), camera.password.clone(), how.tcp);
        src.connect("source-setup", false, move |args| {
            if let Ok(el) = args[1].get::<gst::Element>() {
                if el.find_property("user-id").is_some() {
                    if let Some(u) = &user {
                        el.set_property("user-id", u);
                    }
                    if let Some(p) = &pw {
                        el.set_property("user-pw", p);
                    }
                }
                if el.find_property("protocols").is_some() && tcp {
                    el.set_property_from_str("protocols", "tcp");
                }
                if el.find_property("latency").is_some() {
                    el.set_property("latency", 200u32);
                }
                if el.find_property("timeout").is_some() && el.find_property("tcp-timeout").is_some() {
                    el.set_property("tcp-timeout", 10_000_000u64); // µs: a connection attempt gives up after 10 s
                }
            }
            None
        });
    }
    if encoding.is_some() {
        // Count what the encoder produces, and how often its backlog was full.
        if let Some(pad) = pipeline.by_name("enc").and_then(|e| e.static_pad("src")) {
            let h = health.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                if let Some(b) = info.buffer() {
                    h.encoded.fetch_add(1, Ordering::Relaxed);
                    h.encoded_bytes.fetch_add(b.size() as u64, Ordering::Relaxed);
                }
                gst::PadProbeReturn::Ok
            });
        }
        if let Some(q) = pipeline.by_name("encq") {
            let h = health.clone();
            q.connect("overrun", false, move |_| {
                h.encode_dropped.fetch_add(1, Ordering::Relaxed);
                None
            });
        }
    }
    let Some(sink) = pipeline.by_name("sink").and_then(|s| s.downcast::<gst_app::AppSink>().ok()) else {
        return Outcome::Failed("no appsink".into());
    };
    let generation = health.generation.load(Ordering::Relaxed);
    let (h, n, b) = (health.clone(), newest.clone(), bell.clone());
    let mut count = 0u64;
    sink.set_callbacks(gst_app::AppSinkCallbacks::builder().new_sample(move |s| {
        let sample = s.pull_sample().map_err(|_| gst::FlowError::Eos)?;
        let (Some(caps), Some(buffer)) = (sample.caps(), sample.buffer()) else { return Ok(gst::FlowSuccess::Ok) };
        let Ok(info) = gst_video::VideoInfo::from_caps(caps) else { return Err(gst::FlowError::NotNegotiated) };
        let Ok(frame) = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info) else { return Err(gst::FlowError::Error) };
        let (w, ht) = (info.width() as usize, info.height() as usize);
        let stride = frame.plane_stride()[0] as usize;
        let Ok(y) = frame.plane_data(0) else { return Err(gst::FlowError::Error) };
        // The gray plane, packed. Reuse the buffer the step loop has not taken yet, if any.
        let mut slot = n.lock().unwrap();
        let mut pixels = slot.take().map(|f| f.pixels).unwrap_or_default();
        pixels.resize(w * ht, 0);
        for r in 0..ht {
            pixels[r * w..(r + 1) * w].copy_from_slice(&y[r * stride..r * stride + w]);
        }
        count += 1;
        *slot = Some(Frame { pixels, width: w, height: ht, number: count, generation });
        drop(slot);
        let t = now_ms();
        h.decoded.fetch_add(1, Ordering::Relaxed);
        h.last_frame_ms.store(t, Ordering::Relaxed);
        h.width.store(w as u64, Ordering::Relaxed);
        h.height.store(ht as u64, Ordering::Relaxed);
        if h.state() != State::Up {
            h.up_since_ms.store(t, Ordering::Relaxed);
            h.set(State::Up);
        }
        b.ring();
        Ok(gst::FlowSuccess::Ok)
    }).build());

    if let Err(e) = pipeline.set_state(gst::State::Playing) {
        let _ = pipeline.set_state(gst::State::Null);
        return Outcome::Failed(format!("could not start: {e}"));
    }
    let bus = pipeline.bus().expect("a pipeline has a bus");
    let started = Instant::now();
    let outcome = loop {
        if stop.load(Ordering::Relaxed) {
            break Outcome::Stopped;
        }
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(200)) {
            use gst::MessageView;
            match msg.view() {
                MessageView::Eos(_) => {
                    break if is_live(target) { Outcome::Failed("the stream ended".into()) } else { Outcome::Ended };
                }
                // GStreamer's errors name elements and the problem, not the credentials.
                MessageView::Error(e) => break Outcome::Failed(e.error().to_string()),
                _ => {}
            }
        }
        // A stream that sends nothing for `stall`, from its start or its last frame, has stalled.
        let last = health.last_frame_ms.load(Ordering::Relaxed);
        let quiet = if health.state() == State::Up && last > 0 {
            Duration::from_millis(now_ms().saturating_sub(last))
        } else {
            started.elapsed()
        };
        if quiet > how.stall {
            break Outcome::Failed(format!("no frame for {:.0} s", quiet.as_secs_f64()));
        }
    };
    let _ = pipeline.set_state(gst::State::Null);
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reconnect_continues_the_segment_numbers() {
        let mut r = [0u8; 8];
        getrandom::fill(&mut r).unwrap();
        let d = std::env::temp_dir().join(format!("tsb-seg-{}", u64::from_le_bytes(r)));
        let pattern = format!("{}/gate/%05d.mkv", d.display());
        assert_eq!(next_index(&pattern).unwrap(), 0); // and the folder is made
        for i in [0, 1, 7] {
            std::fs::write(d.join(format!("gate/{i:05}.mkv")), b"x").unwrap();
        }
        std::fs::write(d.join("gate/notes.txt"), b"x").unwrap();
        assert_eq!(next_index(&pattern).unwrap(), 8);
    }

    #[test]
    fn camera_names_are_safe_in_paths() {
        assert_eq!(safe("gate/../../etc"), "gate_.._.._etc");
        assert_eq!(safe("Dock 3"), "Dock_3");
    }
}
