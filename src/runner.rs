//! The loop that serves a set of cameras: wait for new frames, take each camera's
//! newest, step the tracker once for all of them, and report. The set can change while
//! it runs, which is how a coordinator moves cameras between boxes.
//!
//! With [`with_motion`](Runner::with_motion), each frame is first checked for motion,
//! and a camera that is quiet and follows nothing is not looked at. With
//! [`with_recording`](Runner::with_recording), what each recording camera showed is
//! written to its activity log, and a sweeper thread deletes recordings by
//! [`crate::retention`]'s rules.
//!
//! Every camera's day or night mode and its IR switches are followed by
//! [`crate::light`]: each report says which mode the camera is in (`light`) and whether
//! it is `settling` after a switch. A settling camera is not looked at, and its report
//! carries the objects it showed before the switch, so they keep their ids.
//!
//! A camera with `safety` in its config also runs the safety rules ([`crate::safety`]) on
//! its objects: line of fire, red zone, barricade closure and PPE, in metres on the floor.
//! Their verdicts are in its report (`safety`), on `/status`, on `/v1/chat/completions` and
//! in `tamserve_safety_*` metrics. A drift check posted to `POST /safety/<camera>/drift`
//! is applied on the camera's next step.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::camera::{now_ms, Bell, Health, Opening, Source, State};
use crate::config::Camera;
use crate::light::{mean_luma, LightMeter, LightOptions, Mode};
use crate::motion::{Gate, MotionOptions};
use crate::restream::{Chunk, Tap};
use crate::lens::{CalibState, DriftEstimate};
use crate::retention::{ActivityLog, Policy, Sweep};
use crate::safety::{CameraSafety, SafetyCam, SafetySpec};
use crate::{BBox, BoxTracker, Detector, GrayFrame, Result};

/// One camera's objects on one frame.
#[derive(Serialize, Clone, Debug)]
pub struct Report {
    pub camera: String,
    /// The frame's number on its current connection.
    pub frame: u64,
    /// When the step that produced it finished, in ms since 1970.
    pub time_ms: u64,
    pub objects: Vec<Object>,
    /// The camera's mode: "day", "night" (IR: its colour stream is gray too) or "unknown"
    /// (a gray stream, or not decided yet). Draw on the gray frame when it is not "day".
    pub light: Mode,
    /// Within a few seconds of the light switching: `objects` are the ones from before
    /// the switch, not looked for again until it settles.
    pub settling: bool,
    /// The safety rules' verdicts, for a camera with `safety` in its config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety: Option<CameraSafety>,
}

#[derive(Serialize, Clone, Debug)]
pub struct Object {
    pub id: u32,
    pub label: String,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: [f32; 4],
}

/// What the metrics and the HTTP endpoint read: each camera's health and latest report,
/// and the loop's counters.
#[derive(Default)]
pub struct Shared {
    pub cameras: Mutex<BTreeMap<String, View>>,
    pub steps: AtomicU64,
    pub step_us_total: AtomicU64,
    pub step_us_last: AtomicU64,
    pub looks: AtomicU64,
    pub started_ms: AtomicU64,
    pub recordings: Mutex<Recordings>,
    /// Drift checks posted to `POST /safety/<camera>/drift`, waiting for the camera's next step.
    pub drift: Mutex<HashMap<String, DriftEstimate>>,
}

/// The sweeper's counts, for the metrics.
#[derive(Default, Clone)]
pub struct Recordings {
    pub on: bool,
    pub last: Option<Sweep>,
    pub deleted_expired: u64,
    pub deleted_space: u64,
    pub deleted_bytes: u64,
    pub errors: u64,
    pub last_error: String,
}

#[derive(Clone)]
pub struct View {
    pub label: String,
    pub health: Arc<Health>,
    pub latest: Option<Report>,
    /// Where the latest frame moved, when `[motion]` is on.
    pub motion: Vec<BBox>,
    pub live: Option<Arc<Tap>>,
    /// Whether the camera runs the safety rules.
    pub safety: bool,
}

struct Motion {
    opts: MotionOptions,
    gate: bool,
    hold: u32,
    cameras: HashMap<String, Gate>,
    masks: HashMap<String, Vec<[f32; 4]>>,
}

struct Lights {
    opts: LightOptions,
    cameras: HashMap<String, LightMeter>,
}

/// Serves cameras on this box.
pub struct Runner {
    tracker: BoxTracker<Detector, String>,
    sources: BTreeMap<String, Source>,
    opening: Opening,
    bell: Arc<Bell>,
    rung: u64,
    generation: HashMap<String, u64>,
    motion: Option<Motion>,
    light: Lights,
    record: Option<PathBuf>,
    logs: HashMap<String, ActivityLog>,
    safety: HashMap<String, SafetyCam>,
    pub shared: Arc<Shared>,
}

impl Runner {
    pub fn new(tracker: BoxTracker<Detector, String>, opening: Opening) -> Runner {
        let shared = Arc::new(Shared::default());
        shared.started_ms.store(now_ms(), Ordering::Relaxed);
        Runner { tracker, sources: BTreeMap::new(), opening, bell: Arc::new(Bell::default()), rung: 0,
                 generation: HashMap::new(), motion: None,
                 light: Lights { opts: LightOptions::default(), cameras: HashMap::new() },
                 record: None, logs: HashMap::new(), safety: HashMap::new(), shared }
    }

    /// How day, night and IR switches are found ([`crate::light`]). Without this, the
    /// defaults.
    pub fn with_light(mut self, opts: LightOptions) -> Result<Runner> {
        opts.check()?;
        self.light = Lights { opts, cameras: HashMap::new() };
        Ok(self)
    }

    /// Find motion on every frame; with `gate`, a camera that has not moved for more than
    /// `hold` frames and follows nothing is not looked at.
    pub fn with_motion(mut self, opts: MotionOptions, gate: bool, hold: u32) -> Result<Runner> {
        opts.check()?;
        self.motion = Some(Motion { opts, gate, hold, cameras: HashMap::new(), masks: HashMap::new() });
        Ok(self)
    }

    /// Record cameras with `record = true` to `dir` in `segment`-long pieces, and sweep
    /// `dir` by `policy` every `every`. Call before [`set_cameras`](Runner::set_cameras).
    pub fn with_recording(mut self, dir: PathBuf, segment: Duration, policy: Policy, every: Duration) -> Result<Runner> {
        crate::retention::private_dir(&dir)?;
        self.opening.record = Some(crate::camera::RecordTo { dir: dir.clone(), segment });
        self.record = Some(dir.clone());
        self.shared.recordings.lock().unwrap().on = true;
        let shared = Arc::downgrade(&self.shared);
        std::thread::Builder::new().name("recordings sweeper".into()).spawn(move || loop {
            std::thread::sleep(every);
            let Some(shared) = shared.upgrade() else { return };
            let result = crate::retention::sweep(&dir, &policy, now_ms(), false);
            let mut r = shared.recordings.lock().unwrap();
            match result {
                Ok(sw) => {
                    for d in &sw.deleted {
                        match d.why {
                            crate::retention::Why::Expired => r.deleted_expired += 1,
                            crate::retention::Why::Space => r.deleted_space += 1,
                        }
                    }
                    r.deleted_bytes += sw.deleted_bytes;
                    r.last = Some(sw);
                }
                Err(e) => {
                    r.errors += 1;
                    r.last_error = e.to_string();
                    eprintln!("tamserve-boxes: sweeping the recordings failed: {e}");
                }
            }
        })?;
        Ok(self)
    }

    /// The cameras on this box.
    pub fn cameras(&self) -> Vec<String> {
        self.sources.keys().cloned().collect()
    }

    /// Serve exactly `wanted`: open the new ones, close the ones no longer wanted.
    /// Cameras already open stay open. Returns the names opened and closed.
    pub fn set_cameras(&mut self, wanted: &[Camera]) -> Result<(Vec<String>, Vec<String>)> {
        let names: std::collections::BTreeSet<&str> = wanted.iter().map(|c| c.name.as_str()).collect();
        let gone: Vec<String> = self.sources.keys().filter(|n| !names.contains(n.as_str())).cloned().collect();
        for n in &gone {
            self.sources.remove(n); // stops its thread
            self.tracker.drop_camera(n);
            self.generation.remove(n);
            self.light.cameras.remove(n);
            self.safety.remove(n);
            self.shared.drift.lock().unwrap().remove(n);
            if let Some(m) = &mut self.motion {
                m.cameras.remove(n);
                m.masks.remove(n);
            }
            if let Some(mut log) = self.logs.remove(n) {
                let _ = log.flush(now_ms());
            }
            self.shared.cameras.lock().unwrap().remove(n);
        }
        let mut opened = Vec::new();
        for c in wanted {
            if self.sources.contains_key(&c.name) {
                continue;
            }
            let safety = c.safety.as_deref().map(SafetySpec::load).transpose()?;
            let s = Source::open(c.clone(), &self.opening, self.bell.clone())?;
            self.shared.cameras.lock().unwrap().insert(c.name.clone(),
                View { label: c.label(), health: s.health.clone(), latest: None, motion: Vec::new(), live: s.live.clone(),
                       safety: safety.is_some() });
            if let Some(spec) = safety {
                self.safety.insert(c.name.clone(), SafetyCam::new(spec));
            }
            if let Some(m) = &mut self.motion {
                m.masks.insert(c.name.clone(), c.motion_mask.clone());
            }
            if let (true, Some(dir)) = (c.record, &self.record) {
                self.logs.insert(c.name.clone(), ActivityLog::new(dir, &c.name));
            }
            self.sources.insert(c.name.clone(), s);
            opened.push(c.name.clone());
        }
        Ok((opened, gone))
    }

    /// Stop every camera at once, so that their recordings close their open segments in
    /// parallel, and write what the activity logs hold.
    pub fn close(mut self) {
        self.sources.values().for_each(Source::stop_soon);
        let t = now_ms();
        for log in self.logs.values_mut() {
            let _ = log.flush(t);
        }
        self.sources.clear(); // joins each camera's thread
    }

    /// Whether every camera has stopped for good (files read to their end).
    pub fn all_ended(&self) -> bool {
        !self.sources.is_empty() && self.sources.values().all(|s| s.health.state() == State::Ended)
    }

    /// Wait up to `timeout` for new frames, then one step over every camera that has one.
    pub fn step(&mut self, timeout: Duration) -> Result<Vec<Report>> {
        self.rung = self.bell.wait(self.rung, timeout);
        let mut taken: Vec<(String, crate::camera::Frame)> = Vec::new();
        for (name, s) in &self.sources {
            if let Some(f) = s.take() {
                // A new connection: the frames before it are not this camera's past any more.
                if self.generation.insert(name.clone(), f.generation).is_some_and(|g| g != f.generation) {
                    self.tracker.drop_camera(name);
                    if let Some(g) = self.motion.as_mut().and_then(|m| m.cameras.get_mut(name)) {
                        g.reset();
                    }
                    if let Some(l) = self.light.cameras.get_mut(name) {
                        l.reconnected();
                    }
                    if let Some(sc) = self.safety.get_mut(name) {
                        sc.reset();
                    }
                }
                s.health.used.fetch_add(1, Ordering::Relaxed);
                taken.push((name.clone(), f));
            }
        }
        if taken.is_empty() {
            return Ok(Vec::new());
        }
        let t0 = Instant::now();
        let frames: Vec<(String, GrayFrame)> = taken.iter()
            .map(|(n, f)| Ok((n.clone(), GrayFrame::new(&f.pixels, f.width, f.height)?)))
            .collect::<Result<_>>()?;
        let mut quiet: HashSet<String> = HashSet::new();
        let mut moved: HashMap<String, Vec<BBox>> = HashMap::new();
        if let Some(m) = &mut self.motion {
            for (name, f) in &frames {
                let (opts, hold, masks) = (&m.opts, m.hold, &m.masks);
                let g = m.cameras.entry(name.clone()).or_insert_with(|| {
                    Gate::new(MotionOptions { mask: masks.get(name).cloned().unwrap_or_default(), ..opts.clone() }, hold)
                });
                if g.next(f) {
                    quiet.insert(name.clone());
                }
                let h = &self.sources[name].health;
                h.moving.store(!g.quiet(), Ordering::Relaxed);
                h.lighting_resets.store(g.motion.lighting_resets, Ordering::Relaxed);
                if g.last.moved() {
                    h.motion_steps.fetch_add(1, Ordering::Relaxed);
                }
                moved.insert(name.clone(), g.last.boxes.clone());
            }
        }
        // Day, night, and the light switching: a settling camera is not looked at.
        let now = now_ms();
        let mut lights: HashMap<String, crate::light::Light> = HashMap::with_capacity(frames.len());
        for ((name, f), (_, raw)) in frames.iter().zip(&taken) {
            let lighting = self.motion.as_ref().and_then(|m| m.cameras.get(name)).is_some_and(|g| g.last.lighting);
            let opts = &self.light.opts;
            let meter = self.light.cameras.entry(name.clone()).or_insert_with(|| LightMeter::new(opts.clone()));
            let l = meter.next(now, raw.chroma, mean_luma(f), lighting);
            let h = &self.sources[name].health;
            h.light_mode.store(l.mode.code(), Ordering::Relaxed);
            h.light_switches.store(meter.switches, Ordering::Relaxed);
            h.settling.store(l.settling, Ordering::Relaxed);
            lights.insert(name.clone(), l);
        }
        let looked: Vec<(String, GrayFrame)> = frames.iter().filter(|(n, _)| !lights[n].settling).cloned().collect();
        let gate = self.motion.as_ref().is_some_and(|m| m.gate);
        let looks_before = self.tracker.looks();
        let mut found = self.tracker.step_regions(&looked, &|c| gate && quiet.contains(c),
                                                  &|c| moved.get(c).cloned().unwrap_or_default())?;
        for c in self.tracker.skipped() {
            self.sources[c].health.looks_skipped.fetch_add(1, Ordering::Relaxed);
        }
        let us = t0.elapsed().as_micros() as u64;
        let sh = &self.shared;
        sh.steps.fetch_add(1, Ordering::Relaxed);
        sh.step_us_total.fetch_add(us, Ordering::Relaxed);
        sh.step_us_last.store(us, Ordering::Relaxed);
        sh.looks.fetch_add(self.tracker.looks() - looks_before, Ordering::Relaxed);
        let t = now_ms();
        let mut reports = Vec::with_capacity(taken.len());
        let mut views = sh.cameras.lock().unwrap();
        for (name, f) in &taken {
            let light = lights[name];
            let objects = match found.remove(name) {
                Some(seen) => seen.iter().map(|(&id, o)| Object { id, label: o.label.clone(), score: o.score, bbox: o.bbox }).collect(),
                // Settling: what it showed before the switch.
                None => views.get(name).and_then(|v| v.latest.as_ref()).map(|r| r.objects.clone()).unwrap_or_default(),
            };
            let safety = match self.safety.get_mut(name) {
                // Settling: nothing was looked at, so the rules keep their last verdicts.
                Some(_) if light.settling => views.get(name).and_then(|v| v.latest.as_ref()).and_then(|r| r.safety.clone()),
                Some(sc) => {
                    let drift = sh.drift.lock().unwrap().remove(name);
                    let dets = sc.detections(objects.iter().map(|o| (o.label.as_str(), o.score, o.bbox)));
                    Some(sc.process(name, &dets, drift.as_ref()))
                }
                None => None,
            };
            let r: Report = Report { camera: name.clone(), frame: f.number, time_ms: t, objects,
                                     light: light.mode, settling: light.settling, safety };
            let motion = moved.remove(name).unwrap_or_default();
            if let Some(log) = self.logs.get_mut(name) {
                let mut labels: Vec<String> = r.objects.iter().map(|o| o.label.clone()).collect();
                labels.sort();
                labels.dedup();
                let moving = self.motion.is_some() && !quiet.contains(name);
                if let Err(e) = log.note(t, moving, &labels) {
                    self.sources[name].health.fail(format!("cannot write the activity log: {e}"));
                }
            }
            if let Some(v) = views.get_mut(name) {
                v.latest = Some(r.clone());
                v.motion = motion;
            }
            reports.push(r);
        }
        Ok(reports)
    }
}

impl crate::http::Source for Shared {
    fn metrics(&self) -> String {
        use crate::http::label;
        use std::fmt::Write;
        let mut m = String::new();
        let cams = self.cameras.lock().unwrap();
        let series: [(&str, &str, &str); 20] = [
            ("tamserve_camera_encoded_frames_total", "counter", "frames the camera's encoder produced"),
            ("tamserve_camera_encoded_bytes_total", "counter", "bytes the camera's encoder produced"),
            ("tamserve_camera_encode_dropped_total", "counter", "times the encoder's backlog was full and a frame was dropped instead of stalling decoding"),
            ("tamserve_camera_up", "gauge", "1 while frames are arriving"),
            ("tamserve_camera_frames_decoded_total", "counter", "frames the decoder handed over"),
            ("tamserve_camera_frames_used_total", "counter", "frames a step used; the rest were replaced by newer ones first"),
            ("tamserve_camera_reconnects_total", "counter", "times the stream was opened again after failing or stalling"),
            ("tamserve_camera_frame_age_seconds", "gauge", "seconds since the newest frame"),
            ("tamserve_camera_objects", "gauge", "objects the latest step showed"),
            ("tamserve_camera_state", "gauge", "1 for the camera's state: connecting, up, down, ended"),
            ("tamserve_camera_moving", "gauge", "1 while motion is held open ([motion])"),
            ("tamserve_camera_motion_steps_total", "counter", "steps on which the frame moved"),
            ("tamserve_camera_looks_skipped_total", "counter", "looks not made: quiet, with nothing followed"),
            ("tamserve_camera_lighting_resets_total", "counter", "times the whole picture changed at once and motion started over"),
            ("tamserve_camera_light_mode", "gauge", "0 unknown (a gray stream), 1 day, 2 night (IR)"),
            ("tamserve_camera_light_switches_total", "counter", "times the light switched (IR cut filter, floodlights) and the camera settled"),
            ("tamserve_camera_settling", "gauge", "1 while the camera settles after a switch and is not looked at"),
            ("tamserve_camera_segments_total", "counter", "recorded segments started"),
            ("tamserve_camera_live_viewers", "gauge", "viewers of GET /live/<camera>.ts"),
            ("tamserve_camera_live_bytes_total", "counter", "bytes sent to live viewers, counted once per chunk"),
        ];
        for (name, kind, help) in series {
            let _ = writeln!(m, "# HELP {name} {help}\n# TYPE {name} {kind}");
            for (cam, v) in cams.iter() {
                let h = &v.health;
                let c = label(cam);
                let _ = match name {
                    "tamserve_camera_up" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", u8::from(h.state() == State::Up)),
                    "tamserve_camera_frames_decoded_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.decoded.load(Ordering::Relaxed)),
                    "tamserve_camera_frames_used_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.used.load(Ordering::Relaxed)),
                    "tamserve_camera_reconnects_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.reconnects.load(Ordering::Relaxed)),
                    "tamserve_camera_frame_age_seconds" => match h.frame_age() {
                        Some(a) => writeln!(m, "{name}{{camera=\"{c}\"}} {a:.3}"),
                        None => Ok(()),
                    },
                    "tamserve_camera_encoded_frames_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.encoded.load(Ordering::Relaxed)),
                    "tamserve_camera_encoded_bytes_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.encoded_bytes.load(Ordering::Relaxed)),
                    "tamserve_camera_encode_dropped_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.encode_dropped.load(Ordering::Relaxed)),
                    "tamserve_camera_objects" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", v.latest.as_ref().map_or(0, |r| r.objects.len())),
                    "tamserve_camera_moving" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", u8::from(h.moving.load(Ordering::Relaxed))),
                    "tamserve_camera_motion_steps_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.motion_steps.load(Ordering::Relaxed)),
                    "tamserve_camera_looks_skipped_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.looks_skipped.load(Ordering::Relaxed)),
                    "tamserve_camera_lighting_resets_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.lighting_resets.load(Ordering::Relaxed)),
                    "tamserve_camera_light_mode" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.light_mode.load(Ordering::Relaxed)),
                    "tamserve_camera_light_switches_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.light_switches.load(Ordering::Relaxed)),
                    "tamserve_camera_settling" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", u8::from(h.settling.load(Ordering::Relaxed))),
                    "tamserve_camera_segments_total" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", h.segments.load(Ordering::Relaxed)),
                    "tamserve_camera_live_viewers" => match &v.live {
                        Some(t) => writeln!(m, "{name}{{camera=\"{c}\"}} {}", t.viewers()),
                        None => Ok(()),
                    },
                    "tamserve_camera_live_bytes_total" => match &v.live {
                        Some(t) => writeln!(m, "{name}{{camera=\"{c}\"}} {}", t.bytes.load(Ordering::Relaxed)),
                        None => Ok(()),
                    },
                    _ => {
                        for s in [State::Connecting, State::Up, State::Down, State::Ended] {
                            let _ = writeln!(m, "{name}{{camera=\"{c}\",state=\"{}\"}} {}", s.as_str(), u8::from(h.state() == s));
                        }
                        Ok(())
                    }
                };
            }
        }
        let safe: Vec<(&String, &CameraSafety)> = cams.iter().filter_map(|(n, v)| v.latest.as_ref()?.safety.as_ref().map(|s| (n, s))).collect();
        if !safe.is_empty() {
            let _ = writeln!(m, "# HELP tamserve_safety_line_of_fire people in a forklift's line of fire\n# TYPE tamserve_safety_line_of_fire gauge");
            for (cam, s) in &safe {
                let n = s.report.workers.iter().filter(|w| w.line_of_fire == crate::grammar::Tri::Yes).count();
                let _ = writeln!(m, "tamserve_safety_line_of_fire{{camera=\"{}\"}} {n}", label(cam));
            }
            let _ = writeln!(m, "# HELP tamserve_safety_red_zone people in a red zone\n# TYPE tamserve_safety_red_zone gauge");
            for (cam, s) in &safe {
                let n = s.report.workers.iter().filter(|w| w.red_zone == crate::grammar::Tri::Yes).count();
                let _ = writeln!(m, "tamserve_safety_red_zone{{camera=\"{}\"}} {n}", label(cam));
            }
            let _ = writeln!(m, "# HELP tamserve_safety_perimeter_breach 1 while the barricade is open; absent while unknown\n# TYPE tamserve_safety_perimeter_breach gauge");
            for (cam, s) in &safe {
                if let Some(b) = s.report.perimeter.breach {
                    let _ = writeln!(m, "tamserve_safety_perimeter_breach{{camera=\"{}\"}} {}", label(cam), u8::from(b));
                }
            }
            let _ = writeln!(m, "# HELP tamserve_safety_calibration 1 for the floor calibration's state: ok, corrected, unknown\n# TYPE tamserve_safety_calibration gauge");
            for (cam, s) in &safe {
                for (st, name) in [(CalibState::Ok, "ok"), (CalibState::Corrected, "corrected"), (CalibState::Unknown, "unknown")] {
                    let _ = writeln!(m, "tamserve_safety_calibration{{camera=\"{}\",state=\"{name}\"}} {}", label(cam), u8::from(s.report.calibration == st));
                }
            }
            let _ = writeln!(m, "# HELP tamserve_safety_ppe people by helmet and vest verdict, after filling short gaps\n# TYPE tamserve_safety_ppe gauge");
            for (cam, s) in &safe {
                for (item, pick) in [("helmet", 0usize), ("vest", 1)] {
                    for (v, name) in [(crate::grammar::Ppe::Compliant, "compliant"), (crate::grammar::Ppe::Violation, "violation"), (crate::grammar::Ppe::Unknown, "unknown")] {
                        let n = s.ppe_fused.iter().filter(|w| (if pick == 0 { w.helmet } else { w.vest }).verdict == v).count();
                        let _ = writeln!(m, "tamserve_safety_ppe{{camera=\"{}\",item=\"{item}\",verdict=\"{name}\"}} {n}", label(cam));
                    }
                }
            }
        }
        let steps = self.steps.load(Ordering::Relaxed);
        let _ = writeln!(m, "# HELP tamserve_steps_total steps the loop has run\n# TYPE tamserve_steps_total counter\ntamserve_steps_total {steps}");
        let _ = writeln!(m, "# HELP tamserve_step_seconds_total time spent in steps\n# TYPE tamserve_step_seconds_total counter\ntamserve_step_seconds_total {:.6}",
                         self.step_us_total.load(Ordering::Relaxed) as f64 / 1e6);
        let _ = writeln!(m, "# HELP tamserve_step_seconds_last the latest step's time\n# TYPE tamserve_step_seconds_last gauge\ntamserve_step_seconds_last {:.6}",
                         self.step_us_last.load(Ordering::Relaxed) as f64 / 1e6);
        let _ = writeln!(m, "# HELP tamserve_looks_total frames the detector has looked at\n# TYPE tamserve_looks_total counter\ntamserve_looks_total {}",
                         self.looks.load(Ordering::Relaxed));
        let r = self.recordings.lock().unwrap();
        if r.on {
            let (bytes, segments, pinned) = r.last.as_ref().map_or((0, 0, 0), |s| (s.bytes - s.deleted_bytes, s.segments - s.deleted.len(), s.pinned));
            let _ = writeln!(m, "# HELP tamserve_recordings_bytes what the recordings take, at the latest sweep\n# TYPE tamserve_recordings_bytes gauge\ntamserve_recordings_bytes {bytes}");
            let _ = writeln!(m, "# HELP tamserve_recordings_segments segments kept, at the latest sweep\n# TYPE tamserve_recordings_segments gauge\ntamserve_recordings_segments {segments}");
            let _ = writeln!(m, "# HELP tamserve_recordings_pinned segments pinned, at the latest sweep\n# TYPE tamserve_recordings_pinned gauge\ntamserve_recordings_pinned {pinned}");
            let _ = writeln!(m, "# HELP tamserve_recordings_deleted_total segments deleted, by why\n# TYPE tamserve_recordings_deleted_total counter\n\
                                 tamserve_recordings_deleted_total{{why=\"expired\"}} {}\ntamserve_recordings_deleted_total{{why=\"space\"}} {}",
                             r.deleted_expired, r.deleted_space);
            let _ = writeln!(m, "# HELP tamserve_recordings_sweep_errors_total sweeps that failed\n# TYPE tamserve_recordings_sweep_errors_total counter\ntamserve_recordings_sweep_errors_total {}", r.errors);
        }
        m
    }

    fn drift(&self, camera: &str, est: DriftEstimate) -> bool {
        if !self.cameras.lock().unwrap().get(camera).is_some_and(|v| v.safety) {
            return false;
        }
        self.drift.lock().unwrap().insert(camera.to_string(), est);
        true
    }

    fn live(&self, camera: &str) -> std::result::Result<std::sync::mpsc::Receiver<Chunk>, crate::http::LiveRefused> {
        let tap = self.cameras.lock().unwrap().get(camera).and_then(|v| v.live.clone()).ok_or(crate::http::LiveRefused::NoSuchStream)?;
        tap.subscribe().ok_or(crate::http::LiveRefused::Full)
    }

    fn status(&self) -> serde_json::Value {
        let cams = self.cameras.lock().unwrap();
        let cameras: serde_json::Map<String, serde_json::Value> = cams.iter().map(|(n, v)| {
            let h = &v.health;
            (n.clone(), serde_json::json!({
                "source": v.label,
                "state": h.state().as_str(),
                "frames_decoded": h.decoded.load(Ordering::Relaxed),
                "frames_used": h.used.load(Ordering::Relaxed),
                "reconnects": h.reconnects.load(Ordering::Relaxed),
                "frame_age_seconds": h.frame_age(),
                "size": [h.width.load(Ordering::Relaxed), h.height.load(Ordering::Relaxed)],
                "last_error": h.last_error(),
                "encoded_frames": h.encoded.load(Ordering::Relaxed),
                "encoded_bytes": h.encoded_bytes.load(Ordering::Relaxed),
                "encode_dropped": h.encode_dropped.load(Ordering::Relaxed),
                "latest": v.latest,
                "moving": h.moving.load(Ordering::Relaxed),
                "motion": v.motion,
                "looks_skipped": h.looks_skipped.load(Ordering::Relaxed),
                "light": (["unknown", "day", "night"].get(h.light_mode.load(Ordering::Relaxed) as usize)),
                "light_switches": h.light_switches.load(Ordering::Relaxed),
                "settling": h.settling.load(Ordering::Relaxed),
                "segments": h.segments.load(Ordering::Relaxed),
                "live_viewers": v.live.as_ref().map(|t| t.viewers()),
            }))
        }).collect();
        let r = self.recordings.lock().unwrap();
        let recordings = r.on.then(|| serde_json::json!({
            "last_sweep": r.last.as_ref().map(|s| serde_json::json!({
                "time_ms": s.time_ms, "segments": s.segments - s.deleted.len(), "bytes": s.bytes - s.deleted_bytes, "pinned": s.pinned,
                "deleted": s.deleted.len(), "deleted_bytes": s.deleted_bytes})),
            "deleted_expired": r.deleted_expired, "deleted_space": r.deleted_space, "errors": r.errors, "last_error": r.last_error,
        }));
        serde_json::json!({
            "cameras": cameras,
            "steps": self.steps.load(Ordering::Relaxed),
            "looks": self.looks.load(Ordering::Relaxed),
            "step_seconds_last": self.step_us_last.load(Ordering::Relaxed) as f64 / 1e6,
            "uptime_seconds": (now_ms() - self.started_ms.load(Ordering::Relaxed)) as f64 / 1e3,
            "recordings": recordings,
        })
    }
}
