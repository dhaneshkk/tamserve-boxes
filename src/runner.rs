//! The loop that serves a set of cameras: wait for new frames, take each camera's
//! newest, step the tracker once for all of them, and report. The set can change while
//! it runs, which is how a coordinator moves cameras between boxes.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::camera::{now_ms, Bell, Health, Opening, Source, State};
use crate::config::Camera;
use crate::{BoxTracker, Detector, GrayFrame, Result};

/// One camera's objects on one frame.
#[derive(Serialize, Clone, Debug)]
pub struct Report {
    pub camera: String,
    /// The frame's number on its current connection.
    pub frame: u64,
    /// When the step that produced it finished, in ms since 1970.
    pub time_ms: u64,
    pub objects: Vec<Object>,
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
}

#[derive(Clone)]
pub struct View {
    pub label: String,
    pub health: Arc<Health>,
    pub latest: Option<Report>,
}

/// Serves cameras on this box.
pub struct Runner {
    tracker: BoxTracker<Detector, String>,
    sources: BTreeMap<String, Source>,
    opening: Opening,
    bell: Arc<Bell>,
    rung: u64,
    generation: HashMap<String, u64>,
    pub shared: Arc<Shared>,
}

impl Runner {
    pub fn new(tracker: BoxTracker<Detector, String>, opening: Opening) -> Runner {
        let shared = Arc::new(Shared::default());
        shared.started_ms.store(now_ms(), Ordering::Relaxed);
        Runner { tracker, sources: BTreeMap::new(), opening, bell: Arc::new(Bell::default()), rung: 0,
                 generation: HashMap::new(), shared }
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
            self.shared.cameras.lock().unwrap().remove(n);
        }
        let mut opened = Vec::new();
        for c in wanted {
            if self.sources.contains_key(&c.name) {
                continue;
            }
            let s = Source::open(c.clone(), &self.opening, self.bell.clone())?;
            self.shared.cameras.lock().unwrap().insert(c.name.clone(),
                View { label: c.label(), health: s.health.clone(), latest: None });
            self.sources.insert(c.name.clone(), s);
            opened.push(c.name.clone());
        }
        Ok((opened, gone))
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
        let looks_before = self.tracker.looks();
        let found = self.tracker.step(&frames)?;
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
            let objects = found[name].iter().map(|(&id, o)| Object { id, label: o.label.clone(), score: o.score, bbox: o.bbox }).collect();
            let r = Report { camera: name.clone(), frame: f.number, time_ms: t, objects };
            if let Some(v) = views.get_mut(name) {
                v.latest = Some(r.clone());
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
        let series: [(&str, &str, &str); 7] = [
            ("tamserve_camera_up", "gauge", "1 while frames are arriving"),
            ("tamserve_camera_frames_decoded_total", "counter", "frames the decoder handed over"),
            ("tamserve_camera_frames_used_total", "counter", "frames a step used; the rest were replaced by newer ones first"),
            ("tamserve_camera_reconnects_total", "counter", "times the stream was opened again after failing or stalling"),
            ("tamserve_camera_frame_age_seconds", "gauge", "seconds since the newest frame"),
            ("tamserve_camera_objects", "gauge", "objects the latest step showed"),
            ("tamserve_camera_state", "gauge", "1 for the camera's state: connecting, up, down, ended"),
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
                    "tamserve_camera_objects" => writeln!(m, "{name}{{camera=\"{c}\"}} {}", v.latest.as_ref().map_or(0, |r| r.objects.len())),
                    _ => {
                        for s in [State::Connecting, State::Up, State::Down, State::Ended] {
                            let _ = writeln!(m, "{name}{{camera=\"{c}\",state=\"{}\"}} {}", s.as_str(), u8::from(h.state() == s));
                        }
                        Ok(())
                    }
                };
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
        m
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
                "latest": v.latest,
            }))
        }).collect();
        serde_json::json!({
            "cameras": cameras,
            "steps": self.steps.load(Ordering::Relaxed),
            "looks": self.looks.load(Ordering::Relaxed),
            "step_seconds_last": self.step_us_last.load(Ordering::Relaxed) as f64 / 1e6,
            "uptime_seconds": (now_ms() - self.started_ms.load(Ordering::Relaxed)) as f64 / 1e3,
        })
    }
}
