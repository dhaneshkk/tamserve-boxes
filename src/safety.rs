//! The safety rules for one camera, as the runner uses them.
//!
//! A camera with `safety = "path/to/spec.json"` in its config gets a [`SafetyCam`]: the rule
//! engine ([`crate::grammar`]), the PPE fusion that fills short UNKNOWN gaps from the
//! camera's own last clear look, and a label queue for fine-tuning. Each step, the tracker's
//! objects go in and a [`CameraSafety`] comes out in the camera's report, so it shows on
//! `/status`, on `/v1/chat/completions` and as `tamserve_safety_*` metrics.
//!
//! The spec is one JSON file (the notebook's `export_rust_golden` writes one: its extra
//! `episodes` are ignored):
//!
//! ```json
//! { "site": {...}, "calib": {"K": [[...]], "D": [...], "H": [[...]]},
//!   "ppe": {"helmet": {...}, "vest": {...}},          // optional
//!   "backend": "native",                              // or "datalog" with "clauses"
//!   "clauses": [...],
//!   "labels": {"worker": "person"},                   // detector label -> rule class, optional
//!   "fusion": {"own_window_frames": 20},              // optional
//!   "label_queue": {"file": "/var/lib/tamserve/labels.jsonl", "per_hour": 100},
//!   "perimeter": true }                               // false: no barricade in view
//! ```
//!
//! Leave out `ppe` until the detector knows `helmet` and `vest`: with a stock COCO model
//! nobody ever has a helmet box, so everyone in plain view would be a violation.
//!
//! The drift check runs outside (it needs OpenCV) and posts its result to
//! `POST /safety/<camera>/drift`; until a check fails, the calibration is trusted.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::datalog::Clause;
use crate::grammar::{Backend, Detection, FusedPpe, Grid, LabelCandidate, LabelQueue, PpeFusion, PpeHead, Report, SafetyEngine, Site};
use crate::lens::{Calibration, DriftEstimate, Lens, Mat3};
use crate::{Error, Result};

#[derive(Clone, Debug, Deserialize)]
pub struct CalibSpec {
    #[serde(rename = "K")]
    pub k: Mat3,
    #[serde(rename = "D")]
    pub d: [f64; 5],
    #[serde(rename = "H")]
    pub h: Mat3,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PpeSpec {
    pub helmet: PpeHead,
    pub vest: PpeHead,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FusionSpec {
    /// How old the camera's own last clear PPE look may be to fill an UNKNOWN, in frames
    /// (0 turns fusion off).
    #[serde(default = "twenty")]
    pub own_window_frames: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LabelQueueSpec {
    /// Append candidates here, one JSON object per line.
    pub file: PathBuf,
    #[serde(default = "hundred")]
    pub per_hour: usize,
    #[serde(default = "twenty")]
    pub gap_frames: i64,
    #[serde(default = "quarter")]
    pub margin: f32,
    #[serde(default = "ten")]
    pub fps: f64,
}

fn twenty() -> i64 {
    20
}
fn hundred() -> usize {
    100
}
fn quarter() -> f32 {
    0.25
}
fn ten() -> f64 {
    10.0
}

/// One camera's safety settings.
#[derive(Clone, Debug, Deserialize)]
pub struct SafetySpec {
    pub site: Site,
    pub calib: CalibSpec,
    #[serde(default)]
    pub ppe: Option<PpeSpec>,
    /// "native" (default) or "datalog".
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub clauses: Vec<Clause>,
    /// Detector label to rule class (person, forklift, helmet, vest, barrier_post); labels
    /// not listed are used as they are.
    #[serde(default)]
    pub labels: HashMap<String, String>,
    #[serde(default)]
    pub fusion: Option<FusionSpec>,
    #[serde(default)]
    pub label_queue: Option<LabelQueueSpec>,
    /// Whether this camera watches a barricade (default true). Without one, no posts are
    /// ever seen and the closure test would always say "open": set false, and the
    /// perimeter is reported as unknown.
    #[serde(default = "yes")]
    pub perimeter: bool,
}

fn yes() -> bool {
    true
}

impl SafetySpec {
    pub fn load(path: &Path) -> Result<SafetySpec> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Invalid(format!("cannot read the safety spec {}: {e}", path.display())))?;
        let spec: SafetySpec = serde_json::from_str(&text).map_err(|e| Error::Invalid(format!("the safety spec {}: {e}", path.display())))?;
        spec.check().map_err(|e| Error::Invalid(format!("the safety spec {}: {e}", path.display())))?;
        Ok(spec)
    }

    /// The mistakes worth catching before a camera starts.
    pub fn check(&self) -> std::result::Result<(), String> {
        let lens = Lens { k: self.calib.k, d: self.calib.d };
        if !lens.monotone(4096, 4096) && !lens.monotone(640, 480) {
            return Err("the lens model folds inside the picture: recalibrate".into());
        }
        let s = &self.site;
        if s.cell <= 0.0 || s.floor_w <= 0.0 || s.floor_h <= 0.0 || s.hi <= s.lo || s.hi > s.s_max {
            return Err("site: cell, floor size and hysteresis (lo < hi <= s_max) must make sense".into());
        }
        let (w, h) = ((s.floor_w / s.cell) as i64, (s.floor_h / s.cell) as i64);
        let [x0, y0, x1, y1] = s.fill_box;
        if x0 < 0 || y0 < 0 || x1 >= w || y1 >= h || x0 >= x1 || y0 >= y1 {
            return Err(format!("site: fill_box {:?} must lie inside the {w} x {h} grid", s.fill_box));
        }
        match self.backend.as_deref() {
            None | Some("native") => Ok(()),
            Some("datalog") if self.clauses.is_empty() => Err("backend \"datalog\" needs \"clauses\"".into()),
            Some("datalog") => Ok(()),
            Some(b) => Err(format!("backend {b:?}: \"native\" or \"datalog\"")),
        }
    }
}

/// A person's PPE after fusion.
#[derive(Clone, Debug, Serialize)]
pub struct FusedWorker {
    pub track: u32,
    pub helmet: FusedPpe,
    pub vest: FusedPpe,
}

/// What the safety rules decided on one frame.
#[derive(Clone, Debug, Serialize)]
pub struct CameraSafety {
    #[serde(flatten)]
    pub report: Report,
    /// Helmet and vest after filling short gaps from the camera's own last clear look.
    pub ppe_fused: Vec<FusedWorker>,
    /// Crops queued for labelling on this frame.
    pub label_candidates: usize,
}

/// One camera's rules, fusion and label queue.
pub struct SafetyCam {
    spec: SafetySpec,
    engine: SafetyEngine,
    fusion: PpeFusion,
    queue: Option<LabelQueue>,
    pub queued_total: u64,
}

fn engine(spec: &SafetySpec) -> SafetyEngine {
    let calib = Calibration::new(Lens { k: spec.calib.k, d: spec.calib.d }, spec.calib.h, spec.site.drift_ok_px);
    let ppe = spec.ppe.as_ref().map(|p| [p.helmet.clone(), p.vest.clone()]);
    let backend = match spec.backend.as_deref() {
        Some("datalog") => Backend::Datalog(spec.clauses.clone()),
        _ => Backend::Native,
    };
    SafetyEngine::new(Grid::new(spec.site.clone()), calib, ppe, backend)
}

impl SafetyCam {
    pub fn new(spec: SafetySpec) -> Self {
        let queue = spec.label_queue.as_ref().map(|q| LabelQueue::new(q.gap_frames, q.margin, q.per_hour, q.fps));
        SafetyCam { engine: engine(&spec), fusion: PpeFusion::new(true, false, 0.8), queue, queued_total: 0, spec }
    }

    /// Start over: a new connection's frames are not the old one's future.
    pub fn reset(&mut self) {
        let drift = self.engine.calib.clone();
        *self = SafetyCam { queued_total: self.queued_total, ..SafetyCam::new(self.spec.clone()) };
        // A reconnect does not move the camera: keep what the drift check last found.
        self.engine.calib = drift;
    }

    /// The tracker's objects, as the rules' detections (labels mapped through the spec).
    pub fn detections<'a>(&self, objects: impl IntoIterator<Item = (&'a str, f32, [f32; 4])>) -> Vec<Detection> {
        objects.into_iter().map(|(label, score, b)| Detection {
            cls: self.spec.labels.get(label).cloned().unwrap_or_else(|| label.to_string()),
            conf: score as f64,
            bbox: [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64],
        }).collect()
    }

    /// One frame. `camera` names the label queue's lines.
    pub fn process(&mut self, camera: &str, dets: &[Detection], drift: Option<&DriftEstimate>) -> CameraSafety {
        let mut report = self.engine.process(dets, drift);
        if !self.spec.perimeter {
            let p = &mut report.perimeter;
            (p.breach, p.raw_open, p.gap_floor_xy, p.escape_path) = (None, None, None, None);
        }
        let window = self.spec.fusion.as_ref().map_or(20, |f| f.own_window_frames);
        let f = report.frame;
        let ppe_fused = if window > 0 {
            self.fusion.observe(f, &report.workers, &[]);
            let live: std::collections::HashSet<u32> = report.workers.iter().map(|w| w.track).collect();
            let out = report.workers.iter().map(|w| {
                let [helmet, vest] = self.fusion.fused(f, w, window);
                FusedWorker { track: w.track, helmet, vest }
            }).collect();
            if f % 100 == 0 {
                self.fusion.retain(|t| live.contains(&t));
            }
            out
        } else {
            report.workers.iter().map(|w| FusedWorker { track: w.track, helmet: crate::grammar::fuse_ppe(w.helmet, None, f, 0),
                                                        vest: crate::grammar::fuse_ppe(w.vest, None, f, 0) }).collect()
        };
        let mut label_candidates = 0;
        if let (Some(q), Some(spec)) = (&mut self.queue, &self.spec.label_queue) {
            let c = q.consider(f, &report.workers);
            label_candidates = c.len();
            self.queued_total += c.len() as u64;
            if !c.is_empty() {
                if let Err(e) = append(&spec.file, camera, &c) {
                    eprintln!("tamserve-boxes: cannot write the label queue {}: {e}", spec.file.display());
                }
            }
        }
        CameraSafety { report, ppe_fused, label_candidates }
    }
}

fn append(path: &Path, camera: &str, c: &[LabelCandidate]) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut buf = String::new();
    for x in c {
        let mut v = serde_json::to_value(x).map_err(std::io::Error::other)?;
        v["camera"] = serde_json::Value::String(camera.to_string());
        buf.push_str(&v.to_string());
        buf.push('\n');
    }
    f.write_all(buf.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SafetySpec {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vg360_grammar_golden.json");
        SafetySpec::load(Path::new(p)).expect("the golden file is a valid spec")
    }

    #[test]
    fn a_spec_with_mistakes_is_refused() {
        let mut s = spec();
        s.backend = Some("datalog".into());
        s.clauses.clear();
        assert!(s.check().is_err());
        let mut s = spec();
        s.site.fill_box = [0, 0, 400, 10];
        assert!(s.check().is_err());
        let mut s = spec();
        s.calib.d = [-0.9, 0.0, 0.0, 0.0, 0.0];
        assert!(s.check().is_err());
    }

    #[test]
    fn no_barricade_means_the_perimeter_is_unknown_not_open() {
        let mut s = spec();
        s.perimeter = false;
        let mut cam = SafetyCam::new(s);
        let r = (0..40).map(|_| cam.process("yard", &[], None)).last().unwrap();
        assert_eq!((r.report.perimeter.breach, r.report.perimeter.raw_open), (None, None));
        let mut cam = SafetyCam::new(spec()); // with the default, no posts in view reads as open
        let r = (0..40).map(|_| cam.process("yard", &[], None)).last().unwrap();
        assert_eq!(r.report.perimeter.breach, Some(true));
    }

    #[test]
    fn objects_go_in_and_verdicts_come_out() {
        let mut s = spec();
        s.labels.insert("worker".into(), "person".into());
        let dir = std::env::temp_dir().join(format!("tamserve-labels-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("q.jsonl");
        s.label_queue = Some(LabelQueueSpec { file: file.clone(), per_hour: 1000, gap_frames: 1, margin: 0.5, fps: 10.0 });
        let mut cam = SafetyCam::new(s);
        let d = cam.detections([("worker", 0.9f32, [300.0f32, 200.0, 312.0, 230.0])]);
        assert_eq!(d[0].cls, "person");
        let mut last = None;
        for _ in 0..40 {
            last = Some(cam.process("gate", &d, None));
        }
        let r = last.unwrap();
        assert_eq!(r.report.workers.len(), 1);
        assert_eq!(r.ppe_fused.len(), 1);
        let json = serde_json::to_value(&r).unwrap();
        assert!(json.get("workers").is_some() && json.get("perimeter").is_some() && json.get("ppe_fused").is_some());
        // a person with no helmet or vest finds, always visible: the queue writes lines naming the camera
        let lines = std::fs::read_to_string(&file).unwrap_or_default();
        assert!(lines.lines().all(|l| l.contains("\"camera\":\"gate\"")));
        cam.reset();
        assert!(cam.process("gate", &d, None).report.workers.is_empty()); // tracks start over after a reconnect
        let _ = std::fs::remove_dir_all(&dir);
    }
}
