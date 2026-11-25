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
//!   "perimeter": true,                                // false: no barricade in view
//!   "ppe_look": {"model": "ppe_gray.onnx", "most": 4},  // main streams only
//!   "drift": {"every_frames": 10, "reference_dir": "refs"} }
//! ```
//!
//! Leave out `ppe` until the detector knows `helmet` and `vest`: with a stock COCO model
//! nobody ever has a helmet box, so everyone in plain view would be a violation.
//!
//! With `"drift": {"reference_dir": "refs"}` the box checks every `every_frames` frames
//! whether the camera has moved ([`crate::drift`]): against a reference per light mode,
//! taken with `POST /safety/<camera>/reference` (or `auto_reference`) while the view is the
//! calibrated one. A checker elsewhere can still post to `POST /safety/<camera>/drift`; a
//! posted result wins over the box's own for that frame.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::datalog::Clause;
use crate::drift::{DriftChecker, DriftParams, Reference};
use crate::grammar::{Backend, Detection, FusedPpe, Grid, LabelCandidate, LabelQueue, PpeCrop, PpeFusion, PpeHead, Report, SafetyEngine, Site};
use crate::lens::{CalibState, Calibration, DriftEstimate, Lens, Mat3};
use crate::{Detect, Detector, DetectorOptions, Error, GrayFrame, Result};

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

/// A second look for helmets and vests: each step, the head-and-torso crops of up to `most`
/// people, cut from the camera's own frame at full resolution, through a PPE model.
///
/// It pays on a main stream. There the main detector sees the whole frame shrunk to 416 px,
/// so a far worker's head is a few pixels, while the crop goes in at full size. In
/// simulation, helmet MCC was 0.58 at substream resolution and 0.95 on 4x crops. On a
/// 640 px substream a crop has no more pixels than the frame: leave it out.
#[derive(Clone, Debug, Deserialize)]
pub struct PpeLookSpec {
    /// The PPE model (an ONNX with `helmet` and `vest` among its classes, folded to one
    /// channel), relative to the spec's folder.
    pub model: PathBuf,
    /// Its class names in order (a fine-tuned model's `.names.json`); default COCO's.
    #[serde(default)]
    pub names: Option<Vec<String>>,
    /// The most people looked at per step; the rest wait their turn.
    #[serde(default = "four")]
    pub most: usize,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub threshold: Option<f32>,
    /// The model's input side.
    #[serde(default = "size416")]
    pub size: usize,
}

fn four() -> usize {
    4
}
fn size416() -> usize {
    416
}

/// The box's own drift check.
#[derive(Clone, Debug, Deserialize)]
pub struct DriftSpec {
    /// Check every this many frames.
    #[serde(default = "ten_u64")]
    pub every_frames: u64,
    /// Where references are kept (`<light mode>.pgm` and `.json`), relative to the spec's
    /// folder. Without it, references live only until the box restarts.
    #[serde(default)]
    pub reference_dir: Option<PathBuf>,
    /// Take a light mode's reference by itself the first time the mode is seen while the
    /// calibration is trusted. Only safe when the camera is known to be where it was
    /// calibrated when the box starts.
    #[serde(default)]
    pub auto_reference: bool,
    #[serde(flatten)]
    pub params: DriftParams,
}

fn ten_u64() -> u64 {
    10
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
    /// A second look for helmets and vests on full-resolution crops (see [`PpeLookSpec`]).
    #[serde(default)]
    pub ppe_look: Option<PpeLookSpec>,
    /// The box's own drift check (see [`DriftSpec`]).
    #[serde(default)]
    pub drift: Option<DriftSpec>,
}

fn yes() -> bool {
    true
}

impl SafetySpec {
    pub fn load(path: &Path) -> Result<SafetySpec> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Invalid(format!("cannot read the safety spec {}: {e}", path.display())))?;
        let mut spec: SafetySpec = serde_json::from_str(&text).map_err(|e| Error::Invalid(format!("the safety spec {}: {e}", path.display())))?;
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        if let Some(look) = &mut spec.ppe_look {
            if look.model.is_relative() {
                look.model = base.join(&look.model);
            }
        }
        if let Some(d) = &mut spec.drift {
            if let Some(r) = &mut d.reference_dir {
                if r.is_relative() {
                    *r = base.join(&*r);
                }
            }
        }
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
        if let Some(look) = &self.ppe_look {
            if self.ppe.is_none() {
                return Err("ppe_look needs the \"ppe\" heads that judge its finds".into());
            }
            if !look.model.is_file() {
                return Err(format!("ppe_look: no model at {}", look.model.display()));
            }
            if look.most == 0 {
                return Err("ppe_look: most must be at least 1".into());
            }
        }
        if let Some(d) = &self.drift {
            let p = &d.params;
            if d.every_frames == 0 || !(0.0..=1.0).contains(&p.rho_min) || !(0.0..=1.0).contains(&p.contrast_min)
                || !(0.0..=1.0).contains(&p.overlap_min) || p.work_width < 64 {
                return Err("drift: every_frames >= 1, rho_min, contrast_min and overlap_min in 0..1, work_width >= 64".into());
            }
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
    /// People looked at again for PPE on this frame, and since the start.
    pub ppe_looks: usize,
    pub ppe_looks_total: u64,
    /// The box's own drift check: the last one's result (kept until the next), and the light
    /// modes that have a reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift_last: Option<DriftEstimate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub drift_references: Vec<String>,
}

/// The PPE model and whose turn it is.
struct Looker {
    det: Detector,
    most: usize,
    last: HashMap<u32, u64>,
    step: u64,
}

/// One camera's rules, fusion, label queue and PPE look.
pub struct SafetyCam {
    spec: SafetySpec,
    engine: SafetyEngine,
    fusion: PpeFusion,
    queue: Option<LabelQueue>,
    looker: Option<Looker>,
    looks_now: usize,
    drift: Option<(DriftSpec, DriftChecker, HashMap<String, Reference>)>,
    drift_last: Option<DriftEstimate>,
    want_reference: bool,
    frames: u64,
    pub queued_total: u64,
    /// People looked at again for PPE, since the start.
    pub ppe_looks_total: u64,
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
    /// Loads the PPE model when the spec has a `ppe_look`.
    pub fn new(spec: SafetySpec) -> Result<Self> {
        let queue = spec.label_queue.as_ref().map(|q| LabelQueue::new(q.gap_frames, q.margin, q.per_hour, q.fps));
        let looker = match &spec.ppe_look {
            None => None,
            Some(l) => {
                let mut o = DetectorOptions { size: l.size, threads: l.threads, ..DetectorOptions::default() };
                if let Some(n) = &l.names {
                    o.names = n.clone();
                }
                if let Some(t) = l.threshold {
                    o.threshold = t;
                }
                Some(Looker { det: Detector::new(&l.model, o)?, most: l.most, last: HashMap::new(), step: 0 })
            }
        };
        let drift = match &spec.drift {
            None => None,
            Some(d) => {
                let mut refs = HashMap::new();
                if let Some(dir) = &d.reference_dir {
                    for mode in ["unknown", "day", "night"] {
                        if let Some(r) = DriftChecker::load(dir, mode)? {
                            refs.insert(mode.to_string(), r);
                        }
                    }
                }
                Some((d.clone(), DriftChecker::new(Lens { k: spec.calib.k, d: spec.calib.d }, d.params.clone()), refs))
            }
        };
        Ok(SafetyCam { engine: engine(&spec), fusion: PpeFusion::new(true, false, 0.8), queue, looker, looks_now: 0, drift, drift_last: None,
                       want_reference: false, frames: 0, queued_total: 0, ppe_looks_total: 0, spec })
    }

    /// Start over: a new connection's frames are not the old one's future. The PPE model
    /// and the counters stay.
    pub fn reset(&mut self) {
        // A reconnect does not move the camera: keep what the drift check last found.
        let calib = self.engine.calib.clone();
        self.engine = engine(&self.spec);
        self.engine.calib = calib;
        self.fusion = PpeFusion::new(true, false, 0.8);
        if let Some(l) = &mut self.looker {
            l.last.clear();
        }
    }

    /// The floor calibration as the last frame left it: lens, H, drift correction G and state.
    pub fn calibration(&self) -> &Calibration {
        &self.engine.calib
    }

    /// The spec's calibration, as loaded.
    pub fn calib_spec(&self) -> &CalibSpec {
        &self.spec.calib
    }

    /// Take the next frame as the reference for its light mode (the view must be the
    /// calibrated one, or one whose correction the calibration already holds).
    pub fn request_reference(&mut self) -> bool {
        self.want_reference = self.drift.is_some();
        self.want_reference
    }

    /// The box's own drift check on this frame, when one is due: `mode` is the light mode,
    /// `boxes` what is tracked (left out of the fit). Takes a reference first when asked to,
    /// or by itself with `auto_reference`. Returns the check's result to pass to
    /// [`SafetyCam::process`].
    pub fn drift_step(&mut self, pixels: &[u8], width: usize, height: usize, mode: &str, boxes: &[[f64; 4]]) -> Option<DriftEstimate> {
        let trusted = self.engine.calib.state != CalibState::Unknown;
        let g_now = self.engine.calib.g;
        let (spec, chk, refs) = self.drift.as_mut()?;
        self.frames += 1;
        let auto = spec.auto_reference && trusted && !refs.contains_key(mode);
        if std::mem::take(&mut self.want_reference) || auto {
            let r = chk.reference(pixels, width, height, boxes, g_now);
            if let Some(dir) = &spec.reference_dir {
                if let Err(e) = DriftChecker::save(&r, dir, mode) {
                    eprintln!("tamserve-boxes: cannot save the drift reference in {}: {e}", dir.display());
                }
            }
            refs.insert(mode.to_string(), r);
            return None;
        }
        if !self.frames.is_multiple_of(spec.every_frames) {
            return None;
        }
        let e = chk.check(refs.get(mode)?, pixels, width, height, boxes);
        self.drift_last = Some(e.clone());
        Some(e)
    }

    /// Look again for helmets and vests on full-resolution crops of up to `most` people.
    /// `ids[i]` is the tracker's id of `dets[i]`. People never looked at go first, then
    /// whoever was looked at longest ago, smaller ones first. For each person looked at,
    /// the PPE model's finds replace the main detector's helmet and vest finds there;
    /// everyone else keeps the main detector's. Returns how many were looked at.
    pub fn look_ppe(&mut self, pixels: &[u8], width: usize, height: usize, ids: &[u32], dets: &mut Vec<Detection>) -> Result<usize> {
        self.looks_now = 0;
        let Some(l) = &mut self.looker else { return Ok(0) };
        l.step += 1;
        let mut who: Vec<usize> = (0..dets.len()).filter(|&i| dets[i].cls == "person" && i < ids.len()).collect();
        who.sort_by(|&a, &b| {
            let (la, lb) = (l.last.get(&ids[a]).copied().unwrap_or(0), l.last.get(&ids[b]).copied().unwrap_or(0));
            let (ha, hb) = (dets[a].bbox[3] - dets[a].bbox[1], dets[b].bbox[3] - dets[b].bbox[1]);
            la.cmp(&lb).then(ha.total_cmp(&hb))
        });
        let mut crops: Vec<PpeCrop> = Vec::new();
        let mut frames: Vec<GrayFrame> = Vec::new();
        for &i in &who {
            if crops.len() == l.most {
                break;
            }
            let c = PpeCrop::for_person(&dets[i].bbox, 1.0, width as f64, height as f64);
            let (x, y, w, h) = c.pixels();
            let (x, y) = (x as usize, y as usize);
            let (w, h) = ((w as usize).min(width.saturating_sub(x)), (h as usize).min(height.saturating_sub(y)));
            if w < 4 || h < 4 || pixels.len() < (y + h - 1) * width + x + w {
                continue;
            }
            frames.push(GrayFrame { pixels: &pixels[y * width + x..], width: w, height: h, stride: width });
            crops.push(c);
            l.last.insert(ids[i], l.step);
        }
        if crops.is_empty() {
            return Ok(0);
        }
        let finds = l.det.detect(&frames)?;
        let inside = |d: &Detection| {
            let (cx, cy) = ((d.bbox[0] + d.bbox[2]) / 2.0, (d.bbox[1] + d.bbox[3]) / 2.0);
            crops.iter().any(|c| c.rect[0] <= cx && cx <= c.rect[2] && c.rect[1] <= cy && cy <= c.rect[3])
        };
        dets.retain(|d| !((d.cls == "helmet" || d.cls == "vest") && inside(d)));
        let names = l.det.names();
        for (c, f) in crops.iter().zip(&finds) {
            for k in 0..f.len() {
                let name = &names[f.labels[k]];
                let cls = self.spec.labels.get(name).cloned().unwrap_or_else(|| name.clone());
                if cls == "helmet" || cls == "vest" {
                    let b = f.boxes[k];
                    dets.push(c.map_back(&Detection { cls, conf: f.scores[k] as f64, bbox: [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64] }));
                }
            }
        }
        if l.last.len() > 4096 {
            let keep = l.step.saturating_sub(10_000);
            l.last.retain(|_, s| *s >= keep);
        }
        self.ppe_looks_total += crops.len() as u64;
        self.looks_now = crops.len();
        Ok(crops.len())
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
        let ppe_looks = std::mem::take(&mut self.looks_now);
        let mut drift_references: Vec<String> = self.drift.as_ref().map_or(vec![], |d| d.2.keys().cloned().collect());
        drift_references.sort();
        CameraSafety { report, ppe_fused, label_candidates, ppe_looks, ppe_looks_total: self.ppe_looks_total,
                       drift_last: self.drift_last.clone(), drift_references }
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
    fn the_box_takes_a_reference_corrects_a_knock_and_goes_unknown_when_blind() {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/drift");
        let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("drift_cases.json")).unwrap()).unwrap();
        let frame = |n: &str| crate::drift::read_pgm(&std::fs::read(d.join(n)).unwrap()).unwrap();
        let mut s = spec();
        // the drift frames are 320 x 240: their lens and floor homography
        s.calib.k = serde_json::from_value(cases["lens"]["K"].clone()).unwrap();
        s.calib.d = serde_json::from_value(cases["lens"]["D"].clone()).unwrap();
        s.calib.h = serde_json::from_value(cases["H"].clone()).unwrap();
        let refs = std::env::temp_dir().join(format!("tamserve-refs-{}", std::process::id()));
        s.drift = Some(DriftSpec { every_frames: 1, reference_dir: Some(refs.clone()), auto_reference: false, params: DriftParams::default() });
        let mut cam = SafetyCam::new(s.clone()).unwrap();
        let (w, h, px) = frame("drift_ref.pgm");
        assert!(cam.drift_step(&px, w, h, "day", &[]).is_none(), "no reference yet: nothing to check against");
        assert!(cam.request_reference());
        assert!(cam.drift_step(&px, w, h, "day", &[]).is_none(), "the reference is taken, not checked");
        assert!(refs.join("day.pgm").is_file() && refs.join("day.json").is_file());
        // a 3 degree knock: measured and corrected
        let (w, h, px) = frame("drift_tilt_3deg.pgm");
        let e = cam.drift_step(&px, w, h, "day", &[]).expect("a check every frame");
        assert!(e.valid && e.drift_px.unwrap() > 3.0);
        let r = cam.process("gate", &[], Some(&e));
        assert_eq!(r.report.calibration, CalibState::Corrected);
        assert_eq!(r.drift_references, vec!["day".to_string()]);
        // at night there is no reference: no check
        assert!(cam.drift_step(&px, w, h, "night", &[]).is_none());
        // the lens covered: not confirmed, every floor rule unknown
        let (w, h, px) = frame("drift_lens_covered.pgm");
        let e = cam.drift_step(&px, w, h, "day", &[]).unwrap();
        assert!(!e.valid);
        let r = cam.process("gate", &[], Some(&e));
        assert_eq!(r.report.calibration, CalibState::Unknown);
        assert_eq!(r.report.system_alerts, vec!["calibration-lost"]);
        // a box restarted finds its reference again
        let mut again = SafetyCam::new(s).unwrap();
        let (w, h, px) = frame("drift_pan_3deg.pgm");
        assert!(again.drift_step(&px, w, h, "day", &[]).unwrap().valid);
        let _ = std::fs::remove_dir_all(&refs);
    }

    #[test]
    fn no_barricade_means_the_perimeter_is_unknown_not_open() {
        let mut s = spec();
        s.perimeter = false;
        let mut cam = SafetyCam::new(s).unwrap();
        let r = (0..40).map(|_| cam.process("yard", &[], None)).last().unwrap();
        assert_eq!((r.report.perimeter.breach, r.report.perimeter.raw_open), (None, None));
        let mut cam = SafetyCam::new(spec()).unwrap(); // with the default, no posts in view reads as open
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
        let mut cam = SafetyCam::new(s).unwrap();
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
