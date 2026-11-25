//! IOGP Life-Saving Rule alerts for one camera: the tracker's objects go through the
//! floor-plane scene graph ([`vg360_scenegraph`]) into the rules written and approved in
//! iogp-editor ([`iogp_runtime`]).
//!
//! A camera with `iogp = "path/to/cam.iogp.json"` in its config gets an [`IogpCam`]:
//!
//! ```json
//! { "rules": "config.json",          // iogp-editor's /export/config.json (schema iogp-alert-rules/v1)
//!   "site": "site.json",             // vg360-scenegraph site file: decks, zones, occluders, projection
//!   "site_camera": "CAM-02",         // the camera's name in the site file; default the config's name
//!   "labels": {"car": "vehicle", "truck": "vehicle", "helmet": "hardhat"},  // detector label -> catalog detector
//!   "params": {"coast_max_s": 2.0} } // vg360_scenegraph::Params overrides, optional
//! ```
//!
//! Paths are relative to the spec's folder. When the camera also has `safety`, its lens, floor
//! homography and drift state are used: a deck in the site file without a `homography` (and a
//! camera without a `projection`) takes the safety calibration's `H`, the site camera takes its
//! lens when it has none, the drift correction `G` is applied every frame, and while the
//! calibration is UNKNOWN no position fact is emitted (a `calibration-loss` signal on every zone instead; iogp-editor's catalog has the noun),
//! the same gate as the safety rules' `lost_never_clear`.
//!
//! The verdicts are in the camera's report (`iogp`), on `/status`, on `/v1/chat/completions`
//! and as `tamserve_iogp_*` metrics.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vg360_scenegraph::geom::Lens as SgLens;
use vg360_scenegraph::{Det, Params, Policy, SceneBuilder, Site};

use crate::lens::{CalibState, Mat3};
use crate::safety::CalibSpec;
use crate::{Error, Result};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IogpSpec {
    pub rules: PathBuf,
    pub site: PathBuf,
    #[serde(default)]
    pub site_camera: Option<String>,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    #[serde(default)]
    pub params: Option<Params>,
}

impl IogpSpec {
    /// Read the spec and check that its rules and site load for `camera`.
    pub fn load(path: &Path, camera: &str, calib: Option<&CalibSpec>) -> Result<IogpSpec> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Invalid(format!("cannot read the iogp spec {}: {e}", path.display())))?;
        let mut spec: IogpSpec = serde_json::from_str(&text).map_err(|e| Error::Invalid(format!("the iogp spec {}: {e}", path.display())))?;
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        for p in [&mut spec.rules, &mut spec.site] {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        IogpCam::new(&spec, camera, calib).map_err(|e| Error::Invalid(format!("the iogp spec {}: {e}", path.display())))?;
        Ok(spec)
    }
}

/// An alert that holds now.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ActiveAlert {
    pub rule: String,
    pub subject: String,
    pub alert: String,
    pub lsr: u8,
    pub severity: Option<String>,
}

/// What the rules decided on one frame.
#[derive(Clone, Debug, Serialize)]
pub struct IogpReport {
    /// Raised and cleared on this frame, with the proof of each raised one.
    pub events: Vec<iogp_runtime::AlertEvent>,
    pub active: Vec<ActiveAlert>,
    /// False while the floor calibration is UNKNOWN: no position facts were emitted.
    pub positions_trusted: bool,
    pub nodes: usize,
    pub coasting: usize,
    pub facts: usize,
    pub raised_total: u64,
    pub cleared_total: u64,
    /// The rule set's revision, from config.json.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Policy warnings about the rule set (shown, not fatal).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// One camera's scene graph and rule engine.
pub struct IogpCam {
    engine: iogp_runtime::Engine,
    scene: SceneBuilder,
    labels: HashMap<String, String>,
    lsr: HashMap<String, (String, u8, Option<String>)>,
    revision: Option<String>,
    warnings: Vec<String>,
    raised_total: u64,
    cleared_total: u64,
}

fn to_sg(calib: &CalibSpec) -> SgLens {
    SgLens { k: calib.k, d: calib.d }
}

impl IogpCam {
    /// `calib`: the camera's safety calibration, if it has one.
    pub fn new(spec: &IogpSpec, camera: &str, calib: Option<&CalibSpec>) -> std::result::Result<IogpCam, String> {
        let rules_text = std::fs::read_to_string(&spec.rules).map_err(|e| format!("cannot read the rules {}: {e}", spec.rules.display()))?;
        let rules = iogp_runtime::load_config(&rules_text).map_err(|e| e.to_string())?;
        let engine = iogp_runtime::Engine::from_config(&rules_text).map_err(|e| e.to_string())?;
        let site_text = std::fs::read_to_string(&spec.site).map_err(|e| format!("cannot read the site {}: {e}", spec.site.display()))?;
        let mut site = Site::from_json(&site_text).map_err(|e| e.to_string())?;
        let name = spec.site_camera.as_deref().unwrap_or(camera);
        if let (Some(c), Some(sc)) = (calib, site.cameras.iter_mut().find(|c| c.name == name)) {
            if sc.lens.is_none() {
                sc.lens = Some(to_sg(c));
            }
            if sc.projection.is_none() {
                for d in sc.decks.iter_mut().filter(|d| d.homography.is_none()) {
                    d.homography = Some(c.h);
                }
            }
        }
        let cam = site.camera(name).map_err(|e| e.to_string())?;
        let policy = Policy::from_rules(&engine.rules);
        let lsr = rules.rules.iter().map(|r| (r.id.clone(), (r.alert.noun.clone(), r.alert.lsr, r.alert.severity.clone()))).collect();
        let warnings = policy.warnings.clone();
        let scene = SceneBuilder::new(cam, spec.params.clone().unwrap_or_default(), policy);
        Ok(IogpCam { engine, scene, labels: spec.labels.clone(), lsr, revision: rules.revision, warnings, raised_total: 0, cleared_total: 0 })
    }

    /// Start over on a new connection: the tracker's ids start again, so the scene graph's tracks
    /// go. The rule engine stays, so an alert that is up is not raised a second time.
    pub fn reset(&mut self) {
        let (cam, params, policy) = (self.scene.cam.clone(), self.scene.params.clone(), self.scene.policy.clone());
        self.scene = SceneBuilder::new(cam, params, policy);
    }

    /// One frame. `t` in seconds; `objects` are (track id, label, score, box) in the frame's
    /// pixels; `calib` the safety calibration's state and drift correction, when the camera has one.
    pub fn process<'a>(&mut self, t: f64, objects: impl IntoIterator<Item = (u32, &'a str, f32, [f32; 4])>,
                       calib: Option<(CalibState, Mat3)>) -> IogpReport {
        let trusted = calib.is_none_or(|(s, _)| s != CalibState::Unknown);
        if let Some((_, g)) = calib {
            self.scene.set_view_correction(g);
        }
        let labels = &self.labels;
        let dets: Vec<Det> = objects.into_iter().map(|(id, label, score, b)| Det {
            track_id: u64::from(id),
            class: labels.get(label).map_or(label, String::as_str),
            bbox: b.map(f64::from),
            score: f64::from(score),
        }).collect();
        self.scene.step(t, &dets, trusted, &mut self.engine);
        let events = self.engine.end_events();
        for e in &events {
            match e.kind.as_str() {
                "raised" => self.raised_total += 1,
                "cleared" => self.cleared_total += 1,
                _ => {}
            }
        }
        let active = self.engine.active().into_iter().map(|(rule, subject)| {
            let (alert, lsr, severity) = self.lsr.get(&rule).cloned().unwrap_or_default();
            ActiveAlert { rule, subject, alert, lsr, severity }
        }).collect();
        let st = self.scene.stats;
        IogpReport { events, active, positions_trusted: trusted, nodes: st.nodes, coasting: st.coasting, facts: st.facts,
                     raised_total: self.raised_total, cleared_total: self.cleared_total, revision: self.revision.clone(),
                     warnings: self.warnings.clone() }
    }
}
