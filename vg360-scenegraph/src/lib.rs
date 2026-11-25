//! # vg360-scenegraph
//!
//! Turns tracked detections (YOLOX-Nano + ByteTrack, from tamserve-boxes) into the objects and facts
//! that `iogp-runtime` evaluates, computed on calibrated deck planes instead of in pixels:
//!
//! * one homography per deck (image polygon → metres), derived from a 3x4 camera matrix or given;
//! * a constant-velocity Kalman filter per track in deck metres, with pixel noise pushed through the
//!   homography Jacobian, so far-field workers carry larger uncertainty;
//! * hidden feet (occluder polygon or image edge) recovered from box top + person aspect;
//! * occluded people coasted for a bounded time, keeping what they last wore and carried;
//! * suspended-load drop zones from the projection matrix over a height range;
//! * `in` / `near` / `under` decided three-valued and emitted with the polarity the rules need
//!   ([`policy`]); `on` and the associations keep iogp-editor ingest's image geometry;
//! * camera_guard's moved/blocked flag suppresses position facts and raises a calibration signal.
//!
//! ```no_run
//! # use vg360_scenegraph::*;
//! let cfg = std::fs::read_to_string("config.json").unwrap();
//! let mut engine = iogp_runtime::Engine::from_config(&cfg).unwrap();
//! let site = Site::from_json(&std::fs::read_to_string("site.json").unwrap()).unwrap();
//! let mut sg = SceneBuilder::new(site.camera("CAM-02").unwrap(), Params::default(), Policy::from_rules(&engine.rules));
//! let dets = [Det { track_id: 7, class: "person", bbox: [600.0, 300.0, 660.0, 470.0], score: 0.9 }];
//! sg.step(12.4, &dets, true, &mut engine);
//! for e in engine.end() { println!("{} {} {}", e.kind, e.alert, e.subject); }
//! ```

pub mod eval;
pub mod fit;
pub mod geom;
pub mod graph;
pub mod policy;
pub mod sink;
pub mod site;
pub mod track;

pub use graph::{Det, Node, Params, PpeBand, Role, SceneBuilder, Stats};
pub use policy::{Emit, Policy, RelNames};
pub use sink::{FactSink, NullSink, ObjKind};
pub use site::{Camera, Site};
