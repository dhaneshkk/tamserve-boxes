//! Replay a golden episode file through the safety rules and time them.
//!
//! ```text
//! cargo run --release --example grammar_replay -- tests/data/vg360_grammar_golden.json
//! ```
//!
//! Prints per-frame cost for the native rules and the Datalog rules (with proofs), and the
//! first barricade-breach proof it meets.

use std::time::Instant;

use serde::Deserialize;
use tamserve_boxes::datalog::Clause;
use tamserve_boxes::grammar::{Backend, Detection, Grid, PpeHead, SafetyEngine, Site};
use tamserve_boxes::lens::{Calibration, DriftEstimate, Lens, Mat3};

#[derive(Deserialize)]
struct CalibJson {
    #[serde(rename = "K")]
    k: Mat3,
    #[serde(rename = "D")]
    d: [f64; 5],
    #[serde(rename = "H")]
    h: Mat3,
}

#[derive(Deserialize)]
struct Frame {
    dets: Vec<Detection>,
    drift: Option<DriftEstimate>,
}

#[derive(Deserialize)]
struct Episode {
    name: String,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Golden {
    site: Site,
    calib: CalibJson,
    ppe: std::collections::HashMap<String, PpeHead>,
    clauses: Vec<Clause>,
    episodes: Vec<Episode>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "tests/data/vg360_grammar_golden.json".into());
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut shown = false;
    for ep in &g.episodes {
        for (name, backend) in [("native", Backend::Native), ("datalog", Backend::Datalog(g.clauses.clone()))] {
            let calib = Calibration::new(Lens { k: g.calib.k, d: g.calib.d }, g.calib.h, g.site.drift_ok_px);
            let ppe = Some([g.ppe["helmet"].clone(), g.ppe["vest"].clone()]);
            let mut e = SafetyEngine::new(Grid::new(g.site.clone()), calib, ppe, backend);
            let mut times = Vec::with_capacity(ep.frames.len());
            for fr in &ep.frames {
                let t = Instant::now();
                let r = e.process(&fr.dets, fr.drift.as_ref());
                times.push(t.elapsed().as_secs_f64() * 1e3);
                if let (false, Some(true), Some(path)) = (shown, r.perimeter.breach, &r.perimeter.escape_path) {
                    println!("first breach proof (frame {}): escape path {path:?}, gap at {:?}", r.frame, r.perimeter.gap_floor_xy);
                    shown = true;
                }
            }
            times.sort_by(f64::total_cmp);
            println!("{:32} {:8} median {:.3} ms/frame, p95 {:.3} ms", ep.name, name, times[times.len() / 2],
                     times[times.len() * 95 / 100]);
        }
    }
    Ok(())
}
