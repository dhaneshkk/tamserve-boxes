//! The safety rules against the Python reference, frame by frame.
//!
//! `tests/data/vg360_grammar_golden.json` comes from the `vg360_grammar` notebook
//! (`export_rust_golden`): the site, the lens and floor calibration, the helmet and vest
//! models, the rule clauses, and two simulated episodes with every finds list, every drift
//! check, and what the Python engine decided on every frame. One episode has a barricade
//! breach and a camera knocked 3 degrees part way (drift-corrected from then on); in the
//! other the lens is covered part way (the floor rules go UNKNOWN).

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;
use tamserve_boxes::datalog::{valid_path, Clause};
use tamserve_boxes::grammar::{Backend, Detection, Grid, Ppe, PpeHead, SafetyEngine, Site, Tri};
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
struct ExpWorker {
    track: u32,
    cell: [i64; 2],
    xy: [f64; 2],
    lof: Value,
    red: Value,
    helmet: Ppe,
    vest: Ppe,
}

#[derive(Deserialize)]
struct Expected {
    calib: String,
    breach: Option<bool>,
    raw_open: Option<bool>,
    posts: usize,
    gap: Option<[f64; 2]>,
    escape_path: Option<Vec<[i64; 2]>>,
    workers: Vec<ExpWorker>,
}

#[derive(Deserialize)]
struct Frame {
    dets: Vec<Detection>,
    drift: Option<DriftEstimate>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Episode {
    name: String,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Golden {
    site: Site,
    static_grid: Vec<Vec<u8>>,
    static_barrier: Vec<Vec<u8>>,
    calib: CalibJson,
    ppe: HashMap<String, PpeHead>,
    clauses: Vec<Clause>,
    episodes: Vec<Episode>,
}

fn golden() -> Golden {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vg360_grammar_golden.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("golden file")).expect("golden json")
}

fn tri(v: &Value) -> Tri {
    match v {
        Value::Bool(true) => Tri::Yes,
        Value::Bool(false) => Tri::No,
        _ => Tri::Unknown,
    }
}

fn engine(g: &Golden, backend: Backend) -> SafetyEngine {
    let lens = Lens { k: g.calib.k, d: g.calib.d };
    assert!(lens.monotone(640, 480));
    let calib = Calibration::new(lens, g.calib.h, g.site.drift_ok_px);
    let ppe = Some([g.ppe["helmet"].clone(), g.ppe["vest"].clone()]);
    SafetyEngine::new(Grid::new(g.site.clone()), calib, ppe, backend)
}

#[derive(Default, Debug)]
struct Diff {
    frames: usize,
    workers: usize,
    calib: usize,
    breach: usize,
    raw_open: usize,
    posts: usize,
    gap: usize,
    path: usize,
    path_invalid: usize,
    ids: usize,
    xy_max: f64,
    lof: usize,
    red: usize,
    ppe: usize,
}

fn replay(g: &Golden, ep: &Episode, backend: Backend) -> Diff {
    let mut e = engine(g, backend);
    let mut d = Diff::default();
    for fr in &ep.frames {
        let r = e.process(&fr.dets, fr.drift.as_ref());
        let x = &fr.expected;
        d.frames += 1;
        d.calib += (serde_json::to_value(r.calibration).unwrap() != Value::String(x.calib.clone())) as usize;
        d.breach += (r.perimeter.breach != x.breach) as usize;
        d.raw_open += (r.perimeter.raw_open != x.raw_open) as usize;
        d.posts += (r.perimeter.posts_persistent != x.posts) as usize;
        d.gap += match (r.perimeter.gap_floor_xy, x.gap) {
            (None, None) => 0,
            (Some(a), Some(b)) => ((a[0] - b[0]).abs() > 1e-6 || (a[1] - b[1]).abs() > 1e-6) as usize,
            _ => 1,
        };
        if let Some(p) = &r.perimeter.escape_path {
            let bar = e.barrier_mask_now();
            d.path_invalid += !valid_path(&e.grid, p, &bar) as usize;
        }
        if let (Some(xp), Some(rp)) = (&x.escape_path, &r.perimeter.escape_path) {
            d.path += (xp.iter().map(|c| (c[0], c[1])).collect::<Vec<_>>() != *rp) as usize;
        }
        let ids: Vec<u32> = r.workers.iter().map(|w| w.track).collect();
        let xids: Vec<u32> = x.workers.iter().map(|w| w.track).collect();
        if ids != xids {
            d.ids += 1;
            continue;
        }
        for (w, xw) in r.workers.iter().zip(&x.workers) {
            d.workers += 1;
            d.xy_max = d.xy_max.max((w.floor_xy[0] - xw.xy[0]).abs()).max((w.floor_xy[1] - xw.xy[1]).abs());
            d.ids += ((w.cell.0, w.cell.1) != (xw.cell[0], xw.cell[1])) as usize;
            d.lof += (w.line_of_fire != tri(&xw.lof)) as usize;
            d.red += (w.red_zone != tri(&xw.red)) as usize;
            d.ppe += (w.helmet != xw.helmet) as usize + (w.vest != xw.vest) as usize;
        }
    }
    d
}

#[test]
fn the_grid_is_the_python_grid() {
    let g = golden();
    let grid = Grid::new(g.site.clone());
    assert_eq!(grid.tokens, g.static_grid);
    assert_eq!(grid.barrier, g.static_barrier.iter().map(|r| r.iter().map(|&v| v != 0).collect::<Vec<_>>()).collect::<Vec<_>>());
}

#[test]
fn undistortion_matches_the_reference() {
    // The Python reference's Newton inversion on the golden lens. On the floor it agrees with OpenCV 5's
    // undistortPoints(criteria=COUNT|EPS) to 2e-6 px; at the picture's corners OpenCV 4's fixed-point
    // iteration does not converge, which is why both sides invert the model with Newton.
    let g = golden();
    let lens = Lens { k: g.calib.k, d: g.calib.d };
    for ((u, v), (x, y)) in [((320.0, 240.0), (320.0, 240.0)), ((3.0, 4.0), (-5.128257430836413, -2.051320989518615)), ((636.0, 477.0), (644.2120069060795, 483.1590051795596)), ((100.0, 420.0), (46.60105555573108, 463.6900454544018)), ((610.0, 25.0), (635.2513867489433, 6.279144306817869)), ((250.0, 300.0), (246.0019640801382, 303.4268879313101))] {
        let [a, b] = lens.undistort(u, v);
        assert!((a - x).abs() < 1e-9 && (b - y).abs() < 1e-9, "({u},{v}) -> ({a},{b}), reference ({x},{y})");
        let [u2, v2] = lens.distort(a, b);
        assert!((u2 - u).abs() < 1e-8 && (v2 - v).abs() < 1e-8, "round trip ({u},{v}) -> ({u2},{v2})");
    }
}

#[test]
fn replay_matches_python_frame_by_frame() {
    let g = golden();
    for ep in &g.episodes {
        let clauses = g.clauses.clone();
        for (name, backend) in [("datalog", Backend::Datalog(clauses)), ("native", Backend::Native)] {
            let d = replay(&g, ep, backend);
            eprintln!("{} [{name}]: {d:?}", ep.name);
            assert!(d.frames == ep.frames.len() && d.workers > 0);
            assert_eq!((d.calib, d.breach, d.raw_open, d.posts, d.gap, d.ids), (0, 0, 0, 0, 0, 0), "{} [{name}]", ep.name);
            assert_eq!((d.lof, d.red, d.ppe), (0, 0, 0), "{} [{name}]", ep.name);
            assert!(d.xy_max < 1e-6, "{} [{name}]: floor positions differ by {}", ep.name, d.xy_max);
            assert_eq!((d.path, d.path_invalid), (0, 0), "{} [{name}]", ep.name);
        }
    }
}
