//! The drift check against frames with a known answer.
//!
//! `tests/data/drift/` comes from the `vg360_grammar` notebook (`export_drift_frames`): a
//! reference frame of the calibrated view and frames of the same floor with the camera
//! knocked (1° to 8° of tilt and pan), re-lit (as an IR-like picture), steamed, with the lens
//! covered, and turned 30° away, at 320 x 240 with people about. For each, the true
//! correction, floor points to score one in centimetres, and what OpenCV's own
//! `findTransformECC` reaches on the same undistorted pictures.

use serde::Deserialize;
use tamserve_boxes::drift::{read_pgm, DriftChecker, DriftParams, NO_MOVE};
use tamserve_boxes::lens::{apply_h, matmul, Lens, Mat3};

#[derive(Deserialize)]
struct LensJ {
    #[serde(rename = "K")]
    k: Mat3,
    #[serde(rename = "D")]
    d: [f64; 5],
}

#[derive(Deserialize)]
struct Ecc {
    floor_err_cm: Option<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    frame: String,
    expect_valid: bool,
    floor_pts: Vec<[f64; 2]>,
    pixels: Vec<[f64; 2]>,
    boxes: Vec<[f64; 4]>,
    uncorrected_floor_err_cm: f64,
    opencv_ecc: Ecc,
}

#[derive(Deserialize)]
struct Cases {
    lens: LensJ,
    #[serde(rename = "H")]
    h: Mat3,
    reference: String,
    cases: Vec<Case>,
}

fn dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/drift")
}

fn frame(name: &str) -> (usize, usize, Vec<u8>) {
    read_pgm(&std::fs::read(dir().join(name)).unwrap()).expect("a PGM")
}

#[test]
fn knocks_are_corrected_and_blind_views_are_not_confirmed() {
    let c: Cases = serde_json::from_str(&std::fs::read_to_string(dir().join("drift_cases.json")).unwrap()).unwrap();
    let lens = Lens { k: c.lens.k, d: c.lens.d };
    let mut chk = DriftChecker::new(lens.clone(), DriftParams::default());
    let (w, h, px) = frame(&c.reference);
    let r = chk.reference(&px, w, h, &[], NO_MOVE);
    let mut worst = 0.0f64;
    for case in &c.cases {
        let (w, h, px) = frame(&case.frame);
        let t = std::time::Instant::now();
        let e = chk.check(&r, &px, w, h, &case.boxes);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let err = e.g.map(|g| {
            let hg = matmul(&c.h, &g);
            let s: f64 = case.pixels.iter().zip(&case.floor_pts).map(|(p, q)| {
                let [u, v] = lens.undistort(p[0], p[1]);
                let f = apply_h(&hg, u, v);
                ((f[0] - q[0]).powi(2) + (f[1] - q[1]).powi(2)).sqrt()
            }).sum();
            100.0 * s / case.pixels.len() as f64
        });
        eprintln!("{:22} valid {:5} (want {:5})  rho {:.3}  px {:6}  drift {:>6.1} px  floor {:>6} cm (uncorrected {:6.1}, OpenCV ECC {:?})  {ms:.1} ms",
                  case.name, e.valid, case.expect_valid, e.ratio, e.inliers, e.drift_px.unwrap_or(f64::NAN),
                  err.map_or("-".into(), |v| format!("{v:.1}")), case.uncorrected_floor_err_cm, case.opencv_ecc.floor_err_cm);
        assert_eq!(e.valid, case.expect_valid, "{}", case.name);
        if let (Some(err), Some(cv)) = (err, case.opencv_ecc.floor_err_cm) {
            assert!(err <= cv + 1.5, "{}: {err:.1} cm against OpenCV's {cv:.1}", case.name);
            worst = worst.max(err);
        }
    }
    assert!(worst > 0.0);
}

#[test]
fn a_saved_reference_loads_back_the_same() {
    let c: Cases = serde_json::from_str(&std::fs::read_to_string(dir().join("drift_cases.json")).unwrap()).unwrap();
    let lens = Lens { k: c.lens.k, d: c.lens.d };
    let mut chk = DriftChecker::new(lens, DriftParams::default());
    let (w, h, px) = frame(&c.reference);
    let r = chk.reference(&px, w, h, &[], NO_MOVE);
    let tmp = std::env::temp_dir().join(format!("tamserve-drift-{}", std::process::id()));
    DriftChecker::save(&r, &tmp, "day").unwrap();
    let back = DriftChecker::load(&tmp, "day").unwrap().expect("saved");
    assert!(DriftChecker::load(&tmp, "night").unwrap().is_none());
    let (w, h, px) = frame("drift_tilt_3deg.pgm");
    let (a, b) = (chk.check(&r, &px, w, h, &[]), chk.check(&back, &px, w, h, &[]));
    assert!(a.valid && b.valid);
    assert!((a.drift_px.unwrap() - b.drift_px.unwrap()).abs() < 0.5, "{:?} vs {:?}", a.drift_px, b.drift_px);
    let _ = std::fs::remove_dir_all(&tmp);
}
