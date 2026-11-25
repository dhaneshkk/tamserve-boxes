//! IOGP rules on the tracker's objects: scene graph + iogp-runtime, with the safety calibration.
use std::path::{Path, PathBuf};

use tamserve_boxes::iogp::{IogpCam, IogpSpec};
use tamserve_boxes::lens::{CalibState, Lens, Mat3, IDENTITY};
use tamserve_boxes::safety::CalibSpec;

fn golden() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vg360_grammar_golden.json")).unwrap()).unwrap()
}
fn calib() -> CalibSpec { serde_json::from_value(golden()["calib"].clone()).unwrap() }

fn inv3(m: &Mat3) -> Mat3 {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |a: usize, b: usize, e: usize, f: usize| m[a][b] * m[e][f] - m[a][f] * m[e][b];
    [[c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
     [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
     [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d]]
}

/// The raw (distorted) pixel where floor point (x, y) is seen.
fn pixel(c: &CalibSpec, x: f64, y: f64) -> [f64; 2] {
    let hi = inv3(&c.h);
    let q = [0, 1, 2].map(|r| hi[r][0] * x + hi[r][1] * y + hi[r][2]);
    Lens { k: c.k, d: c.d }.distort(q[0] / q[2], q[1] / q[2])
}

fn person_at(c: &CalibSpec, x: f64, y: f64) -> [f32; 4] {
    let [u, v] = pixel(c, x, y);
    [(u - 12.0) as f32, (v - 60.0) as f32, (u + 12.0) as f32, v as f32]
}

fn setup(tag: &str) -> (PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("tsb-iogp-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/vg360-scenegraph/tests/fixtures/config.json"), d.join("rules.json")).unwrap();
    // one deck, no homography of its own: it takes the safety calibration's H and lens
    std::fs::write(d.join("site.json"), r#"{"cameras": [{"name": "gate", "image_size": [640, 480],
        "decks": [{"id": "floor", "image_polygon": [[0,0],[640,0],[640,480],[0,480]]}],
        "zones": [{"id": "Guard-1", "type": "machine-guard-zone", "deck": "floor", "floor_polygon": [[8,8],[12,8],[12,12],[8,12]]}]}]}"#).unwrap();
    let spec = d.join("gate.iogp.json");
    std::fs::write(&spec, r#"{"rules": "rules.json", "site": "site.json", "labels": {"worker": "person"}}"#).unwrap();
    (d, spec)
}

fn run(spec: &Path, state: CalibState, g: Mat3, label: &str) -> (Vec<String>, bool) {
    let c = calib();
    let s = IogpSpec::load(spec, "gate", Some(&c)).unwrap();
    let mut cam = IogpCam::new(&s, "gate", Some(&c)).unwrap();
    let b = person_at(&c, 10.0, 10.0);
    assert!(b[0] > 0.0 && b[2] < 640.0 && b[1] > 0.0 && b[3] < 476.0, "{b:?}");
    let (mut raised, mut trusted) = (vec![], true);
    for i in 0..62 {
        let r = cam.process(i as f64 * 0.1, [(1u32, label, 0.9f32, b)], Some((state, g)));
        raised.extend(r.events.iter().filter(|e| e.kind == "raised").map(|e| format!("{} {} {}", e.rule, e.lsr, e.subject)));
        trusted = r.positions_trusted;
        if i == 61 && state != CalibState::Unknown {
            assert!(r.active.iter().any(|a| a.rule == "LSR1" && a.lsr == 1 && a.subject == "gate/1"), "{:?}", r.active);
        }
    }
    (raised, trusted)
}

#[test]
fn machine_guard_alert_through_the_safety_calibration() {
    let (d, spec) = setup("ok");
    let (raised, trusted) = run(&spec, CalibState::Ok, IDENTITY, "person");
    assert!(trusted);
    assert!(raised.iter().any(|r| r == "LSR1 1 gate/1"), "{raised:?}");
    // a fine-tuned model's label goes through `labels`
    let (raised, _) = run(&spec, CalibState::Ok, IDENTITY, "worker");
    assert!(raised.iter().any(|r| r.starts_with("LSR1")), "{raised:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn unknown_calibration_decides_nothing_about_the_floor() {
    let (d, spec) = setup("unknown");
    let (raised, trusted) = run(&spec, CalibState::Unknown, IDENTITY, "person");
    assert!(!trusted);
    assert!(!raised.iter().any(|r| r.starts_with("LSR1")), "{raised:?}");
    let _ = std::fs::remove_dir_all(&d);
}

/// A measured move: the camera turned, so the worker is seen `dx` px to the right of where the
/// calibration would put them. Without G they fall outside the zone; with G they are back in it.
#[test]
fn drift_correction_reaches_the_rules() {
    let (d, spec) = setup("drift");
    let c = calib();
    let s = IogpSpec::load(&spec, "gate", Some(&c)).unwrap();
    let lens = Lens { k: c.k, d: c.d };
    let b = person_at(&c, 10.0, 10.0);
    let foot_und = lens.undistort(((b[0] + b[2]) / 2.0) as f64, b[3] as f64);
    let dx = 90.0;
    let raw = lens.distort(foot_und[0] + dx, foot_und[1]);
    let moved = [(raw[0] - 12.0) as f32, (raw[1] - 60.0) as f32, (raw[0] + 12.0) as f32, raw[1] as f32];
    assert!(moved[0] > 0.0 && moved[2] < 640.0, "{moved:?}");
    // G maps the current undistorted view to the calibrated one
    let g: Mat3 = [[1.0, 0.0, -dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let lsr1 = |state: CalibState, g: Mat3| -> bool {
        let mut cam = IogpCam::new(&s, "gate", Some(&c)).unwrap();
        let mut hit = false;
        for i in 0..62 {
            let r = cam.process(i as f64 * 0.1, [(1u32, "person", 0.9f32, moved)], Some((state, g)));
            hit |= r.events.iter().any(|e| e.kind == "raised" && e.rule == "LSR1");
        }
        hit
    };
    assert!(!lsr1(CalibState::Ok, IDENTITY), "uncorrected, the moved view should put the worker outside the zone");
    assert!(lsr1(CalibState::Corrected, g), "corrected, the worker is back in the zone");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_bad_spec_is_refused_with_its_reason() {
    let (d, spec) = setup("bad");
    std::fs::write(&spec, r#"{"rules": "missing.json", "site": "site.json"}"#).unwrap();
    let e = IogpSpec::load(&spec, "gate", Some(&calib())).unwrap_err().to_string();
    assert!(e.contains("rules"), "{e}");
    std::fs::write(&spec, r#"{"rules": "rules.json", "site": "site.json", "site_camera": "nope"}"#).unwrap();
    let e = IogpSpec::load(&spec, "gate", Some(&calib())).unwrap_err().to_string();
    assert!(e.contains("nope"), "{e}");
    // no safety calibration and no homography of its own: the deck cannot be placed
    std::fs::write(&spec, r#"{"rules": "rules.json", "site": "site.json"}"#).unwrap();
    let e = IogpSpec::load(&spec, "gate", None).unwrap_err().to_string();
    assert!(e.contains("homography"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}
