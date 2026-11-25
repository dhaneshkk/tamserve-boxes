//! cargo run --release --example bench
#[path = "../tests/common/mod.rs"] mod common;
use common::*;
use vg360_scenegraph::*;

fn main() {
    let mut e = iogp_runtime::Engine::from_config(&fixture_config()).unwrap();
    let p = projection();
    for n in [5u64, 20, 50] {
        let mut sg = builder("[]", Params::default(), &e);
        let mut dets: Vec<Det> = (0..n).map(|i| {
            let (x, y) = (-8.0 + (i % 8) as f64 * 2.0, 8.0 + (i / 8) as f64 * 1.2);
            Det { track_id: i, class: if i % 5 == 0 { "vehicle" } else { "person" }, bbox: bbox(&p, x, y, 0.0, 0.5, 1.75), score: 0.9 }
        }).collect();
        dets.push(Det { track_id: 999, class: "suspended_load", bbox: bbox(&p, 0.0, 11.0, 5.0, 1.2, 0.8), score: 0.9 });
        let frames = 20_000;
        let mut sink = NullSink::default();
        let t0 = std::time::Instant::now();
        for k in 0..frames { sg.step(k as f64 * 0.1, &dets, true, &mut sink); }
        let sg_us = t0.elapsed().as_secs_f64() * 1e6 / frames as f64;
        let t0 = std::time::Instant::now();
        for k in 0..frames { sg.step(k as f64 * 0.1, &dets, true, &mut e); let _ = e.end(); }
        let both_us = t0.elapsed().as_secs_f64() * 1e6 / frames as f64;
        println!("{:>3} detections: scene graph {:6.1} µs/frame, + iogp-runtime {:6.1} µs/frame, {} facts", n + 1, sg_us, both_us, sink.facts);
    }
}
