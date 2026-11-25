//! How well the motion gate predicts the detector, on a recorded video: the detector looks
//! at every frame (the truth: did it find anything?) and the gate says, of each frame,
//! whether it would have woken the detector. Prints the counts, MCC and recall as JSON.
//!
//!   gate_mcc MODEL VIDEO [HOLD] [THRESHOLD]
//!
//! A false negative is a frame with something on it that the gate would have slept
//! through. In the runner that costs less than it seems, since a camera that follows
//! something is looked at whether it moves or not; but an object that arrives without
//! the gate seeing it move is missed until it does. Tune `hold` and `threshold` on your
//! own night and day footage until recall is where it must be, then read MCC and the
//! share of looks saved.
use std::sync::Arc;
use std::time::Duration;

use tamserve_boxes::camera::{self, Bell, Opening, Source, State};
use tamserve_boxes::config::{Camera, DecodeSection};
use tamserve_boxes::motion::{Gate, MotionOptions};
use tamserve_boxes::{Detect, Detector, DetectorOptions, GrayFrame};

fn mcc(tp: f64, fp: f64, tn: f64, fn_: f64) -> f64 {
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if d == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / d }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (model, video) = (a.get(1).ok_or("gate_mcc MODEL VIDEO [HOLD] [THRESHOLD]")?, a.get(2).ok_or("a VIDEO is needed")?);
    let hold: u32 = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(10);
    let opts = MotionOptions { threshold: a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(30), ..Default::default() };
    let mut det = Detector::new(model, DetectorOptions { threads: Some(2), ..Default::default() })?;
    camera::init(Default::default())?;
    let cam = Camera { name: "eval".into(), url: video.clone(), onvif: None, user: None, password: None, look_every: None, weight: 1.0,
                       record: false, live: false, encode: false, motion_mask: Vec::new(), safety: None };
    let bell = Arc::new(Bell::default());
    // At the file's own frame rate, as a camera sends it; frames that arrive while the
    // detector is busy are replaced, as in the runner.
    let how = Opening::from(&DecodeSection { realtime: true, ..Default::default() });
    let src = Source::open(cam, &how, bell.clone())?;
    let mut gate = Gate::new(opts, hold);
    let (mut tp, mut fp, mut tn, mut fn_) = (0f64, 0f64, 0f64, 0f64);
    let mut seen = 0;
    loop {
        seen = bell.wait(seen, Duration::from_millis(500));
        let Some(f) = src.take() else {
            if src.health.state() == State::Ended {
                break;
            }
            continue;
        };
        let g = GrayFrame::new(&f.pixels, f.width, f.height)?;
        let awake = !gate.next(&g);
        let found = !det.detect(&[g])?[0].is_empty();
        match (awake, found) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, false) => tn += 1.0,
            (false, true) => fn_ += 1.0,
        }
    }
    let n = tp + fp + tn + fn_;
    println!("{}", serde_json::json!({
        "frames": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn_,
        "mcc": mcc(tp, fp, tn, fn_),
        "recall": if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 1.0 },
        "looks_saved": if n > 0.0 { (tn + fn_) / n } else { 0.0 },
        "note": "frames replaced while the detector was busy are not counted",
    }));
    Ok(())
}
