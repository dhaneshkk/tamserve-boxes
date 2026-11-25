//! One camera read once and shared: followed with the motion gate on, recorded to
//! segments, and watched live, from a clip made here (a ball moving, then a still frame).
//! Needs YOLOX-Nano (TAMSERVE_YOLOX_ONNX, as tests/model.rs), OpenVINO, and GStreamer
//! with x264enc (gstreamer1.0-plugins-ugly) to make the clip; skipped without them.
#![cfg(feature = "cameras")]

use std::time::{Duration, Instant};

use gstreamer::prelude::*;
use tamserve_boxes::camera::{self, Opening};
use tamserve_boxes::config::{Camera, DecodeSection, Decoder};
use tamserve_boxes::motion::MotionOptions;
use tamserve_boxes::retention::{self, Policy};
use tamserve_boxes::runner::Runner;
use tamserve_boxes::{BoxTracker, DetectorOptions, LookEvery, TrackerOptions};

fn clip(path: &std::path::Path) -> bool {
    gstreamer::init().unwrap();
    if gstreamer::ElementFactory::find("x264enc").is_none() {
        return false;
    }
    let p = gstreamer::parse::launch(&format!(
        "concat name=c ! x264enc key-int-max=25 tune=zerolatency ! h264parse ! mp4mux ! filesink location={} \
         videotestsrc pattern=ball num-buffers=100 ! video/x-raw,width=320,height=180,framerate=25/1 ! c. \
         videotestsrc pattern=black num-buffers=100 ! video/x-raw,width=320,height=180,framerate=25/1 ! c.",
        path.display())).unwrap();
    p.set_state(gstreamer::State::Playing).unwrap();
    let bus = p.bus().unwrap();
    let ok = bus.timed_pop_filtered(gstreamer::ClockTime::from_seconds(30),
        &[gstreamer::MessageType::Eos, gstreamer::MessageType::Error])
        .is_some_and(|m| m.type_() == gstreamer::MessageType::Eos);
    p.set_state(gstreamer::State::Null).unwrap();
    ok
}

#[test]
fn one_connection_is_followed_recorded_and_watched() {
    let Ok(model) = std::env::var("TAMSERVE_YOLOX_ONNX") else { return eprintln!("skipped: TAMSERVE_YOLOX_ONNX not set") };
    let dir = std::env::temp_dir().join(format!("tsb-recording-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let video = dir.join("clip.mp4");
    if !clip(&video) {
        return eprintln!("skipped: no x264enc to make the clip");
    }
    let gray = dir.join("gray.onnx");
    tamserve_boxes::fold::gray_model(&model, &gray).unwrap();

    camera::init(Decoder::Software).unwrap();
    let tracker = BoxTracker::from_model(&gray, DetectorOptions { threads: Some(1), ..Default::default() },
        TrackerOptions { look_every: LookEvery::All(1), ..Default::default() }).unwrap();
    let rec = dir.join("rec");
    let policy = Policy { all: 1, motion: 1, objects: 1, labels: Default::default(), pre: 0, post: 0, max_bytes: None };
    let mut runner = Runner::new(tracker, Opening::from(&DecodeSection { realtime: true, ..Default::default() }))
        // alpha 0.1: the ball's trace leaves the average within the clip's 4 still seconds
        // (at 0.01 it lingers longer than that: extra looks, never fewer).
        .with_motion(MotionOptions { alpha: 0.1, ..Default::default() }, true, 5).unwrap()
        .with_recording(rec.clone(), Duration::from_secs(2), policy, Duration::from_secs(3600)).unwrap();
    let cam = Camera { name: "gate".into(), url: video.display().to_string(), onvif: None, user: None, password: None,
                       look_every: None, weight: 1.0, record: true, live: true, encode: false, motion_mask: Vec::new(), safety: None, iogp: None };
    runner.set_cameras(&[cam]).unwrap();
    let (health, tap) = {
        let views = runner.shared.cameras.lock().unwrap();
        (views["gate"].health.clone(), views["gate"].live.clone().unwrap())
    };
    let viewer = tap.subscribe().unwrap();
    let until = Instant::now() + Duration::from_secs(60);
    let (mut steps, mut late) = (0, None);
    while !runner.all_ended() && Instant::now() < until {
        runner.step(Duration::from_millis(200)).unwrap();
        steps += 1;
        if steps == 60 {
            late = tap.subscribe(); // 2.4 s in: mid-picture-group
        }
    }
    assert!(runner.all_ended(), "the clip was not read to its end: {}", health.last_error());
    drop(runner); // stops the camera: its last segment is closed

    let ord = std::sync::atomic::Ordering::Relaxed;
    assert!(health.motion_steps.load(ord) > 50, "the ball moves for 4 s");
    assert!(health.looks_skipped.load(ord) > 20, "the still frame costs no looks once the gate is quiet");

    let segs = &retention::scan(&rec).unwrap()["gate"];
    assert!(segs.len() >= 3, "8 s in 2 s segments: {segs:?}");
    for s in segs {
        let b = std::fs::read(&s.path).unwrap();
        assert!(!b.is_empty() && b[0] == 0x47 && b.len().is_multiple_of(188), "{:?} is not MPEG-TS", s.path);
    }
    let (activity, covered) = retention::read_activity(&rec, "gate");
    assert!(activity.iter().any(|a| a.motion), "{activity:?}");
    assert!(covered > 0);

    // A late viewer starts with the tables and then a keyframe, not mid-picture-group.
    let first = late.expect("a second viewer").try_recv().expect("its start");
    let pid = |p: &[u8]| (((p[1] & 0x1f) as u16) << 8) | p[2] as u16;
    assert_eq!(pid(&first[..188]), 0, "the PAT first");
    let key = &first[376..564];
    assert!((key[3] >> 4) & 2 != 0 && key[5] & 0x40 != 0, "then a packet marked random-access: a keyframe");

    let live: Vec<u8> = viewer.try_iter().flat_map(|c| c.to_vec()).collect();
    assert!(live.len() > 10_000 && live[0] == 0x47 && live.len().is_multiple_of(188), "{} live bytes", live.len());
    let _ = std::fs::remove_dir_all(&dir);
}
