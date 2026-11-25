//! Day, then IR night, through the real decoder: a clip that is colour for 2 s and then
//! the same picture with its colour taken out (what a camera's colour stream carries in
//! IR mode). The chroma score is read off the decoder's own planes, and the light meter
//! sees one switch, day to night. Needs GStreamer with x264enc; no model.
#![cfg(feature = "cameras")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use gstreamer::prelude::*;
use tamserve_boxes::camera::{self, Bell, Opening, Source, State};
use tamserve_boxes::config::{Camera, DecodeSection, Decoder};
use tamserve_boxes::light::{mean_luma, LightMeter, LightOptions, Mode};
use tamserve_boxes::GrayFrame;

fn clip(path: &std::path::Path) -> bool {
    gstreamer::init().unwrap();
    if gstreamer::ElementFactory::find("x264enc").is_none() {
        return false;
    }
    let p = gstreamer::parse::launch(&format!(
        "concat name=c ! x264enc key-int-max=25 tune=zerolatency ! h264parse ! mp4mux ! filesink location={} \
         videotestsrc pattern=smpte num-buffers=50 ! video/x-raw,format=I420,width=320,height=180,framerate=25/1 ! c. \
         videotestsrc pattern=smpte num-buffers=50 ! video/x-raw,format=I420,width=320,height=180,framerate=25/1 ! \
           videobalance saturation=0 ! videoconvert ! video/x-raw,format=I420 ! c.",
        path.display())).unwrap();
    p.set_state(gstreamer::State::Playing).unwrap();
    let ok = p.bus().unwrap().timed_pop_filtered(gstreamer::ClockTime::from_seconds(30),
        &[gstreamer::MessageType::Eos, gstreamer::MessageType::Error])
        .is_some_and(|m| m.type_() == gstreamer::MessageType::Eos);
    p.set_state(gstreamer::State::Null).unwrap();
    ok
}

#[test]
fn day_then_ir_night_is_one_switch() {
    let dir = std::env::temp_dir().join(format!("tsb-light-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let video = dir.join("dusk.mp4");
    if !clip(&video) {
        return eprintln!("skipped: no x264enc to make the clip");
    }
    camera::init(Decoder::Software).unwrap();
    let cam = Camera { name: "dusk".into(), url: video.display().to_string(), onvif: None, user: None, password: None,
                       look_every: None, weight: 1.0, record: false, live: false, encode: false, motion_mask: Vec::new(), safety: None };
    let src = Source::open(cam, &Opening::from(&DecodeSection { realtime: true, ..Default::default() }),
                           Arc::new(Bell::default())).unwrap();

    let mut seen: Vec<(u64, Option<f32>, f32)> = Vec::new();
    let until = Instant::now() + Duration::from_secs(30);
    while src.health.state() != State::Ended && Instant::now() < until {
        if let Some(f) = src.take() {
            let g = GrayFrame::new(&f.pixels, f.width, f.height).unwrap();
            seen.push((f.number, f.chroma, mean_luma(&g)));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(seen.len() > 60, "frames read: {}", seen.len());

    let early: Vec<f32> = seen.iter().filter(|s| s.0 <= 40).map(|s| s.1.expect("colour planes")).collect();
    let late: Vec<f32> = seen.iter().filter(|s| s.0 > 60).map(|s| s.1.expect("colour planes")).collect();
    assert!(early.iter().all(|&c| c > 20.0), "colour bars are far from neutral: {early:?}");
    assert!(late.iter().all(|&c| c < 1.0), "with the colour taken out, chroma is neutral: {late:?}");

    let mut m = LightMeter::new(LightOptions::default());
    let mut switches = Vec::new();
    for (n, chroma, luma) in &seen {
        let l = m.next(n * 40, *chroma, *luma, false);
        if l.switched {
            switches.push((*n, l.mode));
        }
    }
    assert_eq!(m.mode(), Mode::Night);
    assert_eq!(switches.len(), 1, "{switches:?}");
    let (at, mode) = switches[0];
    assert_eq!(mode, Mode::Night);
    assert!((51..=60).contains(&at), "found within confirm frames of the switch at frame 51: {at}");
    let _ = std::fs::remove_dir_all(&dir);
}
