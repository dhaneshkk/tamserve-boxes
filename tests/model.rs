//! With YOLOX-Nano itself. Set TAMSERVE_YOLOX_ONNX to YOLOX's released yolox_nano.onnx
//! (https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx)
//! and TAMSERVE_DOG_JPG to YOLOX's assets/dog.jpg; without them these tests are skipped.
//! OpenVINO's C library must be on LD_LIBRARY_PATH (or OPENVINO_INSTALL_DIR set).

use tamserve_boxes::{image, BoxTracker, Detect, Detector, DetectorOptions, GrayFrame, LookEvery, TrackerOptions};

fn inputs() -> Option<(String, Vec<u8>, usize, usize)> {
    let (model, dog) = (std::env::var("TAMSERVE_YOLOX_ONNX").ok()?, std::env::var("TAMSERVE_DOG_JPG").ok()?);
    let img = ::image::open(dog).expect("TAMSERVE_DOG_JPG is not a picture").to_luma8();
    let (w, h) = img.dimensions();
    Some((model, img.into_raw(), w as usize, h as usize))
}

fn gray_model(src: &str) -> std::path::PathBuf {
    // One file per call: the tests run at once, and one must not read another's half-written model.
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dst = std::env::temp_dir().join(format!("tamserve-boxes-test-{}-{n}.onnx", std::process::id()));
    tamserve_boxes::fold::gray_model(src, &dst).unwrap();
    dst
}

fn classes() -> DetectorOptions {
    DetectorOptions { threads: Some(1), precision: Some("f32".into()),
                      classes: Some(vec!["dog".into(), "bicycle".into(), "car".into()]), ..Default::default() }
}

#[test]
fn yolox_nano_finds_on_gray_frames() {
    let Some((model, gray, w, h)) = inputs() else { return eprintln!("skipped: TAMSERVE_YOLOX_ONNX / TAMSERVE_DOG_JPG not set") };
    let mut det = Detector::new(gray_model(&model), classes()).unwrap();
    let f = &det.detect(&[GrayFrame::new(&gray, w, h).unwrap()]).unwrap()[0];
    let mut found: Vec<&str> = (0..f.len()).filter(|&i| f.scores[i] > 0.5).map(|i| det.names()[f.labels[i]].as_str()).collect();
    found.sort();
    assert_eq!(found, ["bicycle", "car", "dog"]);
    let dog = (0..f.len()).find(|&i| det.names()[f.labels[i]] == "dog").unwrap();
    let want = [131.0, 209.0, 338.0, 547.0];
    assert!(f.boxes[dog].iter().zip(want).all(|(a, b)| (a - b).abs() < 25.0), "{:?}", f.boxes[dog]);
}

#[test]
fn a_batch_gives_what_each_frame_gives_alone() {
    let Some((model, gray, w, h)) = inputs() else { return };
    let mut det = Detector::new(gray_model(&model), classes()).unwrap();
    let flipped: Vec<u8> = gray.chunks(w).flat_map(|r| r.iter().rev().copied().collect::<Vec<_>>()).collect();
    let (a, b) = (GrayFrame::new(&gray, w, h).unwrap(), GrayFrame::new(&flipped, w, h).unwrap());
    let both = det.detect(&[a, b, a]).unwrap();
    assert_eq!(both[0], det.detect(&[a]).unwrap()[0]);
    assert_eq!(both[1], det.detect(&[b]).unwrap()[0]);
    assert_eq!(both[0], both[2]);
}

#[test]
fn yolox_nano_follows_a_panning_camera() {
    let Some((model, gray, w, h)) = inputs() else { return };
    let det = Detector::new(gray_model(&model), classes()).unwrap();
    let mut t: BoxTracker<_, &str> = BoxTracker::new(det, TrackerOptions { look_every: LookEvery::All(3), ..Default::default() }).unwrap();
    let (fw, fh) = (1280, 720);
    let mut last = Default::default();
    for k in 0..24 {
        let mut f = vec![120u8; fw * fh];
        let x0 = 40 + 6 * k;
        for y in 0..h.min(fh - 60) {
            f[(60 + y) * fw + x0..(60 + y) * fw + x0 + w].copy_from_slice(&gray[y * w..(y + 1) * w]);
        }
        last = t.step(&[("cam", GrayFrame::new(&f, fw, fh).unwrap())]).unwrap().remove("cam").unwrap();
    }
    let ids: Vec<u32> = last.keys().copied().collect();
    let mut labels: Vec<String> = last.values().map(|f| f.label.clone()).collect();
    labels.sort();
    assert_eq!((ids, labels), (vec![1, 2, 3], vec!["bicycle".to_string(), "car".into(), "dog".into()]));
}

#[test]
fn letterbox_matches_opencv_on_a_known_case() {
    // cv2.resize of this 4 x 3 ramp to 3 x 2 with INTER_LINEAR gives these pixels (OpenCV 5.0).
    let src = [0u8, 40, 80, 120, 10, 50, 90, 130, 20, 60, 100, 140];
    let mut out = [0u8; 6];
    image::resize_linear(&GrayFrame::new(&src, 4, 3).unwrap(), 3, 2, &mut out);
    assert_eq!(out, [9, 63, 116, 24, 78, 131]);
}

#[test]
fn splitting_a_look_across_requests_changes_no_frame() {
    let Some((model, gray, w, h)) = inputs() else { return };
    let path = gray_model(&model);
    let flipped: Vec<u8> = gray.chunks(w).flat_map(|r| r.iter().rev().copied().collect::<Vec<_>>()).collect();
    let frames: Vec<GrayFrame> = (0..7).map(|i| GrayFrame::new(if i % 2 == 0 { &gray } else { &flipped }, w, h).unwrap()).collect();
    let mut one = Detector::new(&path, classes()).unwrap();
    let want = one.detect(&frames).unwrap();
    for requests in [2, 3, 4] {
        let mut many = Detector::new(&path, DetectorOptions { requests, threads: Some(2), ..classes() }).unwrap();
        assert_eq!(many.detect(&frames).unwrap(), want, "{requests} requests");
    }
}

/// The dog picture shrunk to a third and put on a 1920 x 1080 frame: what a whole-frame
/// look finds, and what a look cropped to a square around the dog finds.
#[test]
fn a_region_finds_what_is_small_on_a_big_frame() {
    let Some((model, gray, w, h)) = inputs() else { return };
    let (sw, sh) = (w / 3, h / 3);
    let mut small = vec![0u8; sw * sh];
    image::resize_linear(&GrayFrame::new(&gray, w, h).unwrap(), sw, sh, &mut small);
    let (fw, fh, x0, y0) = (1920, 1080, 1300, 700);
    let mut frame = vec![120u8; fw * fh];
    for y in 0..sh {
        frame[(y0 + y) * fw + x0..(y0 + y) * fw + x0 + sw].copy_from_slice(&small[y * sw..(y + 1) * sw]);
    }
    let det = Detector::new(gray_model(&model), classes()).unwrap();
    let opts = TrackerOptions { look_every: LookEvery::All(1), regions: Some(Default::default()), ..Default::default() };
    let mut t: BoxTracker<_, &str> = BoxTracker::new(det, opts).unwrap();
    let f = GrayFrame::new(&frame, fw, fh).unwrap();
    // Where the dog is in the small picture, a third of [131, 209, 338, 547], moved onto the frame.
    let dog = [x0 as f32 + 44.0, y0 as f32 + 70.0, x0 as f32 + 113.0, y0 as f32 + 182.0];
    let whole = t.detector.detect(&[f]).unwrap().remove(0);
    let whole_dogs = (0..whole.len()).filter(|&i| t.detector.names()[whole.labels[i]] == "dog" && whole.scores[i] > 0.5).count();
    t.step_regions(&[("cam", f)], &|_| false, &|_| vec![]).unwrap(); // the first look is whole
    // Something new is on trial until a second look sees it again.
    t.step_regions(&[("cam", f)], &|_| false, &|_| vec![dog]).unwrap();
    let seen = t.step_regions(&[("cam", f)], &|_| false, &|_| vec![dog]).unwrap().remove("cam").unwrap();
    let found: Vec<_> = seen.values().filter(|o| o.label == "dog").collect();
    eprintln!("whole frame: {whole_dogs} dog(s) over 0.5; cropped: {:?}", found.iter().map(|o| (o.score, o.bbox)).collect::<Vec<_>>());
    assert_eq!(whole_dogs, 0, "at under a quarter of its size in the whole frame's look, the third-size dog is too small");
    assert_eq!(found.len(), 1, "followed from the cropped look");
    // Within what yolox_nano_finds_on_gray_frames allows the dog's box, at a third of it.
    assert!(found[0].bbox.iter().zip(dog).all(|(a, b)| (a - b).abs() < 25.0), "{:?} vs {dog:?}", found[0].bbox);
}
