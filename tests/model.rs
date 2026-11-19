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
    let dst = std::env::temp_dir().join(format!("tamserve-boxes-test-{}.onnx", std::process::id()));
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
