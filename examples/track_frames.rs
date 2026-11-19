//! Follow objects through raw gray videos (one file per camera: W x H frames back to back)
//! and print, per step, each camera's objects as JSON: to compare with tamserve.boxes.
//!   track_frames MODEL WIDTH HEIGHT LOOK_EVERY cam0.raw cam1.raw ...
use tamserve_boxes::{BoxTracker, DetectorOptions, GrayFrame, LookEvery, TrackerOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (w, h, every): (usize, usize, u32) = (a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let videos: Vec<Vec<u8>> = a[5..].iter().map(std::fs::read).collect::<Result<_, _>>()?;
    let n = videos[0].len() / (w * h);
    let opts = DetectorOptions { threads: Some(2), precision: Some("f32".into()),
                                 classes: Some(vec!["dog".into(), "bicycle".into(), "car".into()]), ..Default::default() };
    let mut t: BoxTracker<_, usize> = BoxTracker::from_model(&a[1], opts,
        TrackerOptions { look_every: LookEvery::All(every), ..Default::default() })?;
    for k in 0..n {
        let frames: Vec<(usize, GrayFrame)> = videos.iter().enumerate()
            .map(|(c, v)| Ok((c, GrayFrame::new(&v[k * w * h..(k + 1) * w * h], w, h)?)))
            .collect::<Result<_, tamserve_boxes::Error>>()?;
        let out = t.step(&frames)?;
        let cams: Vec<String> = (0..videos.len()).map(|c| {
            let objs: Vec<String> = out[&c].iter().map(|(id, f)| format!("[{id},\"{}\",{:.3},{:.2},{:.2},{:.2},{:.2}]",
                f.label, f.score, f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3])).collect();
            format!("[{}]", objs.join(","))
        }).collect();
        println!("[{}]", cams.join(","));
    }
    Ok(())
}
