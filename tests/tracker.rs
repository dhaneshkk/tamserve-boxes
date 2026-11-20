//! The Python tests of BoxTracker (tests/test_boxes.py), ported: the same scripts, the same answers.

use std::collections::HashMap;

use tamserve_boxes::{BBox, BoxTracker, Detect, Finds, GrayFrame, LookEvery, Result, TrackerOptions};

/// Stands in for a Detector: `finds(camera, step)` gives each look's finds. A frame
/// carries its camera in pixel 0 and its step in pixel 1.
struct Scripted<F: Fn(u8, u8) -> Vec<(BBox, f32, usize)>> {
    finds: F,
    names: Vec<String>,
    looks: Vec<Vec<u8>>,
}

fn scripted<F: Fn(u8, u8) -> Vec<(BBox, f32, usize)>>(finds: F) -> Scripted<F> {
    Scripted { finds, names: vec!["person".into(), "car".into()], looks: vec![] }
}

impl<F: Fn(u8, u8) -> Vec<(BBox, f32, usize)>> Detect for Scripted<F> {
    fn names(&self) -> &[String] {
        &self.names
    }
    fn detect(&mut self, frames: &[GrayFrame]) -> Result<Vec<Finds>> {
        self.looks.push(frames.iter().map(|f| f.pixels[0]).collect());
        Ok(frames.iter().map(|f| {
            let mut out = Finds::default();
            for (b, s, k) in (self.finds)(f.pixels[0], f.pixels[1]) {
                out.boxes.push(b);
                out.scores.push(s);
                out.labels.push(k);
            }
            out
        }).collect())
    }
}

fn frame(cam: u8, step: u8, h: usize) -> Vec<u8> {
    let mut f = vec![0u8; 200 * h];
    f[0] = cam;
    f[1] = step;
    f
}

fn walk(step: u8) -> BBox {
    let x = 20.0 + 5.0 * step as f32;
    [x, 30.0, x + 30.0, 100.0]
}

fn opts(look_every: u32, hold: u32) -> TrackerOptions<u8> {
    TrackerOptions { look_every: LookEvery::All(look_every), hold, ..Default::default() }
}

/// Steps through `steps` rounds on `cams`: each step's ids per camera, sorted.
fn run<D: Detect>(t: &mut BoxTracker<D, u8>, steps: u8, cams: &[u8]) -> Vec<HashMap<u8, Vec<u32>>> {
    (0..steps).map(|k| {
        let pixels: Vec<Vec<u8>> = cams.iter().map(|&c| frame(c, k, 120)).collect();
        let frames: Vec<(u8, GrayFrame)> = cams.iter().zip(&pixels).map(|(&c, p)| (c, GrayFrame::new(p, 200, 120).unwrap())).collect();
        t.step(&frames).unwrap().into_iter().map(|(c, o)| (c, o.keys().copied().collect())).collect()
    }).collect()
}

#[test]
fn an_object_keeps_its_id_and_has_it_back_after_a_short_absence() {
    let mut t = BoxTracker::new(scripted(|_, k| if (5..9).contains(&k) { vec![] } else { vec![(walk(k), 0.9, 0)] }), opts(1, 10)).unwrap();
    let ids: Vec<Vec<u32>> = run(&mut t, 14, &[0]).into_iter().map(|m| m[&0].clone()).collect();
    let want: Vec<Vec<u32>> = (0..14).map(|k| if (5..9).contains(&k) { vec![] } else { vec![1] }).collect();
    assert_eq!(ids, want);
}

#[test]
fn an_object_gone_longer_than_hold_comes_back_under_a_new_id() {
    let mut t = BoxTracker::new(scripted(|_, k| if (5..=10).contains(&k) { vec![] } else { vec![(walk(k), 0.9, 0)] }), opts(1, 3)).unwrap();
    let ids = run(&mut t, 13, &[0]);
    assert_eq!(ids[4][&0], vec![1]);
    assert_eq!(ids[12][&0], vec![2]);
}

#[test]
fn boxes_move_between_looks() {
    let mut t = BoxTracker::new(scripted(|_, k| vec![(walk(k), 0.9, 0)]), opts(3, 30)).unwrap();
    let mut lefts = vec![];
    for k in 0..16u8 {
        let p = frame(0, k, 120);
        let out = t.step(&[(0, GrayFrame::new(&p, 200, 120).unwrap())]).unwrap();
        lefts.push(out[&0][&1].bbox[0]);
    }
    assert!(lefts[9..].windows(2).all(|w| w[1] > w[0]), "carried forward on the frames no look sees: {lefts:?}");
    assert!((lefts[15] - walk(15)[0]).abs() < 6.0);
}

#[test]
fn a_new_object_needs_a_second_look_except_on_a_cameras_first() {
    let car = [120.0, 40.0, 170.0, 80.0];
    let mut t = BoxTracker::new(scripted(move |_, k| {
        let mut v = vec![(walk(k), 0.9, 0)];
        if [3, 6, 7].contains(&k) { v.push((car, 0.9, 1)); }
        v
    }), opts(1, 30)).unwrap();
    let ids = run(&mut t, 9, &[0]);
    assert_eq!(ids[0][&0], vec![1]);                         // the first look's finds are shown at once
    assert_eq!((ids[3][&0].clone(), ids[4][&0].clone()), (vec![1], vec![1])); // seen once, then not: a false find
    assert_eq!((ids[6][&0].clone(), ids[7][&0].clone()), (vec![1], vec![1, 3]));
}

#[test]
fn weak_finds_only_confirm_what_is_followed() {
    let mut t = BoxTracker::new(scripted(|_, k| vec![(walk(k), if k < 3 { 0.9 } else { 0.3 }, 0), ([150.0, 10.0, 190.0, 60.0], 0.3, 1)]),
                                opts(1, 0)).unwrap();
    assert!(run(&mut t, 8, &[0]).iter().all(|m| m[&0] == vec![1]));
}

#[test]
fn looks_are_spread_across_cameras() {
    let mut t = BoxTracker::new(scripted(|_, _| vec![]), opts(3, 30)).unwrap();
    run(&mut t, 12, &[0, 1, 2, 3, 4, 5]);
    let looks = &t.detector.looks;
    assert_eq!(looks[0].len(), 6); // every camera's first frame
    assert!(looks[1..].iter().all(|l| l.len() == 2)); // then 6 cameras / 3 frames a step
    for c in 0..6u8 {
        for k in [3, 6] {
            assert_eq!(looks[k..k + 3].iter().filter(|l| l.contains(&c)).count(), 1);
        }
    }
}

#[test]
fn each_camera_can_have_its_own_look_every() {
    let o = TrackerOptions { look_every: LookEvery::Each { cameras: HashMap::from([(1u8, 1)]), default: 3 }, ..Default::default() };
    let mut t = BoxTracker::new(scripted(|_, _| vec![]), o).unwrap();
    run(&mut t, 9, &[0, 1]);
    let looks = &t.detector.looks;
    assert!(looks.iter().all(|l| l.contains(&1)));
    assert_eq!(looks.iter().filter(|l| l.contains(&0)).count(), 3);
}

#[test]
fn a_camera_that_changes_resolution_starts_afresh() {
    let mut t = BoxTracker::new(scripted(|_, k| vec![(walk(k), 0.9, 0)]), opts(1, 30)).unwrap();
    run(&mut t, 3, &[0]);
    let p = vec![0u8; 200 * 130];
    let out = t.step(&[(0, GrayFrame::new(&p, 200, 130).unwrap())]).unwrap();
    assert_eq!(out[&0].keys().copied().collect::<Vec<_>>(), vec![1]); // id 1 again: a new start
}

#[test]
fn bad_settings_are_rejected() {
    assert!(BoxTracker::new(scripted(|_, _| vec![]), opts(0, 30)).is_err());
    let each = TrackerOptions::<u8> { look_every: LookEvery::Each { cameras: HashMap::from([(0, 0)]), default: 3 }, ..Default::default() };
    assert!(BoxTracker::new(scripted(|_, _| vec![]), each).is_err());
    assert!(GrayFrame::new(&[0u8; 10], 4, 4).is_err());
}

/// Steps with a gate: `quiet(camera, step)`. Returns each step's ids for camera 1.
fn run_gated<D: Detect>(t: &mut BoxTracker<D, u8>, steps: u8, cams: &[u8], quiet: impl Fn(u8, u8) -> bool) -> Vec<Vec<u32>> {
    (0..steps).map(|k| {
        let pixels: Vec<Vec<u8>> = cams.iter().map(|&c| frame(c, k, 120)).collect();
        let frames: Vec<(u8, GrayFrame)> = cams.iter().zip(&pixels).map(|(&c, p)| (c, GrayFrame::new(p, 200, 120).unwrap())).collect();
        let out = t.step_gated(&frames, &|c| quiet(*c, k)).unwrap();
        out[&1].keys().copied().collect()
    }).collect()
}

#[test]
fn a_gate_that_is_never_quiet_changes_nothing() {
    let finds = |cam: u8, step: u8| if cam == 1 && step < 9 { vec![(walk(step), 0.9, 0)] } else { vec![] };
    let mut plain = BoxTracker::new(scripted(finds), opts(3, 4)).unwrap();
    let mut gated = BoxTracker::new(scripted(finds), opts(3, 4)).unwrap();
    let want: Vec<Vec<u32>> = run(&mut plain, 20, &[1, 2]).into_iter().map(|m| m[&1].clone()).collect();
    assert_eq!(run_gated(&mut gated, 20, &[1, 2], |_, _| false), want);
    assert_eq!(plain.detector.looks, gated.detector.looks);
}

#[test]
fn a_quiet_camera_with_nothing_followed_is_not_looked_at() {
    // Nothing is ever there; the camera is quiet from step 2 on, and moves at step 12.
    let mut t = BoxTracker::new(scripted(|_, _| vec![]), opts(1, 4)).unwrap();
    run_gated(&mut t, 16, &[1], |_, k| (2..12).contains(&k));
    let looked: Vec<usize> = t.detector.looks.iter().map(|l| l.len()).collect();
    assert_eq!(looked.len(), 16);
    assert_eq!(looked.iter().sum::<usize>(), 6, "looks only on steps 0, 1 and 12 to 15: {looked:?}");
    assert!(looked[2..12].iter().all(|&n| n == 0));
}

#[test]
fn a_quiet_camera_still_follows_what_it_has() {
    // A person stands still (no motion) from step 3: they are followed while seen, and
    // only once they are gone and held out does the gate stop the looks.
    let finds = |_: u8, step: u8| if step < 8 { vec![([50.0, 30.0, 80.0, 100.0], 0.9, 0)] } else { vec![] };
    let mut t = BoxTracker::new(scripted(finds), opts(1, 3)).unwrap();
    let ids = run_gated(&mut t, 20, &[1], |_, k| k >= 3);
    assert!(ids[..8].iter().all(|i| i == &vec![1]), "{ids:?}");
    let looked: Vec<usize> = t.detector.looks.iter().map(|l| l.len()).collect();
    // Seen up to step 7; held unseen through step 11 (hold 3 + the step it is dropped on); then quiet.
    assert!(looked[..12].iter().all(|&n| n == 1), "{looked:?}");
    assert!(looked[12..].iter().all(|&n| n == 0), "{looked:?}");
    assert_eq!(t.skipped(), &[1]);
}
