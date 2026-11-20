//! Motion, found cheaply on a small gray copy of each frame, so that the detector is
//! not run on cameras where nothing moves. Frigate's idea, done on the Y plane this box
//! already has.
//!
//! Each frame is shrunk by area averaging to `height` rows (100 by default) and compared
//! with a running average of the frames before it. Pixels further than `threshold` from
//! the average are moved pixels; they are grown by one pixel so that near neighbours
//! join, and grouped into connected regions. Regions with fewer than `min_area` moved
//! small pixels are noise. When more than `lighting` of the picture changes at once (an IR
//! cut filter switching at dusk, floodlights, lightning, the camera's exposure jumping)
//! it is the light, not motion: the average is set to the new frame and nothing is
//! reported, rather than waking the detector with one region the size of the picture.
//!
//! The defaults are Frigate's: 100 rows, threshold 30, 10 pixels, alpha 0.01, 0.8.
//! `alpha` is per frame the runner uses, so at 25 frames a second the average forgets in
//! about 4 s. Something bright that stood still for a while and then left stays in the
//! average for a while: the gate stays awake longer. It errs towards looking, never
//! towards missing.

use crate::{BBox, GrayFrame};

/// How motion is found. `Default` is Frigate's.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionOptions {
    /// Rows of the small copy that motion is found on.
    pub height: usize,
    /// How far a pixel must be from the running average (0 to 255) to have moved.
    pub threshold: u8,
    /// The smallest region, in pixels of the small copy, that counts as motion.
    pub min_area: usize,
    /// How fast the average follows the picture: each frame weighs `alpha`.
    pub alpha: f32,
    /// The share of the picture (0 to 1) beyond which a change is the light, not motion.
    pub lighting: f32,
    /// Parts of the picture where motion is not looked for (a clock, trees, a road
    /// beyond the fence), as [left, top, right, bottom] shares of its width and height.
    pub mask: Vec<[f32; 4]>,
}

impl Default for MotionOptions {
    fn default() -> Self {
        MotionOptions { height: 100, threshold: 30, min_area: 10, alpha: 0.01, lighting: 0.8, mask: Vec::new() }
    }
}

impl MotionOptions {
    pub fn check(&self) -> crate::Result<()> {
        let bad = if self.height < 8 {
            "motion.height must be at least 8"
        } else if !(self.alpha > 0.0 && self.alpha <= 1.0) {
            "motion.alpha must be above 0 and at most 1"
        } else if !(self.lighting > 0.0 && self.lighting <= 1.0) {
            "motion.lighting must be above 0 and at most 1"
        } else if self.mask.iter().any(|m| !(m.iter().all(|v| (0.0..=1.0).contains(v)) && m[0] < m[2] && m[1] < m[3])) {
            "a motion mask is [left, top, right, bottom], each from 0 to 1, left < right and top < bottom"
        } else {
            return Ok(());
        };
        Err(crate::Error::Invalid(bad.into()))
    }
}

/// What one frame showed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Motion {
    /// Where things moved, in the frame's own pixels, the largest first.
    pub boxes: Vec<BBox>,
    /// The share of the small copy that changed, 0 to 1.
    pub changed: f32,
    /// The whole picture changed: the light, taken as no motion.
    pub lighting: bool,
}

impl Motion {
    pub fn moved(&self) -> bool {
        !self.boxes.is_empty()
    }
}

/// Finds motion on one camera's frames, one after another.
pub struct MotionDetector {
    pub opts: MotionOptions,
    average: Vec<f32>,
    /// The small copy's size, and the frame size it was made for.
    small: (usize, usize),
    frame: (usize, usize),
    pixels: Vec<u8>,
    /// The small pixels the mask leaves out, and how many it leaves in.
    masked: Vec<bool>,
    open: usize,
    mask: Vec<bool>,
    grown: Vec<bool>,
    seen: Vec<bool>,
    stack: Vec<usize>,
    /// Times the whole picture changed and the average was started again.
    pub lighting_resets: u64,
}

impl MotionDetector {
    pub fn new(opts: MotionOptions) -> MotionDetector {
        MotionDetector { opts, average: Vec::new(), small: (0, 0), frame: (0, 0), pixels: Vec::new(), masked: Vec::new(), open: 0, mask: Vec::new(),
                         grown: Vec::new(), seen: Vec::new(), stack: Vec::new(), lighting_resets: 0 }
    }

    /// Forget the past: the next frame only starts the average.
    pub fn reset(&mut self) {
        self.average.clear();
    }

    /// The next frame. The first one, and the first after the frame size changes, only
    /// start the average and show no motion.
    pub fn next(&mut self, f: &GrayFrame) -> Motion {
        if self.frame != (f.width, f.height) {
            let h = self.opts.height.min(f.height).max(1);
            let w = ((f.width * h) as f64 / f.height as f64).round().max(1.0) as usize;
            self.small = (w, h);
            self.frame = (f.width, f.height);
            self.average.clear();
            // A small pixel is masked if its centre is in a masked part.
            self.masked = (0..w * h).map(|i| {
                let (x, y) = (((i % w) as f32 + 0.5) / w as f32, ((i / w) as f32 + 0.5) / h as f32);
                self.opts.mask.iter().any(|m| x >= m[0] && x < m[2] && y >= m[1] && y < m[3])
            }).collect();
            self.open = self.masked.iter().filter(|&&m| !m).count().max(1);
        }
        let (w, h) = self.small;
        shrink(f, w, h, &mut self.pixels);
        if self.average.len() != w * h {
            self.average = self.pixels.iter().map(|&p| p as f32).collect();
            return Motion::default();
        }
        let t = self.opts.threshold as f32;
        self.mask.clear();
        self.mask.extend(self.pixels.iter().zip(&self.average).zip(&self.masked).map(|((&p, &a), &out)| !out && (p as f32 - a).abs() > t));
        let changed = self.mask.iter().filter(|&&m| m).count() as f32 / self.open as f32;
        if changed > self.opts.lighting {
            self.average.iter_mut().zip(&self.pixels).for_each(|(a, &p)| *a = p as f32);
            self.lighting_resets += 1;
            return Motion { boxes: Vec::new(), changed, lighting: true };
        }
        let alpha = self.opts.alpha;
        self.average.iter_mut().zip(&self.pixels).for_each(|(a, &p)| *a += alpha * (p as f32 - *a));
        grow(&self.mask, w, h, &mut self.grown);
        let regions = regions(&self.grown, &self.mask, w, h, &mut self.seen, &mut self.stack);
        let (sx, sy) = (f.width as f32 / w as f32, f.height as f32 / h as f32);
        let mut found: Vec<(usize, BBox)> = regions.into_iter()
            .filter(|r| r.0 >= self.opts.min_area)
            .map(|(area, [x0, y0, x1, y1])| (area, [x0 as f32 * sx, y0 as f32 * sy, (x1 + 1) as f32 * sx, (y1 + 1) as f32 * sy]))
            .collect();
        found.sort_by_key(|r| std::cmp::Reverse(r.0));
        Motion { boxes: found.into_iter().map(|r| r.1).collect(), changed, lighting: false }
    }
}

/// `f` shrunk to `w` x `h` by averaging the pixels each small pixel covers.
fn shrink(f: &GrayFrame, w: usize, h: usize, out: &mut Vec<u8>) {
    out.clear();
    out.resize(w * h, 0);
    let mut sums = vec![0u32; w];
    let cols: Vec<(usize, usize)> = (0..w).map(|x| (x * f.width / w, ((x + 1) * f.width / w).max(x * f.width / w + 1))).collect();
    for y in 0..h {
        let (r0, r1) = (y * f.height / h, ((y + 1) * f.height / h).max(y * f.height / h + 1));
        sums.iter_mut().for_each(|s| *s = 0);
        for r in r0..r1 {
            let row = f.row(r);
            for (s, &(c0, c1)) in sums.iter_mut().zip(&cols) {
                *s += row[c0..c1].iter().map(|&p| p as u32).sum::<u32>();
            }
        }
        for (x, (&s, &(c0, c1))) in sums.iter().zip(&cols).enumerate() {
            let n = ((r1 - r0) * (c1 - c0)) as u32;
            out[y * w + x] = ((s + n / 2) / n) as u8;
        }
    }
}

/// The mask grown by one pixel in every direction (a 3 x 3 dilation).
fn grow(mask: &[bool], w: usize, h: usize, out: &mut Vec<bool>) {
    out.clear();
    out.resize(w * h, false);
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        out[yy * w + xx] = true;
                    }
                }
            }
        }
    }
}

/// The grown mask's connected regions (8 neighbours): (moved pixels in it, before the
/// growing, so that growth does not make specks look big; [x0, y0, x1, y1] inclusive).
fn regions(mask: &[bool], moved: &[bool], w: usize, h: usize, seen: &mut Vec<bool>, stack: &mut Vec<usize>) -> Vec<(usize, [usize; 4])> {
    seen.clear();
    seen.resize(w * h, false);
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let (mut area, mut b) = (0usize, [start % w, start / w, start % w, start / w]);
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            area += usize::from(moved[i]);
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = yy * w + xx;
                    if mask[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        out.push((area, b));
    }
    out
}

/// Whether a camera is quiet: it has shown no motion for more than `hold` frames.
/// A quiet camera with nothing followed is not looked at (see `BoxTracker::step_gated`).
pub struct Gate {
    pub motion: MotionDetector,
    pub hold: u32,
    /// Frames since motion was last seen; None before the first.
    since: Option<u32>,
    pub last: Motion,
}

impl Gate {
    pub fn new(opts: MotionOptions, hold: u32) -> Gate {
        Gate { motion: MotionDetector::new(opts), hold, since: None, last: Motion::default() }
    }

    /// The next frame; returns whether the camera is quiet.
    pub fn next(&mut self, f: &GrayFrame) -> bool {
        self.last = self.motion.next(f);
        self.since = if self.last.moved() { Some(0) } else { self.since.map(|s| s.saturating_add(1)) };
        self.quiet()
    }

    pub fn quiet(&self) -> bool {
        self.since.is_none_or(|s| s > self.hold)
    }

    /// A new connection: start over.
    pub fn reset(&mut self) {
        self.motion.reset();
        self.since = None;
        self.last = Motion::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, fill: u8, square: Option<(usize, usize, usize, u8)>) -> Vec<u8> {
        let mut p = vec![fill; w * h];
        if let Some((x0, y0, s, v)) = square {
            for y in y0..(y0 + s).min(h) {
                for x in x0..(x0 + s).min(w) {
                    p[y * w + x] = v;
                }
            }
        }
        p
    }

    fn run(d: &mut MotionDetector, p: &[u8], w: usize, h: usize) -> Motion {
        d.next(&GrayFrame::new(p, w, h).unwrap())
    }

    #[test]
    fn a_still_picture_has_no_motion() {
        let mut d = MotionDetector::new(MotionOptions::default());
        let p = frame(640, 360, 90, Some((100, 100, 40, 200)));
        for _ in 0..20 {
            assert!(!run(&mut d, &p, 640, 360).moved());
        }
    }

    #[test]
    fn a_moving_square_is_found_where_it_is() {
        let mut d = MotionDetector::new(MotionOptions::default());
        run(&mut d, &frame(640, 360, 40, None), 640, 360);
        let m = run(&mut d, &frame(640, 360, 40, Some((300, 120, 60, 220))), 640, 360);
        assert!(m.moved() && !m.lighting);
        assert_eq!(m.boxes.len(), 1);
        let b = m.boxes[0];
        // Within a small pixel and the one-pixel growth (6.4 frame pixels each) of the square.
        assert!((b[0] - 300.0).abs() < 14.0 && (b[1] - 120.0).abs() < 14.0, "{b:?}");
        assert!((b[2] - 360.0).abs() < 14.0 && (b[3] - 180.0).abs() < 14.0, "{b:?}");
    }

    #[test]
    fn specks_are_noise() {
        let mut d = MotionDetector::new(MotionOptions::default());
        run(&mut d, &frame(640, 360, 40, None), 640, 360);
        // 4 x 4 frame pixels: well under one small pixel's worth of change.
        assert!(!run(&mut d, &frame(640, 360, 40, Some((200, 200, 4, 255))), 640, 360).moved());
    }

    #[test]
    fn the_light_changing_is_not_motion() {
        let mut d = MotionDetector::new(MotionOptions::default());
        run(&mut d, &frame(640, 360, 30, None), 640, 360);
        let m = run(&mut d, &frame(640, 360, 160, None), 640, 360); // the IR cut filter, at dusk
        assert!(m.lighting && !m.moved());
        assert_eq!(d.lighting_resets, 1);
        // The average starts from the new light, so the frames after it are still.
        assert!(!run(&mut d, &frame(640, 360, 160, None), 640, 360).moved());
    }

    #[test]
    fn motion_under_the_mask_is_not_motion() {
        // The square's part of the picture is masked; the same square elsewhere still moves.
        let mut d = MotionDetector::new(MotionOptions { mask: vec![[0.4, 0.25, 0.65, 0.6]], ..Default::default() });
        run(&mut d, &frame(640, 360, 40, None), 640, 360);
        assert!(!run(&mut d, &frame(640, 360, 40, Some((300, 120, 60, 220))), 640, 360).moved());
        assert!(run(&mut d, &frame(640, 360, 40, Some((40, 40, 60, 220))), 640, 360).moved());
        assert!(MotionOptions { mask: vec![[0.5, 0.0, 0.4, 1.0]], ..Default::default() }.check().is_err());
    }

    #[test]
    fn a_new_size_starts_over() {
        let mut d = MotionDetector::new(MotionOptions::default());
        run(&mut d, &frame(640, 360, 40, None), 640, 360);
        assert!(!run(&mut d, &frame(320, 180, 200, None), 320, 180).moved());
    }

    #[test]
    fn shrinking_averages() {
        let p = [0u8, 10, 20, 30, 40, 50, 60, 70];
        let mut out = Vec::new();
        shrink(&GrayFrame::new(&p, 4, 2).unwrap(), 2, 1, &mut out);
        assert_eq!(out, [(10 + 40 + 50 + 2) / 4, (20 + 30 + 60 + 70 + 2) / 4]);
    }

    #[test]
    fn the_gate_holds_open_after_motion() {
        let mut g = Gate::new(MotionOptions::default(), 3);
        let still = frame(640, 360, 40, None);
        let moved = frame(640, 360, 40, Some((300, 120, 60, 220)));
        assert!(g.next(&GrayFrame::new(&still, 640, 360).unwrap()), "quiet before any motion");
        assert!(!g.next(&GrayFrame::new(&moved, 640, 360).unwrap()));
        // The square's frame now partly weighs in the average, so the still frames that
        // follow differ from it by less than the threshold: no new motion.
        for _ in 0..3 {
            assert!(!g.next(&GrayFrame::new(&still, 640, 360).unwrap()), "held open");
        }
        assert!(g.next(&GrayFrame::new(&still, 640, 360).unwrap()), "quiet again after `hold` frames");
    }
}
