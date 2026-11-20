//! Looking where it matters: a look cropped to squares around what moved and what is
//! followed, instead of the whole frame shrunk to the model's input. Frigate's regions.
//!
//! On a 1920 x 1080 main stream the whole frame goes into YOLOX-Nano's 416 x 416 at under
//! a quarter of its size, and a person 60 px tall comes out 13 px tall, too small to find.
//! A 416 square around them goes in at full size. Each region is a look of its own, so
//! this is about finding small things on big frames, not about saving looks: at most
//! `most` regions are made, and when they would cover more than `full_share` of the
//! frame, or there is nothing to centre them on, the whole frame is looked at instead.
//! On a 640 x 360 sub-stream no 416 square fits, so every look is whole.
//!
//! A find that touches a region's edge inside the frame may be a part of something cut
//! off by the crop: it is left out, and that camera's next look is the whole frame.

use crate::{iou, BBox, Finds};

/// How regions are made.
#[derive(Clone, Debug, PartialEq)]
pub struct Regions {
    /// The smallest region's side, in frame pixels: the model's input size.
    pub size: usize,
    /// The most regions in one look of one camera.
    pub most: usize,
    /// How much a box is grown around what it holds (0.5: by half of its larger side).
    pub margin: f32,
    /// When the regions would cover more than this share of the frame, look at it whole.
    pub full_share: f32,
}

impl Default for Regions {
    fn default() -> Self {
        Regions { size: 416, most: 2, margin: 0.5, full_share: 0.5 }
    }
}

/// A square of the frame: left, top, side.
pub type Square = [usize; 3];

/// The start of a `side` square centred on `c`, kept inside `0..extent`.
/// `proofs/Regions.lean`: with `side <= extent`, the square is inside.
pub fn place(c: f32, side: usize, extent: usize) -> usize {
    let start = (c - side as f32 / 2.0).round().max(0.0) as usize;
    start.min(extent - side)
}

fn square_around(b: &BBox, grow: f32, r: &Regions, w: usize, h: usize) -> Option<Square> {
    let longest = (b[2] - b[0]).max(b[3] - b[1]).max(0.0);
    let side = r.size.max((longest * (1.0 + grow)).ceil() as usize);
    (side <= w.min(h)).then(|| [place((b[0] + b[2]) / 2.0, side, w), place((b[1] + b[3]) / 2.0, side, h), side])
}

fn overlap(a: &Square, b: &Square) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[2] && b[1] < a[1] + a[2]
}

/// The squares to look at on a `w` x `h` frame for `boxes` (what moved, what is
/// followed), or None to look at the whole frame.
pub fn plan(w: usize, h: usize, boxes: &[BBox], r: &Regions) -> Option<Vec<Square>> {
    if boxes.is_empty() || w.min(h) < r.size {
        return None;
    }
    let mut squares: Vec<Square> = boxes.iter().map(|b| square_around(b, r.margin, r, w, h)).collect::<Option<_>>()?;
    // Overlapping squares become one around both, until none overlap.
    'merging: loop {
        for i in 0..squares.len() {
            for j in i + 1..squares.len() {
                if overlap(&squares[i], &squares[j]) {
                    let (a, b) = (squares[i], squares.swap_remove(j));
                    let u = [a[0].min(b[0]) as f32, a[1].min(b[1]) as f32,
                             (a[0] + a[2]).max(b[0] + b[2]) as f32, (a[1] + a[2]).max(b[1] + b[2]) as f32];
                    squares[i] = square_around(&u, 0.0, r, w, h)?;
                    continue 'merging;
                }
            }
        }
        break;
    }
    let area: usize = squares.iter().map(|s| s[2] * s[2]).sum();
    if squares.len() > r.most || area as f32 > r.full_share * (w * h) as f32 {
        return None;
    }
    squares.sort();
    Some(squares)
}

/// What the looks at `parts` (each square's finds, in the square's own pixels) found on
/// the whole frame: moved into the frame's pixels, the ones cut by an edge inside the
/// frame left out, and duplicates from overlapping squares removed (the same class
/// overlapping by more than half: the more confident stays). Also whether any was cut.
pub fn join(parts: &[(Finds, Square)], w: usize, h: usize) -> (Finds, bool) {
    const EDGE: f32 = 2.0;
    let mut all: Vec<(BBox, f32, usize)> = Vec::new();
    let mut cut = false;
    for (f, [x0, y0, side]) in parts {
        let (x0, y0, side) = (*x0 as f32, *y0 as f32, *side as f32);
        let inner = [x0 > 0.0, y0 > 0.0, x0 + side < w as f32, y0 + side < h as f32];
        for i in 0..f.len() {
            let b = f.boxes[i];
            let at_edge = [b[0] <= EDGE, b[1] <= EDGE, b[2] >= side - EDGE, b[3] >= side - EDGE];
            if inner.iter().zip(at_edge).any(|(&i, e)| i && e) {
                cut = true;
                continue;
            }
            all.push(([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], f.scores[i], f.labels[i]));
        }
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut out = Finds::default();
    for (b, s, l) in all {
        let dup = (0..out.len()).any(|k| out.labels[k] == l && iou(&out.boxes[k], &b) > 0.5);
        if !dup {
            out.boxes.push(b);
            out.scores.push(s);
            out.labels.push(l);
        }
    }
    (out, cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finds(v: &[(BBox, f32, usize)]) -> Finds {
        Finds { boxes: v.iter().map(|x| x.0).collect(), scores: v.iter().map(|x| x.1).collect(), labels: v.iter().map(|x| x.2).collect() }
    }

    #[test]
    fn a_small_thing_gets_a_square_of_the_model_size_around_it() {
        let r = Regions::default();
        let sq = plan(1920, 1080, &[[900.0, 500.0, 930.0, 560.0]], &r).unwrap();
        assert_eq!(sq, vec![[707, 322, 416]]);
    }

    #[test]
    fn squares_stay_inside_the_frame() {
        let r = Regions::default();
        let sq = plan(1920, 1080, &[[1900.0, 1060.0, 1915.0, 1078.0], [0.0, 0.0, 10.0, 10.0]], &r).unwrap();
        assert_eq!(sq, vec![[0, 0, 416], [1504, 664, 416]]);
    }

    #[test]
    fn overlapping_squares_become_one() {
        let r = Regions::default();
        let sq = plan(1920, 1080, &[[900.0, 500.0, 930.0, 560.0], [1000.0, 520.0, 1030.0, 580.0]], &r).unwrap();
        assert_eq!(sq.len(), 1);
        let [x, y, s] = sq[0];
        assert!(x <= 900 && y <= 500 && x + s >= 1030 && y + s >= 580, "{sq:?}");
    }

    #[test]
    fn the_whole_frame_when_regions_would_not_help() {
        let r = Regions::default();
        assert_eq!(plan(1920, 1080, &[], &r), None, "nothing to centre on");
        assert_eq!(plan(640, 360, &[[10.0, 10.0, 20.0, 20.0]], &r), None, "no 416 square fits");
        assert_eq!(plan(1920, 1080, &[[0.0, 0.0, 900.0, 900.0]], &r), None, "too big to crop");
        let three = [[100.0, 100.0, 110.0, 110.0], [900.0, 500.0, 910.0, 510.0], [1700.0, 900.0, 1710.0, 910.0]];
        assert_eq!(plan(1920, 1080, &three, &r), None, "more than `most`");
        let r = Regions { full_share: 0.3, ..Regions::default() };
        assert_eq!(plan(1280, 720, &three[..2], &r), None, "two 416 squares are 38% of 1280 x 720");
    }

    #[test]
    fn finds_move_into_the_frame_and_cut_ones_are_left_out() {
        let parts = [(finds(&[([100.0, 100.0, 140.0, 200.0], 0.9, 0), ([380.0, 50.0, 416.0, 120.0], 0.8, 0)]), [700, 300, 416])];
        let (f, cut) = join(&parts, 1920, 1080);
        assert_eq!(f.boxes, vec![[800.0, 400.0, 840.0, 500.0]]);
        assert!(cut, "the second touches the square's right edge, which is inside the frame");
        // At the frame's own edge, a find is whole.
        let (f, cut) = join(&[(finds(&[([0.0, 10.0, 30.0, 60.0], 0.9, 0)]), [0, 300, 416])], 1920, 1080);
        assert_eq!((f.len(), cut), (1, false));
    }

    #[test]
    fn the_same_thing_seen_twice_is_one_find() {
        let parts = [(finds(&[([10.0, 10.0, 50.0, 90.0], 0.7, 0)]), [500, 300, 416]),
                     (finds(&[([12.0, 12.0, 52.0, 92.0], 0.9, 0), ([12.0, 12.0, 52.0, 92.0], 0.6, 1)]), [498, 298, 416])];
        let (f, _) = join(&parts, 1920, 1080);
        assert_eq!(f.scores, vec![0.9, 0.6], "the more confident of the two, and the other class kept");
    }
}
