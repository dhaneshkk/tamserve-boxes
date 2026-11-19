//! Frames as YOLOX takes them: gray, scaled to fit a square, top left, padded with grey.

use crate::{Error, GrayFrame, Result};

/// The grey YOLOX pads its letterbox with.
pub const PAD: f32 = 114.0;

/// An H x W x 3 RGB frame as gray, weighed as BT.601 weighs the channels (what
/// `tamserve.boxes` does with RGB frames). The Y plane of a decoded frame is already
/// gray and needs none of this.
pub fn rgb_to_gray(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    if rgb.len() != width * height * 3 {
        return Err(Error::Invalid(format!("{} bytes is not a {width} x {height} RGB frame", rgb.len())));
    }
    Ok(rgb
        .chunks_exact(3)
        .map(|p| (p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114 + 0.5) as u8)
        .collect())
}

/// Where each output pixel reads from along a row, as OpenCV's INTER_LINEAR works it
/// out: the source column left of the pixel's centre, and the two weights in 1/2048ths.
/// At the edges OpenCV reads the edge pixel alone (weights 2048 and 0).
fn taps(src: usize, dst: usize) -> Vec<(usize, i32, i32)> {
    raw_taps(src, dst)
        .into_iter()
        .map(|(s, frac)| {
            let (s, frac) = if s < 0 { (0, 0.0) } else if s as usize >= src - 1 { (src as i64 - 1, 0.0) } else { (s, frac) };
            let (a, b) = weights(frac);
            (s as usize, a, b)
        })
        .collect()
}

/// As [`taps`], but for rows: OpenCV clamps the row it reads at the picture's edges and
/// keeps the weights as they were worked out, where along a row it zeroes them. The row
/// above is returned unclamped (it can be -1 or the last row).
fn row_taps(src: usize, dst: usize) -> Vec<(i64, i32, i32)> {
    raw_taps(src, dst).into_iter().map(|(s, frac)| { let (a, b) = weights(frac); (s, a, b) }).collect()
}

/// OpenCV's positions: scale = 1 / (dst / src) in double, each position in float.
fn raw_taps(src: usize, dst: usize) -> Vec<(i64, f32)> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|x| {
            let f = ((x as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            (s as i64, f - s)
        })
        .collect()
}

/// Each weight rounded to the nearest 1/2048th (half to even) on its own, as OpenCV does.
fn weights(frac: f32) -> (i32, i32) {
    (((1.0 - frac) * 2048.0).round_ties_even() as i32, (frac * 2048.0).round_ties_even() as i32)
}

/// `frame` resized to `width` x `height` with bilinear weights, as OpenCV's
/// `cv2.resize(..., INTER_LINEAR)` computes them for 8-bit pictures (which is what
/// YOLOX uses), bit for bit: weights in 1/2048ths and OpenCV's own rounding. Writes `out`
/// row by row.
pub fn resize_linear(frame: &GrayFrame, width: usize, height: usize, out: &mut [u8]) {
    assert!(out.len() >= width * height);
    let xs = taps(frame.width, width);
    let ys = row_taps(frame.height, height);
    let mut rows: [Vec<i32>; 2] = [vec![0; width], vec![0; width]];
    let mut cached = [usize::MAX; 2];
    let horizontal = |y: usize, buf: &mut Vec<i32>| {
        let src = frame.row(y);
        for (o, &(s, a, b)) in buf.iter_mut().zip(&xs) {
            let right = if s + 1 < src.len() { src[s + 1] } else { src[s] };
            *o = src[s] as i32 * a + right as i32 * b;
        }
    };
    let last = frame.height as i64 - 1;
    for (y, &(s0, a, b)) in ys.iter().enumerate() {
        let (s, below) = (s0.clamp(0, last) as usize, (s0 + 1).clamp(0, last) as usize);
        for (slot, want) in [(0, s), (1, below)] {
            if cached[slot] != want {
                if cached[1 - slot] == want {
                    let (a, b) = rows.split_at_mut(1);
                    if slot == 0 { a[0].copy_from_slice(&b[0]) } else { b[0].copy_from_slice(&a[0]) }
                } else {
                    horizontal(want, &mut rows[slot]);
                }
                cached[slot] = want;
            }
        }
        let line = &mut out[y * width..(y + 1) * width];
        // OpenCV's vertical pass for 8-bit pictures rounds in two stages: each row's sum is
        // shifted down 4 bits and multiplied by its weight keeping the high 16 bits (its
        // vectorised code's v_mul_hi), and the two are added with 2 and shifted 2.
        let hi = |s: i32, w: i32| ((s >> 4).clamp(-32768, 32767) * w) >> 16;
        for (x, o) in line.iter_mut().enumerate() {
            *o = ((hi(rows[0][x], a) + hi(rows[1][x], b) + 2) >> 2).clamp(0, 255) as u8;
        }
    }
}

/// The size a frame is scaled to so that it fits `size` x `size`, and the scale.
/// Rounds half to even, as Python's `round` does, so both versions agree.
pub fn fit(width: usize, height: usize, size: usize) -> (usize, usize, f32) {
    let r = (size as f64 / height as f64).min(size as f64 / width as f64);
    let w = ((width as f64 * r).round_ties_even() as usize).max(1);
    let h = ((height as f64 * r).round_ties_even() as usize).max(1);
    (w, h, r as f32)
}

/// One frame letterboxed into `out` (size x size floats, already filled with [`PAD`]):
/// scaled to fit, at the top left. `scratch` holds the scaled picture. Returns the scale.
pub fn letterbox(frame: &GrayFrame, size: usize, out: &mut [f32], scratch: &mut Vec<u8>) -> f32 {
    let (w, h, r) = fit(frame.width, frame.height, size);
    scratch.resize(w * h, 0);
    resize_linear(frame, w, h, scratch);
    for y in 0..h {
        for (o, &p) in out[y * size..y * size + w].iter_mut().zip(&scratch[y * w..(y + 1) * w]) {
            *o = p as f32;
        }
    }
    r
}
