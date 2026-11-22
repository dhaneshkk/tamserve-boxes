//! Has the camera moved? A drift check on the box itself, without OpenCV.
//!
//! The check aligns the current frame with a reference frame taken when the floor
//! calibration was right, and hands [`crate::lens::Calibration::apply_drift`] what it
//! found: the correction `G` when the view was confirmed, or "not confirmed", which makes
//! every floor rule UNKNOWN.
//!
//! How:
//! * **Undistort first.** The frame is box-averaged down to about 320 px wide and
//!   undistorted with the camera's lens. In undistorted pixels any move of the camera shows
//!   the floor moved by a homography, which [`crate::lens::Calibration`] can correct exactly.
//! * **Direct alignment, not features.** The homography is fitted by maximising the
//!   correlation coefficient between the reference and the warped current frame (ECC,
//!   Evangelidis and Psarakis 2008, as OpenCV's `findTransformECC`), coarse to fine over
//!   three levels, starting from the best whole-pixel shift at the coarsest. Correlation
//!   does not care about brightness or contrast, so a dimmer or brighter frame still aligns.
//! * **Moving things are masked.** The boxes of what is tracked (people, forklifts) are
//!   left out of the fit.
//! * **Not confirmed** when the correlation stays below `rho_min`, too few pixels overlap,
//!   or the frame has lost most of its contrast. Correlation alone is not enough there: a
//!   lens almost covered still leaves a faint copy of the scene, which correlates as well
//!   as a clear one. So the frame's contrast must be at least `contrast_min` of the
//!   reference's.
//! * **One reference per light mode.** An IR night picture does not look like the day one;
//!   take a reference in each (`POST /safety/<camera>/reference`, or `auto_reference`).
//!
//! On the notebook's test frames (`tests/drift.rs`) it brings a 3° knock from 130 cm of
//! floor error back to the calibration's own 5 cm, as OpenCV's ECC does on the same frames.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::lens::{apply_h, matmul, DriftEstimate, Lens, Mat3, IDENTITY};
use crate::Result;

/// How the check decides.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftParams {
    /// The correlation below which the view is not confirmed.
    #[serde(default = "rho_min")]
    pub rho_min: f64,
    /// The share of the reference's contrast the frame must keep.
    #[serde(default = "contrast_min")]
    pub contrast_min: f64,
    /// The share of the reference's pixels that must take part in the fit.
    #[serde(default = "overlap_min")]
    pub overlap_min: f64,
    /// The width the frame is reduced to (about), in pixels.
    #[serde(default = "work_width")]
    pub work_width: usize,
}

fn rho_min() -> f64 {
    0.9
}
fn contrast_min() -> f64 {
    0.25
}
fn overlap_min() -> f64 {
    0.25
}
fn work_width() -> usize {
    320
}

impl Default for DriftParams {
    fn default() -> Self {
        DriftParams { rho_min: rho_min(), contrast_min: contrast_min(), overlap_min: overlap_min(), work_width: work_width() }
    }
}

/// A grey picture as floats, with which pixels hold picture.
#[derive(Clone, Debug)]
struct Img {
    w: usize,
    h: usize,
    px: Vec<f32>,
    ok: Vec<bool>,
}

impl Img {
    fn half(&self) -> Img {
        let (w, h) = (self.w / 2, self.h / 2);
        let mut px = vec![0.0; w * h];
        let mut ok = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = [(2 * y) * self.w + 2 * x, (2 * y) * self.w + 2 * x + 1, (2 * y + 1) * self.w + 2 * x, (2 * y + 1) * self.w + 2 * x + 1];
                ok[y * w + x] = i.iter().all(|&k| self.ok[k]);
                px[y * w + x] = i.iter().map(|&k| self.px[k]).sum::<f32>() / 4.0;
            }
        }
        Img { w, h, px, ok }
    }

    #[inline]
    fn sample(&self, x: f64, y: f64) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (self.w - 1) as f64 || y > (self.h - 1) as f64 {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let idx = [y0 * self.w + x0, y0 * self.w + x1, y1 * self.w + x0, y1 * self.w + x1];
        if !idx.iter().all(|&k| self.ok[k]) {
            return None;
        }
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let p = |k: usize| self.px[idx[k]];
        Some((p(0) * (1.0 - fx) + p(1) * fx) * (1.0 - fy) + (p(2) * (1.0 - fx) + p(3) * fx) * fy)
    }

    /// Central-difference gradients (zero, and not ok, at the border or next to a hole).
    fn gradients(&self) -> (Img, Img) {
        let mut gx = Img { w: self.w, h: self.h, px: vec![0.0; self.px.len()], ok: vec![false; self.px.len()] };
        let mut gy = gx.clone();
        for y in 1..self.h.saturating_sub(1) {
            for x in 1..self.w.saturating_sub(1) {
                let i = y * self.w + x;
                if self.ok[i - 1] && self.ok[i + 1] && self.ok[i - self.w] && self.ok[i + self.w] {
                    gx.px[i] = (self.px[i + 1] - self.px[i - 1]) / 2.0;
                    gy.px[i] = (self.px[i + self.w] - self.px[i - self.w]) / 2.0;
                    gx.ok[i] = true;
                    gy.ok[i] = true;
                }
            }
        }
        (gx, gy)
    }
}

/// A reference view: its undistorted pyramid, and the calibration's correction when it
/// was taken (a reference taken after a corrected knock carries that correction).
#[derive(Clone, Debug)]
pub struct Reference {
    levels: Vec<Img>,
    /// The calibration's `G` when the reference was taken.
    pub g_at_capture: Mat3,
    frame: (usize, usize),
}

/// For one frame size: the reduction factor k, the lens on the reduced picture, and for each
/// work pixel where to read it in the reduced, still distorted picture.
type WorkMap = ((usize, usize), usize, Lens, Vec<Option<(f32, f32)>>);

/// One camera's drift check.
#[derive(Clone, Debug)]
pub struct DriftChecker {
    lens: Lens,
    pub params: DriftParams,
    map: Option<WorkMap>,
}

fn invert(m: &Mat3) -> Option<Mat3> {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if d.abs() < 1e-12 {
        return None;
    }
    let c = |a: usize, b: usize, e: usize, f: usize| m[a][b] * m[e][f] - m[a][f] * m[e][b];
    Some([
        [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
        [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
        [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
    ])
}

/// Solve the 8 x 8 system `a x = b` (symmetric positive definite) by Gaussian elimination
/// with partial pivoting.
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for c in 0..8 {
        let p = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-18 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let pivot = a[c];
        for r in c + 1..8 {
            let f = a[r][c] / pivot[c];
            for (v, p) in a[r].iter_mut().zip(&pivot).skip(c) {
                *v -= f * p;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 8];
    for c in (0..8).rev() {
        let s: f64 = (c + 1..8).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}

/// What one alignment found.
#[derive(Clone, Debug)]
struct Fit {
    w: Mat3,
    rho: f64,
    n: usize,
    contrast: f64,
}

impl DriftChecker {
    pub fn new(lens: Lens, params: DriftParams) -> Self {
        DriftChecker { lens, params, map: None }
    }

    /// Work pixels to frame pixels: x_frame = k x_work + (k - 1) / 2.
    fn to_frame(k: usize) -> Mat3 {
        let (k, o) = (k as f64, (k as f64 - 1.0) / 2.0);
        [[k, 0.0, o], [0.0, k, o], [0.0, 0.0, 1.0]]
    }

    fn ensure_map(&mut self, w: usize, h: usize) {
        if self.map.as_ref().is_some_and(|m| m.0 == (w, h)) {
            return;
        }
        let k = (w as f64 / self.params.work_width as f64).round().max(1.0) as usize;
        let (sw, sh) = (w / k, h / k);
        // the lens on the reduced picture: the same distortion, the camera matrix scaled
        let t = Self::to_frame(k);
        let ti = invert(&t).expect("a scale");
        let ks = matmul(&ti, &self.lens.k);
        let small = Lens { k: ks, d: self.lens.d };
        let mut m = Vec::with_capacity(sw * sh);
        for y in 0..sh {
            for x in 0..sw {
                let [u, v] = small.distort(x as f64, y as f64);
                m.push((u >= 0.0 && v >= 0.0 && u <= (sw - 1) as f64 && v <= (sh - 1) as f64).then_some((u as f32, v as f32)));
            }
        }
        self.map = Some(((w, h), k, small, m));
    }

    /// The frame, reduced, undistorted, with `boxes` (frame pixels, as detected) masked.
    fn work(&mut self, pixels: &[u8], w: usize, h: usize, boxes: &[[f64; 4]]) -> Img {
        self.ensure_map(w, h);
        let ((_, _), k, _, map) = self.map.as_ref().expect("map");
        let (sw, sh) = (w / k, h / k);
        // box average k x k
        let mut red = Img { w: sw, h: sh, px: vec![0.0; sw * sh], ok: vec![true; sw * sh] };
        let kk = (k * k) as f32;
        for y in 0..sh {
            for x in 0..sw {
                let mut s = 0u32;
                for dy in 0..*k {
                    let row = &pixels[(y * k + dy) * w + x * k..(y * k + dy) * w + x * k + k];
                    s += row.iter().map(|&p| p as u32).sum::<u32>();
                }
                red.px[y * sw + x] = s as f32 / kk;
            }
        }
        let mut out = Img { w: sw, h: sh, px: vec![0.0; sw * sh], ok: vec![false; sw * sh] };
        for (i, m) in map.iter().enumerate() {
            if let Some((u, v)) = m {
                if let Some(p) = red.sample(*u as f64, *v as f64) {
                    out.px[i] = p;
                    out.ok[i] = true;
                }
            }
        }
        // moving things out: each box's corners undistorted into work pixels, grown by 3
        let ti = invert(&Self::to_frame(*k)).expect("a scale");
        for b in boxes {
            let corners = [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]].map(|[x, y]| {
                let [u, v] = self.lens.undistort(x, y);
                apply_h(&ti, u, v)
            });
            let (x0, x1) = corners.iter().fold((f64::MAX, f64::MIN), |(a, z), c| (a.min(c[0]), z.max(c[0])));
            let (y0, y1) = corners.iter().fold((f64::MAX, f64::MIN), |(a, z), c| (a.min(c[1]), z.max(c[1])));
            let (x0, y0) = (((x0 - 3.0).max(0.0)) as usize, ((y0 - 3.0).max(0.0)) as usize);
            let (x1, y1) = (((x1 + 3.0).max(0.0) as usize).min(sw), ((y1 + 3.0).max(0.0) as usize).min(sh));
            for y in y0..y1 {
                for x in x0..x1 {
                    out.ok[y * sw + x] = false;
                }
            }
        }
        out
    }

    fn pyramid(img: Img) -> Vec<Img> {
        let mut v = vec![img];
        for _ in 0..2 {
            let n = v.last().expect("a level").half();
            v.push(n);
        }
        v
    }

    /// Take the current frame as the reference (with the calibration's current `G`).
    pub fn reference(&mut self, pixels: &[u8], w: usize, h: usize, boxes: &[[f64; 4]], g_now: Mat3) -> Reference {
        let img = self.work(pixels, w, h, boxes);
        Reference { levels: Self::pyramid(img), g_at_capture: g_now, frame: (w, h) }
    }

    /// The best whole-pixel shift at the coarsest level, by correlation.
    fn shift(t: &Img, i: &Img, reach: i64) -> (f64, f64) {
        let mut best = (f64::MIN, 0i64, 0i64);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (mut n, mut st, mut si, mut stt, mut sii, mut sti) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
                for y in 0..t.h as i64 {
                    let yy = y + dy;
                    if yy < 0 || yy >= i.h as i64 {
                        continue;
                    }
                    for x in 0..t.w as i64 {
                        let xx = x + dx;
                        if xx < 0 || xx >= i.w as i64 {
                            continue;
                        }
                        let (a, b) = ((y as usize) * t.w + x as usize, (yy as usize) * i.w + xx as usize);
                        if t.ok[a] && i.ok[b] {
                            let (p, q) = (t.px[a] as f64, i.px[b] as f64);
                            n += 1.0;
                            st += p;
                            si += q;
                            stt += p * p;
                            sii += q * q;
                            sti += p * q;
                        }
                    }
                }
                if n < 0.25 * (t.w * t.h) as f64 {
                    continue;
                }
                let cov = sti - st * si / n;
                let den = ((stt - st * st / n) * (sii - si * si / n)).sqrt();
                let r = if den > 0.0 { cov / den } else { f64::MIN };
                if r > best.0 {
                    best = (r, dx, dy);
                }
            }
        }
        (best.1 as f64, best.2 as f64)
    }

    /// ECC at one level, from `w` (reference pixels -> current pixels).
    fn ecc(t: &Img, i: &Img, mut w: Mat3, iters: usize, step: usize) -> Option<Fit> {
        let (gx, gy) = i.gradients();
        let mut last = None;
        let mut done = false;
        for it in 0..=iters {
            // the reference's pixels that land on picture in the current frame
            let (mut tv, mut iv, mut jac) = (Vec::new(), Vec::new(), Vec::<[f64; 8]>::new());
            for y in (0..t.h).step_by(step) {
                for x in (0..t.w).step_by(step) {
                    let k = y * t.w + x;
                    if !t.ok[k] {
                        continue;
                    }
                    let (xf, yf) = (x as f64, y as f64);
                    let den = w[2][0] * xf + w[2][1] * yf + 1.0;
                    if den.abs() < 1e-9 {
                        continue;
                    }
                    let u = (w[0][0] * xf + w[0][1] * yf + w[0][2]) / den;
                    let v = (w[1][0] * xf + w[1][1] * yf + w[1][2]) / den;
                    let (Some(p), Some(ax), Some(ay)) = (i.sample(u, v), gx.sample(u, v), gy.sample(u, v)) else { continue };
                    let (ax, ay) = (ax as f64, ay as f64);
                    let g = ax * u + ay * v;
                    jac.push([ax * xf / den, ax * yf / den, ax / den, ay * xf / den, ay * yf / den, ay / den, -g * xf / den, -g * yf / den]);
                    tv.push(t.px[k] as f64);
                    iv.push(p as f64);
                }
            }
            let n = tv.len();
            if n < 64 {
                return None;
            }
            let (mt, mi) = (tv.iter().sum::<f64>() / n as f64, iv.iter().sum::<f64>() / n as f64);
            for v in tv.iter_mut() {
                *v -= mt;
            }
            for v in iv.iter_mut() {
                *v -= mi;
            }
            let (tn, inn) = (tv.iter().map(|v| v * v).sum::<f64>().sqrt(), iv.iter().map(|v| v * v).sum::<f64>().sqrt());
            let corr: f64 = tv.iter().zip(&iv).map(|(a, b)| a * b).sum();
            let rho = if tn > 0.0 && inn > 0.0 { corr / (tn * inn) } else { 0.0 };
            let contrast = if tn > 0.0 { inn / tn } else { 0.0 };
            last = Some(Fit { w, rho, n: n * step * step, contrast });
            if it == iters || done {
                break;
            }
            let mut hes = [[0.0f64; 8]; 8];
            let (mut ip, mut tp) = ([0.0f64; 8], [0.0f64; 8]);
            for (r, j) in jac.iter().enumerate() {
                for a in 0..8 {
                    ip[a] += j[a] * iv[r];
                    tp[a] += j[a] * tv[r];
                    for b in a..8 {
                        hes[a][b] += j[a] * j[b];
                    }
                }
            }
            let upper = hes;
            for (a, row) in hes.iter_mut().enumerate() {
                for (b, v) in row.iter_mut().enumerate().take(a) {
                    *v = upper[b][a];
                }
            }
            let hip = solve8(hes, ip)?;
            let lambda_n = inn * inn - ip.iter().zip(&hip).map(|(a, b)| a * b).sum::<f64>();
            let lambda_d = corr - tp.iter().zip(&hip).map(|(a, b)| a * b).sum::<f64>();
            if lambda_d <= 0.0 {
                return last; // OpenCV stops here too: the last fit stands, judged by its correlation
            }
            let lambda = lambda_n / lambda_d;
            let mut ep = [0.0f64; 8];
            for (r, j) in jac.iter().enumerate() {
                let e = lambda * tv[r] - iv[r];
                for a in 0..8 {
                    ep[a] += j[a] * e;
                }
            }
            let dp = solve8(hes, ep)?;
            w[0][0] += dp[0];
            w[0][1] += dp[1];
            w[0][2] += dp[2];
            w[1][0] += dp[3];
            w[1][1] += dp[4];
            w[1][2] += dp[5];
            w[2][0] += dp[6];
            w[2][1] += dp[7];
            // converged when the corners of the picture move by less than a hundredth of a pixel
            let shift = [(0.0, 0.0), (t.w as f64, 0.0), (0.0, t.h as f64), (t.w as f64, t.h as f64)].iter().map(|&(x, y)| {
                let (a, b) = (dp[0] * x + dp[1] * y + dp[2], dp[3] * x + dp[4] * y + dp[5]);
                let c = (dp[6] * x + dp[7] * y) * (x.abs() + y.abs());
                (a * a + b * b).sqrt() + c.abs()
            }).fold(0.0f64, f64::max);
            done = shift < 0.01; // one more pass scores the final warp
        }
        last
    }

    /// Check a frame against a reference. `boxes`: what is tracked now (masked).
    pub fn check(&mut self, r: &Reference, pixels: &[u8], w: usize, h: usize, boxes: &[[f64; 4]]) -> DriftEstimate {
        let bad = DriftEstimate { valid: false, g: None, drift_px: None, inliers: 0, ratio: 0.0 };
        if (w, h) != r.frame {
            return bad;
        }
        let cur = Self::pyramid(self.work(pixels, w, h, boxes));
        let (tc, ic) = (&r.levels[2], &cur[2]);
        let (dx, dy) = Self::shift(tc, ic, (tc.w as i64 / 6).max(3));
        let mut wm: Mat3 = [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]];
        let mut fit = None;
        for lvl in (0..3).rev() {
            if lvl < 2 {
                // a level up: reference and current pixels both double
                wm = [[wm[0][0], wm[0][1], 2.0 * wm[0][2]], [wm[1][0], wm[1][1], 2.0 * wm[1][2]], [wm[2][0] / 2.0, wm[2][1] / 2.0, 1.0]];
            }
            match Self::ecc(&r.levels[lvl], &cur[lvl], wm, if lvl == 0 { 15 } else { 40 }, if lvl == 0 { 2 } else { 1 }) {
                Some(f) => {
                    wm = f.w;
                    fit = Some(f);
                }
                None => return bad,
            }
        }
        let Some(f) = fit else { return bad };
        let total = r.levels[0].ok.iter().filter(|&&o| o).count().max(1);
        let ((_, _), k, _, _) = self.map.as_ref().expect("map");
        let t = Self::to_frame(*k);
        let (Some(ti), Some(wi)) = (invert(&t), invert(&f.w)) else { return bad };
        // current -> reference, in frame pixels; then onto the calibrated view
        let g_ref = matmul(&matmul(&t, &wi), &ti);
        let g = matmul(&r.g_at_capture, &g_ref);
        let gi = invert(&g);
        // how far the calibrated view's points moved: median over a grid, frame pixels
        let lv = &r.levels[0];
        let mut d: Vec<f64> = Vec::new();
        for y in (0..lv.h).step_by(8) {
            for x in (0..lv.w).step_by(8) {
                if lv.ok[y * lv.w + x] {
                    let q = apply_h(&t, x as f64, y as f64);
                    if let Some(gi) = &gi {
                        let p = apply_h(gi, q[0], q[1]);
                        d.push(((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt());
                    }
                }
            }
        }
        d.sort_by(f64::total_cmp);
        let drift = d.get(d.len() / 2).copied();
        let valid = f.rho >= self.params.rho_min && f.contrast >= self.params.contrast_min
            && (f.n as f64) >= self.params.overlap_min * total as f64 && gi.is_some() && drift.is_some_and(|v| v.is_finite());
        DriftEstimate { valid, g: valid.then_some(g), drift_px: drift, inliers: f.n, ratio: f.rho }
    }

    /// Save a reference to `dir` as `<mode>.pgm` (the frame reduced, before undistortion, is
    /// not kept: the undistorted work picture is) and `<mode>.json`.
    pub fn save(r: &Reference, dir: &Path, mode: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let l = &r.levels[0];
        let mut bytes = format!("P5\n{} {}\n255\n", l.w, l.h).into_bytes();
        bytes.extend(l.px.iter().zip(&l.ok).map(|(&p, &o)| if o { p.round().clamp(1.0, 255.0) as u8 } else { 0 }));
        std::fs::write(dir.join(format!("{mode}.pgm")), bytes)?;
        let meta = serde_json::json!({"G": r.g_at_capture, "frame": [r.frame.0, r.frame.1]});
        std::fs::write(dir.join(format!("{mode}.json")), meta.to_string())
    }

    /// Load a saved reference, if both files are there. Pixels saved as 0 are holes.
    pub fn load(dir: &Path, mode: &str) -> Result<Option<Reference>> {
        let (p, j) = (dir.join(format!("{mode}.pgm")), dir.join(format!("{mode}.json")));
        if !p.is_file() || !j.is_file() {
            return Ok(None);
        }
        let bad = |m: &str| crate::Error::Invalid(format!("drift reference {}: {m}", p.display()));
        let bytes = std::fs::read(&p).map_err(|e| bad(&e.to_string()))?;
        let (w, h, px) = read_pgm(&bytes).ok_or_else(|| bad("not a binary 8-bit PGM"))?;
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&j).map_err(|e| bad(&e.to_string()))?)
            .map_err(|e| bad(&e.to_string()))?;
        let g: Mat3 = serde_json::from_value(meta["G"].clone()).map_err(|e| bad(&e.to_string()))?;
        let fr: [usize; 2] = serde_json::from_value(meta["frame"].clone()).map_err(|e| bad(&e.to_string()))?;
        let img = Img { w, h, px: px.iter().map(|&p| p as f32).collect(), ok: px.iter().map(|&p| p != 0).collect() };
        Ok(Some(Reference { levels: Self::pyramid(img), g_at_capture: g, frame: (fr[0], fr[1]) }))
    }
}

/// A binary 8-bit PGM (P5): width, height, pixels.
pub fn read_pgm(b: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 && i < b.len() {
        while i < b.len() && (b[i] as char).is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < b.len() && !(b[i] as char).is_ascii_whitespace() {
            i += 1;
        }
        fields.push(std::str::from_utf8(&b[s..i]).ok()?.to_string());
    }
    if fields.len() < 4 || fields[0] != "P5" || fields[3] != "255" {
        return None;
    }
    let (w, h): (usize, usize) = (fields[1].parse().ok()?, fields[2].parse().ok()?);
    let data = b.get(i + 1..i + 1 + w * h)?.to_vec();
    Some((w, h, data))
}

/// The identity: nothing moved.
pub const NO_MOVE: Mat3 = IDENTITY;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve8_inverts_a_system() {
        let mut a = [[0.0; 8]; 8];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = if i == j { 4.0 } else { 1.0 / (1.0 + (i + j) as f64) };
            }
        }
        let x = [1.0, -2.0, 3.0, 0.5, 0.0, -1.0, 2.0, 7.0];
        let b: Vec<f64> = (0..8).map(|i| (0..8).map(|j| a[i][j] * x[j]).sum()).collect();
        let s = solve8(a, b.try_into().unwrap()).unwrap();
        assert!(s.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-12));
    }

    #[test]
    fn pgm_round_trip() {
        let b = b"P5\n# c\n3 2\n255\n\x01\x02\x03\x04\x05\x06";
        assert_eq!(read_pgm(b), Some((3, 2, vec![1, 2, 3, 4, 5, 6])));
        assert_eq!(read_pgm(b"P2\n1 1\n255\n1"), None);
    }
}
