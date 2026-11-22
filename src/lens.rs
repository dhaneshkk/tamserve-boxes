//! From pixels to metres on the floor: the lens, the floor homography, and a guard that
//! stops every floor rule when the camera no longer looks where it was calibrated.
//!
//! A wide CCTV lens bends straight lines; a floor homography alone cannot follow the bend
//! and puts people tens of centimetres to a metre away from where they stand, worst near
//! the edges of the picture. So a point is first undistorted with the lens model, then
//! mapped to the floor:
//!
//! ```text
//! floor = H · G · undistort(u, v)
//! ```
//!
//! `H` is the homography fitted at installation from floor markings (in undistorted pixels).
//! `G` is the drift correction: when the camera is knocked, the floor seen now and the floor
//! seen at calibration differ by a homography between the two undistorted views, so a
//! correction found by matching the current frame against a reference frame is exact for
//! anything on the floor. `G` is the identity until a drift check finds otherwise.
//!
//! The drift check itself (feature matching and a robust homography between frames) runs
//! where OpenCV is, and hands a [`DriftEstimate`] to [`Calibration::apply_drift`]. When it
//! cannot confirm the view (lens covered, steam, camera turned away) the state becomes
//! [`CalibState::Unknown`] and the rules report UNKNOWN instead of "clear":
//! `proofs/Grammar.lean`, `lost_never_clear`.
//!
//! The undistortion inverts OpenCV's lens model (`k1 k2 p1 p2 k3`) with Newton's method, to
//! 1e-9 px. Do not use OpenCV 4's `undistortPoints` for floor geometry on a strong barrel
//! lens: its default 5 iterations leave tens of centimetres of error, and its fixed-point
//! iteration does not converge at all near the corners of the picture.

use serde::{Deserialize, Serialize};

/// A 3 x 3 matrix, row by row.
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `a · b`.
pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// A homography applied to a point.
#[inline]
pub fn apply_h(h: &Mat3, x: f64, y: f64) -> [f64; 2] {
    let q0 = x * h[0][0] + y * h[0][1] + h[0][2];
    let q1 = x * h[1][0] + y * h[1][1] + h[1][2];
    let q2 = x * h[2][0] + y * h[2][1] + h[2][2];
    [q0 / q2, q1 / q2]
}

/// A pinhole camera with OpenCV's distortion coefficients `k1 k2 p1 p2 k3`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lens {
    /// The camera matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
    #[serde(rename = "K")]
    pub k: Mat3,
    /// `k1 k2 p1 p2 k3`.
    #[serde(rename = "D")]
    pub d: [f64; 5],
}

impl Lens {
    /// The ideal pixel (in the same camera matrix) of an observed one: Newton's method on the
    /// forward model, stopping below 1e-9 px. The fixed-point iteration of OpenCV 4's
    /// `undistortPoints` oscillates near the corners of a strong barrel lens and never finds
    /// the point; Newton converges wherever the model is one-to-one ([`Lens::monotone`]).
    /// Same arithmetic, in the same order, as the Python reference's `und_pts`.
    pub fn undistort(&self, u: f64, v: f64) -> [f64; 2] {
        if self.d.iter().all(|&c| c == 0.0) {
            return [u, v];
        }
        let (fx, fy, cx, cy) = (self.k[0][0], self.k[1][1], self.k[0][2], self.k[1][2]);
        let [k1, k2, p1, p2, k3] = self.d;
        let (xd, yd) = ((u - cx) / fx, (v - cy) / fy);
        let (mut x, mut y) = (xd, yd);
        for _ in 0..50 {
            let r2 = x * x + y * y;
            let cd = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            let dc = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2;
            let ex = x * cd + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd;
            let ey = y * cd + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd;
            let (tx, ty) = (ex * fx, ey * fy);
            if (tx * tx + ty * ty).sqrt() < 1e-9 {
                break;
            }
            let a = cd + 2.0 * x * x * dc + 2.0 * p1 * y + 6.0 * p2 * x;
            let b = 2.0 * x * y * dc + 2.0 * p1 * x + 2.0 * p2 * y;
            let d = cd + 2.0 * y * y * dc + 6.0 * p1 * y + 2.0 * p2 * x;
            let det = a * d - b * b;
            let (nx, ny) = (x - (d * ex - b * ey) / det, y - (a * ey - b * ex) / det);
            x = nx;
            y = ny;
        }
        [fx * x + cx, fy * y + cy]
    }

    /// OpenCV's forward model on normalized coordinates.
    pub fn distort_normalized(&self, x: f64, y: f64) -> [f64; 2] {
        let [k1, k2, p1, p2, k3] = self.d;
        let r2 = x * x + y * y;
        let (r4, r6) = (r2 * r2, r2 * r2 * r2);
        let cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let (a1, a2, a3) = (2.0 * x * y, r2 + 2.0 * x * x, r2 + 2.0 * y * y);
        [x * cdist + p1 * a1 + p2 * a2, y * cdist + p1 * a3 + p2 * a1]
    }

    /// The observed pixel of an ideal one: the inverse of [`Lens::undistort`].
    pub fn distort(&self, u: f64, v: f64) -> [f64; 2] {
        let (fx, fy, cx, cy) = (self.k[0][0], self.k[1][1], self.k[0][2], self.k[1][2]);
        let [xd, yd] = self.distort_normalized((u - cx) / fx, (v - cy) / fy);
        [xd * fx + cx, yd * fy + cy]
    }

    /// Whether the radial model keeps increasing out to 1.3 x the picture's half-diagonal.
    /// A fitted polynomial can turn back near the edges, and undistortion then folds the
    /// picture onto itself: refuse such a calibration.
    pub fn monotone(&self, width: usize, height: usize) -> bool {
        let [k1, k2, _, _, k3] = self.d;
        let rmax = 1.3 * ((width as f64 / 2.0).powi(2) + (height as f64 / 2.0).powi(2)).sqrt() / self.k[0][0];
        let rd = |r: f64| r * (1.0 + k1 * r * r + k2 * r.powi(4) + k3 * r.powi(6));
        (0..400).map(|i| rmax * i as f64 / 399.0).collect::<Vec<_>>().windows(2).all(|w| rd(w[1]) > rd(w[0]))
    }
}

/// How far the floor geometry can be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CalibState {
    /// The camera sees what it saw at calibration.
    Ok,
    /// It moved, and the move was measured: `G` corrects it.
    Corrected,
    /// The view could not be confirmed: no floor rule may decide anything.
    Unknown,
}

/// What a drift check found: the homography from the current undistorted view to the
/// reference one, and whether enough features agreed on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftEstimate {
    pub valid: bool,
    #[serde(rename = "G")]
    pub g: Option<Mat3>,
    /// The median shift of the reference features, in pixels.
    pub drift_px: Option<f64>,
    pub inliers: usize,
    pub ratio: f64,
}

/// One camera's floor calibration and its drift state.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub lens: Lens,
    /// Undistorted pixels to floor metres, fitted at installation.
    pub h: Mat3,
    /// The drift correction, current view to reference view.
    pub g: Mat3,
    pub state: CalibState,
    /// Below this median shift (pixels) the camera has not moved.
    pub ok_px: f64,
    hg: Mat3,
}

impl Calibration {
    pub fn new(lens: Lens, h: Mat3, ok_px: f64) -> Self {
        Calibration { lens, h, g: IDENTITY, state: CalibState::Ok, ok_px, hg: h }
    }

    /// A pixel of the current frame on the floor, in metres.
    #[inline]
    pub fn to_floor(&self, u: f64, v: f64) -> [f64; 2] {
        let [x, y] = self.lens.undistort(u, v);
        apply_h(&self.hg, x, y)
    }

    /// Take a drift check's result. `proofs/Grammar.lean`: `gate`.
    pub fn apply_drift(&mut self, est: &DriftEstimate) {
        match (est.valid, est.drift_px, est.g) {
            (true, Some(d), _) if d <= self.ok_px => {
                self.state = CalibState::Ok;
                self.g = IDENTITY;
            }
            (true, Some(_), Some(g)) => {
                self.state = CalibState::Corrected;
                self.g = g;
            }
            _ => self.state = CalibState::Unknown, // keep the last G: it is only used again once trusted
        }
        self.hg = matmul(&self.h, &self.g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide() -> Lens {
        Lens { k: [[230.0, 0.0, 320.0], [0.0, 230.0, 240.0], [0.0, 0.0, 1.0]], d: [-0.32, 0.10, 0.0005, -0.0004, 0.0] }
    }

    #[test]
    fn undistort_inverts_distort_over_the_picture() {
        let l = wide();
        for &(u, v) in &[(320.0, 240.0), (5.0, 5.0), (635.0, 470.0), (100.0, 400.0), (600.0, 30.0)] {
            let [x, y] = l.undistort(u, v);
            let [u2, v2] = l.distort(x, y);
            assert!((u2 - u).abs() < 1e-6 && (v2 - v).abs() < 1e-6, "{u},{v} -> {x},{y} -> {u2},{v2}");
        }
    }

    #[test]
    fn a_folding_model_is_refused() {
        assert!(wide().monotone(640, 480));
        let bad = Lens { d: [-0.6, 0.0, 0.0, 0.0, 0.0], ..wide() };
        assert!(!bad.monotone(640, 480));
    }

    #[test]
    fn drift_gate() {
        let mut c = Calibration::new(wide(), IDENTITY, 3.0);
        let g = [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        c.apply_drift(&DriftEstimate { valid: true, g: Some(g), drift_px: Some(5.0), inliers: 900, ratio: 0.8 });
        assert_eq!(c.state, CalibState::Corrected);
        c.apply_drift(&DriftEstimate { valid: false, g: None, drift_px: None, inliers: 8, ratio: 0.1 });
        assert_eq!(c.state, CalibState::Unknown);
        assert_eq!(c.g, g);
        c.apply_drift(&DriftEstimate { valid: true, g: Some(g), drift_px: Some(0.5), inliers: 900, ratio: 0.9 });
        assert_eq!((c.state, c.g), (CalibState::Ok, IDENTITY));
    }
}
