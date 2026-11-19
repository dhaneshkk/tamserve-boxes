//! ByteTrack's Kalman filter: centre x, y, aspect ratio and height, and their
//! velocities. The same numbers as `tamserve.boxes._Kalman`.

#![allow(clippy::needless_range_loop)] // matrix code reads best with its indices

use crate::BBox;

const POS: f64 = 1.0 / 20.0;
const VEL: f64 = 1.0 / 160.0;

type M8 = [[f64; 8]; 8];

#[derive(Clone, Debug)]
pub(crate) struct Kalman {
    x: [f64; 8],
    p: M8,
}

fn measure(b: &BBox) -> [f64; 4] {
    let (w, h) = ((b[2] - b[0]) as f64, (b[3] - b[1]) as f64);
    [b[0] as f64 + w / 2.0, b[1] as f64 + h / 2.0, w / h.max(1e-6), h.max(1e-6)]
}

fn diag_sq<const N: usize>(v: [f64; N]) -> [[f64; N]; N] {
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        m[i][i] = v[i] * v[i];
    }
    m
}

impl Kalman {
    pub(crate) fn new(b: &BBox) -> Self {
        let z = measure(b);
        let h = z[3];
        Kalman {
            x: [z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0, 0.0],
            p: diag_sq([2.0 * POS * h, 2.0 * POS * h, 1e-2, 2.0 * POS * h,
                        10.0 * VEL * h, 10.0 * VEL * h, 1e-5, 10.0 * VEL * h]),
        }
    }

    /// One frame forward: position += velocity, and the uncertainty grows.
    pub(crate) fn predict(&mut self) {
        let h = self.x[3];
        let q = [POS * h, POS * h, 1e-2, POS * h, VEL * h, VEL * h, 1e-5, VEL * h];
        for i in 0..4 {
            self.x[i] += self.x[i + 4];
        }
        // P = F P Fᵀ + Q, with F = [[I, I], [0, I]]: rows i < 4 gain row i + 4, then columns likewise.
        let p = &mut self.p;
        for i in 0..4 {
            for j in 0..8 {
                p[i][j] += p[i + 4][j];
            }
        }
        for row in p.iter_mut() {
            for j in 0..4 {
                row[j] += row[j + 4];
            }
        }
        for i in 0..8 {
            p[i][i] += q[i] * q[i];
        }
    }

    /// Correct with a box a look found.
    pub(crate) fn update(&mut self, b: &BBox) {
        let h = self.x[3];
        let r = [POS * h, POS * h, 1e-1, POS * h];
        // S = H P Hᵀ + R: the top-left 4 x 4 of P.
        let mut s = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = self.p[i][j];
            }
            s[i][i] += r[i] * r[i];
        }
        // K = P Hᵀ S⁻¹ (8 x 4): solve S Kᵀ = H P, which is the top 4 rows of P.
        let mut hp = [[0.0; 8]; 4];
        hp.copy_from_slice(&self.p[..4]);
        let kt = solve4(s, hp);
        let z = measure(b);
        let y: Vec<f64> = (0..4).map(|i| z[i] - self.x[i]).collect();
        for i in 0..8 {
            self.x[i] += (0..4).map(|k| kt[k][i] * y[k]).sum::<f64>();
        }
        // P -= K H P = K (top 4 rows of P)
        let old = self.p;
        for i in 0..8 {
            for j in 0..8 {
                self.p[i][j] = old[i][j] - (0..4).map(|k| kt[k][i] * old[k][j]).sum::<f64>();
            }
        }
    }

    pub(crate) fn bbox(&self) -> BBox {
        let (cx, cy, a, h) = (self.x[0], self.x[1], self.x[2], self.x[3]);
        let w = a * h;
        [(cx - w / 2.0) as f32, (cy - h / 2.0) as f32, (cx + w / 2.0) as f32, (cy + h / 2.0) as f32]
    }
}

/// X with S X = B, for a symmetric positive definite 4 x 4 S, by Gaussian elimination
/// with partial pivoting.
fn solve4(mut s: [[f64; 4]; 4], mut b: [[f64; 8]; 4]) -> [[f64; 8]; 4] {
    for c in 0..4 {
        let piv = (c..4).max_by(|&i, &j| s[i][c].abs().total_cmp(&s[j][c].abs())).unwrap();
        s.swap(c, piv);
        b.swap(c, piv);
        for r in c + 1..4 {
            let f = s[r][c] / s[c][c];
            for k in c..4 {
                s[r][k] -= f * s[c][k];
            }
            for k in 0..8 {
                b[r][k] -= f * b[c][k];
            }
        }
    }
    for c in (0..4).rev() {
        for k in 0..8 {
            let mut v = b[c][k];
            for j in c + 1..4 {
                v -= s[c][j] * b[j][k];
            }
            b[c][k] = v / s[c][c];
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_moving_steadily_is_predicted_ahead() {
        let mut k = Kalman::new(&[10.0, 10.0, 40.0, 90.0]);
        for t in 1..20 {
            k.predict();
            let d = 4.0 * t as f32;
            k.update(&[10.0 + d, 10.0, 40.0 + d, 90.0]);
        }
        k.predict();
        let b = k.bbox();
        assert!((b[0] - (10.0 + 80.0)).abs() < 1.5, "{b:?}");
        assert!((b[3] - 90.0).abs() < 0.5);
    }
}
