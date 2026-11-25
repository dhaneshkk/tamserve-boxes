//! Constant-velocity Kalman filter in deck coordinates (metres), fixed-size, no allocation.

use crate::geom::{M2, P2};

#[derive(Clone, Copy, Debug)]
pub struct Kf { pub x: [f64; 4], pub p: [[f64; 4]; 4] }

impl Kf {
    pub fn new(z: P2, r: &M2) -> Kf {
        let mut p = [[0.0; 4]; 4];
        p[0][0] = r[0][0]; p[0][1] = r[0][1]; p[1][0] = r[1][0]; p[1][1] = r[1][1];
        p[2][2] = 4.0; p[3][3] = 4.0; // (2 m/s)^2 prior on velocity
        Kf { x: [z.0, z.1, 0.0, 0.0], p }
    }
    pub fn pos(&self) -> P2 { (self.x[0], self.x[1]) }
    pub fn pos_cov(&self) -> M2 { [[self.p[0][0], self.p[0][1]], [self.p[1][0], self.p[1][1]]] }

    /// Predict by dt with white-acceleration noise of std `sa` (m/s^2).
    pub fn predict(&mut self, dt: f64, sa: f64) {
        if dt <= 0.0 { return; }
        let x = &mut self.x;
        x[0] += dt * x[2];
        x[1] += dt * x[3];
        // P = F P F^T + Q, F = [[I, dt I], [0, I]]
        let p = self.p;
        let mut n = [[0.0; 4]; 4];
        for i in 0..4 { for j in 0..4 {
            let f = |r: usize, c: usize| -> f64 { if r == c { 1.0 } else if c == r + 2 { dt } else { 0.0 } };
            let mut s = 0.0;
            for a in 0..4 { for b in 0..4 { s += f(i, a) * p[a][b] * f(j, b); } }
            n[i][j] = s;
        } }
        let q = sa * sa;
        let (dt2, dt3, dt4) = (dt * dt, dt * dt * dt, dt * dt * dt * dt);
        for k in 0..2 {
            n[k][k] += q * dt4 / 4.0;
            n[k][k + 2] += q * dt3 / 2.0;
            n[k + 2][k] += q * dt3 / 2.0;
            n[k + 2][k + 2] += q * dt2;
        }
        self.p = n;
    }

    /// Squared Mahalanobis distance of measurement z (cov r) from the prediction.
    pub fn mahalanobis2(&self, z: P2, r: &M2) -> f64 {
        let s = [[self.p[0][0] + r[0][0], self.p[0][1] + r[0][1]], [self.p[1][0] + r[1][0], self.p[1][1] + r[1][1]]];
        let d = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if d <= 1e-18 { return f64::INFINITY; }
        let (ex, ey) = (z.0 - self.x[0], z.1 - self.x[1]);
        (ex * (s[1][1] * ex - s[0][1] * ey) + ey * (-s[1][0] * ex + s[0][0] * ey)) / d
    }

    pub fn update(&mut self, z: P2, r: &M2) {
        let p = self.p;
        let s = [[p[0][0] + r[0][0], p[0][1] + r[0][1]], [p[1][0] + r[1][0], p[1][1] + r[1][1]]];
        let d = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if d <= 1e-18 { return; }
        let si = [[s[1][1] / d, -s[0][1] / d], [-s[1][0] / d, s[0][0] / d]];
        // K = P H^T S^-1, H = [I 0]
        let mut k = [[0.0; 2]; 4];
        for i in 0..4 { for j in 0..2 { k[i][j] = p[i][0] * si[0][j] + p[i][1] * si[1][j]; } }
        let (ex, ey) = (z.0 - self.x[0], z.1 - self.x[1]);
        for i in 0..4 { self.x[i] += k[i][0] * ex + k[i][1] * ey; }
        let mut n = p;
        for i in 0..4 { for j in 0..4 { n[i][j] = p[i][j] - (k[i][0] * p[0][j] + k[i][1] * p[1][j]); } }
        // keep symmetric
        for i in 0..4 { for j in (i + 1)..4 { let m = 0.5 * (n[i][j] + n[j][i]); n[i][j] = m; n[j][i] = m; } }
        self.p = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converges_on_constant_velocity_and_grows_when_coasting() {
        let r = [[0.04, 0.0], [0.0, 0.04]];
        let mut kf = Kf::new((0.0, 0.0), &r);
        for i in 1..=50 { kf.predict(0.1, 0.5); kf.update((0.1 * i as f64, 0.0), &r); }
        assert!((kf.x[2] - 1.0).abs() < 0.1, "vx = {}", kf.x[2]);
        let before = kf.pos_cov()[0][0];
        for _ in 0..10 { kf.predict(0.1, 0.5); }
        assert!(kf.pos_cov()[0][0] > before);
        assert!((kf.x[0] - 6.0).abs() < 0.3);
    }
}
