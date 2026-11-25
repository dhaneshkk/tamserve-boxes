//! Small fixed-size geometry: homographies (with Jacobian), 3x4 camera matrices, polygons, capsules.
//! Everything is on the stack; nothing here allocates.

pub type P2 = (f64, f64);
pub type M2 = [[f64; 2]; 2];
pub type M3 = [[f64; 3]; 3];

pub fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn inv3(m: &M3) -> Option<M3> {
    let d = det3(m);
    if d.abs() < 1e-12 { return None; }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
        [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
        [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
    ])
}

pub fn mul3v(m: &M3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Image pixel -> deck metres. `None` on the horizon (w = 0). A fitted homography's overall sign
/// is arbitrary (-H is the same map), so negative w is fine; which pixels are on a deck is the
/// deck's image polygon's job.
pub fn h_apply(h: &M3, p: P2) -> Option<P2> {
    let [x, y, w] = mul3v(h, [p.0, p.1, 1.0]);
    if w.abs() <= 1e-12 { return None; }
    Some((x / w, y / w))
}

/// Jacobian d(X,Y)/d(u,v) of the homography at pixel p.
pub fn h_jacobian(h: &M3, p: P2) -> Option<M2> {
    let [x, y, w] = mul3v(h, [p.0, p.1, 1.0]);
    if w.abs() <= 1e-12 { return None; }
    let w2 = w * w;
    Some([
        [(h[0][0] * w - x * h[2][0]) / w2, (h[0][1] * w - x * h[2][1]) / w2],
        [(h[1][0] * w - y * h[2][0]) / w2, (h[1][1] * w - y * h[2][1]) / w2],
    ])
}

/// J diag(sx^2, sy^2) J^T: pixel noise pushed onto the floor.
pub fn propagate(j: &M2, sx: f64, sy: f64) -> M2 {
    let (a, b) = (sx * sx, sy * sy);
    let e = |r: usize, c: usize| j[r][0] * a * j[c][0] + j[r][1] * b * j[c][1];
    [[e(0, 0), e(0, 1)], [e(1, 0), e(1, 1)]]
}

/// Largest eigenvalue of a symmetric 2x2.
pub fn lambda_max(m: &M2) -> f64 {
    let (a, b, d) = (m[0][0], 0.5 * (m[0][1] + m[1][0]), m[1][1]);
    let tr = 0.5 * (a + d);
    tr + (0.25 * (a - d) * (a - d) + b * b).sqrt()
}

/// Standard deviation of the projection of covariance `m` onto unit direction `u`.
pub fn sigma_along(m: &M2, u: P2) -> f64 {
    (u.0 * u.0 * m[0][0] + u.0 * u.1 * (m[0][1] + m[1][0]) + u.1 * u.1 * m[1][1]).max(0.0).sqrt()
}

pub fn add2(a: &M2, b: &M2) -> M2 { [[a[0][0] + b[0][0], a[0][1] + b[0][1]], [a[1][0] + b[1][0], a[1][1] + b[1][1]]] }

/// Plane Z = z0 of a 3x4 camera P: the image->plane homography (inverse of [p1 p2 z0*p3+p4]).
pub fn plane_homography(p: &[[f64; 4]; 3], z0: f64) -> Option<M3> {
    let g: M3 = [0, 1, 2].map(|r| [p[r][0], p[r][1], z0 * p[r][2] + p[r][3]]);
    inv3(&g)
}

/// Back-project pixel `px` through P and intersect the ray with the horizontal plane Z = z.
pub fn ray_at_height(p: &[[f64; 4]; 3], px: P2, z: f64) -> Option<P2> {
    let m: M3 = [0, 1, 2].map(|r| [p[r][0], p[r][1], p[r][2]]);
    let mi = inv3(&m)?;
    let c = mul3v(&mi, [-p[0][3], -p[1][3], -p[2][3]]); // camera centre
    let d = mul3v(&mi, [px.0, px.1, 1.0]);
    if d[2].abs() < 1e-12 { return None; }
    let s = (z - c[2]) / d[2];
    Some((c[0] + s * d[0], c[1] + s * d[1]))
}


/// OpenCV's pinhole + `k1 k2 p1 p2 k3` lens. `undistort` is tamserve-boxes' `lens::Lens::undistort`
/// (Newton on the forward model to 1e-9 px), so both put a pixel at the same floor point.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Lens {
    #[serde(rename = "K")] pub k: M3,
    #[serde(rename = "D")] pub d: [f64; 5],
}

impl Lens {
    pub fn undistort(&self, u: f64, v: f64) -> P2 {
        if self.d.iter().all(|&c| c == 0.0) { return (u, v); }
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
            if (tx * tx + ty * ty).sqrt() < 1e-9 { break; }
            let a = cd + 2.0 * x * x * dc + 2.0 * p1 * y + 6.0 * p2 * x;
            let b = 2.0 * x * y * dc + 2.0 * p1 * x + 2.0 * p2 * y;
            let d = cd + 2.0 * y * y * dc + 6.0 * p1 * y + 2.0 * p2 * x;
            let det = a * d - b * b;
            let (nx, ny) = (x - (d * ex - b * ey) / det, y - (a * ey - b * ex) / det);
            x = nx;
            y = ny;
        }
        (fx * x + cx, fy * y + cy)
    }
    /// The observed pixel of an ideal one (OpenCV's forward model).
    pub fn distort(&self, u: f64, v: f64) -> P2 {
        let (fx, fy, cx, cy) = (self.k[0][0], self.k[1][1], self.k[0][2], self.k[1][2]);
        let [k1, k2, p1, p2, k3] = self.d;
        let (x, y) = ((u - cx) / fx, (v - cy) / fy);
        let r2 = x * x + y * y;
        let c = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let (xd, yd) = (x * c + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), y * c + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
        (xd * fx + cx, yd * fy + cy)
    }
}

pub fn matmul3(a: &M3, b: &M3) -> M3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}
pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Ray-casting point-in-polygon (same rule as iogp-editor's ingest).
pub fn in_polygon(poly: &[[f64; 2]], p: P2) -> bool {
    let n = poly.len();
    if n < 3 { return false; }
    let (mut c, mut j) = (false, n - 1);
    for i in 0..n {
        let (xi, yi, xj, yj) = (poly[i][0], poly[i][1], poly[j][0], poly[j][1]);
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi { c = !c; }
        j = i;
    }
    c
}

pub fn seg_dist(p: P2, a: P2, b: P2) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 < 1e-18 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) };
    ((p.0 - a.0 - t * dx).powi(2) + (p.1 - a.1 - t * dy).powi(2)).sqrt()
}

/// Distance from p to the polygon's boundary.
pub fn boundary_dist(poly: &[[f64; 2]], p: P2) -> f64 {
    let n = poly.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let j = (i + 1) % n;
        best = best.min(seg_dist(p, (poly[i][0], poly[i][1]), (poly[j][0], poly[j][1])));
    }
    best
}

/// Distance from p to a polyline (the axis of a capsule).
pub fn polyline_dist(pts: &[P2], p: P2) -> f64 {
    match pts.len() {
        0 => f64::INFINITY,
        1 => ((p.0 - pts[0].0).powi(2) + (p.1 - pts[0].1).powi(2)).sqrt(),
        _ => pts.windows(2).map(|w| seg_dist(p, w[0], w[1])).fold(f64::INFINITY, f64::min),
    }
}

pub type BBox = [f64; 4];
pub fn foot(b: &BBox) -> P2 { ((b[0] + b[2]) / 2.0, b[3]) }
pub fn centre(b: &BBox) -> P2 { ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) }
pub fn inside(b: &BBox, p: P2) -> bool { p.0 >= b[0] && p.0 <= b[2] && p.1 >= b[1] && p.1 <= b[3] }
pub fn area(b: &BBox) -> f64 { (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0) }
pub fn inter(a: &BBox, b: &BBox) -> f64 { area(&[a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn jacobian_matches_finite_difference() {
        let h: M3 = [[0.02, 0.001, -3.0], [0.0005, 0.03, -5.0], [0.0001, 0.0008, 1.0]];
        let p = (640.0, 420.0);
        let j = h_jacobian(&h, p).unwrap();
        let e = 1e-3;
        let (a, b) = (h_apply(&h, (p.0 + e, p.1)).unwrap(), h_apply(&h, (p.0 - e, p.1)).unwrap());
        assert!(((a.0 - b.0) / (2.0 * e) - j[0][0]).abs() < 1e-7);
        let (a, b) = (h_apply(&h, (p.0, p.1 + e)).unwrap(), h_apply(&h, (p.0, p.1 - e)).unwrap());
        assert!(((a.1 - b.1) / (2.0 * e) - j[1][1]).abs() < 1e-7);
    }
    #[test]
    fn negated_homography_is_the_same_map() {
        let h: M3 = [[0.02, 0.001, -3.0], [0.0005, 0.03, -5.0], [0.0001, 0.0008, 1.0]];
        let n: M3 = h.map(|r| r.map(|v| -v));
        let p = (640.0, 420.0);
        let (a, b) = (h_apply(&h, p).unwrap(), h_apply(&n, p).unwrap());
        assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
        let (ja, jb) = (h_jacobian(&h, p).unwrap(), h_jacobian(&n, p).unwrap());
        assert!((ja[0][0] - jb[0][0]).abs() < 1e-12 && (ja[1][1] - jb[1][1]).abs() < 1e-12);
    }
    #[test]
    fn plane_homography_inverts_projection() {
        // camera 10 m up looking down-forward
        let p = [[800.0, 0.0, 640.0, 0.0], [0.0, -400.0, 360.0 + 300.0, -4000.0], [0.0, 0.5, 0.866, 10.0]];
        for z0 in [0.0, 4.5] {
            let h = plane_homography(&p, z0).unwrap();
            let w = [3.0, 7.0, z0, 1.0];
            let q = [0, 1, 2].map(|r| (0..4).map(|c| p[r][c] * w[c]).sum::<f64>());
            let px = (q[0] / q[2], q[1] / q[2]);
            let back = h_apply(&h, px).unwrap();
            assert!((back.0 - 3.0).abs() < 1e-6 && (back.1 - 7.0).abs() < 1e-6);
            let r = ray_at_height(&p, px, z0).unwrap();
            assert!((r.0 - 3.0).abs() < 1e-6 && (r.1 - 7.0).abs() < 1e-6);
        }
    }
    #[test]
    fn undistort_inverts_distort() {
        let l = Lens { k: [[230.0, 0.0, 320.0], [0.0, 230.0, 240.0], [0.0, 0.0, 1.0]], d: [-0.32, 0.10, 0.0005, -0.0004, 0.0] };
        for &(u, v) in &[(320.0, 240.0), (5.0, 5.0), (635.0, 470.0), (100.0, 400.0)] {
            let (x, y) = l.undistort(u, v);
            let (u2, v2) = l.distort(x, y);
            assert!((u2 - u).abs() < 1e-6 && (v2 - v).abs() < 1e-6);
        }
    }
}
