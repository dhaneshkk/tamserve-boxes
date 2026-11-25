#![allow(dead_code)]
//! A synthetic camera: 10 m up, 8 m behind the deck origin, looking at (0, 10, 0).
use vg360_scenegraph::*;

pub fn projection() -> [[f64; 4]; 3] {
    let c: [f64; 3] = [0.0, -8.0, 10.0];
    let f = { let v = [0.0 - c[0], 10.0 - c[1], 0.0 - c[2]]; let n = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt(); [v[0]/n, v[1]/n, v[2]/n] };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    let r = cross(f, [0.0, 0.0, 1.0]);
    let rn = (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt();
    let r = [r[0]/rn, r[1]/rn, r[2]/rn];
    let d = cross(f, r);
    let rot = [r, d, f];
    let t = [0, 1, 2].map(|i| -(rot[i][0]*c[0] + rot[i][1]*c[1] + rot[i][2]*c[2]));
    let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]];
    let mut p = [[0.0; 4]; 3];
    for i in 0..3 { for j in 0..4 {
        p[i][j] = (0..3).map(|m| k[i][m] * if j < 3 { rot[m][j] } else { t[m] }).sum();
    } }
    p
}

pub fn project(p: &[[f64; 4]; 3], w: [f64; 3]) -> (f64, f64) {
    let q = [0, 1, 2].map(|r| p[r][0]*w[0] + p[r][1]*w[1] + p[r][2]*w[2] + p[r][3]);
    (q[0] / q[2], q[1] / q[2])
}

/// Pixel box of an upright object of `width` x `height` metres standing on the deck at (x, y),
/// optionally floating `z0` metres up.
pub fn bbox(p: &[[f64; 4]; 3], x: f64, y: f64, z0: f64, width: f64, height: f64) -> [f64; 4] {
    let pts = [[x - width/2.0, y, z0], [x + width/2.0, y, z0], [x - width/2.0, y, z0 + height], [x + width/2.0, y, z0 + height], [x, y, z0], [x, y, z0 + height]];
    let px: Vec<(f64, f64)> = pts.iter().map(|w| project(p, *w)).collect();
    let foot = px[4];
    let top = px[5];
    let (xmin, xmax) = px.iter().fold((f64::MAX, f64::MIN), |(a, b), q| (a.min(q.0), b.max(q.0)));
    [xmin, top.1, xmax, foot.1]
}

pub fn site_json(occluders: &str) -> String {
    let p = projection();
    format!(r#"{{"cameras": [{{
        "name": "CAM-02", "image_size": [1280, 720], "projection": {proj},
        "occluders": {occluders},
        "decks": [{{"id": "main", "elevation_m": 0.0, "image_polygon": [[0,0],[1280,0],[1280,720],[0,720]]}}],
        "zones": [
          {{"id": "Guard-1", "type": "machine-guard-zone", "deck": "main", "floor_polygon": [[-3,8],[3,8],[3,14],[-3,14]]}},
          {{"id": "Rig-Floor-East", "type": "height-zone", "deck": "main", "floor_polygon": [[4,8],[10,8],[10,14],[4,14]]}}
        ]}}]}}"#, proj = serde_json::to_string(&p).unwrap())
}

pub fn fixture_config() -> String {
    include_str!("../fixtures/config.json").to_string()
}

pub fn builder(occluders: &str, params: Params, engine: &iogp_runtime::Engine) -> SceneBuilder {
    let site = Site::from_json(&site_json(occluders)).unwrap();
    SceneBuilder::new(site.camera("CAM-02").unwrap(), params, Policy::from_rules(&engine.rules))
}
