//! Site calibration from measurements: a worksheet of floor marks in → a site file out.
//!
//! On the rig, for each camera and each deck it sees, put down (or pick existing) marks whose
//! deck position is known in metres: deck plate corners, painted lines, bolt heads, from the rig
//! drawings or a tape. Read each mark's pixel off a still of the camera's detection stream (the
//! same resolution tamserve-boxes decodes). Four marks fit a homography; six to ten spread over
//! the deck give a check on it. For drop zones under suspended loads, also mark points at known
//! heights (a handrail top, a column band, a container corner) to fit the 3x4 camera matrix.
//!
//! ```json
//! { "cameras": [ {
//!     "name": "CAM-03", "image_size": [640, 360],
//!     "lens": {"K": [[...]], "D": [...]},                       // optional; marks are then raw pixels
//!     "decks": [ { "id": "drill-floor", "elevation_m": 0.0,
//!                  "image_polygon": [[u,v], ...],               // where this deck is in the picture
//!                  "marks": [ {"px": [u, v], "m": [x, y]}, ... ] } ],
//!     "height_marks": [ {"px": [u, v], "m": [x, y, z]}, ... ],  // optional, >= 6, two heights or more
//!     "occluders": [ [[u,v], ...] ],
//!     "zones": [ { "id": "Rig-Floor-East", "type": "height-zone", "deck": "drill-floor",
//!                  "floor_polygon": [[x,y], ...] }               // metres, from the drawings, or
//!              /* "image_polygon": [[u,v], ...]  drawn on the still: mapped through the deck's H */ ] } ] }
//! ```
//!
//! Every fit is checked: the RMS and worst residual of the marks, in pixels and metres, must stay
//! under the limits, or the camera is refused with the marks that disagree.

use crate::geom::{h_apply, in_polygon, Lens, M3, P2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Worksheet {
    #[serde(default, rename = "_comment")] pub comment: Option<serde_json::Value>,
    pub cameras: Vec<WsCamera>,
    #[serde(default)]
    pub limits: Limits,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// worst floor residual of a deck mark, metres
    pub max_floor_m: f64,
    /// worst image residual of a height mark, pixels
    pub max_px: f64,
}
impl Default for Limits { fn default() -> Self { Limits { max_floor_m: 0.15, max_px: 3.0 } } }

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WsCamera {
    pub name: String,
    #[serde(default)] pub description: Option<String>,
    #[serde(default)] pub image_size: Option<[f64; 2]>,
    #[serde(default)] pub lens: Option<Lens>,
    #[serde(default)] pub decks: Vec<WsDeck>,
    #[serde(default)] pub height_marks: Vec<Mark3>,
    #[serde(default)] pub occluders: Vec<Vec<[f64; 2]>>,
    #[serde(default)] pub zones: Vec<WsZone>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WsDeck {
    pub id: String,
    #[serde(default)] pub note: Option<String>,
    #[serde(default)] pub elevation_m: f64,
    pub image_polygon: Vec<[f64; 2]>,
    #[serde(default)] pub marks: Vec<Mark2>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mark2 { pub px: [f64; 2], pub m: [f64; 2], #[serde(default)] pub note: Option<String> }

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mark3 { pub px: [f64; 2], pub m: [f64; 3] }

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WsZone {
    pub id: String,
    #[serde(default)] pub note: Option<String>,
    #[serde(rename = "type")] pub zone: String,
    pub deck: String,
    #[serde(default)] pub floor_polygon: Option<Vec<[f64; 2]>>,
    #[serde(default)] pub image_polygon: Option<Vec<[f64; 2]>>,
}

/// How well a camera fitted.
#[derive(Clone, Debug, Serialize)]
pub struct FitReport {
    pub camera: String,
    pub decks: Vec<DeckFit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection: Option<ProjFit>,
}
#[derive(Clone, Debug, Serialize)]
pub struct DeckFit { pub deck: String, pub marks: usize, pub rms_m: f64, pub worst_m: f64, pub worst_mark: usize }
#[derive(Clone, Debug, Serialize)]
pub struct ProjFit { pub marks: usize, pub rms_px: f64, pub worst_px: f64, pub worst_mark: usize, pub decks_agree_m: f64 }

// ------------------------------------------------------------------ linear algebra

/// Least squares A x = b by the normal equations (n <= 12), Gaussian elimination with pivoting.
fn lstsq(rows: &[(Vec<f64>, f64)], n: usize) -> Option<Vec<f64>> {
    let mut m = vec![vec![0.0; n + 1]; n];
    for (a, b) in rows {
        for i in 0..n {
            for j in 0..n { m[i][j] += a[i] * a[j]; }
            m[i][n] += a[i] * b;
        }
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-12 { return None; }
        m.swap(c, p);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..=n { m[r][k] -= f * m[c][k]; }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// Hartley normalisation: centroid to the origin, mean distance sqrt(2) (or sqrt(3) in 3D).
fn normaliser<const D: usize>(pts: &[[f64; D]]) -> ([f64; D], f64) {
    let n = pts.len() as f64;
    let mut c = [0.0; D];
    for p in pts { for k in 0..D { c[k] += p[k] / n; } }
    let mean = pts.iter().map(|p| (0..D).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>().sqrt()).sum::<f64>() / n;
    (c, if mean < 1e-12 { 1.0 } else { (D as f64).sqrt() / mean })
}

/// Homography image -> floor from >= 4 pairs (normalised inhomogeneous DLT).
pub fn fit_homography(px: &[P2], m: &[P2]) -> Option<M3> {
    if px.len() < 4 || px.len() != m.len() { return None; }
    let a: Vec<[f64; 2]> = px.iter().map(|p| [p.0, p.1]).collect();
    let b: Vec<[f64; 2]> = m.iter().map(|p| [p.0, p.1]).collect();
    let ((ca, sa), (cb, sb)) = (normaliser(&a), normaliser(&b));
    let mut rows = vec![];
    for (p, q) in a.iter().zip(&b) {
        let (x, y) = ((p[0] - ca[0]) * sa, (p[1] - ca[1]) * sa);
        let (u, v) = ((q[0] - cb[0]) * sb, (q[1] - cb[1]) * sb);
        rows.push((vec![x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u));
        rows.push((vec![0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v));
    }
    let h = lstsq(&rows, 8)?;
    let hn: M3 = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]];
    let ta: M3 = [[sa, 0.0, -sa * ca[0]], [0.0, sa, -sa * ca[1]], [0.0, 0.0, 1.0]];
    let tbi: M3 = [[1.0 / sb, 0.0, cb[0]], [0.0, 1.0 / sb, cb[1]], [0.0, 0.0, 1.0]];
    Some(mul(&mul(&tbi, &hn), &ta))
}

/// 3x4 camera matrix world -> image from >= 6 pairs not all on one plane.
pub fn fit_projection(px: &[P2], w: &[[f64; 3]]) -> Option<[[f64; 4]; 3]> {
    if px.len() < 6 || px.len() != w.len() { return None; }
    let a: Vec<[f64; 2]> = px.iter().map(|p| [p.0, p.1]).collect();
    let ((ca, sa), (cw, sw)) = (normaliser(&a), normaliser(w));
    let mut rows = vec![];
    for (p, q) in a.iter().zip(w) {
        let (u, v) = ((p[0] - ca[0]) * sa, (p[1] - ca[1]) * sa);
        let (x, y, z) = ((q[0] - cw[0]) * sw, (q[1] - cw[1]) * sw, (q[2] - cw[2]) * sw);
        rows.push((vec![x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z], u));
        rows.push((vec![0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z], v));
    }
    let p = lstsq(&rows, 11)?;
    let pn = [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]], [p[8], p[9], p[10], 1.0]];
    // P = Ta^-1 Pn Tw
    let tai: M3 = [[1.0 / sa, 0.0, ca[0]], [0.0, 1.0 / sa, ca[1]], [0.0, 0.0, 1.0]];
    let tw = [[sw, 0.0, 0.0, -sw * cw[0]], [0.0, sw, 0.0, -sw * cw[1]], [0.0, 0.0, sw, -sw * cw[2]], [0.0, 0.0, 0.0, 1.0]];
    let mut pt = [[0.0; 4]; 3];
    for r in 0..3 { for c in 0..4 { pt[r][c] = (0..4).map(|k| pn[r][k] * tw[k][c]).sum(); } }
    let mut out = [[0.0; 4]; 3];
    for r in 0..3 { for c in 0..4 { out[r][c] = (0..3).map(|k| tai[r][k] * pt[k][c]).sum(); } }
    Some(out)
}

fn mul(a: &M3, b: &M3) -> M3 { [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())) }

pub fn project(p: &[[f64; 4]; 3], w: [f64; 3]) -> Option<P2> {
    let q = [0, 1, 2].map(|r| p[r][0] * w[0] + p[r][1] * w[1] + p[r][2] * w[2] + p[r][3]);
    if q[2].abs() < 1e-12 { return None; }
    Some((q[0] / q[2], q[1] / q[2]))
}

// ------------------------------------------------------------------ worksheet -> site

/// Fit every camera of the worksheet; returns the site file's JSON and one report per camera.
/// `catalog`: iogp-editor's catalog JSON, to check that each zone instance it names is on its camera.
pub fn build_site(ws: &Worksheet, catalog: Option<&serde_json::Value>) -> Result<(serde_json::Value, Vec<FitReport>), Vec<String>> {
    let mut errors = vec![];
    let mut cams = vec![];
    let mut reports = vec![];
    for c in &ws.cameras {
        match fit_camera(c, &ws.limits) {
            Ok((v, r)) => { cams.push(v); reports.push(r); }
            Err(mut e) => errors.append(&mut e),
        }
    }
    if let Some(cat) = catalog {
        for zi in cat["zone_instances"].as_array().into_iter().flatten() {
            let (name, zone, cam) = (zi["name"].as_str().unwrap_or(""), zi["zone"].as_str().unwrap_or(""), zi["camera"].as_str());
            let Some(cam) = cam else { continue };
            match ws.cameras.iter().find(|c| c.name == cam) {
                None => errors.push(format!("catalog zone {name} is on {cam}, which the worksheet does not have")),
                Some(c) => match c.zones.iter().find(|z| z.id == name) {
                    None => errors.push(format!("{cam}: catalog zone {name} ({zone}) is not drawn")),
                    Some(z) if z.zone != zone => errors.push(format!("{cam}/{name}: type {} but the catalog says {zone}", z.zone)),
                    _ => {}
                },
            }
        }
        let known: Vec<&str> = cat["zones"].as_array().into_iter().flatten().filter_map(|z| z["noun"].as_str()).collect();
        for c in &ws.cameras { for z in &c.zones {
            if !known.contains(&z.zone.as_str()) { errors.push(format!("{}/{}: zone type {} is not in the catalog", c.name, z.id, z.zone)); }
        } }
    }
    if errors.is_empty() { Ok((serde_json::json!({ "cameras": cams }), reports)) } else { Err(errors) }
}

fn fit_camera(c: &WsCamera, lim: &Limits) -> Result<(serde_json::Value, FitReport), Vec<String>> {
    let mut errs = vec![];
    let und = |p: [f64; 2]| -> P2 { match &c.lens { Some(l) => l.undistort(p[0], p[1]), None => (p[0], p[1]) } };
    let mut decks_out = vec![];
    let mut fits = vec![];
    let mut hs: Vec<(String, M3)> = vec![];
    if c.decks.is_empty() { errs.push(format!("{}: no decks", c.name)); }
    for d in &c.decks {
        let tag = format!("{}/{}", c.name, d.id);
        if d.image_polygon.len() < 3 { errs.push(format!("{tag}: image_polygon needs at least 3 points")); continue; }
        if d.marks.len() < 4 { errs.push(format!("{tag}: {} floor marks, need at least 4 (6-10 spread over the deck to check the fit)", d.marks.len())); continue; }
        for (i, mk) in d.marks.iter().enumerate() {
            if !in_polygon(&d.image_polygon, (mk.px[0], mk.px[1])) { errs.push(format!("{tag}: mark {i} at {:?} px is outside the deck's image_polygon", mk.px)); }
        }
        let px: Vec<P2> = d.marks.iter().map(|m| und(m.px)).collect();
        let fl: Vec<P2> = d.marks.iter().map(|m| (m.m[0], m.m[1])).collect();
        let Some(h) = fit_homography(&px, &fl) else { errs.push(format!("{tag}: the marks are degenerate (three or more on one line?)")); continue };
        let res: Vec<f64> = px.iter().zip(&fl).map(|(p, q)| h_apply(&h, *p).map_or(f64::INFINITY, |r| ((r.0 - q.0).powi(2) + (r.1 - q.1).powi(2)).sqrt())).collect();
        let (worst_mark, worst) = res.iter().cloned().enumerate().fold((0, 0.0), |b, (i, r)| if r > b.1 { (i, r) } else { b });
        let rms = (res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64).sqrt();
        if d.marks.len() == 4 { errs.push(format!("{tag}: only 4 marks fit exactly and cannot be checked; add 2 or more")); }
        if worst > lim.max_floor_m { errs.push(format!("{tag}: mark {worst_mark} is {worst:.3} m off the fit (limit {} m): re-measure it or its pixel", lim.max_floor_m)); }
        fits.push(DeckFit { deck: d.id.clone(), marks: d.marks.len(), rms_m: rms, worst_m: worst, worst_mark });
        hs.push((d.id.clone(), h));
        decks_out.push(serde_json::json!({ "id": d.id, "elevation_m": d.elevation_m, "image_polygon": d.image_polygon, "homography": h }));
    }
    let mut projection = None;
    let mut proj_fit = None;
    if !c.height_marks.is_empty() {
        let px: Vec<P2> = c.height_marks.iter().map(|m| und(m.px)).collect();
        let w: Vec<[f64; 3]> = c.height_marks.iter().map(|m| m.m).collect();
        let zs: Vec<f64> = w.iter().map(|p| p[2]).collect();
        let spread = zs.iter().cloned().fold(f64::MIN, f64::max) - zs.iter().cloned().fold(f64::MAX, f64::min);
        if c.height_marks.len() < 6 { errs.push(format!("{}: {} height marks, need at least 6", c.name, c.height_marks.len())); }
        else if spread < 1.0 { errs.push(format!("{}: height marks span {spread:.2} m in height; need at least 1 m (two or more levels)", c.name)); }
        else if let Some(p) = fit_projection(&px, &w) {
            let res: Vec<f64> = px.iter().zip(&w).map(|(q, x)| project(&p, *x).map_or(f64::INFINITY, |r| ((r.0 - q.0).powi(2) + (r.1 - q.1).powi(2)).sqrt())).collect();
            let (worst_mark, worst) = res.iter().cloned().enumerate().fold((0, 0.0), |b, (i, r)| if r > b.1 { (i, r) } else { b });
            let rms = (res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64).sqrt();
            if worst > lim.max_px { errs.push(format!("{}: height mark {worst_mark} is {worst:.1} px off the camera fit (limit {} px)", c.name, lim.max_px)); }
            // the deck homographies and P must describe the same floor: compare on each deck's marks
            let mut agree: f64 = 0.0;
            for d in &c.decks {
                let Some((_, h)) = hs.iter().find(|x| x.0 == d.id) else { continue };
                if let Some(hp) = crate::geom::plane_homography(&p, d.elevation_m) {
                    for mk in &d.marks {
                        if let (Some(a), Some(b)) = (h_apply(h, und(mk.px)), h_apply(&hp, und(mk.px))) {
                            agree = agree.max(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt());
                        }
                    }
                }
            }
            if agree > lim.max_floor_m * 2.0 { errs.push(format!("{}: the camera matrix and the deck homographies disagree by {agree:.2} m: are the height marks and floor marks in the same coordinates?", c.name)); }
            projection = Some(p);
            proj_fit = Some(ProjFit { marks: c.height_marks.len(), rms_px: rms, worst_px: worst, worst_mark, decks_agree_m: agree });
        } else { errs.push(format!("{}: the height marks are degenerate", c.name)); }
    }
    let mut zones_out = vec![];
    for z in &c.zones {
        let tag = format!("{}/{}", c.name, z.id);
        if z.floor_polygon.is_some() == z.image_polygon.is_some() {
            errs.push(format!("{tag}: give floor_polygon (metres) or image_polygon (pixels), one of them"));
            continue;
        }
        let Some((_, h)) = hs.iter().find(|x| x.0 == z.deck) else {
            if !c.decks.iter().any(|d| d.id == z.deck) { errs.push(format!("{tag}: unknown deck {}", z.deck)); }
            continue;
        };
        let floor: Vec<[f64; 2]> = match (&z.floor_polygon, &z.image_polygon) {
            (Some(f), None) => f.clone(),
            (None, Some(ip)) => match ip.iter().map(|p| h_apply(h, und(*p)).map(|q| [q.0, q.1])).collect::<Option<Vec<_>>>() {
                Some(v) => v,
                None => { errs.push(format!("{tag}: image_polygon reaches the horizon")); continue; }
            },
            _ => { errs.push(format!("{tag}: give floor_polygon (metres) or image_polygon (pixels), one of them")); continue; }
        };
        if floor.len() < 3 { errs.push(format!("{tag}: polygon needs at least 3 points")); continue; }
        zones_out.push(serde_json::json!({ "id": z.id, "type": z.zone, "deck": z.deck, "floor_polygon": floor }));
    }
    if !errs.is_empty() { return Err(errs); }
    let mut v = serde_json::json!({ "name": c.name, "decks": decks_out, "occluders": c.occluders, "zones": zones_out });
    if let Some(s) = c.image_size { v["image_size"] = serde_json::json!(s); }
    if let Some(l) = &c.lens { v["lens"] = serde_json::to_value(l).unwrap(); }
    if let Some(p) = projection { v["projection"] = serde_json::json!(p); }
    Ok((v, FitReport { camera: c.name.clone(), decks: fits, projection: proj_fit }))
}
