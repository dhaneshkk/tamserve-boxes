//! Fitting a site from measured marks: exact on clean marks, refused on bad ones.
mod common;
use common::*;
use vg360_scenegraph::fit::{build_site, Worksheet};
use vg360_scenegraph::geom::h_apply;

fn marks(p: &[[f64; 4]; 3], pts: &[(f64, f64)], noise: f64) -> Vec<serde_json::Value> {
    pts.iter().enumerate().map(|(i, &(x, y))| {
        let q = project(p, [x, y, 0.0]);
        let j = if i % 2 == 0 { noise } else { -noise };
        serde_json::json!({"px": [q.0 + j, q.1 - j], "m": [x, y]})
    }).collect()
}

fn worksheet(floor_marks: Vec<serde_json::Value>, height: Vec<serde_json::Value>) -> Worksheet {
    serde_json::from_value(serde_json::json!({"cameras": [{
        "name": "CAM-02", "image_size": [1280, 720],
        "decks": [{"id": "main", "image_polygon": [[0,0],[1280,0],[1280,720],[0,720]], "marks": floor_marks}],
        "height_marks": height,
        "zones": [
            {"id": "Crane-Deck", "type": "lifting-zone", "deck": "main", "floor_polygon": [[-3,8],[3,8],[3,14],[-3,14]]},
            {"id": "Moonpool", "type": "restricted-area", "deck": "main", "image_polygon": [[540,300],[740,300],[740,400],[540,400]]}
        ]}]})).unwrap()
}

const FLOOR: [(f64, f64); 8] = [(-6.0, 6.0), (6.0, 6.0), (-6.0, 16.0), (6.0, 16.0), (0.0, 9.0), (3.0, 12.0), (-4.0, 11.0), (1.0, 15.0)];

#[test]
fn clean_marks_fit_exactly() {
    let p = projection();
    let (site, rep) = build_site(&worksheet(marks(&p, &FLOOR, 0.0), vec![]), None).unwrap();
    assert!(rep[0].decks[0].worst_m < 1e-6, "{rep:?}");
    let s = vg360_scenegraph::Site::from_json(&site.to_string()).unwrap();
    let cam = s.camera("CAM-02").unwrap();
    let q = project(&p, [2.5, 10.0, 0.0]);
    let f = h_apply(&cam.decks[0].h, q).unwrap();
    assert!((f.0 - 2.5).abs() < 1e-6 && (f.1 - 10.0).abs() < 1e-6, "{f:?}");
    // an image-drawn zone lands on the floor through the fitted H
    let moon = cam.zones.iter().find(|z| z.id == "Moonpool").unwrap();
    let (_, poly) = moon.floor.as_ref().unwrap();
    let want = h_apply(&cam.decks[0].h, (540.0, 400.0)).unwrap();
    assert!(poly.iter().any(|v| (v[0] - want.0).abs() < 1e-9 && (v[1] - want.1).abs() < 1e-9));
}

#[test]
fn half_pixel_noise_stays_within_limits() {
    let p = projection();
    let (_, rep) = build_site(&worksheet(marks(&p, &FLOOR, 0.5), vec![]), None).unwrap();
    assert!(rep[0].decks[0].worst_m < 0.15 && rep[0].decks[0].worst_m > 0.0, "{rep:?}");
}

#[test]
fn a_wrong_mark_is_named() {
    let p = projection();
    let mut m = marks(&p, &FLOOR, 0.0);
    m[5]["m"] = serde_json::json!([3.0, 13.0]); // measured 1 m off
    let e = build_site(&worksheet(m, vec![]), None).unwrap_err();
    assert!(e.iter().any(|x| x.contains("CAM-02/main") && x.contains("off the fit")), "{e:?}");
}

#[test]
fn too_few_marks_are_refused() {
    let p = projection();
    let e = build_site(&worksheet(marks(&p, &FLOOR[..3], 0.0), vec![]), None).unwrap_err();
    assert!(e.iter().any(|x| x.contains("need at least 4")), "{e:?}");
    let e = build_site(&worksheet(marks(&p, &FLOOR[..4], 0.0), vec![]), None).unwrap_err();
    assert!(e.iter().any(|x| x.contains("cannot be checked")), "{e:?}");
}

#[test]
fn camera_matrix_from_height_marks() {
    let p = projection();
    let pts3 = [[-6.0, 6.0, 0.0], [6.0, 6.0, 0.0], [-6.0, 16.0, 0.0], [6.0, 16.0, 0.0], [-6.0, 6.0, 1.1], [6.0, 16.0, 1.1], [0.0, 12.0, 3.0], [-4.0, 14.0, 2.0]];
    let h: Vec<serde_json::Value> = pts3.iter().map(|w| { let q = project(&p, *w); serde_json::json!({"px": [q.0, q.1], "m": w}) }).collect();
    let (site, rep) = build_site(&worksheet(marks(&p, &FLOOR, 0.0), h), None).unwrap();
    let pr = rep[0].projection.as_ref().unwrap();
    assert!(pr.worst_px < 1e-4 && pr.decks_agree_m < 1e-4, "{pr:?}");
    let pf: [[f64; 4]; 3] = serde_json::from_value(site["cameras"][0]["projection"].clone()).unwrap();
    let (a, b) = (project(&pf, [1.0, 9.0, 5.0]), project(&p, [1.0, 9.0, 5.0]));
    assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4);
    // all on one level: refused
    let flat: Vec<serde_json::Value> = pts3.iter().map(|w| { let q = project(&p, [w[0], w[1], 0.0]); serde_json::json!({"px": [q.0, q.1], "m": [w[0], w[1], 0.0]}) }).collect();
    let e = build_site(&worksheet(marks(&p, &FLOOR, 0.0), flat), None).unwrap_err();
    assert!(e.iter().any(|x| x.contains("two or more levels")), "{e:?}");
}

#[test]
fn catalog_zones_must_be_drawn_on_their_camera() {
    let p = projection();
    let cat = serde_json::json!({"zones": [{"noun": "lifting-zone"}, {"noun": "restricted-area"}],
        "zone_instances": [{"name": "Crane-Deck", "zone": "lifting-zone", "camera": "CAM-02"}, {"name": "Rig-Floor-East", "zone": "height-zone", "camera": "CAM-03"}]});
    let e = build_site(&worksheet(marks(&p, &FLOOR, 0.0), vec![]), Some(&cat)).unwrap_err();
    assert!(e.iter().any(|x| x.contains("Rig-Floor-East") && x.contains("CAM-03")), "{e:?}");
}
