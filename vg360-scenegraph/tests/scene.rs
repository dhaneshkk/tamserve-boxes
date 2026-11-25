mod common;
use common::*;
use vg360_scenegraph::*;

fn engine() -> iogp_runtime::Engine { iogp_runtime::Engine::from_config(&fixture_config()).unwrap() }

fn facts_of(sg: &mut SceneBuilder, t: f64, dets: &[Det], ok: bool) -> iogp_runtime::Frame {
    let mut f = iogp_runtime::Frame::default();
    sg.step(t, dets, ok, &mut f);
    f
}
fn has(f: &iogp_runtime::Frame, r: &str, a: &str, b: &str) -> bool { f.facts.iter().any(|x| x.0 == r && x.1 == a && x.2 == b) }

#[test]
fn policy_from_fixture_rules() {
    let e = engine();
    let p = Policy::from_rules(&e.rules);
    assert_eq!((p.inside, p.near, p.under), (Emit::Possible, Emit::Possible, Emit::Possible));
    assert_eq!(p.near_m, 3.0);
    assert!(p.near_classes.contains(&"person".to_string()) && p.near_classes.contains(&"vehicle".to_string()));
    assert!(p.warnings.is_empty(), "{:?}", p.warnings);
}

#[test]
fn negated_only_relation_uses_definite() {
    let p = Policy::from_conditions([("in", true, None, Some("person"), None)], RelNames::default());
    assert_eq!(p.inside, Emit::Definite);
    let p = Policy::from_conditions([("near", false, Some(3.0), Some("person"), None), ("near", true, Some(5.0), None, None)], RelNames::default());
    assert_eq!(p.near, Emit::Possible);
    assert_eq!(p.near_m, 5.0);
    assert_eq!(p.warnings.len(), 2);
}

#[test]
fn floor_positions_are_metric() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let dets = [Det { track_id: 1, class: "person", bbox: bbox(&p, 1.0, 10.0, 0.0, 0.5, 1.75), score: 0.9 }];
    let f = facts_of(&mut sg, 0.0, &dets, true);
    let n = sg.nodes()[0];
    // the box's x-centre is not exactly the foot's projection under perspective (head and feet are
    // at different scales), so X carries a few centimetres of bias; Y comes from the box bottom
    assert!((n.floor.0 - 1.0).abs() < 0.05 && (n.floor.1 - 10.0).abs() < 1e-6, "{:?}", n.floor);
    assert!(has(&f, "in", "CAM-02/1", "Guard-1"));
    assert!(!has(&f, "in", "CAM-02/1", "Rig-Floor-East"));
}

#[test]
fn far_workers_carry_more_uncertainty() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let dets = [
        Det { track_id: 1, class: "person", bbox: bbox(&p, 0.0, 6.0, 0.0, 0.5, 1.75), score: 0.9 },
        Det { track_id: 2, class: "person", bbox: bbox(&p, 0.0, 30.0, 0.0, 0.5, 1.75), score: 0.9 },
    ];
    facts_of(&mut sg, 0.0, &dets, true);
    let (a, b) = (sg.nodes()[0], sg.nodes()[1]);
    assert!(b.cov[1][1] > 4.0 * a.cov[1][1], "near {:?} far {:?}", a.cov, b.cov);
}

#[test]
fn near_is_metric_and_raises_after_duration() {
    let mut e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let mut raised = None;
    for i in 0..30 {
        let t = i as f64 * 0.1;
        let dets = [
            Det { track_id: 1, class: "person", bbox: bbox(&p, 0.0, 12.0, 0.0, 0.5, 1.75), score: 0.9 },
            Det { track_id: 2, class: "vehicle", bbox: bbox(&p, 2.5, 12.0, 0.0, 2.0, 2.2), score: 0.9 },
            Det { track_id: 3, class: "vehicle", bbox: bbox(&p, -8.0, 12.0, 0.0, 2.0, 2.2), score: 0.9 },
        ];
        sg.step(t, &dets, true, &mut e);
        for ev in e.end() { if ev.kind == "raised" && ev.rule == "NEAR" { raised.get_or_insert((t, ev.subject.clone())); } }
    }
    let (t, s) = raised.expect("NEAR alert");
    assert_eq!(s, "CAM-02/1");
    assert!(t >= 2.0 - 1e-9, "raised at {t}");
    let f = facts_of(&mut sg, 3.0, &[
        Det { track_id: 1, class: "person", bbox: bbox(&p, 0.0, 12.0, 0.0, 0.5, 1.75), score: 0.9 },
        Det { track_id: 3, class: "vehicle", bbox: bbox(&p, -8.0, 12.0, 0.0, 2.0, 2.2), score: 0.9 },
    ], true);
    assert!(!has(&f, "near", "CAM-02/1", "CAM-02/3"));
}

#[test]
fn under_uses_drop_zone_not_box_bottom() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    // load 1.2 m wide, 0.8 m tall, bottom 5.6 m above (2, 12)
    let load = Det { track_id: 9, class: "suspended_load", bbox: bbox(&p, 2.0, 12.0, 5.6, 1.2, 0.8), score: 0.9 };
    let below = Det { track_id: 1, class: "person", bbox: bbox(&p, 2.0, 12.0, 0.0, 0.5, 1.75), score: 0.9 };
    let aside = Det { track_id: 2, class: "person", bbox: bbox(&p, 6.0, 12.0, 0.0, 0.5, 1.75), score: 0.9 };
    let f = facts_of(&mut sg, 0.0, &[load, below, aside], true);
    assert!(has(&f, "under", "CAM-02/1", "CAM-02/9"));
    assert!(!has(&f, "under", "CAM-02/2", "CAM-02/9"));
}

#[test]
fn line_of_fire_alert_end_to_end() {
    let mut e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let mut got = false;
    for i in 0..30 {
        let dets = [
            Det { track_id: 9, class: "suspended_load", bbox: bbox(&p, 2.0, 12.0, 5.6, 1.2, 0.8), score: 0.9 },
            Det { track_id: 1, class: "person", bbox: bbox(&p, 2.0, 12.0, 0.0, 0.5, 1.75), score: 0.9 },
        ];
        sg.step(i as f64 * 0.1, &dets, true, &mut e);
        got |= e.end().iter().any(|ev| ev.kind == "raised" && ev.rule == "LSR6");
    }
    assert!(got);
}

/// A worker walks behind equipment for 1.5 s. With coasting the 5 s machine-guard alert still fires;
/// without it the duration timer restarts and the alert is missed.
#[test]
fn coasting_keeps_duration_timers_through_occlusion() {
    let p = projection();
    let foot = project(&p, [0.0, 11.0, 0.0]);
    let occ = format!("[[[{},{}],[{},{}],[{},{}],[{},{}]]]", foot.0 - 60.0, foot.1 - 40.0, foot.0 + 60.0, foot.1 - 40.0, foot.0 + 60.0, foot.1 + 40.0, foot.0 - 60.0, foot.1 + 40.0);
    let run = |coast: f64| -> bool {
        let mut e = engine();
        let mut sg = builder(&occ, Params { coast_max_s: coast, ..Params::default() }, &e);
        let mut got = false;
        for i in 0..62 {
            let t = i as f64 * 0.1;
            let y = 11.0;
            // feet visible until 1.5 s, feet hidden 1.5-2.0 s, fully hidden 2.0-3.5 s
            let mut b = bbox(&p, 0.0, y, 0.0, 0.5, 1.75);
            if t >= 1.5 - 1e-9 && t < 2.0 - 1e-9 { b[3] = foot.1 - 30.0; }
            let dets: Vec<Det> = if t >= 2.0 - 1e-9 && t < 3.5 - 1e-9 { vec![] } else { vec![Det { track_id: 1, class: "person", bbox: b, score: 0.9 }] };
            sg.step(t, &dets, true, &mut e);
            got |= e.end().iter().any(|ev| ev.kind == "raised" && ev.rule == "LSR1");
        }
        got
    };
    assert!(run(2.0), "coasting should keep the alert");
    assert!(!run(0.0), "without coasting the timer restarts");
}

/// While coasting, the harness the worker last wore is re-asserted: no working-at-height alert.
#[test]
fn coasting_worker_keeps_ppe() {
    let p = projection();
    let foot = project(&p, [7.0, 11.0, 0.0]);
    let occ = format!("[[[{},{}],[{},{}],[{},{}],[{},{}]]]", foot.0 - 60.0, foot.1 - 40.0, foot.0 + 60.0, foot.1 - 40.0, foot.0 + 60.0, foot.1 + 40.0, foot.0 - 60.0, foot.1 + 40.0);
    let mut e = engine();
    let mut sg = builder(&occ, Params::default(), &e);
    let mut alerts = vec![];
    for i in 0..60 {
        let t = i as f64 * 0.1;
        let mut b = bbox(&p, 7.0, 11.0, 0.0, 0.5, 1.75);
        let hidden_feet = t >= 1.0 - 1e-9;
        if hidden_feet { b[3] = foot.1 - 30.0; }
        let harness = [b[0] + 4.0, b[1] + (b[3] - b[1]) * 0.25, b[2] - 4.0, b[1] + (b[3] - b[1]) * 0.55];
        let dets: Vec<Det> = if t >= 2.0 - 1e-9 && t < 3.8 { vec![] } else {
            vec![Det { track_id: 1, class: "person", bbox: b, score: 0.9 }, Det { track_id: 5, class: "harness", bbox: harness, score: 0.8 }]
        };
        sg.step(t, &dets, true, &mut e);
        for ev in e.end() { if ev.kind == "raised" { alerts.push(ev.rule.clone()); } }
    }
    assert!(!alerts.iter().any(|r| r == "LSR9" || r == "NAMED" || r == "CAMERA"), "{alerts:?}");
}

#[test]
fn moved_camera_suppresses_positions_and_signals() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let dets = [Det { track_id: 1, class: "person", bbox: bbox(&p, 1.0, 10.0, 0.0, 0.5, 1.75), score: 0.9 }];
    let f = facts_of(&mut sg, 0.0, &dets, false);
    assert!(!f.facts.iter().any(|x| x.0 == "in" || x.0 == "near" || x.0 == "under"));
    assert!(has(&f, "have", "Guard-1", "CAM-02/calibration-lost"));
    assert!(f.objects.iter().any(|o| o.signal.as_deref() == Some("calibration-lost")));
}

#[test]
fn ppe_band_rejects_helmet_at_waist() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let b = bbox(&p, 1.0, 10.0, 0.0, 0.5, 1.75);
    let h = b[3] - b[1];
    let at_head = [b[0] + 2.0, b[1], b[2] - 2.0, b[1] + 0.12 * h];
    let at_waist = [b[0] + 2.0, b[1] + 0.5 * h, b[2] - 2.0, b[1] + 0.62 * h];
    let f = facts_of(&mut sg, 0.0, &[Det { track_id: 1, class: "person", bbox: b, score: 0.9 }, Det { track_id: 2, class: "hardhat", bbox: at_head, score: 0.9 }], true);
    assert!(has(&f, "wear", "CAM-02/1", "CAM-02/2"));
    let f = facts_of(&mut sg, 0.1, &[Det { track_id: 1, class: "person", bbox: b, score: 0.9 }, Det { track_id: 2, class: "hardhat", bbox: at_waist, score: 0.9 }], true);
    assert!(!has(&f, "wear", "CAM-02/1", "CAM-02/2"));
}

/// A site file written for iogp-editor ingest (no decks) still works: image-space zones.
#[test]
fn ingest_site_files_load_unchanged() {
    let site = Site::from_json(r#"{"cameras": [{"name": "CAM-03", "px_per_metre": 40, "zones": [{"id": "Rig-Floor-East", "type": "height-zone", "polygon": [[0,0],[400,0],[400,480],[0,480]]}]}]}"#).unwrap();
    let mut sg = SceneBuilder::new(site.camera("CAM-03").unwrap(), Params::default(), Policy::default());
    let mut f = iogp_runtime::Frame::default();
    sg.step(0.0, &[Det { track_id: 7, class: "person", bbox: [100.0, 250.0, 160.0, 420.0], score: 0.9 }], true, &mut f);
    assert!(has(&f, "in", "CAM-03/7", "Rig-Floor-East"));
}

/// With a lens, boxes come in distorted pixels; the floor point must be the undistorted one's.
#[test]
fn lens_is_undone_before_the_homography() {
    let p = projection();
    let lens = vg360_scenegraph::geom::Lens { k: [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], d: [-0.25, 0.08, 0.0, 0.0, 0.0] };
    let mut site: serde_json::Value = serde_json::from_str(&site_json("[]")).unwrap();
    site["cameras"][0]["lens"] = serde_json::json!({"K": lens.k, "D": lens.d});
    let site: Site = serde_json::from_value(site).unwrap();
    let mut sg = SceneBuilder::new(site.camera("CAM-02").unwrap(), Params::default(), Policy::default());
    // near the left edge of the picture, where a barrel lens bends most
    let b = bbox(&p, -7.0, 6.0, 0.0, 0.5, 1.75);
    assert!(b[0] > 0.0);
    let foot = lens.distort((b[0] + b[2]) / 2.0, b[3]);
    let dist_box = [foot.0 - 10.0, foot.1 - 80.0, foot.0 + 10.0, foot.1];
    let mut f = iogp_runtime::Frame::default();
    sg.step(0.0, &[Det { track_id: 1, class: "person", bbox: dist_box, score: 0.9 }], true, &mut f);
    let n = sg.nodes()[0];
    // the same box seen through a lens-free camera
    let mut plain = SceneBuilder::new(Site::from_json(&site_json("[]")).unwrap().camera("CAM-02").unwrap(), Params::default(), Policy::default());
    plain.step(0.0, &[Det { track_id: 1, class: "person", bbox: b, score: 0.9 }], true, &mut f);
    let want = plain.nodes()[0].floor;
    assert!((n.floor.0 - want.0).abs() < 1e-6 && (n.floor.1 - want.1).abs() < 1e-6, "{:?} vs {:?}", n.floor, want);
    // and without undoing the lens it would be off by a lot
    let mut raw = SceneBuilder::new(Site::from_json(&site_json("[]")).unwrap().camera("CAM-02").unwrap(), Params::default(), Policy::default());
    raw.step(0.0, &[Det { track_id: 1, class: "person", bbox: dist_box, score: 0.9 }], true, &mut f);
    let off = raw.nodes()[0].floor;
    assert!(((off.0 - want.0).powi(2) + (off.1 - want.1).powi(2)).sqrt() > 0.3, "{off:?}");
}

/// A measured camera move G keeps people where they are on the floor.
#[test]
fn drift_correction_keeps_floor_positions() {
    let e = engine();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let b = bbox(&p, 1.0, 10.0, 0.0, 0.5, 1.75);
    // the camera turned: everything is 25 px right of where it was at calibration
    let moved = [b[0] + 25.0, b[1], b[2] + 25.0, b[3]];
    sg.set_view_correction([[1.0, 0.0, -25.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let mut f = iogp_runtime::Frame::default();
    sg.step(0.0, &[Det { track_id: 1, class: "person", bbox: moved, score: 0.9 }], true, &mut f);
    let n = sg.nodes()[0];
    assert!((n.floor.0 - 1.0).abs() < 0.05 && (n.floor.1 - 10.0).abs() < 1e-6, "{:?}", n.floor);
}

#[test]
fn params_load_from_json_with_defaults() {
    let p: Params = serde_json::from_str(r#"{"coast_max_s": 1.0, "human_classes": ["person", "worker"]}"#).unwrap();
    assert_eq!(p.coast_max_s, 1.0);
    assert_eq!(p.k_sigma, 2.0);
    assert!(serde_json::from_str::<Params>(r#"{"coast": 1.0}"#).is_err());
}
