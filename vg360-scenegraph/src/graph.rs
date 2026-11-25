//! Per-camera scene-graph builder: detections in, iogp-runtime objects and facts out.
//!
//! Steady state allocates nothing: node, pair and track buffers are reserved up front and reused,
//! track id strings are rewritten in place, and pair search is sort-and-sweep over a reused index
//! buffer (`tests/no_alloc.rs` checks this with a counting allocator).

use crate::geom::*;
use crate::policy::{decide, Policy};
use crate::site::Camera;
use crate::sink::{FactSink, ObjKind};
use crate::track::Kf;
use std::fmt::Write as _;

/// One detection from YOLOX-Nano + ByteTrack (pixel coordinates of the frame the boxes were drawn on).
#[derive(Clone, Copy, Debug)]
pub struct Det<'a> { pub track_id: u64, pub class: &'a str, pub bbox: BBox, pub score: f64 }

/// A PPE class and where on the person its centre must fall (fractions of box height from the top).
#[derive(Clone, Debug)]
pub struct PpeBand { pub class: String, pub top: f64, pub bottom: f64 }

#[derive(Clone, Debug)]
pub struct Params {
    /// k in "within k standard deviations" for the three-valued relations
    pub k_sigma: f64,
    /// pixel std of a foot point = frac * box height + floor
    pub px_sigma_frac: f64,
    pub px_sigma_floor: f64,
    /// std multiplier when the foot point was estimated (feet hidden)
    pub est_sigma_mul: f64,
    /// width / height of a standing person, for recovering a hidden foot point
    pub person_aspect: f64,
    /// a box bottom within this many pixels of the image bottom is treated as cut off
    pub edge_px: f64,
    /// keep an occluded track alive (Kalman-predicted) for this long
    pub coast_max_s: f64,
    /// forget a track not seen for this long
    pub drop_after_s: f64,
    /// white-acceleration std for the floor Kalman filter, m/s^2
    pub accel_sigma: f64,
    /// Mahalanobis^2 gate; a jump beyond it restarts the filter (id switch upstream)
    pub gate_chi2: f64,
    /// height range of a suspended load above the deck, metres
    pub load_height_m: [f64; 2],
    pub load_samples: usize,
    /// added to the load's estimated half-width for the drop zone
    pub load_margin_m: f64,
    /// share of a PPE box that must lie inside the person box (iogp-editor ingest uses 0.6)
    pub ppe_containment: f64,
    pub ppe_bands: Vec<PpeBand>,
    pub human_classes: Vec<String>,
    pub load_classes: Vec<String>,
    /// when the camera is reported moved, emit this signal on every zone (None = emit nothing)
    pub calibration_signal: Option<String>,
    pub max_nodes: usize,
    pub max_tracks: usize,
}

impl Default for Params {
    fn default() -> Self {
        let b = |c: &str, t: f64, bo: f64| PpeBand { class: c.into(), top: t, bottom: bo };
        Params {
            k_sigma: 2.0, px_sigma_frac: 0.03, px_sigma_floor: 1.5, est_sigma_mul: 2.5, person_aspect: 0.38, edge_px: 4.0,
            coast_max_s: 2.0, drop_after_s: 2.5, accel_sigma: 1.5, gate_chi2: 13.8,
            load_height_m: [1.0, 12.0], load_samples: 8, load_margin_m: 0.5, ppe_containment: 0.6,
            ppe_bands: vec![b("hardhat", 0.0, 0.3), b("hi_vis_vest", 0.15, 0.7), b("harness", 0.1, 0.8)],
            human_classes: vec!["person".into()], load_classes: vec!["suspended_load".into()],
            calibration_signal: Some("calibration-lost".into()), max_nodes: 128, max_tracks: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role { Human, Load, Other }

/// A node of the frame's graph.
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub slot: usize,
    pub role: Role,
    /// image box (for coasting nodes: the last one seen)
    pub bbox: BBox,
    /// image foot point used (possibly estimated)
    pub foot_px: P2,
    pub deck: Option<usize>,
    pub floor: P2,
    pub cov: M2,
    pub foot_estimated: bool,
    pub coasting: bool,
}

const MAX_HELD: usize = 4;
#[derive(Clone, Debug, Default)]
struct Held { rel: u8, class: String, id: String }

#[derive(Clone, Debug)]
struct Track {
    active: bool,
    key: u64,
    id: String,
    class: String,
    role: Role,
    last_t: f64,
    seen_this_frame: bool,
    kf: Option<Kf>,
    deck: Option<usize>,
    bbox: BBox,
    foot_px: P2,
    occluded: bool,
    held: [Held; MAX_HELD],
    n_held: usize,
}

impl Track {
    fn empty() -> Track {
        Track { active: false, key: 0, id: String::with_capacity(32), class: String::with_capacity(24), role: Role::Other, last_t: 0.0, seen_this_frame: false,
            kf: None, deck: None, bbox: [0.0; 4], foot_px: (0.0, 0.0), occluded: false, held: Default::default(), n_held: 0 }
    }
}

/// Drop zone of one load on one deck: polyline of floor points (one per sampled height) + radius.
#[derive(Clone, Copy, Debug)]
struct DropZone { node: usize, deck: usize, pts: [P2; 16], n: usize, radius: f64 }

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats { pub nodes: usize, pub coasting: usize, pub overflow: usize, pub facts: usize, pub pairs_tested: usize }

pub struct SceneBuilder {
    pub cam: Camera,
    pub params: Params,
    pub policy: Policy,
    tracks: Vec<Track>,
    nodes: Vec<Node>,
    order: Vec<u16>,
    drops: Vec<DropZone>,
    calib_id: String,
    pub stats: Stats,
}

const R_WEAR: u8 = 0; const R_CARRY: u8 = 1; const R_HOLD: u8 = 2; const R_USE: u8 = 3; const R_DRIVE: u8 = 4;

impl SceneBuilder {
    pub fn new(cam: Camera, params: Params, policy: Policy) -> SceneBuilder {
        let calib_id = format!("{}/{}", cam.name, params.calibration_signal.as_deref().unwrap_or("calibration-lost"));
        SceneBuilder {
            tracks: (0..params.max_tracks).map(|_| Track::empty()).collect(),
            nodes: Vec::with_capacity(params.max_nodes),
            order: Vec::with_capacity(params.max_nodes),
            drops: Vec::with_capacity(16),
            cam, params, policy, calib_id, stats: Stats::default(),
        }
    }

    /// The graph built by the last [`SceneBuilder::step`] (for overlays and debugging).
    pub fn nodes(&self) -> &[Node] { &self.nodes }
    pub fn track_id(&self, n: &Node) -> &str { &self.tracks[n.slot].id }
    pub fn track_class(&self, n: &Node) -> &str { &self.tracks[n.slot].class }

    fn role(&self, class: &str) -> Role {
        if self.params.human_classes.iter().any(|c| c == class) { Role::Human }
        else if self.params.load_classes.iter().any(|c| c == class) { Role::Load }
        else { Role::Other }
    }

    fn rel_name(&self, r: u8) -> &str {
        let n = &self.policy.names;
        match r { R_WEAR => &n.wear, R_CARRY => &n.carry, R_HOLD => &n.hold, R_USE => &n.r#use, _ => &n.drive }
    }

    fn slot_for(&mut self, key: u64, t: f64) -> usize {
        if let Some(i) = self.tracks.iter().position(|tr| tr.active && tr.key == key) { return i; }
        let i = self.tracks.iter().position(|tr| !tr.active).unwrap_or_else(|| {
            // full: evict the stalest
            let mut best = 0;
            for (i, tr) in self.tracks.iter().enumerate() { if tr.last_t < self.tracks[best].last_t { best = i; } }
            best
        });
        let cam = &self.cam.name;
        let tr = &mut self.tracks[i];
        tr.active = true; tr.key = key; tr.kf = None; tr.deck = None; tr.n_held = 0; tr.last_t = t;
        tr.id.clear();
        let _ = write!(tr.id, "{cam}/{key}");
        i
    }

    fn deck_of(&self, p: P2) -> Option<usize> { self.cam.decks.iter().position(|d| in_polygon(&d.image_polygon, p)) }

    /// Foot point and whether it was estimated because the feet are hidden.
    fn foot_point(&self, b: &BBox, role: Role) -> (P2, bool) {
        let f = foot(b);
        let cut_by_edge = self.cam.image_size.is_some_and(|s| b[3] >= s[1] - self.params.edge_px);
        let cut_by_occluder = self.cam.occluders.iter().any(|o| in_polygon(o, f));
        if role == Role::Human && (cut_by_edge || cut_by_occluder) {
            let w = b[2] - b[0];
            let est_h = (b[3] - b[1]).max(w / self.params.person_aspect);
            ((f.0, b[1] + est_h), true)
        } else { (f, false) }
    }

    /// Build the graph for one frame and write objects and facts into `sink`.
    /// `camera_ok = false` (camera_guard: moved/blocked) suppresses every position-derived fact.
    pub fn step<S: FactSink>(&mut self, t: f64, dets: &[Det], camera_ok: bool, sink: &mut S) {
        self.nodes.clear();
        self.drops.clear();
        self.stats = Stats::default();
        for tr in self.tracks.iter_mut() { tr.seen_this_frame = false; }
        sink.begin(t, &self.cam.name);

        // --- nodes from detections
        for d in dets {
            let role = self.role(d.class);
            let slot = self.slot_for(d.track_id, t);
            let (foot_px, est) = self.foot_point(&d.bbox, role);
            let deck = if camera_ok { self.deck_of(foot_px) } else { None };
            let (mut floor, mut cov) = ((0.0, 0.0), [[0.0; 2]; 2]);
            let mut deck_used = None;
            if let Some(di) = deck {
                let h = &self.cam.decks[di].h;
                if let (Some(fl), Some(j)) = (h_apply(h, foot_px), h_jacobian(h, foot_px)) {
                    let mut s = self.params.px_sigma_frac * (d.bbox[3] - d.bbox[1]) + self.params.px_sigma_floor;
                    if est { s *= self.params.est_sigma_mul; }
                    let r = propagate(&j, s * 0.5, s); // feet are better localised across than along the image column
                    let (sa, gate) = (self.params.accel_sigma, self.params.gate_chi2);
                    let tr = &mut self.tracks[slot];
                    let kf = match (tr.kf.as_mut(), tr.deck == Some(di)) {
                        (Some(kf), true) => {
                            kf.predict(t - tr.last_t, sa);
                            if kf.mahalanobis2(fl, &r) > gate { *kf = Kf::new(fl, &r); } else { kf.update(fl, &r); }
                            *kf
                        }
                        _ => Kf::new(fl, &r),
                    };
                    tr.kf = Some(kf);
                    floor = kf.pos(); cov = kf.pos_cov(); deck_used = Some(di);
                }
            }
            let tr = &mut self.tracks[slot];
            tr.class.clear(); tr.class.push_str(d.class);
            tr.role = role; tr.last_t = t; tr.seen_this_frame = true; tr.deck = deck_used; tr.bbox = d.bbox; tr.foot_px = foot_px;
            tr.occluded = est;
            if deck_used.is_none() { tr.kf = None; }
            if self.nodes.len() < self.params.max_nodes {
                self.nodes.push(Node { slot, role, bbox: d.bbox, foot_px, deck: deck_used, floor, cov, foot_estimated: est, coasting: false });
            } else { self.stats.overflow += 1; }
        }
        let n_seen = self.nodes.len();

        // --- coast occluded tracks; forget stale ones
        for slot in 0..self.tracks.len() {
            let tr = &mut self.tracks[slot];
            if !tr.active || tr.seen_this_frame { continue; }
            let age = t - tr.last_t;
            if age > self.params.drop_after_s { tr.active = false; continue; }
            if !camera_ok || !tr.occluded || age > self.params.coast_max_s { continue; }
            let (Some(kf), Some(di)) = (tr.kf, tr.deck) else { continue };
            let mut kf = kf;
            kf.predict(age, self.params.accel_sigma); // from the last update; the stored filter is not advanced
            if self.nodes.len() < self.params.max_nodes {
                self.nodes.push(Node { slot, role: tr.role, bbox: tr.bbox, foot_px: tr.foot_px, deck: Some(di), floor: kf.pos(), cov: kf.pos_cov(), foot_estimated: true, coasting: true });
                self.stats.coasting += 1;
            } else { self.stats.overflow += 1; }
        }
        self.stats.nodes = self.nodes.len();

        // --- objects
        for n in &self.nodes {
            let tr = &self.tracks[n.slot];
            sink.object(&tr.id, ObjKind::Class, &tr.class, None);
        }
        for z in &self.cam.zones { sink.object(&z.id, ObjKind::Zone, &z.zone, Some(&self.cam.name)); }

        if camera_ok {
            self.zone_facts(sink);
            self.near_facts(sink);
            self.under_facts(sink);
            self.on_facts(n_seen, sink);
        } else if let Some(sig) = &self.params.calibration_signal {
            sink.object(&self.calib_id, ObjKind::Signal, sig, None);
            for z in &self.cam.zones { sink.fact(&self.policy.names.have, &z.id, &self.calib_id); self.stats.facts += 1; }
        }
        self.association_facts(n_seen, sink);
    }

    fn slack(&self, cov: &M2) -> f64 { self.params.k_sigma * lambda_max(cov).max(0.0).sqrt() }

    fn zone_facts<S: FactSink>(&mut self, sink: &mut S) {
        let rel = &self.policy.names.inside;
        for n in &self.nodes {
            for z in &self.cam.zones {
                let hit = match (&z.floor, n.deck) {
                    (Some((zd, poly)), Some(nd)) => {
                        if *zd != nd { false } else {
                            let d = boundary_dist(poly, n.floor);
                            let margin = if in_polygon(poly, n.floor) { d } else { -d };
                            decide(self.policy.inside, margin, self.slack(&n.cov))
                        }
                    }
                    (Some(_), None) => false,
                    (None, _) => in_polygon(&z.polygon, n.foot_px), // image-space zone: iogp-editor ingest semantics
                };
                if hit { sink.fact(rel, &self.tracks[n.slot].id, &z.id); self.stats.facts += 1; }
            }
        }
    }

    fn near_facts<S: FactSink>(&mut self, sink: &mut S) {
        let r = self.policy.near_m;
        let k = self.params.k_sigma;
        self.order.clear();
        let mut smax: f64 = 0.0;
        for (i, n) in self.nodes.iter().enumerate() {
            if n.deck.is_some() && self.policy.near_applies(&self.tracks[n.slot].class) {
                self.order.push(i as u16);
                smax = smax.max(lambda_max(&n.cov).max(0.0).sqrt());
            }
        }
        let nodes = &self.nodes;
        self.order.sort_unstable_by(|a, b| nodes[*a as usize].floor.0.total_cmp(&nodes[*b as usize].floor.0));
        let window = r + k * 2.0 * smax;
        for ia in 0..self.order.len() {
            let a = &self.nodes[self.order[ia] as usize];
            for ib in (ia + 1)..self.order.len() {
                let b = &self.nodes[self.order[ib] as usize];
                if b.floor.0 - a.floor.0 > window { break; }
                if a.deck != b.deck || a.slot == b.slot { continue; }
                self.stats.pairs_tested += 1;
                let (dx, dy) = (b.floor.0 - a.floor.0, b.floor.1 - a.floor.1);
                let d = (dx * dx + dy * dy).sqrt();
                let sum = add2(&a.cov, &b.cov);
                let s = if d < 1e-9 { lambda_max(&sum).max(0.0).sqrt() } else { sigma_along(&sum, (dx / d, dy / d)) };
                if decide(self.policy.near, r - d, k * s) {
                    let (ida, idb) = (&self.tracks[a.slot].id, &self.tracks[b.slot].id);
                    sink.fact(&self.policy.names.near, ida, idb);
                    sink.fact(&self.policy.names.near, idb, ida);
                    self.stats.facts += 2;
                }
            }
        }
    }

    fn under_facts<S: FactSink>(&mut self, sink: &mut S) {
        // drop zones (needs the 3x4 projection); without it fall back to the image heuristic
        if let Some(p) = &self.cam.projection {
            for (li, ln) in self.nodes.iter().enumerate() {
                if ln.role != Role::Load || ln.coasting { continue; }
                let b = ln.bbox;
                let (bc, bl, br) = (((b[0] + b[2]) / 2.0, b[3]), (b[0], b[3]), (b[2], b[3]));
                for (di, deck) in self.cam.decks.iter().enumerate() {
                    let ns = self.params.load_samples.clamp(2, 16);
                    let mut dz = DropZone { node: li, deck: di, pts: [(0.0, 0.0); 16], n: 0, radius: 0.0 };
                    let [h0, h1] = self.params.load_height_m;
                    let mut half_w: f64 = 0.0;
                    for s in 0..ns {
                        let h = h0 + (h1 - h0) * s as f64 / (ns - 1) as f64;
                        let z = deck.elevation_m + h;
                        if let Some(q) = ray_at_height(p, bc, z) {
                            dz.pts[dz.n] = q; dz.n += 1;
                            if let (Some(l), Some(r)) = (ray_at_height(p, bl, z), ray_at_height(p, br, z)) {
                                half_w = half_w.max(0.5 * ((r.0 - l.0).powi(2) + (r.1 - l.1).powi(2)).sqrt());
                            }
                        }
                    }
                    if dz.n == 0 { continue; }
                    dz.radius = half_w + self.params.load_margin_m;
                    if self.drops.len() < self.drops.capacity() { self.drops.push(dz); }
                }
            }
            for n in self.nodes.iter() {
                if n.role == Role::Load { continue; }
                let Some(nd) = n.deck else { continue };
                for dz in self.drops.iter().filter(|dz| dz.deck == nd) {
                    let d = polyline_dist(&dz.pts[..dz.n], n.floor);
                    if decide(self.policy.under, dz.radius - d, self.slack(&n.cov)) {
                        sink.fact(&self.policy.names.under, &self.tracks[n.slot].id, &self.tracks[self.nodes[dz.node].slot].id);
                        self.stats.facts += 1;
                    }
                }
            }
        } else {
            for a in self.nodes.iter().filter(|n| !n.coasting) {
                for b in self.nodes.iter().filter(|n| n.role == Role::Load && !n.coasting) {
                    if a.slot == b.slot { continue; }
                    let (ba, bb) = (&a.bbox, &b.bbox);
                    // iogp-editor ingest `below_footprint`
                    let w = (bb[2] - bb[0]) * 0.1;
                    let (fx, _) = a.foot_px;
                    if fx >= bb[0] - w && fx <= bb[2] + w && bb[3] < ba[3] && bb[3] <= ba[1] + (ba[3] - ba[1]) * 0.5 {
                        sink.fact(&self.policy.names.under, &self.tracks[a.slot].id, &self.tracks[b.slot].id);
                        self.stats.facts += 1;
                    }
                }
            }
        }
    }

    /// `on` (ingest `standing_on`): foot point inside the lower 60 % of the other box. Image-space.
    fn on_facts<S: FactSink>(&mut self, n_seen: usize, sink: &mut S) {
        for a in &self.nodes[..n_seen] {
            for b in &self.nodes[..n_seen] {
                if a.slot == b.slot { continue; }
                let bb = b.bbox;
                let lower = [bb[0], bb[1] + (bb[3] - bb[1]) * 0.4, bb[2], bb[3]];
                if inside(&lower, a.foot_px) { sink.fact(&self.policy.names.on, &self.tracks[a.slot].id, &self.tracks[b.slot].id); self.stats.facts += 1; }
            }
        }
    }

    /// wear / carry / hold / use / drive, image-space (ingest geometry; PPE also needs its band).
    /// Coasting people re-assert what they last wore and carried: their state is unknown, and
    /// dropping it would turn an occlusion into "does not wear a harness".
    fn association_facts<S: FactSink>(&mut self, n_seen: usize, sink: &mut S) {
        let minc = self.params.ppe_containment;
        for ai in 0..n_seen {
            let a = self.nodes[ai];
            if a.role != Role::Human { continue; }
            let mut keep: [(usize, u8); MAX_HELD] = [(0, 0); MAX_HELD];
            let mut nk = 0usize;
            for bi in 0..n_seen {
                let b = self.nodes[bi];
                if a.slot == b.slot { continue; }
                let (ba, bb) = (a.bbox, b.bbox);
                let mut rels = [false; 5];
                if area(&bb) > 0.0 && inter(&ba, &bb) / area(&bb) >= minc {
                    let bclass = &self.tracks[b.slot].class;
                    rels[R_WEAR as usize] = self.params.ppe_bands.iter().find(|p| &p.class == bclass).map_or(true, |p| {
                        let f = ((bb[1] + bb[3]) / 2.0 - ba[1]) / (ba[3] - ba[1]).max(1e-9);
                        f >= p.top && f <= p.bottom
                    });
                }
                let w = (ba[2] - ba[0]) * 0.15;
                let carry = inside(&[ba[0] - w, ba[1], ba[2] + w, ba[3]], centre(&bb));
                rels[R_CARRY as usize] = carry; rels[R_HOLD as usize] = carry; rels[R_USE as usize] = carry;
                rels[R_DRIVE as usize] = b.role != Role::Human && inside(&bb, centre(&ba));
                for (r, on) in rels.iter().enumerate() {
                    if !on { continue; }
                    sink.fact(self.rel_name(r as u8), &self.tracks[a.slot].id, &self.tracks[b.slot].id);
                    self.stats.facts += 1;
                    if nk < MAX_HELD && r != R_DRIVE as usize { keep[nk] = (b.slot, r as u8); nk += 1; }
                }
            }
            for (i, &(bs, r)) in keep[..nk].iter().enumerate() {
                let (src, dst) = two_mut(&mut self.tracks, bs, a.slot);
                let h = &mut dst.held[i];
                h.rel = r;
                h.id.clear(); h.id.push_str(&src.id);
                h.class.clear(); h.class.push_str(&src.class);
            }
            self.tracks[a.slot].n_held = nk;
        }
        for n in &self.nodes[n_seen..] {
            if n.role != Role::Human { continue; }
            let tr = &self.tracks[n.slot];
            for h in &tr.held[..tr.n_held] {
                sink.object(&h.id, ObjKind::Class, &h.class, None);
                sink.fact(self.rel_name(h.rel), &tr.id, &h.id);
                self.stats.facts += 1;
            }
        }
    }
}

/// (&v[i], &mut v[j]) for i != j.
fn two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&T, &mut T) {
    assert_ne!(i, j);
    if i < j { let (l, r) = v.split_at_mut(j); (&l[i], &mut r[0]) } else { let (l, r) = v.split_at_mut(i); (&r[0], &mut l[j]) }
}
