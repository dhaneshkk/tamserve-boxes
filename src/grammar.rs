//! Safety rules over the floor: what the boxes mean once they stand somewhere in metres.
//!
//! YOLOX finds boxes; this turns them into verdicts a site can act on, with the same
//! behaviour as the Python reference (`vg360_grammar` notebook), step for step:
//!
//! * The floor is a grid of cells (0.5 m). People, forklifts and barricade posts are placed
//!   on it through the lens and the floor homography ([`crate::lens`]).
//! * **Line of fire**: a person within `r_lof` metres of a forklift, unless a barrier cell
//!   lies on the 4-connected ray between them. A 4-connected ray cannot slip between the
//!   diagonal cells of an 8-connected barrier, so a barrier drawn with Bresenham always stops it.
//! * **Red zone**: a person on a cell of the static red zone.
//! * **Barricade**: posts seen in at least `post_min` of the last `post_win` frames are
//!   chained when closer than `link_m`; a 4-connected fill from the hazard that reaches the
//!   edge of its search box means the barricade is open. The widest gap between neighbouring
//!   posts is where it is broken.
//! * **PPE**: a person's box splits into head, torso and legs; helmet and vest finds are
//!   matched to their band. A small model (trained with SAM) reads ten frames of that, and
//!   a per-class conformal threshold decides; an occluded person or an ambiguous set is
//!   UNKNOWN, never compliant (`proofs/Grammar.lean`, `occluded_never_compliant`).
//! * Every alarm goes through a leaky bucket: on and off each take at least `hi - lo`
//!   frames, so it cannot flicker (`no_fast_rise`, `no_fast_fall`).
//! * When the calibration cannot be trusted, nothing about the floor is decided
//!   (`lost_never_clear`).
//!
//! The rules can run natively or as Datalog with a proof for every alarm ([`crate::datalog`]).

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize, Serializer};

use crate::datalog::{self, Clause};
use crate::lens::{CalibState, Calibration, DriftEstimate};

/// A grid cell, `(column, row)`.
pub type Cell = (i64, i64);

pub const FREE: u8 = 0;
pub const RED_ZONE: u8 = 1;
pub const BARRIER: u8 = 2;
pub const HAZARD: u8 = 3;

/// One site's floor: its size and the fixed things on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub floor_w: f64,
    pub floor_h: f64,
    pub cell: f64,
    /// Line-of-fire radius around a forklift, metres.
    pub r_lof: f64,
    /// Two posts closer than this are chained, metres.
    pub link_m: f64,
    pub post_win: i64,
    pub post_min: usize,
    /// Sightings within this distance are one post, metres.
    pub post_r: f64,
    pub s_max: u32,
    pub hi: u32,
    pub lo: u32,
    pub ppe_win: usize,
    pub guardrails: Vec<[[f64; 2]; 2]>,
    pub red_rect: [f64; 4],
    pub spill_c: [f64; 2],
    pub ring_half: f64,
    pub hazard_radius: f64,
    /// The barricade's fill search box, in cells: x0, y0, x1, y1.
    pub fill_box: [i64; 4],
    pub drift_ok_px: f64,
}

/// The site's grid: sizes and the fixed tokens.
#[derive(Clone, Debug)]
pub struct Grid {
    pub site: Site,
    pub w: i64,
    pub h: i64,
    /// Row by row: FREE, RED_ZONE or HAZARD.
    pub tokens: Vec<Vec<u8>>,
    /// Row by row: guardrails.
    pub barrier: Vec<Vec<bool>>,
    pub seed: Cell,
}

impl Grid {
    pub fn new(site: Site) -> Self {
        let (w, h) = ((site.floor_w / site.cell) as i64, (site.floor_h / site.cell) as i64);
        let mut g = Grid { w, h, tokens: vec![vec![FREE; w as usize]; h as usize], barrier: vec![vec![false; w as usize]; h as usize],
                           seed: (0, 0), site };
        for cy in 0..h {
            for cx in 0..w {
                let p = g.c2f((cx, cy));
                let r = g.site.red_rect;
                if r[0] <= p[0] && p[0] <= r[2] && r[1] <= p[1] && p[1] <= r[3] {
                    g.tokens[cy as usize][cx as usize] = RED_ZONE;
                }
                if (p[0] - g.site.spill_c[0]).hypot(p[1] - g.site.spill_c[1]) <= g.site.hazard_radius {
                    g.tokens[cy as usize][cx as usize] = HAZARD;
                }
            }
        }
        for [a, b] in g.site.guardrails.clone() {
            for (cx, cy) in line8(g.f2c(a), g.f2c(b)) {
                g.barrier[cy as usize][cx as usize] = true;
            }
        }
        g.seed = g.f2c(g.site.spill_c);
        g
    }

    /// Floor metres to a cell, clamped to the grid (`proofs/Grammar.lean`: `clamp_in`).
    pub fn f2c(&self, p: [f64; 2]) -> Cell {
        let c = |v: f64, n: i64| ((v / self.site.cell) as i64).max(0).min(n - 1);
        (c(p[0], self.w), c(p[1], self.h))
    }

    /// A cell's centre in metres.
    pub fn c2f(&self, c: Cell) -> [f64; 2] {
        [(c.0 as f64 + 0.5) * self.site.cell, (c.1 as f64 + 0.5) * self.site.cell]
    }
}

/// A thin 8-connected line (Bresenham): how barriers are drawn.
pub fn line8(a: Cell, b: Cell) -> Vec<Cell> {
    let ((mut x0, mut y0), (x1, y1)) = (a, b);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut err, mut out) = (dx + dy, Vec::new());
    loop {
        out.push((x0, y0));
        if (x0, y0) == (x1, y1) {
            return out;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// A 4-connected line: how rays are tested against barriers.
pub fn line4(a: Cell, b: Cell) -> Vec<Cell> {
    let ((mut x, mut y), (x1, y1)) = (a, b);
    let (dx, dy) = ((x1 - x).abs(), (y1 - y).abs());
    let (sx, sy) = (if x1 > x { 1 } else { -1 }, if y1 > y { 1 } else { -1 });
    let (mut ix, mut iy, mut out) = (0i64, 0i64, vec![(x, y)]);
    for _ in 0..dx + dy {
        if (0.5 + ix as f64) * (dy as f64) < (0.5 + iy as f64) * (dx as f64) {
            x += sx;
            ix += 1;
        } else {
            y += sy;
            iy += 1;
        }
        out.push((x, y));
    }
    out
}

/// The leaky bucket's step (`proofs/Grammar.lean`: `stepS`).
#[inline]
pub fn step_s(s: u32, pos: bool, s_max: u32) -> u32 {
    if pos { if s < s_max { s + 1 } else { s_max } } else { s.saturating_sub(1) }
}

/// The alarm after a step: on at `hi`, off at `lo`, unchanged between.
#[inline]
pub fn alarm_next(alarm: bool, s: u32, hi: u32, lo: u32) -> bool {
    if s >= hi { true } else if s <= lo { false } else { alarm }
}

/// A three-valued answer: yes, no, or not known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    Yes,
    No,
    Unknown,
}

impl From<bool> for Tri {
    fn from(b: bool) -> Self {
        if b { Tri::Yes } else { Tri::No }
    }
}

impl Serialize for Tri {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Tri::Yes => s.serialize_bool(true),
            Tri::No => s.serialize_bool(false),
            Tri::Unknown => s.serialize_str("unknown"),
        }
    }
}

/// A PPE verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Ppe {
    Compliant,
    Violation,
    Unknown,
}

/// The PPE decision (`proofs/Grammar.lean`: `decidePPE`): from whether the person is
/// occluded and which labels the conformal set holds.
pub fn decide_ppe(occluded: bool, has_ok: bool, has_violation: bool) -> Ppe {
    match (occluded, has_ok, has_violation) {
        (true, _, _) => Ppe::Unknown,
        (false, true, false) => Ppe::Compliant,
        (false, false, true) => Ppe::Violation,
        _ => Ppe::Unknown,
    }
}

/// One linear layer, `out = w · x + b`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Linear {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

/// The PPE model for one item: standardisation, an MLP with GELU, and the conformal
/// thresholds per class (`q[0]` compliant, `q[1]` violation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PpeHead {
    pub layers: Vec<Linear>,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
    pub q: [f64; 2],
}

fn erf(x: f64) -> f64 {
    // Taylor series near 0, continued fraction for erfc further out: ~1e-15.
    let z = x.abs();
    let r = if z < 2.5 {
        let (mut term, mut sum, z2) = (z, z, z * z);
        for n in 1..200 {
            term *= -z2 / n as f64;
            let t = term / (2 * n + 1) as f64;
            sum += t;
            if t.abs() < 1e-17 {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        let mut f = 0.0;
        for n in (1..=60).rev() {
            f = (n as f64 / 2.0) / (z + f);
        }
        1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + f)
    };
    if x < 0.0 { -r } else { r }
}

impl PpeHead {
    /// Class probabilities `[compliant, violation]` for one feature row.
    pub fn proba(&self, f: &[f32]) -> [f32; 2] {
        let mut x: Vec<f32> = f.iter().zip(&self.mu).zip(&self.sd).map(|((&v, m), s)| ((v as f64 - m) / s) as f32).collect();
        for (li, l) in self.layers.iter().enumerate() {
            let mut y: Vec<f32> = l.w.iter().zip(&l.b).map(|(row, b)| row.iter().zip(&x).map(|(w, v)| w * v).sum::<f32>() + b).collect();
            if li + 1 < self.layers.len() {
                for v in y.iter_mut() {
                    *v = (*v as f64 * 0.5 * (1.0 + erf(*v as f64 / std::f64::consts::SQRT_2))) as f32;
                }
            }
            x = y;
        }
        let m = x[0].max(x[1]);
        let (e0, e1) = ((x[0] - m).exp(), (x[1] - m).exp());
        [e0 / (e0 + e1), e1 / (e0 + e1)]
    }

    pub fn decide(&self, f: &[f32], occluded: bool) -> Ppe {
        self.decide_with_p(f, occluded).0
    }

    /// The verdict and the model's probability of a violation.
    pub fn decide_with_p(&self, f: &[f32], occluded: bool) -> (Ppe, f32) {
        let p = self.proba(f);
        (decide_ppe(occluded, ((1.0f32 - p[0]) as f64) <= self.q[0], ((1.0f32 - p[1]) as f64) <= self.q[1]), p[1])
    }
}

/// One frame of a person's PPE evidence.
#[derive(Clone, Copy, Debug)]
struct Seen {
    detected: bool,
    occluded: bool,
    conf: f64,
    helmet: f64,
    vest: f64,
    box_h: f64,
}

fn ppe_features(hist: &VecDeque<Seen>, win: usize) -> (Vec<f32>, bool) {
    let now = *hist.back().expect("a track has at least one frame");
    let good: Vec<&Seen> = hist.iter().filter(|s| s.detected && !s.occluded).collect();
    let col = |f: &dyn Fn(&Seen) -> f64| -> Vec<f64> { if good.is_empty() { vec![0.0] } else { good.iter().map(|s| f(s)).collect() } };
    let (hc, vc) = (col(&|s| s.helmet), col(&|s| s.vest));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let frac = |v: &[f64]| v.iter().filter(|&&x| x > 0.5).count() as f64 / v.len() as f64;
    let bh = if good.is_empty() { 0.0 } else { mean(&good.iter().map(|s| s.box_h).collect::<Vec<_>>()) };
    let f = [now.helmet, now.vest, now.conf, now.detected as u8 as f64, now.occluded as u8 as f64, mean(&hc), max(&hc), frac(&hc),
             mean(&vc), max(&vc), frac(&vc), good.len() as f64 / win as f64, bh / 50.0];
    (f.iter().map(|&v| v as f32).collect(), now.occluded || !now.detected)
}

/// One find, as the detector gives it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub cls: String,
    pub conf: f64,
    /// left, top, right, bottom in the frame's pixels.
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
}

/// Where to look for a person's helmet and vest on the full-resolution stream.
///
/// The tracker runs on the small grey substream, where a far worker's head is a few pixels:
/// in simulation, helmet MCC was 0.58 there against 0.95 on a 4x crop of the same person from
/// the main stream. So for each confirmed person, crop the head-and-torso band from the
/// matching full-resolution frame, run the PPE detector on the crop, and map its finds back
/// to substream pixels with [`PpeCrop::map_back`]; the engine then takes them like any find.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpeCrop {
    /// left, top, right, bottom in full-resolution pixels.
    pub rect: [f64; 4],
    /// Full-resolution pixels per substream pixel.
    pub scale: f64,
}

impl PpeCrop {
    /// The band from 10 % above the person's box to 65 % down it, with 25 % margin each side,
    /// scaled to the full-resolution frame and clamped to it.
    pub fn for_person(person: &[f64; 4], scale: f64, full_w: f64, full_h: f64) -> Self {
        let [x1, y1, x2, y2] = *person;
        let (h, w) = (y2 - y1, x2 - x1);
        let r = [(x1 - 0.25 * w) * scale, (y1 - 0.10 * h) * scale, (x2 + 0.25 * w) * scale, (y1 + 0.65 * h) * scale];
        PpeCrop { rect: [r[0].max(0.0), r[1].max(0.0), r[2].min(full_w), r[3].min(full_h)], scale }
    }

    /// The crop as whole pixels: x, y, width, height (at least 1 x 1).
    pub fn pixels(&self) -> (u32, u32, u32, u32) {
        let (x0, y0) = (self.rect[0].floor().max(0.0), self.rect[1].floor().max(0.0));
        let (x1, y1) = (self.rect[2].ceil(), self.rect[3].ceil());
        (x0 as u32, y0 as u32, ((x1 - x0) as u32).max(1), ((y1 - y0) as u32).max(1))
    }

    /// A find in crop pixels (origin at [`PpeCrop::pixels`]' corner) back in substream pixels.
    pub fn map_back(&self, det: &Detection) -> Detection {
        let (ox, oy, _, _) = self.pixels();
        let (ox, oy) = (ox as f64, oy as f64);
        let b = det.bbox;
        Detection { cls: det.cls.clone(), conf: det.conf,
                    bbox: [(ox + b[0]) / self.scale, (oy + b[1]) / self.scale, (ox + b[2]) / self.scale, (oy + b[3]) / self.scale] }
    }
}

/// Where a PPE verdict came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PpeSource {
    /// The camera watching the person, this frame.
    Main,
    /// The same camera's last clear look at the person.
    #[serde(rename = "self")]
    OwnHistory,
    /// Another camera that saw the person (matched by floor position), e.g. at an entry.
    Entry,
}

/// A PPE verdict after fusion, with where it came from and how many frames old it is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FusedPpe {
    pub verdict: Ppe,
    pub source: Option<PpeSource>,
    pub age_frames: Option<i64>,
}

/// A remembered decided verdict: what, on which frame, from where.
pub type PpeMemory = (Ppe, i64, PpeSource);

/// The fusion rule (`proofs/Grammar.lean`: `fuse`): the current verdict stands; remembered
/// evidence only fills an UNKNOWN, and only while at most `window` frames old.
pub fn fuse_ppe(main: Ppe, mem: Option<PpeMemory>, frame: i64, window: i64) -> FusedPpe {
    if main != Ppe::Unknown {
        return FusedPpe { verdict: main, source: Some(PpeSource::Main), age_frames: Some(0) };
    }
    match mem {
        Some((v, at, src)) if frame - at <= window => FusedPpe { verdict: v, source: Some(src), age_frames: Some(frame - at) },
        _ => FusedPpe { verdict: Ppe::Unknown, source: None, age_frames: None },
    }
}

/// Remembers each person's last decided helmet and vest verdicts, from their own camera and
/// from other cameras on the same floor, to fill short UNKNOWN gaps (occlusion, a missed box).
///
/// In simulation, the camera's own last clear look within 2 s cut helmet UNKNOWN from 32 % to
/// 1 % (MCC 0.953 to 0.942); an entry camera alone only to 23 %, and adding it to the own
/// history gained nothing. Filled verdicts are the ones to audit: 2.4 % of filled "compliant"
/// were people who had just taken the helmet off. Keep the window short.
#[derive(Debug, Default)]
pub struct PpeFusion {
    /// Use the camera's own last clear look.
    pub use_own: bool,
    /// Use other cameras' verdicts.
    pub use_entry: bool,
    /// How far apart on the floor two cameras' people can be and still be the same person, metres.
    pub assoc_m: f64,
    mem: HashMap<u32, [Option<PpeMemory>; 2]>,
}

impl PpeFusion {
    pub fn new(use_own: bool, use_entry: bool, assoc_m: f64) -> Self {
        PpeFusion { use_own, use_entry, assoc_m, mem: HashMap::new() }
    }

    fn put(&mut self, track: u32, item: usize, v: Ppe, frame: i64, src: PpeSource) {
        let slot = &mut self.mem.entry(track).or_default()[item];
        if slot.is_none_or(|(_, at, _)| frame >= at) {
            *slot = Some((v, frame, src));
        }
    }

    /// Take one frame: this camera's people, and other cameras' people on the same frame.
    pub fn observe(&mut self, frame: i64, main: &[WorkerReport], others: &[WorkerReport]) {
        if self.use_own {
            for w in main {
                for (i, v) in [w.helmet, w.vest].into_iter().enumerate() {
                    if v != Ppe::Unknown {
                        self.put(w.track, i, v, frame, PpeSource::OwnHistory);
                    }
                }
            }
        }
        if self.use_entry && !main.is_empty() {
            for g in others {
                let d = |w: &WorkerReport| ((w.floor_xy[0] - g.floor_xy[0]).powi(2) + (w.floor_xy[1] - g.floor_xy[1]).powi(2)).sqrt();
                let k = (0..main.len()).fold(0, |b, i| if d(&main[i]) < d(&main[b]) { i } else { b });
                if d(&main[k]) <= self.assoc_m {
                    for (i, v) in [g.helmet, g.vest].into_iter().enumerate() {
                        if v != Ppe::Unknown {
                            self.put(main[k].track, i, v, frame, PpeSource::Entry);
                        }
                    }
                }
            }
        }
    }

    /// A person's helmet and vest after fusion.
    pub fn fused(&self, frame: i64, w: &WorkerReport, window: i64) -> [FusedPpe; 2] {
        let mem = self.mem.get(&w.track).copied().unwrap_or_default();
        [fuse_ppe(w.helmet, mem[0], frame, window), fuse_ppe(w.vest, mem[1], frame, window)]
    }

    /// Forget people no longer tracked.
    pub fn retain(&mut self, live: impl Fn(u32) -> bool) {
        self.mem.retain(|&t, _| live(t));
    }
}

/// One crop worth labelling.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LabelCandidate {
    pub frame: i64,
    pub track: u32,
    /// "helmet" or "vest".
    pub item: &'static str,
    /// The person's box on the substream; cut the main-stream crop with [`PpeCrop::for_person`].
    pub person_box: [f64; 4],
    pub p_violation: f32,
    /// "ambiguous" (the conformal set held both labels) or "near_coin_flip".
    pub reason: &'static str,
}

/// Picks the crops to label for fine-tuning on a site's own far-field footage: people in
/// plain view whose PPE verdict was ambiguous or close to a coin flip, at most one crop per
/// person and item every `gap` frames, and at most `per_hour` crops per hour.
///
/// In simulation a queued crop was about ten times likelier to be a model mistake than a
/// random visible person (helmet 23 % against 2.3 %). Label them, add them to
/// `notebooks/finetune_gray.ipynb`'s data, and hold out whole cameras, not frames.
#[derive(Debug)]
pub struct LabelQueue {
    pub gap: i64,
    pub margin: f32,
    pub per_hour: usize,
    pub fps: f64,
    last: HashMap<(u32, usize), i64>,
    taken: VecDeque<i64>,
}

impl LabelQueue {
    pub fn new(gap: i64, margin: f32, per_hour: usize, fps: f64) -> Self {
        LabelQueue { gap, margin, per_hour, fps, last: HashMap::new(), taken: VecDeque::new() }
    }

    /// This frame's candidates.
    pub fn consider(&mut self, frame: i64, workers: &[WorkerReport]) -> Vec<LabelCandidate> {
        let hour = (3600.0 * self.fps) as i64;
        while self.taken.front().is_some_and(|&t| t <= frame - hour) {
            self.taken.pop_front();
        }
        let mut out = Vec::new();
        for w in workers {
            let Some(bbox) = w.bbox else { continue };
            if w.ppe_occluded {
                continue;
            }
            for (i, (item, verdict)) in [("helmet", w.helmet), ("vest", w.vest)].into_iter().enumerate() {
                let Some(p) = w.ppe_p_violation[i] else { continue };
                let reason = if verdict == Ppe::Unknown { "ambiguous" } else if (p - 0.5).abs() < self.margin { "near_coin_flip" } else { continue };
                if self.taken.len() >= self.per_hour || frame - self.last.get(&(w.track, i)).copied().unwrap_or(i64::MIN / 2) < self.gap {
                    continue;
                }
                self.last.insert((w.track, i), frame);
                self.taken.push_back(frame);
                out.push(LabelCandidate { frame, track: w.track, item, person_box: bbox, p_violation: p, reason });
            }
        }
        out
    }
}

struct Track {
    id: u32,
    pos: [f64; 2],
    vel: [f64; 2],
    pred: [f64; 2],
    last: i64,
    hits: u32,
    s: [u32; 2],
    alarm: [bool; 2],
    hist: VecDeque<Seen>,
    ppe: [Ppe; 2],
    ppe_p: [Option<f32>; 2],
    occluded: bool,
    last_box: [f64; 4],
}

struct Post {
    xy: [f64; 2],
    n: u32,
    seen: VecDeque<i64>,
}

/// Which rule engine decides.
#[derive(Clone, Debug)]
pub enum Backend {
    /// Plain code: fastest.
    Native,
    /// The rules as Datalog clauses (yappu-fsm's schema): every alarm carries its derivation.
    Datalog(Vec<Clause>),
}

/// One person on one frame.
#[derive(Clone, Debug, Serialize)]
pub struct WorkerReport {
    pub track: u32,
    pub floor_xy: [f64; 2],
    pub cell: Cell,
    pub line_of_fire: Tri,
    pub red_zone: Tri,
    pub helmet: Ppe,
    pub vest: Ppe,
    /// The person's box on this frame's substream (None when only predicted): where to cut a
    /// [`PpeCrop`] from the main stream.
    pub bbox: Option<[f64; 4]>,
    /// The PPE model's probability of a helmet / vest violation (None without PPE heads).
    pub ppe_p_violation: [Option<f32>; 2],
    /// Whether the PPE view was occluded or the person missed on this frame.
    pub ppe_occluded: bool,
    /// The derivation of each alarm (Datalog backend only).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub proof: Vec<String>,
}

/// The barricade on one frame.
#[derive(Clone, Debug, Serialize)]
pub struct PerimeterReport {
    /// None while warming up or while the calibration is not trusted.
    pub breach: Option<bool>,
    pub warming_up: bool,
    /// This frame's closure test, before the bucket (None while not trusted).
    pub raw_open: Option<bool>,
    pub gap_floor_xy: Option<[f64; 2]>,
    pub posts_persistent: usize,
    /// The escape path from the hazard to the edge (Datalog backend).
    pub escape_path: Option<Vec<Cell>>,
}

/// Everything decided on one frame.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub frame: i64,
    pub workers: Vec<WorkerReport>,
    pub calibration: CalibState,
    pub system_alerts: Vec<&'static str>,
    pub perimeter: PerimeterReport,
}

/// The rules for one camera.
pub struct SafetyEngine {
    pub grid: Grid,
    pub calib: Calibration,
    ppe: Option<[PpeHead; 2]>,
    backend: Backend,
    tracks: Vec<Track>,
    posts: Vec<Post>,
    frame: i64,
    next_id: u32,
    perim_s: u32,
    perim_alarm: bool,
    last_bar: Vec<Vec<bool>>,
}

const LOF: usize = 0;
const RED: usize = 1;

impl SafetyEngine {
    /// `ppe`: the helmet and vest heads, if PPE is to be judged.
    pub fn new(grid: Grid, calib: Calibration, ppe: Option<[PpeHead; 2]>, backend: Backend) -> Self {
        SafetyEngine { grid, calib, ppe, backend, tracks: Vec::new(), posts: Vec::new(), frame: -1, next_id: 0, perim_s: 0, perim_alarm: false, last_bar: Vec::new() }
    }

    fn parts(b: &[f64; 4], by: &HashMap<&str, Vec<&Detection>>) -> (f64, f64) {
        let (x1, y1, x2, h) = (b[0], b[1], b[2], b[3] - b[1]);
        let best = |cls: &str, lo: f64, hi: f64| {
            by.get(cls).map_or(0.0, |ds| {
                ds.iter().fold(0.0f64, |m, d| {
                    let (cx, cy) = ((d.bbox[0] + d.bbox[2]) / 2.0, (d.bbox[1] + d.bbox[3]) / 2.0);
                    if x1 <= cx && cx <= x2 && y1 + lo * h <= cy && cy <= y1 + hi * h { m.max(d.conf) } else { m }
                })
            })
        };
        (best("helmet", -0.05, 0.25), best("vest", 0.15, 0.62))
    }

    fn update_posts(&mut self, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let f = self.frame;
        for p in pts {
            let near = self.posts.iter().enumerate()
                .map(|(i, c)| (i, ((c.xy[0] - p[0]).powi(2) + (c.xy[1] - p[1]).powi(2)).sqrt()))
                .fold(None, |b: Option<(usize, f64)>, (i, d)| match b { Some((_, bd)) if bd <= d => b, _ => Some((i, d)) });
            match near {
                Some((k, d)) if d <= self.grid.site.post_r => {
                    let c = &mut self.posts[k];
                    c.n += 1;
                    let n = c.n.min(20) as f64;
                    c.xy = [c.xy[0] + (p[0] - c.xy[0]) / n, c.xy[1] + (p[1] - c.xy[1]) / n];
                    c.seen.push_back(f);
                }
                _ => self.posts.push(Post { xy: *p, n: 1, seen: VecDeque::from([f]) }),
            }
        }
        let win = self.grid.site.post_win;
        for c in self.posts.iter_mut() {
            while c.seen.front().is_some_and(|&s| s <= f - win) {
                c.seen.pop_front();
            }
        }
        self.posts.retain(|c| !c.seen.is_empty());
        let min = self.grid.site.post_min;
        self.posts.iter().filter(|c| c.seen.iter().collect::<HashSet<_>>().len() >= min).map(|c| c.xy).collect()
    }

    /// The barrier cells of the latest frame (guardrails and the post chain).
    pub fn barrier_mask_now(&self) -> Vec<Vec<bool>> {
        self.last_bar.clone()
    }

    /// Guardrails plus the post chain.
    pub fn barrier_mask(&self, posts: &[[f64; 2]]) -> Vec<Vec<bool>> {
        let mut bar = self.grid.barrier.clone();
        for i in 0..posts.len() {
            for j in i + 1..posts.len() {
                if ((posts[i][0] - posts[j][0]).powi(2) + (posts[i][1] - posts[j][1]).powi(2)).sqrt() <= self.grid.site.link_m {
                    for (cx, cy) in line8(self.grid.f2c(posts[i]), self.grid.f2c(posts[j])) {
                        bar[cy as usize][cx as usize] = true;
                    }
                }
            }
        }
        bar
    }

    /// Whether a 4-connected fill from the hazard reaches the edge of the search box.
    pub fn flood_open(&self, bar: &[Vec<bool>]) -> bool {
        let [x0, y0, x1, y1] = self.grid.site.fill_box;
        let mut seen = HashSet::from([self.grid.seed]);
        let mut q = VecDeque::from([self.grid.seed]);
        while let Some((cx, cy)) = q.pop_front() {
            if cx <= x0 || cy <= y0 || cx >= x1 || cy >= y1 {
                return true;
            }
            for n in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
                if !bar[n.1 as usize][n.0 as usize] && seen.insert(n) {
                    q.push_back(n);
                }
            }
        }
        false
    }

    fn gap(&self, posts: &[[f64; 2]]) -> Option<[f64; 2]> {
        if posts.len() < 3 {
            return None;
        }
        let c = self.grid.site.spill_c;
        let mut p = posts.to_vec();
        p.sort_by(|a, b| (a[1] - c[1]).atan2(a[0] - c[0]).total_cmp(&(b[1] - c[1]).atan2(b[0] - c[0])));
        let n = p.len();
        let d: Vec<f64> = (0..n).map(|i| ((p[i][0] - p[(i + 1) % n][0]).powi(2) + (p[i][1] - p[(i + 1) % n][1]).powi(2)).sqrt()).collect();
        let k = (0..n).fold(0, |b, i| if d[i] > d[b] { i } else { b });
        (d[k] > self.grid.site.link_m).then(|| [(p[k][0] + p[(k + 1) % n][0]) / 2.0, (p[k][1] + p[(k + 1) % n][1]) / 2.0])
    }

    fn native_rules(&self, bar: &[Vec<bool>], fks: &[Cell], workers: &[(u32, Cell)]) -> (bool, Vec<[bool; 2]>) {
        let raw = workers.iter().map(|&(_, wc)| {
            let w = self.grid.c2f(wc);
            let lof = fks.iter().any(|&fc| {
                let f = self.grid.c2f(fc);
                let ray = line4(wc, fc);
                let inner = &ray[1.min(ray.len())..ray.len().saturating_sub(1).max(1.min(ray.len()))];
                (w[0] - f[0]).hypot(w[1] - f[1]) <= self.grid.site.r_lof && !inner.iter().any(|&(x, y)| bar[y as usize][x as usize])
            });
            [lof, self.grid.tokens[wc.1 as usize][wc.0 as usize] == RED_ZONE]
        }).collect();
        (self.flood_open(bar), raw)
    }

    /// One frame: its finds and, when a drift check ran on it, what that found.
    pub fn process(&mut self, dets: &[Detection], drift: Option<&DriftEstimate>) -> Report {
        self.frame += 1;
        let f = self.frame;
        if let Some(e) = drift {
            self.calib.apply_drift(e);
        }
        let gated = self.calib.state == CalibState::Unknown;
        let mut by: HashMap<&str, Vec<&Detection>> = HashMap::new();
        for d in dets {
            by.entry(d.cls.as_str()).or_default().push(d);
        }
        // people on the floor; a box cut off at the bottom has no trustworthy foot point
        struct P { conf: f64, xy: [f64; 2], trunc: bool, h: f64, helmet: f64, vest: f64, bbox: [f64; 4] }
        let people: Vec<P> = by.get("person").map_or(vec![], |ds| ds.iter().map(|d| {
            let [x1, y1, x2, y2] = d.bbox;
            let (h, w) = (y2 - y1, (x2 - x1).max(1e-3));
            let (helmet, vest) = Self::parts(&d.bbox, &by);
            P { conf: d.conf, xy: self.calib.to_floor((x1 + x2) / 2.0, y2), trunc: h / w < 1.9, h, helmet, vest, bbox: d.bbox }
        }).collect());
        // two-stage matching on the floor, in metres
        for t in self.tracks.iter_mut() {
            t.pred = [t.pos[0] + t.vel[0], t.pos[1] + t.vel[1]];
        }
        let mut free: HashSet<usize> = (0..self.tracks.len()).collect();
        let mut matched: Vec<(usize, usize)> = Vec::new();
        for high in [true, false] {
            let taken: HashSet<usize> = matched.iter().map(|m| m.1).collect();
            let cand: Vec<usize> = (0..people.len()).filter(|&i| !taken.contains(&i) && (people[i].conf >= 0.45) == high && people[i].conf >= 0.15).collect();
            let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
            for &t in &free {
                for &i in &cand {
                    let (p, q) = (self.tracks[t].pred, people[i].xy);
                    pairs.push((((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt(), t, i));
                }
            }
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            let mut used = HashSet::new();
            for (d, t, i) in pairs {
                let gate = if people[i].trunc { 2.0 } else { 1.0 };
                if free.contains(&t) && !used.contains(&i) && d <= gate {
                    free.remove(&t);
                    used.insert(i);
                    matched.push((t, i));
                }
            }
        }
        for &(t, i) in &matched {
            let (tr, p) = (&mut self.tracks[t], &people[i]);
            if !p.trunc {
                tr.vel = [0.6 * tr.vel[0] + 0.4 * (p.xy[0] - tr.pos[0]), 0.6 * tr.vel[1] + 0.4 * (p.xy[1] - tr.pos[1])];
                tr.pos = p.xy;
            } else {
                tr.pos = tr.pred;
            }
            tr.last = f;
            tr.hits += 1;
            tr.last_box = p.bbox;
            push_seen(&mut tr.hist, Seen { detected: true, occluded: p.trunc || p.conf < 0.35, conf: p.conf, helmet: p.helmet, vest: p.vest, box_h: p.h },
                      self.grid.site.ppe_win);
        }
        for &t in &free {
            let tr = &mut self.tracks[t];
            tr.pos = tr.pred;
            tr.vel = [tr.vel[0] * 0.8, tr.vel[1] * 0.8];
            push_seen(&mut tr.hist, Seen { detected: false, occluded: true, conf: 0.0, helmet: 0.0, vest: 0.0, box_h: 0.0 }, self.grid.site.ppe_win);
        }
        let taken: HashSet<usize> = matched.iter().map(|m| m.1).collect();
        for (i, p) in people.iter().enumerate() {
            if !taken.contains(&i) && p.conf >= 0.45 && !p.trunc {
                self.next_id += 1;
                let mut hist = VecDeque::new();
                push_seen(&mut hist, Seen { detected: true, occluded: false, conf: p.conf, helmet: p.helmet, vest: p.vest, box_h: p.h }, self.grid.site.ppe_win);
                self.tracks.push(Track { id: self.next_id, pos: p.xy, vel: [0.0; 2], pred: p.xy, last: f, hits: 1, s: [0; 2], alarm: [false; 2],
                                         hist, ppe: [Ppe::Unknown; 2], ppe_p: [None; 2], occluded: false, last_box: p.bbox });
            }
        }
        self.tracks.retain(|t| f - t.last <= 15);
        // forklifts and posts
        let foot = |d: &&Detection| [(d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3]];
        let fks: Vec<[f64; 2]> = by.get("forklift").map_or(vec![], |ds| ds.iter().filter(|d| d.conf >= 0.3).map(|d| { let [u, v] = foot(d); self.calib.to_floor(u, v) }).collect());
        let cur: Vec<[f64; 2]> = by.get("barrier_post").map_or(vec![], |ds| ds.iter().filter(|d| d.conf >= 0.3).map(|d| { let [u, v] = foot(d); self.calib.to_floor(u, v) }).collect());
        let warm = f < self.grid.site.post_win - 1;
        let posts = self.update_posts(&cur);
        let bar = self.barrier_mask(&posts);
        let fk_cells: Vec<Cell> = fks.iter().map(|&x| self.grid.f2c(x)).collect();
        let conf: Vec<usize> = (0..self.tracks.len()).filter(|&i| self.tracks[i].hits >= 3).collect();
        let workers: Vec<(u32, Cell)> = conf.iter().map(|&i| (self.tracks[i].id, self.grid.f2c(self.tracks[i].pos))).collect();
        // rules
        let (mut opened, mut raw, mut proofs, mut path) = (None, vec![None; workers.len()], vec![Vec::new(); workers.len()], None);
        if !gated {
            match &self.backend {
                Backend::Native => {
                    let (o, r) = self.native_rules(&bar, &fk_cells, &workers);
                    opened = Some(o);
                    raw = r.into_iter().map(Some).collect();
                }
                Backend::Datalog(clauses) => {
                    let facts = datalog::grid_facts(&self.grid, &bar, &fk_cells, &workers);
                    let m = datalog::Model::run(clauses, facts);
                    let o = m.has(&["alert", "scene", "perimeter-breach"]);
                    opened = Some(o);
                    for (k, &(tid, _)) in workers.iter().enumerate() {
                        let t = format!("T{tid}");
                        let r = [m.has(&["alert", &t, "line-of-fire"]), m.has(&["alert", &t, "red-zone"])];
                        for (ev, on) in [("line-of-fire", r[0]), ("red-zone", r[1])] {
                            if on {
                                proofs[k].extend(m.explain(&["alert", &t, ev]));
                            }
                        }
                        raw[k] = Some(r);
                    }
                    if o {
                        path = m.escape_path();
                    }
                }
            }
        }
        if !warm && !gated {
            self.perim_s = step_s(self.perim_s, opened == Some(true), self.grid.site.s_max);
            self.perim_alarm = alarm_next(self.perim_alarm, self.perim_s, self.grid.site.hi, self.grid.site.lo);
        }
        let mut out = Vec::new();
        let (s_max, hi, lo, win) = (self.grid.site.s_max, self.grid.site.hi, self.grid.site.lo, self.grid.site.ppe_win);
        for (k, &i) in conf.iter().enumerate() {
            let tr = &mut self.tracks[i];
            if let Some(r) = raw[k] {
                for j in [LOF, RED] {
                    tr.s[j] = step_s(tr.s[j], r[j], s_max);
                    tr.alarm[j] = alarm_next(tr.alarm[j], tr.s[j], hi, lo);
                }
            }
            if let Some(heads) = &self.ppe {
                let (feat, occ) = ppe_features(&tr.hist, win);
                let (h, v) = (heads[0].decide_with_p(&feat, occ), heads[1].decide_with_p(&feat, occ));
                tr.ppe = [h.0, v.0];
                tr.ppe_p = [Some(h.1), Some(v.1)];
                tr.occluded = occ;
            }
            let tri = |a: bool| if gated { Tri::Unknown } else { Tri::from(a) };
            out.push(WorkerReport { track: tr.id, floor_xy: tr.pos, cell: workers[k].1, line_of_fire: tri(tr.alarm[LOF]), red_zone: tri(tr.alarm[RED]),
                                    helmet: tr.ppe[0], vest: tr.ppe[1], bbox: (tr.last == f).then_some(tr.last_box),
                                    ppe_p_violation: tr.ppe_p, ppe_occluded: tr.occluded, proof: std::mem::take(&mut proofs[k]) });
        }
        let gap = if opened == Some(true) { self.gap(&posts) } else { None };
        self.last_bar = bar;
        Report {
            frame: f,
            workers: out,
            calibration: self.calib.state,
            system_alerts: if gated { vec!["calibration-lost"] } else { vec![] },
            perimeter: PerimeterReport { breach: if warm || gated { None } else { Some(self.perim_alarm) }, warming_up: warm, raw_open: opened,
                                         gap_floor_xy: gap, posts_persistent: posts.len(), escape_path: path },
        }
    }
}

fn push_seen(h: &mut VecDeque<Seen>, s: Seen, win: usize) {
    if h.len() == win {
        h.pop_front();
    }
    h.push_back(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(track: u32, xy: [f64; 2], helmet: Ppe) -> WorkerReport {
        WorkerReport { track, floor_xy: xy, cell: (0, 0), line_of_fire: Tri::No, red_zone: Tri::No, helmet, vest: Ppe::Unknown, bbox: None,
                       ppe_p_violation: [None; 2], ppe_occluded: false, proof: vec![] }
    }

    #[test]
    fn fusion_never_hides_a_current_verdict_and_expires() {
        // mirror of fuse_keeps_violation / fuse_stale_is_unknown / fuse_compliant_has_witness
        let ppe = [Ppe::Compliant, Ppe::Violation, Ppe::Unknown];
        let src = PpeSource::Entry;
        for &main in &ppe {
            for &m in &ppe {
                for at in [0i64, 5, 10] {
                    let f = fuse_ppe(main, Some((m, at, src)), 10, 3);
                    if main != Ppe::Unknown {
                        assert_eq!(f.verdict, main);
                    }
                    if f.verdict == Ppe::Compliant {
                        assert!(main == Ppe::Compliant || (main == Ppe::Unknown && m == Ppe::Compliant && 10 - at <= 3));
                    }
                    if main == Ppe::Unknown && 10 - at > 3 {
                        assert_eq!(f.verdict, Ppe::Unknown);
                    }
                }
            }
        }
        // an entry-camera verdict fills a gap for the person standing at the same place, not for another
        let mut fu = PpeFusion::new(false, true, 0.8);
        fu.observe(100, &[worker(7, [10.0, 6.0], Ppe::Unknown), worker(8, [14.0, 6.0], Ppe::Unknown)], &[worker(1, [10.3, 6.1], Ppe::Violation)]);
        assert_eq!(fu.fused(110, &worker(7, [10.0, 6.0], Ppe::Unknown), 20)[0].verdict, Ppe::Violation);
        assert_eq!(fu.fused(110, &worker(8, [14.0, 6.0], Ppe::Unknown), 20)[0].verdict, Ppe::Unknown);
        assert_eq!(fu.fused(130, &worker(7, [10.0, 6.0], Ppe::Unknown), 20)[0].verdict, Ppe::Unknown);
        assert_eq!(fu.fused(110, &worker(7, [10.0, 6.0], Ppe::Compliant), 20)[0].source, Some(PpeSource::Main));
    }

    #[test]
    fn the_label_queue_takes_visible_doubtful_people_and_caps() {
        let mut q = LabelQueue::new(20, 0.25, 3, 10.0);
        let mut w = worker(1, [0.0, 0.0], Ppe::Unknown);
        w.bbox = Some([10.0, 10.0, 20.0, 40.0]);
        w.ppe_p_violation = [Some(0.48), Some(0.02)];
        w.vest = Ppe::Compliant;
        assert_eq!(q.consider(0, std::slice::from_ref(&w)).len(), 1); // helmet ambiguous; vest confidently fine
        assert!(q.consider(5, std::slice::from_ref(&w)).is_empty()); // same person too soon
        w.ppe_occluded = true;
        assert!(q.consider(30, std::slice::from_ref(&w)).is_empty()); // occluded teaches nothing
        w.ppe_occluded = false;
        let mut others: Vec<WorkerReport> = (2..6).map(|t| WorkerReport { track: t, ..w.clone() }).collect();
        others[0].ppe_p_violation = [Some(0.6), Some(0.02)];
        others[0].helmet = Ppe::Violation; // decided, but near a coin flip
        assert_eq!(q.consider(40, &others).len(), 2); // hourly cap of 3 reached
    }

    #[test]
    fn a_ppe_crop_maps_back_onto_the_person() {
        // a 20 x 50 px person on the substream, 4x main stream
        let c = PpeCrop::for_person(&[100.0, 200.0, 120.0, 250.0], 4.0, 2560.0, 1920.0);
        assert_eq!(c.rect, [380.0, 780.0, 500.0, 930.0]);
        assert_eq!(c.pixels(), (380, 780, 120, 150));
        // a helmet found at the top middle of the crop lands on the person's head band
        let h = c.map_back(&Detection { cls: "helmet".into(), conf: 0.9, bbox: [44.0, 20.0, 76.0, 44.0] });
        assert_eq!(h.bbox, [106.0, 200.0, 114.0, 206.0]);
        // clamped at the frame's edge
        let e = PpeCrop::for_person(&[0.0, 0.0, 20.0, 50.0], 4.0, 2560.0, 1920.0);
        assert_eq!(e.rect[..2], [0.0, 0.0]);
    }

    #[test]
    fn a_four_ray_never_crosses_an_eight_line() {
        // every 8-connected diagonal barrier stops every 4-connected ray that crosses it
        let wall: HashSet<Cell> = line8((0, 0), (9, 9)).into_iter().collect();
        for (a, b) in [((0, 5), (5, 0)), ((1, 8), (8, 1)), ((0, 9), (9, 0)), ((2, 7), (7, 3))] {
            assert!(line4(a, b).iter().any(|c| wall.contains(c)), "{a:?}->{b:?} slipped through");
        }
    }

    #[test]
    fn line4_steps_are_four_neighbours() {
        let l = line4((3, 2), (-4, 9));
        assert!(l.windows(2).all(|w| (w[0].0 - w[1].0).abs() + (w[0].1 - w[1].1).abs() == 1));
        assert_eq!((l[0], *l.last().unwrap()), ((3, 2), (-4, 9)));
    }

    #[test]
    fn the_bucket_cannot_flicker() {
        // mirror of no_fast_rise / no_fast_fall: between an OFF state and the next ON, >= hi - lo frames
        let (s_max, hi, lo) = (6, 3, 0);
        let mut seed = 12345u64;
        for _ in 0..2000 {
            let (mut s, mut alarm, mut last_off) = (0u32, false, -1i64);
            for k in 0..80i64 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let s2 = step_s(s, !(seed >> 33).is_multiple_of(3), s_max);
                assert!(s2.abs_diff(s) <= 1 && s2 <= s_max);
                s = s2;
                let new = alarm_next(alarm, s, hi, lo);
                if s <= lo {
                    last_off = k;
                }
                assert!(!(new && !alarm) || k - last_off >= (hi - lo) as i64);
                alarm = new;
            }
        }
    }

    #[test]
    fn occluded_is_never_compliant() {
        for a in [false, true] {
            for b in [false, true] {
                assert_ne!(decide_ppe(true, a, b), Ppe::Compliant);
            }
        }
        assert_eq!(decide_ppe(false, true, true), Ppe::Unknown);
    }

    #[test]
    fn erf_matches_known_values() {
        for (x, e) in [(0.0, 0.0), (0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (2.0, 0.995_322_265_018_952_7),
                       (3.0, 0.999_977_909_503_001_4), (-1.5, -0.966_105_146_475_310_7)] {
            assert!((erf(x) - e).abs() < 1e-13, "erf({x}) = {} not {e}", erf(x));
        }
    }
}
