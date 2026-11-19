//! Following: looks spread across cameras, a Kalman filter per object between looks,
//! and ByteTrack's matching on each look. Step for step what `tamserve.boxes.BoxTracker` does.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

use crate::kalman::Kalman;
use crate::{iou, BBox, Detect, Detector, Error, Finds, Found, GrayFrame, Result};

/// Frames between a camera's looks: the same for all, or some cameras' own.
#[derive(Clone, Debug)]
pub enum LookEvery<K> {
    All(u32),
    /// These cameras' own; the others keep `default`.
    Each { cameras: HashMap<K, u32>, default: u32 },
}

impl<K: Eq + Hash> LookEvery<K> {
    fn of(&self, cam: &K) -> u32 {
        match self {
            LookEvery::All(k) => *k,
            LookEvery::Each { cameras, default } => *cameras.get(cam).unwrap_or(default),
        }
    }

    fn check(&self) -> Result<()> {
        let bad = match self {
            LookEvery::All(k) => *k < 1,
            LookEvery::Each { cameras, default } => *default < 1 || cameras.values().any(|&k| k < 1),
        };
        if bad { Err(Error::Invalid("look_every must be at least 1".into())) } else { Ok(()) }
    }
}

/// How a [`BoxTracker`] follows. `Default` is the Python version's.
#[derive(Clone, Debug)]
pub struct TrackerOptions<K = String> {
    /// Frames between a camera's looks. 1 looks at every frame; 3 costs a third as much
    /// and finds new things up to 2 frames later. Can be changed while running.
    pub look_every: LookEvery<K>,
    /// Frames an object no look has seen is still followed, out of the answer, so that
    /// it has its id back if a look sees it again.
    pub hold: u32,
    /// The confidence a find needs to start something new.
    pub start: f32,
    /// Finds from this confidence are matched first; weaker ones may only confirm an
    /// object already followed (ByteTrack's second round).
    pub high: f32,
}

impl<K> Default for TrackerOptions<K> {
    fn default() -> Self {
        TrackerOptions { look_every: LookEvery::All(3), hold: 30, start: 0.6, high: 0.5 }
    }
}

struct Object {
    kalman: Kalman,
    label: usize,
    score: f32,
    confirmed: bool,
    unseen: u32, // frames since a look last saw it
}

struct Camera {
    objects: BTreeMap<u32, Object>, // id -> object, ids in the order they started
    since_look: Option<u32>,        // frames since the last look; None before the first
    next_id: u32,
    size: Option<(usize, usize)>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { objects: BTreeMap::new(), since_look: None, next_id: 1, size: None } // ids count from 1
    }
}

/// Find and follow objects on every camera; call [`step`](BoxTracker::step) once per round.
pub struct BoxTracker<D: Detect = Detector, K: Eq + Hash + Clone = String> {
    pub detector: D,
    pub opts: TrackerOptions<K>,
    cameras: HashMap<K, Camera>,
    joined: u32, // cameras that have had their first look
}

/// box index -> track index, one to one, the best overlap first, down to `least`.
fn assign(tracks: &[BBox], boxes: &[BBox], least: f32) -> BTreeMap<usize, usize> {
    let mut out = BTreeMap::new();
    if tracks.is_empty() || boxes.is_empty() {
        return out;
    }
    let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(boxes.len() * tracks.len());
    for (i, b) in boxes.iter().enumerate() {
        for (j, t) in tracks.iter().enumerate() {
            pairs.push((iou(b, t), i, j));
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0)); // stable: ties keep box-major order
    let mut taken = vec![false; tracks.len()];
    for (v, i, j) in pairs {
        if v < least {
            break;
        }
        if !out.contains_key(&i) && !taken[j] {
            out.insert(i, j);
            taken[j] = true;
        }
    }
    out
}

/// Σ 1/k, rounded up: how many looks a step should hold, on average.
fn looks_per_step(every: impl Iterator<Item = u32>) -> usize {
    let (mut num, mut den) = (0u128, 1u128);
    for k in every {
        let k = k as u128;
        num = num * k + den;
        den *= k;
        let g = gcd(num, den);
        num /= g;
        den /= g;
    }
    num.div_ceil(den).max(1) as usize
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

impl<K: Eq + Hash + Clone> BoxTracker<Detector, K> {
    /// Build the detector from a 1-channel model, then the tracker.
    pub fn from_model(model: impl AsRef<std::path::Path>, detector: crate::DetectorOptions,
                      opts: TrackerOptions<K>) -> Result<Self> {
        BoxTracker::new(Detector::new(model, detector)?, opts)
    }
}

impl<D: Detect, K: Eq + Hash + Clone> BoxTracker<D, K> {
    pub fn new(detector: D, opts: TrackerOptions<K>) -> Result<Self> {
        opts.look_every.check()?;
        Ok(BoxTracker { detector, opts, cameras: HashMap::new(), joined: 0 })
    }

    /// Forget a camera.
    pub fn drop_camera(&mut self, cam: &K) {
        self.cameras.remove(cam);
    }

    /// The cameras that look this step, spread as tamserve's trackers spread them: a new
    /// camera at once; others once their interval has passed, no more in one step than
    /// the intervals call for on average, those that look most often first.
    fn due(&self, cams: &[&K]) -> Vec<K> {
        let every = |c: &K| self.opts.look_every.of(c);
        let mut out: Vec<K> = cams.iter().filter(|c| self.cameras[**c].since_look.is_none()).map(|c| (*c).clone()).collect();
        let mut due: Vec<&K> = cams.iter().copied()
            .filter(|c| self.cameras[*c].since_look.unwrap_or(0) >= every(c))
            .collect();
        due.sort_by_key(|c| (every(c), std::cmp::Reverse(self.cameras[*c].since_look)));
        let cap = looks_per_step(self.cameras.keys().map(every));
        out.extend(due.into_iter().take(cap).cloned());
        out
    }

    /// `frames`: each camera's newest frame, in any order. Returns, for each camera,
    /// what the latest look on it saw: object id -> [`Found`]. Ids count from 1 on each camera.
    pub fn step(&mut self, frames: &[(K, GrayFrame)]) -> Result<HashMap<K, BTreeMap<u32, Found>>> {
        self.opts.look_every.check()?;
        for (c, f) in frames {
            let shape = (f.height, f.width);
            let cam = self.cameras.entry(c.clone()).or_default();
            if cam.size.is_some_and(|s| s != shape) {
                *cam = Camera::default(); // the camera changed resolution: start it afresh
            }
            cam.size = Some(shape);
            for o in cam.objects.values_mut() {
                o.kalman.predict();
                o.unseen += 1;
            }
        }
        let cams: Vec<&K> = frames.iter().map(|(c, _)| c).collect();
        let looking = self.due(&cams);
        let pictures: Vec<GrayFrame> = looking.iter()
            .map(|c| frames.iter().find(|(k, _)| k == c).unwrap().1)
            .collect();
        let found = self.detector.detect(&pictures)?;
        for (c, finds) in looking.iter().zip(found) {
            let every = self.opts.look_every.of(c);
            let (high, start, hold) = (self.opts.high, self.opts.start, self.opts.hold);
            let cam = self.cameras.get_mut(c).unwrap();
            look(cam, &finds, high, start, hold);
            cam.since_look = Some(match cam.since_look {
                None => {
                    // first look: set this camera's later looks apart from the others'
                    let s = self.joined % every;
                    self.joined += 1;
                    s
                }
                Some(_) => 0,
            });
        }
        let names = self.detector.names();
        let mut out = HashMap::with_capacity(frames.len());
        for (c, _) in frames {
            let cam = self.cameras.get_mut(c).unwrap();
            let since = cam.since_look.unwrap() + 1;
            cam.since_look = Some(since);
            let seen = cam.objects.iter()
                .filter(|(_, o)| o.confirmed && o.unseen < since)
                .map(|(&t, o)| (t, Found { bbox: o.kalman.bbox(), score: o.score, label: names[o.label].clone() }))
                .collect();
            out.insert(c.clone(), seen);
        }
        Ok(out)
    }
}

/// ByteTrack's association, per class: confident finds against everything followed,
/// weak finds against what is left, new objects on trial last.
fn look(cam: &mut Camera, finds: &Finds, high: f32, start: f32, hold: u32) {
    let first = cam.since_look.is_none();
    let labels: BTreeSet<usize> = finds.labels.iter().copied().chain(cam.objects.values().map(|o| o.label)).collect();
    for label in labels {
        let mine: Vec<usize> = (0..finds.len()).filter(|&i| finds.labels[i] == label).collect();
        let b: Vec<BBox> = mine.iter().map(|&i| finds.boxes[i]).collect();
        let s: Vec<f32> = mine.iter().map(|&i| finds.scores[i]).collect();
        let ids: Vec<u32> = cam.objects.iter().filter(|(_, o)| o.label == label).map(|(&t, _)| t).collect();
        let sure: Vec<u32> = ids.iter().copied().filter(|t| cam.objects[t].confirmed).collect();
        let trial: Vec<u32> = ids.iter().copied().filter(|t| !cam.objects[t].confirmed).collect();
        let strong: Vec<usize> = (0..b.len()).filter(|&i| s[i] >= high).collect();
        let weak: Vec<usize> = (0..b.len()).filter(|&i| s[i] < high).collect();
        let mut seen: BTreeSet<u32> = BTreeSet::new();

        let mut take = |tracks: &[u32], idx: &[usize], least: f32, seen: &mut BTreeSet<u32>| -> Vec<usize> {
            let boxes: Vec<BBox> = tracks.iter().map(|t| cam.objects[t].kalman.bbox()).collect();
            let hits = assign(&boxes, &idx.iter().map(|&i| b[i]).collect::<Vec<_>>(), least);
            for (&i, &j) in &hits {
                let o = cam.objects.get_mut(&tracks[j]).unwrap();
                o.kalman.update(&b[idx[i]]);
                o.score = s[idx[i]];
                o.unseen = 0;
                o.confirmed = true;
                seen.insert(tracks[j]);
            }
            (0..idx.len()).filter(|i| !hits.contains_key(i)).map(|i| idx[i]).collect()
        };

        let strong = take(&sure, &strong, 0.2, &mut seen);
        let rest: Vec<u32> = sure.iter().copied().filter(|t| !seen.contains(t)).collect();
        take(&rest, &weak, 0.5, &mut seen);
        let strong = take(&trial, &strong, 0.3, &mut seen);
        for t in &trial {
            if !seen.contains(t) {
                cam.objects.remove(t); // on trial and not seen again: a false find
            }
        }
        for i in strong {
            if s[i] >= start {
                cam.objects.insert(cam.next_id, Object { kalman: Kalman::new(&b[i]), label, score: s[i],
                                                         confirmed: first, unseen: 0 });
                cam.next_id += 1;
            }
        }
    }
    cam.objects.retain(|_, o| o.unseen <= hold);
}
