//! Following: looks spread across cameras, a Kalman filter per object between looks,
//! and ByteTrack's matching on each look. Step for step what `tamserve.boxes.BoxTracker` does.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

use crate::kalman::Kalman;
use crate::regions::{self, Regions, Square};
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
    /// Crop looks to squares around what moved and what is followed (see
    /// [`regions`](crate::regions)); None looks at whole frames, as the Python version does.
    pub regions: Option<Regions>,
}

impl<K> Default for TrackerOptions<K> {
    fn default() -> Self {
        TrackerOptions { look_every: LookEvery::All(3), hold: 30, start: 0.6, high: 0.5, regions: None }
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
    full_next: bool, // a region cut a find: look at the whole frame next time
}

impl Default for Camera {
    fn default() -> Self {
        Camera { objects: BTreeMap::new(), since_look: None, next_id: 1, size: None, full_next: false } // ids count from 1
    }
}

/// Find and follow objects on every camera; call [`step`](BoxTracker::step) once per round.
pub struct BoxTracker<D: Detect = Detector, K: Eq + Hash + Clone = String> {
    pub detector: D,
    pub opts: TrackerOptions<K>,
    cameras: HashMap<K, Camera>,
    joined: u32, // cameras that have had their first look
    looks: u64,  // frames the detector has been given
    skipped: Vec<K>, // cameras the latest step did not look at because they were quiet
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
        Ok(BoxTracker { detector, opts, cameras: HashMap::new(), joined: 0, looks: 0, skipped: Vec::new() })
    }

    /// Frames the detector has been given so far.
    pub fn looks(&self) -> u64 {
        self.looks
    }

    /// The cameras the latest [`step_gated`](BoxTracker::step_gated) would have looked at
    /// but did not, because they were quiet with nothing followed.
    pub fn skipped(&self) -> &[K] {
        &self.skipped
    }

    /// Forget a camera.
    pub fn drop_camera(&mut self, cam: &K) {
        self.cameras.remove(cam);
    }

    /// The cameras that look this step, spread as tamserve's trackers spread them: a new
    /// camera at once; others once their interval has passed, no more in one step than
    /// the intervals call for on average, those that look most often first.
    ///
    /// A camera that is `quiet` and follows nothing (not even an object on trial or held
    /// unseen) is left out once it has had its first look, and stays due: it looks on the
    /// first step that is not quiet.
    fn due(&mut self, cams: &[&K], quiet: &dyn Fn(&K) -> bool) -> Vec<K> {
        let every = |c: &K| self.opts.look_every.of(c);
        let mut out: Vec<K> = cams.iter().filter(|c| self.cameras[**c].since_look.is_none()).map(|c| (*c).clone()).collect();
        let (mut due, idle): (Vec<&K>, Vec<&K>) = cams.iter().copied()
            .filter(|c| self.cameras[*c].since_look.is_some_and(|s| s >= every(c)))
            .partition(|c| !(self.cameras[*c].objects.is_empty() && quiet(c)));
        self.skipped = idle.into_iter().cloned().collect();
        due.sort_by_key(|c| (every(c), std::cmp::Reverse(self.cameras[*c].since_look)));
        let cap = looks_per_step(self.cameras.keys().map(every));
        out.extend(due.into_iter().take(cap).cloned());
        out
    }

    /// `frames`: each camera's newest frame, in any order. Returns, for each camera,
    /// what the latest look on it saw: object id -> [`Found`]. Ids count from 1 on each camera.
    pub fn step(&mut self, frames: &[(K, GrayFrame)]) -> Result<HashMap<K, BTreeMap<u32, Found>>> {
        self.step_gated(frames, &|_| false)
    }

    /// [`step`](BoxTracker::step), but a camera for which `quiet` is true, and that follows
    /// nothing, is not looked at: its frame costs no detection. With `quiet` never true
    /// this is `step` exactly.
    pub fn step_gated(&mut self, frames: &[(K, GrayFrame)], quiet: &dyn Fn(&K) -> bool)
                      -> Result<HashMap<K, BTreeMap<u32, Found>>> {
        self.step_regions(frames, quiet, &|_| Vec::new())
    }

    /// [`step_gated`](BoxTracker::step_gated), and with `regions` set, each look cropped to
    /// squares around `moved(camera)` and the camera's followed objects. Without `regions`
    /// this is `step_gated`, and `moved` is not asked.
    pub fn step_regions(&mut self, frames: &[(K, GrayFrame)], quiet: &dyn Fn(&K) -> bool, moved: &dyn Fn(&K) -> Vec<BBox>)
                        -> Result<HashMap<K, BTreeMap<u32, Found>>> {
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
        let looking = self.due(&cams, quiet);
        // What each looking camera's look is: its whole frame, or squares of it.
        let mut pictures: Vec<GrayFrame> = Vec::with_capacity(looking.len());
        let mut parts: Vec<Option<Vec<Square>>> = Vec::with_capacity(looking.len());
        for c in &looking {
            let f = frames.iter().find(|(k, _)| k == c).unwrap().1;
            let cam = self.cameras.get_mut(c).unwrap();
            let squares = match &self.opts.regions {
                Some(r) if cam.since_look.is_some() && !std::mem::take(&mut cam.full_next) => {
                    let mut around = moved(c);
                    around.extend(cam.objects.values().map(|o| {
                        let b = o.kalman.bbox();
                        [b[0].max(0.0), b[1].max(0.0), b[2].min(f.width as f32), b[3].min(f.height as f32)]
                    }).filter(|b| b[2] > b[0] && b[3] > b[1]));
                    regions::plan(f.width, f.height, &around, r)
                }
                _ => None,
            };
            match &squares {
                None => pictures.push(f),
                Some(sq) => pictures.extend(sq.iter().map(|&[x0, y0, side]| {
                    GrayFrame::with_stride(&f.pixels[y0 * f.stride + x0..], side, side, f.stride).expect("a square inside the frame")
                })),
            }
            parts.push(squares);
        }
        self.looks += pictures.len() as u64;
        let mut found = self.detector.detect(&pictures)?.into_iter();
        let mut joined = Vec::with_capacity(looking.len());
        for (c, squares) in looking.iter().zip(&parts) {
            match squares {
                None => joined.push(found.next().unwrap_or_default()),
                Some(sq) => {
                    let (w, h) = self.cameras[c].size.map(|(h, w)| (w, h)).unwrap();
                    let each: Vec<(Finds, Square)> = sq.iter().map(|&s| (found.next().unwrap_or_default(), s)).collect();
                    let (finds, cut) = regions::join(&each, w, h);
                    self.cameras.get_mut(c).unwrap().full_next = cut;
                    joined.push(finds);
                }
            }
        }
        for (c, finds) in looking.iter().zip(joined) {
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
