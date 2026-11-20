//! Recordings kept by what was in them: every segment for `all_days`, segments with
//! motion for `motion_days`, segments with an object for `objects_days` (or a label's
//! own days), and segments an operator pinned for as long as the pin stays.
//!
//! The camera writes its stream to short MPEG-TS segments, `DIR/<camera>/<start ms>.ts`,
//! without decoding it. The runner writes what each camera showed, as intervals of
//! activity, to `DIR/<camera>/activity/<day>.jsonl`, and with them how far that log
//! covers. A segment's **keep-until** is its end plus the longest retention of any
//! activity within `pre_seconds` before it to `post_seconds` after it, so a clip keeps
//! the seconds that lead up to an event and follow it. A segment past the end of the
//! activity log (the box stopped before writing what it showed) is kept for the longest
//! retention there is, since what it holds is unknown.
//!
//! A sweep deletes:
//! 1. every segment whose keep-until has passed;
//! 2. while the recordings take more than `max_gb`, the segments that would expire
//!    first, in order of keep-until: so a segment with only motion goes before one with
//!    a person, and nothing with an object goes while a motion-only segment is left of
//!    an earlier expiry.
//!
//! Never deleted: a pinned segment, and each camera's newest segment (it may be being
//! written). `proofs/Retention.lean` proves both rules keep these promises.
//!
//! Pins are in `DIR/pins.json`: `[{"camera": "gate", "from": ms, "to": ms, "note": "..."}]`;
//! a pin without `camera` holds every camera.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const DAY_MS: u64 = 86_400_000;

/// How long recordings are kept, in milliseconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    pub all: u64,
    pub motion: u64,
    pub objects: u64,
    /// Labels with their own retention, in place of `objects`.
    pub labels: HashMap<String, u64>,
    pub pre: u64,
    pub post: u64,
    /// The most the recordings may take, in bytes.
    pub max_bytes: Option<u64>,
}

impl Policy {
    /// The longest any segment can be kept for.
    pub fn longest(&self) -> u64 {
        self.labels.values().copied().chain([self.all, self.motion, self.objects]).max().unwrap_or(0)
    }

    /// How long a label is kept for.
    pub fn label(&self, l: &str) -> u64 {
        *self.labels.get(l).unwrap_or(&self.objects)
    }
}

/// What one camera showed from `from` to `to` (ms since 1970).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub from: u64,
    pub to: u64,
    #[serde(default)]
    pub motion: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pin {
    #[serde(default)]
    pub camera: Option<String>,
    pub from: u64,
    pub to: u64,
    #[serde(default)]
    pub note: String,
}

/// One recorded segment.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub camera: String,
    pub path: PathBuf,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
}

fn overlaps(a_from: u64, a_to: u64, b_from: u64, b_to: u64) -> bool {
    a_from <= b_to && b_from <= a_to
}

/// When `s` may be deleted, given its camera's activity and how far that log covers.
pub fn keep_until(s: &Segment, activity: &[Interval], covered: u64, p: &Policy) -> u64 {
    let mut keep = if s.end > covered { p.longest() } else { p.all };
    for a in activity {
        // The activity, widened by the seconds kept before and after it, meets the segment.
        if overlaps(a.from.saturating_sub(p.pre), a.to.saturating_add(p.post), s.start, s.end) {
            if a.motion {
                keep = keep.max(p.motion);
            }
            for l in &a.labels {
                keep = keep.max(p.label(l));
            }
        }
    }
    s.end.saturating_add(keep)
}

pub fn pinned(s: &Segment, pins: &[Pin]) -> bool {
    pins.iter().any(|p| p.camera.as_ref().is_none_or(|c| c == &s.camera) && overlaps(p.from, p.to, s.start, s.end))
}

/// Why a segment goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Why {
    Expired,
    Space,
}

/// What a sweep would delete: indexes into `segments`, each with why.
/// `keep[i]` is segment i's keep-until; `held[i]` says it may not be deleted at all.
pub fn plan(segments: &[Segment], keep: &[u64], held: &[bool], now: u64, max_bytes: Option<u64>) -> Vec<(usize, Why)> {
    let mut out: Vec<(usize, Why)> = (0..segments.len()).filter(|&i| !held[i] && now > keep[i]).map(|i| (i, Why::Expired)).collect();
    if let Some(max) = max_bytes {
        let mut total: u64 = segments.iter().map(|s| s.bytes).sum::<u64>() - out.iter().map(|&(i, _)| segments[i].bytes).sum::<u64>();
        let mut rest: Vec<usize> = (0..segments.len()).filter(|&i| !held[i] && now <= keep[i]).collect();
        rest.sort_by_key(|&i| (keep[i], segments[i].start));
        for i in rest {
            if total <= max {
                break;
            }
            total -= segments[i].bytes;
            out.push((i, Why::Space));
        }
    }
    out
}

/// One camera's activity log: intervals of what it showed, appended as they end.
pub struct ActivityLog {
    dir: PathBuf,
    open: Option<Interval>,
    /// How far apart two notes may be and still be one interval.
    gap: u64,
    /// The longest an interval is held open before it is written.
    longest: u64,
    covered: u64,
}

impl ActivityLog {
    pub fn new(root: &Path, camera: &str) -> ActivityLog {
        ActivityLog { dir: root.join(camera).join("activity"), open: None, gap: 2_000, longest: 10_000, covered: 0 }
    }

    /// What the camera showed at `t`. Nothing moving and nothing seen is not activity.
    pub fn note(&mut self, t: u64, motion: bool, labels: &[String]) -> Result<()> {
        if !motion && labels.is_empty() {
            return self.flush_if(t);
        }
        match &mut self.open {
            Some(a) if t <= a.to + self.gap && t < a.from + self.longest => {
                a.to = a.to.max(t);
                a.motion |= motion;
                for l in labels {
                    if !a.labels.contains(l) {
                        a.labels.push(l.clone());
                    }
                }
            }
            _ => {
                self.flush(t)?;
                let mut ls = labels.to_vec();
                ls.sort();
                ls.dedup();
                self.open = Some(Interval { from: t, to: t, motion, labels: ls });
            }
        }
        Ok(())
    }

    fn flush_if(&mut self, t: u64) -> Result<()> {
        let stale = self.open.as_ref().is_some_and(|a| t > a.to + self.gap);
        if stale || t >= self.covered + self.longest {
            self.flush(t)?;
        }
        Ok(())
    }

    /// Write the open interval, if any, and that the log covers up to `t`.
    pub fn flush(&mut self, t: u64) -> Result<()> {
        private_dir(&self.dir)?;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join(format!("{}.jsonl", t / DAY_MS)))?;
        let mut text = String::new();
        if let Some(a) = self.open.take() {
            text += &serde_json::to_string(&a).map_err(|e| Error::Invalid(e.to_string()))?;
            text.push('\n');
        }
        // Covered up to `t`, less the gap: a note that would join a closed interval cannot come.
        text += &format!("{{\"covered\":{t}}}\n");
        f.write_all(text.as_bytes())?;
        self.covered = t;
        Ok(())
    }
}

/// A camera's activity, read back: the intervals and how far the log covers.
pub fn read_activity(root: &Path, camera: &str) -> (Vec<Interval>, u64) {
    let dir = root.join(camera).join("activity");
    let (mut out, mut covered) = (Vec::new(), 0u64);
    let Ok(entries) = std::fs::read_dir(&dir) else { return (out, covered) };
    for e in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue }; // a cut-off last line
            if let Some(c) = v.get("covered").and_then(|c| c.as_u64()) {
                covered = covered.max(c);
            } else if let Ok(a) = serde_json::from_value::<Interval>(v) {
                covered = covered.max(a.to);
                out.push(a);
            }
        }
    }
    (out, covered)
}

/// The segments under `root`, by camera, oldest first. A segment ends where the next
/// one starts; the newest ends when it was last written.
pub fn scan(root: &Path) -> Result<BTreeMap<String, Vec<Segment>>> {
    let mut out = BTreeMap::new();
    let Ok(cams) = std::fs::read_dir(root) else { return Ok(out) };
    for c in cams.flatten() {
        if !c.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let camera = c.file_name().to_string_lossy().to_string();
        let mut segs: Vec<(u64, PathBuf, std::fs::Metadata)> = std::fs::read_dir(c.path())?.flatten()
            .filter_map(|f| {
                let name = f.file_name().to_string_lossy().to_string();
                let start: u64 = name.strip_suffix(".ts")?.parse().ok()?;
                Some((start, f.path(), f.metadata().ok()?))
            })
            .collect();
        segs.sort_by_key(|s| s.0);
        let mut list = Vec::with_capacity(segs.len());
        for (i, (start, path, meta)) in segs.iter().enumerate() {
            let written = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64).unwrap_or(*start);
            let end = segs.get(i + 1).map(|n| n.0).unwrap_or(written).max(*start);
            list.push(Segment { camera: camera.clone(), path: path.clone(), start: *start, end, bytes: meta.len() });
        }
        if !list.is_empty() {
            out.insert(camera, list);
        }
    }
    Ok(out)
}

/// Make a folder for recordings that only its owner and group can open (mode 750 on
/// Unix): they are pictures of people.
pub fn private_dir(dir: &Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o750);
    b.create(dir)
}

pub fn read_pins(root: &Path) -> Result<Vec<Pin>> {
    match std::fs::read_to_string(root.join("pins.json")) {
        Ok(t) => serde_json::from_str(&t).map_err(|e| Error::Invalid(format!("{}: {e}", root.join("pins.json").display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

/// What a sweep did, or would do.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Sweep {
    pub time_ms: u64,
    pub segments: usize,
    pub bytes: u64,
    pub deleted: Vec<Deleted>,
    pub deleted_bytes: u64,
    pub pinned: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Deleted {
    pub camera: String,
    pub start: u64,
    pub end: u64,
    pub why: Why,
}

/// Sweep `root` at `now`: delete what the policy says may go, unless `dry_run`. A pins
/// file that cannot be read stops the sweep, rather than deleting pinned recordings.
pub fn sweep(root: &Path, p: &Policy, now: u64, dry_run: bool) -> Result<Sweep> {
    let pins = read_pins(root)?;
    let cams = scan(root)?;
    let (mut segments, mut keep, mut held) = (Vec::new(), Vec::new(), Vec::new());
    for (cam, segs) in &cams {
        let (activity, covered) = read_activity(root, cam);
        for (i, s) in segs.iter().enumerate() {
            keep.push(keep_until(s, &activity, covered, p));
            held.push(i + 1 == segs.len() || pinned(s, &pins));
            segments.push(s.clone());
        }
    }
    let mut out = Sweep { time_ms: now, segments: segments.len(), bytes: segments.iter().map(|s| s.bytes).sum(),
                          pinned: segments.iter().filter(|s| pinned(s, &pins)).count(), ..Default::default() };
    for (i, why) in plan(&segments, &keep, &held, now, p.max_bytes) {
        let s = &segments[i];
        if !dry_run {
            match std::fs::remove_file(&s.path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        out.deleted_bytes += s.bytes;
        out.deleted.push(Deleted { camera: s.camera.clone(), start: s.start, end: s.end, why });
    }
    if !dry_run {
        // Activity older than anything that could still be kept is not needed.
        let oldest_day = now.saturating_sub(p.longest() + p.pre + DAY_MS) / DAY_MS;
        for cam in cams.keys() {
            let Ok(days) = std::fs::read_dir(root.join(cam).join("activity")) else { continue };
            for d in days.flatten() {
                let day: Option<u64> = d.file_name().to_string_lossy().strip_suffix(".jsonl").and_then(|s| s.parse().ok());
                if day.is_some_and(|d| d < oldest_day) {
                    let _ = std::fs::remove_file(d.path());
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u64 = 1_000;
    const D: u64 = DAY_MS;

    fn policy() -> Policy {
        Policy { all: D, motion: 3 * D, objects: 14 * D, labels: [("person".to_string(), 30 * D)].into(),
                 pre: 5 * S, post: 5 * S, max_bytes: None }
    }

    fn seg(cam: &str, start: u64, bytes: u64) -> Segment {
        Segment { camera: cam.into(), path: PathBuf::from(format!("/nowhere/{cam}/{start}.ts")), start, end: start + 10 * S, bytes }
    }

    fn act(from: u64, to: u64, motion: bool, labels: &[&str]) -> Interval {
        Interval { from, to, motion, labels: labels.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn each_kind_of_segment_keeps_its_days() {
        let p = policy();
        let a = [act(100 * S, 101 * S, true, &[]), act(200 * S, 201 * S, true, &["car"]), act(300 * S, 301 * S, true, &["person"])];
        let far = u64::MAX;
        assert_eq!(keep_until(&seg("g", 0, 1), &a, far, &p), 10 * S + D);
        assert_eq!(keep_until(&seg("g", 100 * S, 1), &a, far, &p), 110 * S + 3 * D);
        assert_eq!(keep_until(&seg("g", 200 * S, 1), &a, far, &p), 210 * S + 14 * D);
        assert_eq!(keep_until(&seg("g", 300 * S, 1), &a, far, &p), 310 * S + 30 * D);
    }

    #[test]
    fn the_seconds_before_and_after_an_event_are_kept_with_it() {
        let p = policy();
        let a = [act(1_003 * S, 1_004 * S, false, &["person"])];
        // [990, 1000] ends 3 s before the person: within pre_seconds.
        assert_eq!(keep_until(&seg("g", 990 * S, 1), &a, u64::MAX, &p), 1_000 * S + 30 * D);
        // [1010, 1020] starts 6 s after: outside post_seconds.
        assert_eq!(keep_until(&seg("g", 1_010 * S, 1), &a, u64::MAX, &p), 1_020 * S + D);
    }

    #[test]
    fn a_segment_past_the_log_is_kept_longest() {
        let p = policy();
        assert_eq!(keep_until(&seg("g", 500 * S, 1), &[], 505 * S, &p), 510 * S + 30 * D);
    }

    #[test]
    fn space_goes_by_expiry_and_held_segments_stay() {
        let segs = [seg("g", 0, 10), seg("g", 10 * S, 10), seg("g", 20 * S, 10), seg("g", 30 * S, 10)];
        let keep = [5 * D, D, 30 * D, 2 * D];
        let held = [false, false, false, true];
        // Nothing expired at day 0.5; 40 bytes, at most 15: the soonest expiries go first.
        let p = plan(&segs, &keep, &held, D / 2, Some(15));
        assert_eq!(p, vec![(1, Why::Space), (0, Why::Space), (2, Why::Space)]);
        // At day 6, 0 and 1 have expired; 20 bytes left is under 25.
        assert_eq!(plan(&segs, &keep, &held, 6 * D, Some(25)), vec![(0, Why::Expired), (1, Why::Expired)]);
    }

    #[test]
    fn pins_hold_their_camera_or_all() {
        let pins = [Pin { camera: Some("gate".into()), from: 15 * S, to: 16 * S, note: String::new() },
                    Pin { camera: None, from: 100 * S, to: 100 * S, note: String::new() }];
        assert!(pinned(&seg("gate", 10 * S, 1), &pins));
        assert!(!pinned(&seg("yard", 10 * S, 1), &pins));
        assert!(pinned(&seg("yard", 95 * S, 1), &pins));
    }

    #[test]
    fn a_sweep_on_disk() {
        let root = std::env::temp_dir().join(format!("tsb-retention-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("gate")).unwrap();
        let p = Policy { all: 60 * S, motion: 120 * S, objects: 600 * S, labels: HashMap::new(), pre: 0, post: 0, max_bytes: None };
        let t0 = 10 * D;
        for k in 0..6 {
            std::fs::write(root.join("gate").join(format!("{}.ts", t0 + k * 10 * S)), [0u8; 100]).unwrap();
        }
        let mut log = ActivityLog::new(&root, "gate");
        log.note(t0 + 12 * S, true, &[]).unwrap(); // motion in segment 1
        log.note(t0 + 25 * S, false, &["person".into()]).unwrap(); // a person in segment 2
        log.flush(t0 + 60 * S).unwrap();
        std::fs::write(root.join("pins.json"), format!(r#"[{{"from": {}, "to": {}}}]"#, t0 + 35 * S, t0 + 36 * S)).unwrap();
        // At +200 s: segment 0, 4 (60 s) and 1 (motion, 120 s) have expired; 2 has a person;
        // 3 is pinned; 5 is the newest.
        let now = t0 + 200 * S;
        let dry = sweep(&root, &p, now, true).unwrap();
        let gone: Vec<u64> = dry.deleted.iter().map(|d| (d.start - t0) / (10 * S)).collect();
        assert_eq!(gone, vec![0, 1, 4]);
        assert_eq!(std::fs::read_dir(root.join("gate")).unwrap().count(), 7, "a dry run deletes nothing");
        sweep(&root, &p, now, false).unwrap();
        let left: Vec<u64> = scan(&root).unwrap()["gate"].iter().map(|s| (s.start - t0) / (10 * S)).collect();
        assert_eq!(left, vec![2, 3, 5]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_bad_pins_file_stops_the_sweep() {
        let root = std::env::temp_dir().join(format!("tsb-pins-{}", std::process::id()));
        std::fs::create_dir_all(root.join("gate")).unwrap();
        std::fs::write(root.join("pins.json"), "not json").unwrap();
        assert!(sweep(&root, &policy(), 0, false).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_activity_log_joins_close_notes_and_reads_back() {
        let root = std::env::temp_dir().join(format!("tsb-activity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut log = ActivityLog::new(&root, "gate");
        log.note(1_000, true, &[]).unwrap();
        log.note(2_500, false, &["person".into()]).unwrap();
        log.note(9_000, true, &[]).unwrap(); // after the 2 s gap: a new interval
        log.flush(9_100).unwrap();
        let (a, covered) = read_activity(&root, "gate");
        assert_eq!(a, vec![act(1_000, 2_500, true, &["person"]), act(9_000, 9_000, true, &[])]);
        assert_eq!(covered, 9_100);
        let _ = std::fs::remove_dir_all(&root);
    }
}
