//! The camera config file: what to read, how to detect and follow, and where the
//! cameras' passwords are, so that none of them is ever on a command line.
//!
//! ```toml
//! [detector]
//! model = "yolox_nano_gray.onnx"
//! threads = 3          # OpenVINO threads in all
//! requests = 2         # inference requests in flight at once
//! classes = ["person", "car", "truck"]
//!
//! [tracker]
//! look_every = 3
//! hold = 30
//!
//! [decode]
//! decoder = "auto"     # "software", or "vaapi" to decode on the integrated GPU
//! width = 640          # scale every stream to this first (optional)
//! height = 360
//!
//! [http]
//! listen = "127.0.0.1:9300"   # /metrics, /status, /v1/chat/completions
//!
//! [[camera]]
//! name = "gate"
//! url = "rtsp://10.0.0.21/Streaming/Channels/102"
//! user = "admin"
//! password_file = "/etc/tamserve/secrets/gate"   # or password_env = "GATE_PASSWORD"
//! look_every = 1                                  # this camera's own
//! record = true                                   # write segments to [record] dir
//! live = true                                     # GET /live/gate.ts
//!
//! [motion]             # look only where something moves (see `motion`)
//! hold = 10            # frames a camera stays awake after motion
//!
//! [record]             # see `retention`
//! dir = "/var/lib/tamserve/recordings"
//! all_days = 1
//! motion_days = 3
//! objects_days = 14
//! labels = { person = 30 }
//! max_gb = 400
//! ```
//!
//! A password can come from a file (`password_file`), from the environment
//! (`password_env`), or be written in this file (`password`). A file holding a password,
//! this one included, must be readable by its owner only (mode 600 or stricter):
//! otherwise loading refuses it. A password inside the URL (`rtsp://user:pass@...`) is
//! refused too, so that it cannot be copied into a log by mistake.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{DetectorOptions, Error, LookEvery, Result, TrackerOptions};

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DetectorSection {
    pub model: PathBuf,
    pub threads: Option<usize>,
    #[serde(default = "one")]
    pub requests: usize,
    pub classes: Option<Vec<String>>,
    #[serde(default = "size")]
    pub size: usize,
    pub precision: Option<String>,
    pub threshold: Option<f32>,
    /// The model's class names, in order: a fine-tuned model's `.names.json`. Default COCO's.
    pub names: Option<Vec<String>>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackerSection {
    pub look_every: Option<u32>,
    pub hold: Option<u32>,
    pub start: Option<f32>,
    pub high: Option<f32>,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Decoder {
    /// Whatever GStreamer ranks best: VA-API where it is installed and has a device.
    #[default]
    Auto,
    /// The CPU's decoders only.
    Software,
    /// VA-API's, on the integrated GPU; refused at start if there are none.
    Vaapi,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct DecodeSection {
    #[serde(default)]
    pub decoder: Decoder,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// RTSP over TCP (true, the default) or UDP.
    pub tcp: Option<bool>,
    /// Seconds without a frame before a stream counts as stalled and is reopened.
    pub stall_seconds: Option<f64>,
    /// Reconnect delays: the first, and the most it grows to, in seconds.
    pub backoff_first: Option<f64>,
    pub backoff_max: Option<f64>,
    /// Read files at their own frame rate, as a camera would send them (for testing).
    #[serde(default)]
    pub realtime: bool,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct HttpSection {
    pub listen: Option<String>,
    /// A file holding a bearer token the endpoints require (mode 600). None: no token.
    pub token_file: Option<PathBuf>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CameraSection {
    pub name: String,
    pub url: String,
    pub user: Option<String>,
    pub password: Option<String>,
    pub password_file: Option<PathBuf>,
    pub password_env: Option<String>,
    pub look_every: Option<u32>,
    /// What following this camera costs, against others, for the coordinator (default 1).
    pub weight: Option<f64>,
    /// Write this camera's stream to `[record] dir` (default: `[record] all_cameras`).
    pub record: Option<bool>,
    /// Serve this camera's stream on `GET /live/<name>.ts`.
    #[serde(default)]
    pub live: bool,
}

/// Motion found before detecting, so that cameras where nothing moves cost no looks.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct MotionSection {
    /// false: motion is found (for metrics and recording) but every look is made.
    #[serde(default = "yes")]
    pub gate: bool,
    /// Frames a camera stays awake after it last moved.
    #[serde(default = "ten")]
    pub hold: u32,
    pub height: Option<usize>,
    pub threshold: Option<u8>,
    pub min_area: Option<usize>,
    pub alpha: Option<f32>,
    pub lighting: Option<f32>,
}

/// Recording, and how long recordings are kept.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RecordSection {
    pub dir: PathBuf,
    /// Record every camera unless it says `record = false`.
    #[serde(default)]
    pub all_cameras: bool,
    #[serde(default = "ten_f")]
    pub segment_seconds: f64,
    /// Every segment, in days.
    #[serde(default = "one_f")]
    pub all_days: f64,
    /// Segments with motion (needs `[motion]`).
    pub motion_days: Option<f64>,
    /// Segments with any object.
    pub objects_days: Option<f64>,
    /// Labels kept for their own days, in place of `objects_days`.
    #[serde(default)]
    pub labels: HashMap<String, f64>,
    #[serde(default = "five_f")]
    pub pre_seconds: f64,
    #[serde(default = "five_f")]
    pub post_seconds: f64,
    /// The most the recordings may take; the segments that would expire first go.
    pub max_gb: Option<f64>,
    #[serde(default = "sixty_f")]
    pub sweep_seconds: f64,
}

fn yes() -> bool {
    true
}
fn ten() -> u32 {
    10
}
fn one_f() -> f64 {
    1.0
}
fn five_f() -> f64 {
    5.0
}
fn ten_f() -> f64 {
    10.0
}
fn sixty_f() -> f64 {
    60.0
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub detector: DetectorSection,
    #[serde(default)]
    pub tracker: TrackerSection,
    #[serde(default)]
    pub decode: DecodeSection,
    #[serde(default)]
    pub http: HttpSection,
    pub motion: Option<MotionSection>,
    pub record: Option<RecordSection>,
    #[serde(default, rename = "camera")]
    pub cameras: Vec<CameraSection>,
}

fn one() -> usize {
    1
}
fn size() -> usize {
    416
}

/// A camera ready to open: its address without credentials, and the credentials apart.
#[derive(Clone)]
pub struct Camera {
    pub name: String,
    pub url: String,
    pub user: Option<String>,
    pub password: Option<String>,
    pub look_every: Option<u32>,
    pub weight: f64,
    pub record: bool,
    pub live: bool,
}

impl std::fmt::Debug for Camera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the password; the URL is already free of credentials.
        write!(f, "Camera {{ name: {:?}, url: {:?}, user: {:?} }}", self.name, self.url, self.user)
    }
}

impl Camera {
    /// What logs and errors may say about this camera: its name and host, nothing more.
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, host(&self.url))
    }
}

/// The host of a URL, for logs.
pub fn host(url: &str) -> String {
    let rest = url.split_once("://").map(|x| x.1).unwrap_or(url);
    let rest = rest.rsplit_once('@').map(|x| x.1).unwrap_or(rest);
    rest.split(['/', '?']).next().unwrap_or("").to_string()
}

/// A name that is safe as a folder and in a URL path.
fn plain_name(n: &str) -> bool {
    !n.is_empty() && !n.starts_with('.') && n.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

/// Whether only the owner can read `path` (mode 600, 400 or stricter). Not checked where
/// the system has no Unix permissions.
fn owner_only(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = std::fs::metadata(path)?;
        if m.mode() & 0o077 != 0 {
            return Err(Error::Invalid(format!(
                "{} holds a camera password and can be read by others (mode {:o}): chmod 600 it",
                path.display(), m.mode() & 0o777)));
        }
        if m.uid() != process_uid() && m.uid() != 0 {
            return Err(Error::Invalid(format!("{} belongs to another user", path.display())));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn process_uid() -> u32 {
    // The uid of this process, read from /proc to avoid a libc dependency.
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("Uid:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

impl Config {
    /// Read and check a config file. Relative paths in it are taken from its folder.
    pub fn load(path: impl AsRef<Path>) -> Result<(Config, Vec<Camera>)> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let mut cfg: Config = toml::from_str(&text).map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
        let base = path.parent().unwrap_or(Path::new("."));
        if cfg.detector.model.is_relative() {
            cfg.detector.model = base.join(&cfg.detector.model);
        }
        if cfg.detector.requests == 0 {
            return Err(Error::Invalid("detector.requests must be at least 1".into()));
        }
        let mut seen = std::collections::HashSet::new();
        let mut cams = Vec::with_capacity(cfg.cameras.len());
        let mut inline = false;
        for c in &cfg.cameras {
            if !seen.insert(c.name.clone()) {
                return Err(Error::Invalid(format!("two cameras are named {:?}", c.name)));
            }
            if let Some(rest) = c.url.split_once("://").map(|x| x.1) {
                if rest.split('/').next().unwrap_or("").contains('@') {
                    return Err(Error::Invalid(format!(
                        "camera {:?}: put the user and password in `user` and `password_file`, not in the URL", c.name)));
                }
            }
            let sources = [c.password.is_some(), c.password_file.is_some(), c.password_env.is_some()];
            if sources.iter().filter(|&&s| s).count() > 1 {
                return Err(Error::Invalid(format!("camera {:?}: give one of password, password_file, password_env", c.name)));
            }
            let password = if let Some(p) = &c.password {
                inline = true;
                Some(p.clone())
            } else if let Some(f) = &c.password_file {
                let f = if f.is_relative() { base.join(f) } else { f.clone() };
                owner_only(&f)?;
                Some(std::fs::read_to_string(&f)?.trim_end_matches(['\r', '\n']).to_string())
            } else if let Some(v) = &c.password_env {
                Some(std::env::var(v).map_err(|_| Error::Invalid(format!("camera {:?}: ${v} is not set", c.name)))?)
            } else {
                None
            };
            if password.is_some() && c.user.is_none() {
                return Err(Error::Invalid(format!("camera {:?}: a password needs a `user`", c.name)));
            }
            if let Some(0) = c.look_every {
                return Err(Error::Invalid(format!("camera {:?}: look_every must be at least 1", c.name)));
            }
            let record = c.record.unwrap_or(cfg.record.as_ref().is_some_and(|r| r.all_cameras));
            if record && cfg.record.is_none() {
                return Err(Error::Invalid(format!("camera {:?}: record = true needs a [record] section with a dir", c.name)));
            }
            if (record || c.live) && !plain_name(&c.name) {
                return Err(Error::Invalid(format!(
                    "camera {:?}: a camera that records or is watched live names a folder and a URL: use letters, digits, `-`, `_` and `.` only", c.name)));
            }
            cams.push(Camera { name: c.name.clone(), url: c.url.clone(), user: c.user.clone(), password,
                               look_every: c.look_every, weight: c.weight.unwrap_or(1.0), record, live: c.live });
        }
        if inline {
            owner_only(path)?;
        }
        if let Some(t) = &cfg.http.token_file {
            owner_only(&if t.is_relative() { base.join(t) } else { t.clone() })?;
        }
        if let Some(m) = &cfg.motion {
            cfg.motion_options_of(m).check()?;
        }
        if let Some(r) = &mut cfg.record {
            if r.dir.is_relative() {
                r.dir = base.join(&r.dir);
            }
            let days = [Some(r.all_days), r.motion_days, r.objects_days].into_iter().flatten().chain(r.labels.values().copied());
            // Written so that NaN fails too.
            let ok = |v: f64, least: f64| v >= least;
            if !ok(r.segment_seconds, 1.0) || days.into_iter().any(|d| !ok(d, 0.0)) || r.max_gb.is_some_and(|g| g <= 0.0 || g.is_nan()) {
                return Err(Error::Invalid("[record]: segment_seconds must be at least 1, days at least 0, max_gb above 0".into()));
            }
            if r.motion_days.is_some() && cfg.motion.is_none() {
                return Err(Error::Invalid("[record] motion_days needs a [motion] section: motion is what it keeps".into()));
            }
        }
        Ok((cfg, cams))
    }

    pub fn detector_options(&self) -> DetectorOptions {
        let d = &self.detector;
        let mut o = DetectorOptions { threads: d.threads, classes: d.classes.clone(), size: d.size,
                                      precision: d.precision.clone(), requests: d.requests, ..Default::default() };
        if let Some(t) = d.threshold {
            o.threshold = t;
        }
        if let Some(n) = &d.names {
            o.names = n.clone();
        }
        o
    }

    pub fn tracker_options(&self, cams: &[Camera]) -> TrackerOptions<String> {
        let t = &self.tracker;
        let every = t.look_every.unwrap_or(3);
        let own: HashMap<String, u32> = cams.iter().filter_map(|c| Some((c.name.clone(), c.look_every?))).collect();
        let d = TrackerOptions::<String>::default();
        TrackerOptions {
            look_every: if own.is_empty() { LookEvery::All(every) } else { LookEvery::Each { cameras: own, default: every } },
            hold: t.hold.unwrap_or(d.hold),
            start: t.start.unwrap_or(d.start),
            high: t.high.unwrap_or(d.high),
        }
    }

    fn motion_options_of(&self, m: &MotionSection) -> crate::motion::MotionOptions {
        let d = crate::motion::MotionOptions::default();
        crate::motion::MotionOptions {
            height: m.height.unwrap_or(d.height),
            threshold: m.threshold.unwrap_or(d.threshold),
            min_area: m.min_area.unwrap_or(d.min_area),
            alpha: m.alpha.unwrap_or(d.alpha),
            lighting: m.lighting.unwrap_or(d.lighting),
        }
    }

    /// How motion is found, if `[motion]` is on.
    pub fn motion_options(&self) -> Option<crate::motion::MotionOptions> {
        self.motion.as_ref().map(|m| self.motion_options_of(m))
    }

    /// How long recordings are kept, if `[record]` is on.
    pub fn retention(&self) -> Option<crate::retention::Policy> {
        let r = self.record.as_ref()?;
        let ms = |days: f64| (days * 86_400_000.0) as u64;
        let all = ms(r.all_days);
        let motion = r.motion_days.map(ms).unwrap_or(all).max(all);
        let objects = r.objects_days.map(ms).unwrap_or(motion).max(all);
        Some(crate::retention::Policy {
            all, motion, objects,
            labels: r.labels.iter().map(|(k, &v)| (k.clone(), ms(v))).collect(),
            pre: (r.pre_seconds * 1000.0) as u64,
            post: (r.post_seconds * 1000.0) as u64,
            max_bytes: r.max_gb.map(|g| (g * 1e9) as u64),
        })
    }

    /// The bearer token the HTTP endpoints require, if any.
    pub fn http_token(&self, base: &Path) -> Result<Option<String>> {
        match &self.http.token_file {
            None => Ok(None),
            Some(t) => {
                let t = if t.is_relative() { base.join(t) } else { t.clone() };
                Ok(Some(std::fs::read_to_string(t)?.trim().to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn write(dir: &Path, name: &str, text: &str, mode: u32) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        p
    }

    fn dir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("tsb-config-{}-{}", std::process::id(), rand_u64()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn rand_u64() -> u64 {
        let mut b = [0u8; 8];
        getrandom::fill(&mut b).unwrap();
        u64::from_le_bytes(b)
    }

    const HEAD: &str = "[detector]\nmodel = \"m.onnx\"\n";

    #[test]
    fn a_password_file_is_read_and_never_shown() {
        let d = dir();
        write(&d, "gate.pw", "s3cret\n", 0o600);
        let cfg = write(&d, "c.toml", &format!("{HEAD}[[camera]]\nname=\"gate\"\nurl=\"rtsp://10.0.0.21/ch1\"\nuser=\"admin\"\npassword_file=\"gate.pw\"\n"), 0o644);
        let (_, cams) = Config::load(&cfg).unwrap();
        assert_eq!(cams[0].password.as_deref(), Some("s3cret"));
        assert!(!format!("{:?}", cams[0]).contains("s3cret"));
        assert_eq!(cams[0].label(), "gate (10.0.0.21)");
    }

    #[test]
    fn a_password_others_can_read_is_refused() {
        let d = dir();
        write(&d, "gate.pw", "s3cret", 0o644);
        let cfg = write(&d, "c.toml", &format!("{HEAD}[[camera]]\nname=\"gate\"\nurl=\"rtsp://h/x\"\nuser=\"a\"\npassword_file=\"gate.pw\"\n"), 0o600);
        assert!(Config::load(&cfg).unwrap_err().to_string().contains("chmod 600"));
        let inline = write(&d, "i.toml", &format!("{HEAD}[[camera]]\nname=\"gate\"\nurl=\"rtsp://h/x\"\nuser=\"a\"\npassword=\"p\"\n"), 0o640);
        assert!(Config::load(&inline).is_err()); // an inline password makes the config file itself secret
        std::fs::set_permissions(&inline, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(Config::load(&inline).is_ok());
    }

    #[test]
    fn credentials_in_the_url_are_refused() {
        let d = dir();
        let cfg = write(&d, "c.toml", &format!("{HEAD}[[camera]]\nname=\"gate\"\nurl=\"rtsp://admin:pw@10.0.0.21/ch1\"\n"), 0o600);
        assert!(Config::load(&cfg).unwrap_err().to_string().contains("not in the URL"));
    }

    #[test]
    fn mistakes_are_caught() {
        let d = dir();
        for body in ["[[camera]]\nname=\"a\"\nurl=\"rtsp://h/1\"\n[[camera]]\nname=\"a\"\nurl=\"rtsp://h/2\"\n",
                     "[[camera]]\nname=\"a\"\nurl=\"rtsp://h/1\"\npasswrd_file=\"x\"\n",
                     "[[camera]]\nname=\"a\"\nurl=\"rtsp://h/1\"\npassword_env=\"TSB_SURELY_UNSET\"\nuser=\"u\"\n",
                     "[[camera]]\nname=\"a\"\nurl=\"rtsp://h/1\"\nlook_every=0\n"] {
            let cfg = write(&d, "c.toml", &format!("{HEAD}{body}"), 0o600);
            assert!(Config::load(&cfg).is_err(), "{body}");
        }
    }

    #[test]
    fn recording_and_motion_settings() {
        let d = dir();
        let body = "[motion]\nhold = 5\n[record]\ndir = \"rec\"\nall_days = 1\nmotion_days = 3\nobjects_days = 14\nlabels = { person = 30 }\nmax_gb = 2\n\
                    [[camera]]\nname=\"gate\"\nurl=\"rtsp://h/1\"\nrecord=true\nlive=true\n[[camera]]\nname=\"yard\"\nurl=\"rtsp://h/2\"\n";
        let (cfg, cams) = Config::load(write(&d, "c.toml", &format!("{HEAD}{body}"), 0o600)).unwrap();
        assert!(cams[0].record && cams[0].live && !cams[1].record && !cams[1].live);
        assert_eq!(cfg.record.as_ref().unwrap().dir, d.join("rec"));
        let p = cfg.retention().unwrap();
        assert_eq!((p.all, p.motion, p.objects, p.label("person"), p.label("car")),
                   (86_400_000, 3 * 86_400_000, 14 * 86_400_000, 30 * 86_400_000, 14 * 86_400_000));
        assert_eq!(p.max_bytes, Some(2_000_000_000));
        assert_eq!(cfg.motion.as_ref().unwrap().hold, 5);
        assert!(cfg.motion_options().is_some());
    }

    #[test]
    fn recording_mistakes_are_caught() {
        let d = dir();
        for body in ["[[camera]]\nname=\"a\"\nurl=\"rtsp://h/1\"\nrecord=true\n",
                     "[[camera]]\nname=\"../a\"\nurl=\"rtsp://h/1\"\nlive=true\n",
                     "[record]\ndir=\"r\"\nmotion_days=3\n",
                     "[record]\ndir=\"r\"\nsegment_seconds=0\n",
                     "[motion]\nalpha=0\n"] {
            let cfg = write(&d, "c.toml", &format!("{HEAD}{body}"), 0o600);
            assert!(Config::load(&cfg).is_err(), "{body}");
        }
    }

    #[test]
    fn hosts_for_logs() {
        assert_eq!(host("rtsp://10.0.0.5:554/Streaming/Channels/102"), "10.0.0.5:554");
        assert_eq!(host("rtsp://u:p@cam.local/x?y"), "cam.local");
        assert_eq!(host("file:///videos/a.mp4"), "");
    }
}
