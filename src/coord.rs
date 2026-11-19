//! Spreading cameras over boxes: a coordinator assigns each camera to a box, moves the
//! cameras of a box that stops answering to the others, and collects every box's results.
//!
//! No camera address or password crosses the network. Each box has its own camera
//! config, passwords and all; the coordinator knows only camera names (and their weights),
//! and tells each box which of the cameras it knows to serve.
//!
//! The protocol is length-prefixed JSON over TCP. A connection starts with a handshake
//! in which each side proves it knows the shared token without sending it, under a hard
//! deadline, and both derive a session key from it. Every message after that carries an
//! HMAC-SHA256 over its direction, its sequence number and its bytes, so a message cannot
//! be forged, changed, replayed or reordered by anyone on the path. Messages are not
//! encrypted: the results (labels and boxes) can be read on the path, so keep boxes on a
//! private network or a VPN if that matters.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use crate::{now_ms, Error, Result};

type HmacSha256 = Hmac<Sha256>;

pub const PROTOCOL: &str = "tamserve-boxes/1";
const MAGIC: &[u8; 4] = b"TSB1";
const MAX_MESSAGE: usize = 16 << 20;
const HANDSHAKE: Duration = Duration::from_secs(10);
pub const TOKEN_MIN: usize = 32;

fn mac(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(p);
    }
    m.finalize().into_bytes().into()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn nonce() -> [u8; 32] {
    let mut n = [0u8; 32];
    getrandom::fill(&mut n).expect("the system's random source");
    n
}

/// The shared token, from a file only its owner can read, at least 32 characters.
pub fn read_token(path: &Path) -> Result<Vec<u8>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = std::fs::metadata(path)?;
        if m.mode() & 0o077 != 0 {
            return Err(Error::Invalid(format!("{} can be read by others (mode {:o}): chmod 600 it", path.display(), m.mode() & 0o777)));
        }
    }
    let t = std::fs::read_to_string(path)?.trim().as_bytes().to_vec();
    if t.len() < TOKEN_MIN {
        return Err(Error::Invalid(format!("the token must be at least {TOKEN_MIN} characters; make one with \
                                           `head -c 32 /dev/urandom | base64`")));
    }
    Ok(t)
}

/// One side of an authenticated connection.
pub struct Channel {
    stream: TcpStream,
    key: [u8; 32],
    me: u8,
    sent: u64,
    received: u64,
}

fn read_exact(s: &mut TcpStream, buf: &mut [u8], until: Instant) -> Result<()> {
    let left = until.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(Error::Invalid("the handshake took too long".into()));
    }
    s.set_read_timeout(Some(left))?;
    s.read_exact(buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => Error::Invalid("the handshake took too long".into()),
        _ => Error::Io(e),
    })
}

impl Channel {
    /// The coordinator's half: challenge, check the box's proof, prove back. Returns the
    /// channel and the box's name. The whole handshake has a deadline, so a peer that
    /// stalls part way cannot hold the coordinator's door.
    pub fn accept(mut s: TcpStream, token: &[u8]) -> Result<(Channel, String)> {
        let until = Instant::now() + HANDSHAKE;
        s.set_write_timeout(Some(HANDSHAKE))?;
        s.set_nodelay(true)?;
        let theirs_for = nonce();
        s.write_all(MAGIC)?;
        s.write_all(&theirs_for)?;
        let mut len = [0u8; 2];
        read_exact(&mut s, &mut len, until)?;
        let n = u16::from_be_bytes(len) as usize;
        if n == 0 || n > 256 {
            return Err(Error::Invalid("not a tamserve-boxes box".into()));
        }
        let mut rest = vec![0u8; n + 64];
        read_exact(&mut s, &mut rest, until)?;
        let (name, rest) = rest.split_at(n);
        let (theirs, proof) = rest.split_at(32);
        if !ct_eq(proof, &mac(token, &[b"box", &theirs_for, theirs, name])) {
            return Err(Error::Invalid("the box does not know the token".into()));
        }
        s.write_all(&mac(token, &[b"coordinator", theirs, &theirs_for, name]))?;
        let key = mac(token, &[b"session", &theirs_for, theirs]);
        let name = String::from_utf8(name.to_vec()).map_err(|_| Error::Invalid("a box name is UTF-8".into()))?;
        s.set_read_timeout(None)?;
        Ok((Channel { stream: s, key, me: b'C', sent: 0, received: 0 }, name))
    }

    /// The box's half.
    pub fn connect(addr: &str, token: &[u8], name: &str) -> Result<Channel> {
        let mut s = TcpStream::connect(addr)?;
        let until = Instant::now() + HANDSHAKE;
        s.set_write_timeout(Some(HANDSHAKE))?;
        s.set_nodelay(true)?;
        let mut hello = [0u8; 36];
        read_exact(&mut s, &mut hello, until)?;
        if &hello[..4] != MAGIC {
            return Err(Error::Invalid(format!("{addr} is not a tamserve-boxes coordinator")));
        }
        let theirs = &hello[4..];
        let mine = nonce();
        let n = name.as_bytes();
        if n.is_empty() || n.len() > 256 {
            return Err(Error::Invalid("a box name is 1 to 256 bytes".into()));
        }
        let mut msg = (n.len() as u16).to_be_bytes().to_vec();
        msg.extend_from_slice(n);
        msg.extend_from_slice(&mine);
        msg.extend_from_slice(&mac(token, &[b"box", theirs, &mine, n]));
        s.write_all(&msg)?;
        let mut proof = [0u8; 32];
        read_exact(&mut s, &mut proof, until).map_err(|_| Error::Invalid("the coordinator did not accept the token".into()))?;
        if !ct_eq(&proof, &mac(token, &[b"coordinator", &mine, theirs, n])) {
            return Err(Error::Invalid("the coordinator does not know the token".into()));
        }
        let key = mac(token, &[b"session", theirs, &mine]);
        s.set_read_timeout(None)?;
        Ok(Channel { stream: s, key, me: b'B', sent: 0, received: 0 })
    }

    /// A second handle on the same connection, for a writer thread: it signs with the
    /// sending sequence, the original keeps reading.
    pub fn split(&self) -> Result<Channel> {
        Ok(Channel { stream: self.stream.try_clone()?, key: self.key, me: self.me, sent: self.sent, received: self.received })
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> Result<()> {
        Ok(self.stream.set_read_timeout(t)?)
    }

    pub fn send<T: Serialize>(&mut self, msg: &T) -> Result<()> {
        let body = serde_json::to_vec(msg).map_err(|e| Error::Invalid(e.to_string()))?;
        if body.len() > MAX_MESSAGE {
            return Err(Error::Invalid("a message is too large".into()));
        }
        self.sent += 1;
        let tag = mac(&self.key, &[&[self.me], &self.sent.to_be_bytes(), &body]);
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend_from_slice(&body);
        frame.extend_from_slice(&tag);
        self.stream.write_all(&frame)?;
        Ok(())
    }

    pub fn recv<T: for<'de> Deserialize<'de>>(&mut self) -> Result<T> {
        self.recv_inner().map_err(|e| match e {
            Error::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof => Error::Invalid("the connection closed".into()),
            Error::Io(io) if matches!(io.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) =>
                Error::Invalid("nothing heard within the deadline".into()),
            e => e,
        })
    }

    fn recv_inner<T: for<'de> Deserialize<'de>>(&mut self) -> Result<T> {
        let mut len = [0u8; 4];
        self.stream.read_exact(&mut len)?;
        let n = u32::from_be_bytes(len) as usize;
        if n > MAX_MESSAGE {
            return Err(Error::Invalid("a message is too large".into()));
        }
        let mut body = vec![0u8; n + 32];
        self.stream.read_exact(&mut body)?;
        let tag = body.split_off(n);
        let them = if self.me == b'C' { b'B' } else { b'C' };
        let seq = self.received + 1;
        if !ct_eq(&tag, &mac(&self.key, &[&[them], &seq.to_be_bytes(), &body])) {
            return Err(Error::Invalid("a message failed its check: forged, changed, replayed or out of order".into()));
        }
        self.received = seq;
        serde_json::from_slice(&body).map_err(|e| Error::Invalid(format!("a malformed message: {e}")))
    }

    pub fn shutdown(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

/// How one camera is doing on a box, as its heartbeats tell the coordinator.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CameraHealth {
    pub name: String,
    pub state: String,
    pub frame_age_seconds: Option<f64>,
    pub frames_decoded: u64,
    pub reconnects: u64,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FromBox {
    /// First message: what the box can take and which cameras it has a config for.
    Hello { protocol: String, capacity: f64, known: Vec<String> },
    /// Every second: its cameras' health and how busy its loop is (0 to 1).
    Heartbeat { cameras: Vec<CameraHealth>, busy: f64 },
    /// What its steps found.
    Reports { reports: Vec<serde_json::Value> },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FromCoordinator {
    /// Serve exactly these cameras.
    Assign { cameras: Vec<String> },
    /// Nothing to say: the coordinator is still there.
    Ping,
    /// Refused, and why; the box should not retry with the same settings.
    Refused { reason: String },
}

// ----------------------------------------------------------------------------- coordinator

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CoordCamera {
    pub name: String,
    /// What following it costs against other cameras: 1 by default; 2 for one at twice the frame rate.
    #[serde(default = "one")]
    pub weight: f64,
    /// Only these boxes may serve it (default: any box that has it in its config).
    pub boxes: Option<Vec<String>>,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CoordConfig {
    /// Where boxes connect, such as "0.0.0.0:7400".
    pub listen: String,
    pub token_file: std::path::PathBuf,
    /// Seconds without a heartbeat before a box counts as gone and its cameras move.
    #[serde(default = "dead")]
    pub dead_seconds: f64,
    /// Where to serve /metrics, /status and /v1/chat/completions (optional).
    pub http: Option<String>,
    pub http_token_file: Option<std::path::PathBuf>,
    /// Utilisation gap between boxes above which one camera at a time is moved to even them out.
    #[serde(default = "gap")]
    pub rebalance_gap: f64,
    #[serde(default, rename = "camera")]
    pub cameras: Vec<CoordCamera>,
}

fn dead() -> f64 {
    5.0
}
fn gap() -> f64 {
    0.25
}

struct BoxState {
    tx: mpsc::Sender<FromCoordinator>,
    capacity: f64,
    known: BTreeSet<String>,
    heard: Instant,
    assigned: BTreeSet<String>,
    sent: Option<BTreeSet<String>>,
    health: HashMap<String, CameraHealth>,
    busy: f64,
    session: u64,
    connected_ms: u64,
}

/// Everything the coordinator knows, behind one lock.
pub struct Coordinator {
    cfg: CoordConfig,
    boxes: Mutex<BTreeMap<String, BoxState>>,
    latest: Mutex<BTreeMap<String, serde_json::Value>>,
    moved: Mutex<HashMap<String, Instant>>,
    moves: std::sync::atomic::AtomicU64,
    out: Mutex<Box<dyn Write + Send>>,
    sessions: std::sync::atomic::AtomicU64,
}

impl Coordinator {
    pub fn new(cfg: CoordConfig, out: Box<dyn Write + Send>) -> Result<Arc<Coordinator>> {
        let mut names = BTreeSet::new();
        for c in &cfg.cameras {
            if !names.insert(&c.name) {
                return Err(Error::Invalid(format!("two cameras are named {:?}", c.name)));
            }
            if c.weight <= 0.0 {
                return Err(Error::Invalid(format!("camera {:?}: weight must be above 0", c.name)));
            }
        }
        Ok(Arc::new(Coordinator { cfg, boxes: Mutex::new(BTreeMap::new()), latest: Mutex::new(BTreeMap::new()),
                                  moved: Mutex::new(HashMap::new()), moves: 0.into(), out: Mutex::new(out),
                                  sessions: 0.into() }))
    }

    /// Accept boxes on `cfg.listen`, and plan once every half second, until the process ends.
    pub fn serve(self: &Arc<Self>, token: Vec<u8>) -> Result<std::net::SocketAddr> {
        let l = TcpListener::bind(&self.cfg.listen).map_err(|e| Error::Invalid(format!("cannot listen on {}: {e}", self.cfg.listen)))?;
        let addr = l.local_addr()?;
        let me = self.clone();
        let greeting = Arc::new(AtomicUsize::new(0));
        std::thread::Builder::new().name("coordinator door".into()).spawn(move || {
            for conn in l.incoming().flatten() {
                if greeting.load(Ordering::Relaxed) >= 16 {
                    continue; // too many handshakes at once: someone is flooding the door
                }
                greeting.fetch_add(1, Ordering::Relaxed);
                let (me, token, greeting) = (me.clone(), token.clone(), greeting.clone());
                std::thread::spawn(move || {
                    let peer = conn.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                    let hand = Channel::accept(conn, &token);
                    greeting.fetch_sub(1, Ordering::Relaxed);
                    match hand {
                        Ok((ch, name)) => me.session(ch, name, peer),
                        Err(e) => eprintln!("tamserve-boxes coordinator: turned away {peer}: {e}"),
                    }
                });
            }
        })?;
        let me = self.clone();
        std::thread::Builder::new().name("planner".into()).spawn(move || loop {
            me.plan();
            std::thread::sleep(Duration::from_millis(500));
        })?;
        Ok(addr)
    }

    /// One box's connection: its writer on a thread, its messages read here.
    fn session(self: &Arc<Self>, mut ch: Channel, name: String, peer: String) {
        let hello: Result<FromBox> = { ch.set_read_timeout(Some(HANDSHAKE)).ok(); ch.recv() };
        let (capacity, known) = match hello {
            Ok(FromBox::Hello { protocol, capacity, known }) if protocol == PROTOCOL => (capacity, known),
            Ok(FromBox::Hello { protocol, .. }) => {
                let _ = ch.send(&FromCoordinator::Refused { reason: format!("this coordinator speaks {PROTOCOL}, the box {protocol}") });
                return;
            }
            _ => return,
        };
        let dead = Duration::from_secs_f64(self.cfg.dead_seconds);
        let _ = ch.set_read_timeout(Some(dead * 2));
        let (tx, rx) = mpsc::channel::<FromCoordinator>();
        let Ok(mut writer) = ch.split() else { return };
        let id = self.sessions.fetch_add(1, Ordering::Relaxed) + 1;
        {
            let mut boxes = self.boxes.lock().unwrap();
            if let Some(old) = boxes.remove(&name) {
                let _ = old.tx.send(FromCoordinator::Refused { reason: "the same box connected again".into() });
            }
            boxes.insert(name.clone(), BoxState { tx: tx.clone(), capacity: capacity.max(0.0), known: known.into_iter().collect(),
                heard: Instant::now(), assigned: BTreeSet::new(), sent: None, health: HashMap::new(), busy: 0.0,
                session: id, connected_ms: now_ms() });
        }
        eprintln!("tamserve-boxes coordinator: box {name} joined from {peer}, capacity {capacity}");
        std::thread::spawn(move || {
            // Pings keep the box's read deadline from passing when there is nothing to assign.
            loop {
                match rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(m) => {
                        let stop = matches!(m, FromCoordinator::Refused { .. });
                        if writer.send(&m).is_err() || stop {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if writer.send(&FromCoordinator::Ping).is_err() {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            writer.shutdown();
        });
        self.plan();
        loop {
            match ch.recv::<FromBox>() {
                Ok(FromBox::Heartbeat { cameras, busy }) => {
                    let mut boxes = self.boxes.lock().unwrap();
                    if let Some(b) = boxes.get_mut(&name).filter(|b| b.session == id) {
                        b.heard = Instant::now();
                        b.busy = busy;
                        b.health = cameras.into_iter().map(|c| (c.name.clone(), c)).collect();
                    } else {
                        break; // replaced by a newer connection
                    }
                }
                Ok(FromBox::Reports { reports }) => {
                    if let Some(b) = self.boxes.lock().unwrap().get_mut(&name) {
                        b.heard = Instant::now();
                    }
                    let mut out = self.out.lock().unwrap();
                    let mut latest = self.latest.lock().unwrap();
                    for mut r in reports {
                        if let Some(cam) = r.get("camera").and_then(|c| c.as_str()).map(String::from) {
                            r["box"] = serde_json::Value::String(name.clone());
                            let _ = writeln!(out, "{r}");
                            latest.insert(cam, r);
                        }
                    }
                    let _ = out.flush();
                }
                Ok(FromBox::Hello { .. }) => break,
                Err(e) => {
                    eprintln!("tamserve-boxes coordinator: box {name}: {e}");
                    break;
                }
            }
        }
        ch.shutdown();
        let mut boxes = self.boxes.lock().unwrap();
        if boxes.get(&name).is_some_and(|b| b.session == id) {
            boxes.remove(&name);
            eprintln!("tamserve-boxes coordinator: box {name} left; its cameras will move");
        }
        drop(boxes);
        self.plan();
    }

    fn eligible(&self, cam: &CoordCamera, name: &str, b: &BoxState) -> bool {
        b.known.contains(&cam.name) && cam.boxes.as_ref().is_none_or(|bs| bs.iter().any(|x| x == name))
    }

    /// Decide where every camera goes, and tell the boxes whose cameras changed.
    pub fn plan(&self) {
        let dead = Duration::from_secs_f64(self.cfg.dead_seconds);
        let mut boxes = self.boxes.lock().unwrap();
        let gone: Vec<String> = boxes.iter().filter(|(_, b)| b.heard.elapsed() > dead).map(|(n, _)| n.clone()).collect();
        for n in gone {
            if let Some(b) = boxes.remove(&n) {
                eprintln!("tamserve-boxes coordinator: box {n} has been silent for {:.0} s; its {} camera(s) will move",
                          b.heard.elapsed().as_secs_f64(), b.assigned.len());
                let _ = b.tx.send(FromCoordinator::Refused { reason: "silent for too long".into() });
            }
        }
        let weight: HashMap<&str, f64> = self.cfg.cameras.iter().map(|c| (c.name.as_str(), c.weight)).collect();
        let load = |b: &BoxState| b.assigned.iter().map(|c| weight.get(c.as_str()).copied().unwrap_or(1.0)).sum::<f64>();
        // Keep what is still valid; drop cameras a box can no longer serve or the config no longer has.
        for (n, b) in boxes.iter_mut() {
            let keep: BTreeSet<String> = b.assigned.iter()
                .filter(|c| self.cfg.cameras.iter().any(|cc| &cc.name == *c && self.eligible(cc, n, b)))
                .cloned().collect();
            b.assigned = keep;
        }
        let taken: BTreeSet<String> = boxes.values().flat_map(|b| b.assigned.iter().cloned()).collect();
        let mut free: Vec<&CoordCamera> = self.cfg.cameras.iter().filter(|c| !taken.contains(&c.name)).collect();
        free.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(a.name.cmp(&b.name)));
        for cam in free {
            // The eligible box with the most room; with none that has room, the least full.
            let pick = boxes.iter()
                .filter(|(n, b)| self.eligible(cam, n, b))
                .map(|(n, b)| (n.clone(), b.capacity - load(b), (load(b) + cam.weight) / b.capacity.max(1e-9)))
                .max_by(|x, y| {
                    let fits = |v: &(String, f64, f64)| v.1 >= cam.weight;
                    fits(x).cmp(&fits(y)).then(y.2.total_cmp(&x.2)).then(y.0.cmp(&x.0))
                });
            if let Some((n, _, _)) = pick {
                boxes.get_mut(&n).unwrap().assigned.insert(cam.name.clone());
            }
        }
        // Even out: one camera a tick from the fullest box to the emptiest, if the gap is wide
        // and the camera has not moved in the last 30 s.
        let util = |b: &BoxState| load(b) / b.capacity.max(1e-9);
        if boxes.len() > 1 {
            let full = boxes.iter().max_by(|a, b| util(a.1).total_cmp(&util(b.1))).map(|(n, _)| n.clone()).unwrap();
            let empty = boxes.iter().min_by(|a, b| util(a.1).total_cmp(&util(b.1))).map(|(n, _)| n.clone()).unwrap();
            if util(&boxes[&full]) - util(&boxes[&empty]) > self.cfg.rebalance_gap {
                let mut moved = self.moved.lock().unwrap();
                let candidate = boxes[&full].assigned.iter()
                    .filter_map(|c| self.cfg.cameras.iter().find(|cc| &cc.name == c))
                    .filter(|cc| self.eligible(cc, &empty, &boxes[&empty]))
                    .filter(|cc| moved.get(&cc.name).is_none_or(|t| t.elapsed() > Duration::from_secs(30)))
                    .filter(|cc| {
                        // Only if the move narrows the gap.
                        let (fl, el) = (load(&boxes[&full]), load(&boxes[&empty]));
                        let (fc, ec) = (boxes[&full].capacity.max(1e-9), boxes[&empty].capacity.max(1e-9));
                        ((fl - cc.weight) / fc - (el + cc.weight) / ec).abs() < fl / fc - el / ec
                    })
                    .min_by(|a, b| a.weight.total_cmp(&b.weight).then(a.name.cmp(&b.name)))
                    .map(|cc| cc.name.clone());
                if let Some(c) = candidate {
                    boxes.get_mut(&full).unwrap().assigned.remove(&c);
                    boxes.get_mut(&empty).unwrap().assigned.insert(c.clone());
                    moved.insert(c, Instant::now());
                    self.moves.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        for b in boxes.values_mut() {
            if b.sent.as_ref() != Some(&b.assigned) {
                let _ = b.tx.send(FromCoordinator::Assign { cameras: b.assigned.iter().cloned().collect() });
                b.sent = Some(b.assigned.clone());
            }
        }
    }

    /// Camera name -> the box serving it.
    pub fn assignment(&self) -> BTreeMap<String, String> {
        let boxes = self.boxes.lock().unwrap();
        boxes.iter().flat_map(|(n, b)| b.assigned.iter().map(move |c| (c.clone(), n.clone()))).collect()
    }
}

impl crate::http::Source for Coordinator {
    fn metrics(&self) -> String {
        use crate::http::label;
        use std::fmt::Write;
        let boxes = self.boxes.lock().unwrap();
        let mut m = String::new();
        let _ = writeln!(m, "# HELP tamserve_box_cameras cameras assigned to each box\n# TYPE tamserve_box_cameras gauge");
        for (n, b) in boxes.iter() {
            let _ = writeln!(m, "tamserve_box_cameras{{box=\"{}\"}} {}", label(n), b.assigned.len());
        }
        let _ = writeln!(m, "# HELP tamserve_box_busy how busy each box's loop is, 0 to 1\n# TYPE tamserve_box_busy gauge");
        for (n, b) in boxes.iter() {
            let _ = writeln!(m, "tamserve_box_busy{{box=\"{}\"}} {:.3}", label(n), b.busy);
        }
        let _ = writeln!(m, "# HELP tamserve_box_heartbeat_age_seconds seconds since each box was last heard\n# TYPE tamserve_box_heartbeat_age_seconds gauge");
        for (n, b) in boxes.iter() {
            let _ = writeln!(m, "tamserve_box_heartbeat_age_seconds{{box=\"{}\"}} {:.3}", label(n), b.heard.elapsed().as_secs_f64());
        }
        let assigned: BTreeMap<String, String> = boxes.iter().flat_map(|(n, b)| b.assigned.iter().map(move |c| (c.clone(), n.clone()))).collect();
        let _ = writeln!(m, "# HELP tamserve_camera_up 1 while the camera's box reports frames arriving\n# TYPE tamserve_camera_up gauge");
        for c in &self.cfg.cameras {
            let up = assigned.get(&c.name).and_then(|b| boxes[b].health.get(&c.name)).is_some_and(|h| h.state == "up");
            let _ = writeln!(m, "tamserve_camera_up{{camera=\"{}\",box=\"{}\"}} {}", label(&c.name),
                             label(assigned.get(&c.name).map(|s| s.as_str()).unwrap_or("")), u8::from(up));
        }
        let unassigned = self.cfg.cameras.iter().filter(|c| !assigned.contains_key(&c.name)).count();
        let _ = writeln!(m, "# HELP tamserve_cameras_unassigned cameras no box can serve now\n# TYPE tamserve_cameras_unassigned gauge\ntamserve_cameras_unassigned {unassigned}");
        let _ = writeln!(m, "# HELP tamserve_camera_moves_total cameras moved to even out the boxes\n# TYPE tamserve_camera_moves_total counter\ntamserve_camera_moves_total {}",
                         self.moves.load(Ordering::Relaxed));
        m
    }

    fn status(&self) -> serde_json::Value {
        let boxes = self.boxes.lock().unwrap();
        let latest = self.latest.lock().unwrap();
        let assigned: BTreeMap<String, String> = boxes.iter().flat_map(|(n, b)| b.assigned.iter().map(move |c| (c.clone(), n.clone()))).collect();
        let cameras: serde_json::Map<String, serde_json::Value> = self.cfg.cameras.iter().map(|c| {
            let b = assigned.get(&c.name);
            let health = b.and_then(|b| boxes[b].health.get(&c.name));
            (c.name.clone(), serde_json::json!({
                "box": b, "weight": c.weight, "health": health, "latest": latest.get(&c.name),
            }))
        }).collect();
        let bs: serde_json::Map<String, serde_json::Value> = boxes.iter().map(|(n, b)| (n.clone(), serde_json::json!({
            "capacity": b.capacity, "cameras": b.assigned, "busy": b.busy,
            "heartbeat_age_seconds": b.heard.elapsed().as_secs_f64(), "connected_ms": b.connected_ms,
        }))).collect();
        serde_json::json!({"cameras": cameras, "boxes": bs, "moves": self.moves.load(Ordering::Relaxed)})
    }
}
