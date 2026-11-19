//! tamserve-boxes: find and follow boxes on many cameras with YOLOX-Nano on gray frames,
//! on the CPU through OpenVINO.
//!
//!   tamserve-boxes fold yolox_nano.onnx yolox_nano_gray.onnx
//!   tamserve-boxes detect --model yolox_nano_gray.onnx picture.pgm ...
//!   tamserve-boxes run --config cameras.toml
//!   tamserve-boxes box --config cameras.toml --coordinator 10.0.0.2:7400 --token-file token --name box-a
//!   tamserve-boxes coordinator --config coordinator.toml
//!   tamserve-boxes bench --model yolox_nano_gray.onnx --image picture.pgm --cameras 8 --fps 25
//!
//! `run` reads the cameras in its config in this process with GStreamer (the gray plane,
//! the newest frame only, reconnecting with backoff), and prints one JSON line per camera
//! per step; its HTTP endpoint serves metrics. `box` and `coordinator` spread cameras over
//! several boxes. OpenVINO's C library is found at run time: put its folder on
//! LD_LIBRARY_PATH or set OPENVINO_INSTALL_DIR.

use std::collections::HashMap;
#[cfg(feature = "cameras")]
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tamserve_boxes::{BoxTracker, Detect, Detector, DetectorOptions, GrayFrame, LookEvery, TrackerOptions};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage:
  tamserve-boxes fold IN.onnx OUT.onnx
  tamserve-boxes detect --model M [--threads N] [--classes a,b] PICTURE.pgm ...
  tamserve-boxes run --config cameras.toml
  tamserve-boxes box --config cameras.toml --coordinator HOST:PORT --token-file F --name NAME [--capacity W]
  tamserve-boxes coordinator --config coordinator.toml
  tamserve-boxes bench --model M --image PICTURE.pgm [--cameras 4] [--fps 10] [--seconds 20]
                       [--threads 1] [--requests 1] [--look-every 3] [--separate]";

/// Options as `--key value`, repeatable, and the arguments that are not options.
struct Args {
    opts: HashMap<String, Vec<String>>,
    flags: Vec<String>,
    rest: Vec<String>,
}

impl Args {
    fn parse(a: &[String], flags: &[&str]) -> Res<Args> {
        let (mut opts, mut fl, mut rest) = (HashMap::<String, Vec<String>>::new(), vec![], vec![]);
        let mut i = 0;
        while i < a.len() {
            if let Some(k) = a[i].strip_prefix("--") {
                if flags.contains(&k) {
                    fl.push(k.to_string());
                } else {
                    let v = a.get(i + 1).ok_or(format!("--{k} needs a value"))?;
                    opts.entry(k.to_string()).or_default().push(v.clone());
                    i += 1;
                }
            } else {
                rest.push(a[i].clone());
            }
            i += 1;
        }
        Ok(Args { opts, flags: fl, rest })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).and_then(|v| v.last()).map(|s| s.as_str())
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Res<T> {
        match self.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{k}: {v:?} is not a number").into()),
        }
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

fn detector_options(a: &Args) -> Res<DetectorOptions> {
    Ok(DetectorOptions {
        threads: a.get("threads").map(|t| t.parse()).transpose().map_err(|_| "--threads is not a number")?,
        classes: a.get("classes").map(|c| c.split(',').map(|s| s.trim().to_string()).collect()),
        precision: a.get("precision").map(String::from),
        ..Default::default()
    })
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// A binary PGM (P5, 8-bit): (pixels, width, height).
fn read_pgm(path: &str) -> Res<(Vec<u8>, usize, usize)> {
    let data = std::fs::read(path)?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while i < data.len() && data[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < data.len() && data[i] == b'#' {
            while i < data.len() && data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < data.len() && !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&data[s..i]).to_string());
    }
    if fields[0] != "P5" || fields[3] != "255" {
        return Err(format!("{path}: only binary 8-bit PGM (P5, 255) is read").into());
    }
    let (w, h): (usize, usize) = (fields[1].parse()?, fields[2].parse()?);
    let px = data.get(i + 1..i + 1 + w * h).ok_or(format!("{path}: the picture is cut short"))?;
    Ok((px.to_vec(), w, h))
}

fn fold(a: &Args) -> Res<()> {
    let [src, dst] = a.rest.as_slice() else { return Err("fold IN.onnx OUT.onnx".into()) };
    #[cfg(feature = "fold")]
    {
        tamserve_boxes::fold::gray_model(src, dst)?;
        eprintln!("wrote {dst}: 1 channel, any batch");
        Ok(())
    }
    #[cfg(not(feature = "fold"))]
    {
        let _ = (src, dst);
        Err("this build has no `fold` feature".into())
    }
}

fn detect(a: &Args) -> Res<()> {
    let mut det = Detector::new(a.get("model").ok_or("--model is needed")?, detector_options(a)?)?;
    let pictures: Vec<_> = a.rest.iter().map(|p| read_pgm(p)).collect::<Res<_>>()?;
    let frames: Vec<GrayFrame> = pictures.iter().map(|(p, w, h)| GrayFrame::new(p, *w, *h)).collect::<Result<_, _>>()?;
    let names = det.names().to_vec();
    for (path, f) in a.rest.iter().zip(det.detect(&frames)?) {
        let finds: Vec<String> = (0..f.len()).map(|i| {
            let b = f.boxes[i];
            format!("{{\"label\":{},\"score\":{:.4},\"box\":[{:.1},{:.1},{:.1},{:.1}]}}",
                    json_str(&names[f.labels[i]]), f.scores[i], b[0], b[1], b[2], b[3])
        }).collect();
        println!("{{\"picture\":{},\"finds\":[{}]}}", json_str(path), finds.join(","));
    }
    Ok(())
}

/// Log a camera's state when it changes: its name and host only, never its address.
#[cfg(feature = "cameras")]
fn log_changes(shared: &tamserve_boxes::runner::Shared, last: &mut HashMap<String, &'static str>) {
    for (name, v) in shared.cameras.lock().unwrap().iter() {
        let s = v.health.state().as_str();
        if last.insert(name.clone(), s) != Some(s) {
            let why = if s == "down" { format!(": {}", v.health.last_error()) } else { String::new() };
            eprintln!("tamserve-boxes: camera {} is {s}{why}", v.label);
        }
    }
}

#[cfg(feature = "cameras")]
fn run(a: &Args) -> Res<()> {
    use tamserve_boxes::{camera, config::Config, runner::Runner};
    let path = a.get("config").ok_or("run --config cameras.toml (camera passwords live in the config, not on the command line)")?;
    let (cfg, cams) = Config::load(path)?;
    if cams.is_empty() {
        return Err("the config has no [[camera]]".into());
    }
    camera::init(cfg.decode.decoder)?;
    let tracker = BoxTracker::from_model(&cfg.detector.model, cfg.detector_options(), cfg.tracker_options(&cams))?;
    let mut runner = Runner::new(tracker, camera::Opening::from(&cfg.decode));
    runner.set_cameras(&cams)?;
    if let Some(l) = &cfg.http.listen {
        let base = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let at = tamserve_boxes::http::serve(l, cfg.http_token(base)?, runner.shared.clone())?;
        eprintln!("tamserve-boxes: metrics on http://{at}/metrics, status on /status, OpenAI-compatible on /v1/chat/completions");
    }
    let mut states = HashMap::new();
    let stdout = std::io::stdout();
    loop {
        let reports = runner.step(Duration::from_millis(500))?;
        log_changes(&runner.shared, &mut states);
        let mut out = stdout.lock();
        for r in &reports {
            writeln!(out, "{}", serde_json::to_string(r)?)?;
        }
        out.flush()?;
        if reports.is_empty() && runner.all_ended() {
            log_changes(&runner.shared, &mut states);
            return Ok(());
        }
    }
}

#[cfg(not(feature = "cameras"))]
fn run(_: &Args) -> Res<()> {
    Err("this build has no `cameras` feature".into())
}

/// CPU seconds (user + system) and peak resident memory in bytes of a process, from /proc.
fn usage(pid: u32) -> (f64, u64) {
    let tick = 100.0; // USER_HZ on Linux
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let after = stat.rsplit_once(')').map(|x| x.1).unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    let cpu = if f.len() > 13 { (f[11].parse::<f64>().unwrap_or(0.0) + f[12].parse::<f64>().unwrap_or(0.0)) / tick } else { 0.0 };
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let rss = status.lines().find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|k| k.parse::<u64>().ok()).unwrap_or(0) * 1024;
    (cpu, rss)
}

/// Frames for synthetic cameras: `picture` moving right over a smooth background, 720p.
fn synthetic(picture: &(Vec<u8>, usize, usize), seed: usize, n: usize) -> Vec<Vec<u8>> {
    let (w, h) = (1280usize, 720usize);
    let (p, pw, ph) = picture;
    let bg: Vec<u8> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); (90 + (x * 60 / w) as u32 + (y * 40 / h) as u32) as u8 }).collect();
    (0..n).map(|k| {
        let mut f = bg.clone();
        let x0 = (40 + 4 * k + 37 * seed) % (w - pw);
        for y in 0..(*ph).min(h - 60) {
            f[(60 + y) * w + x0..(60 + y) * w + x0 + pw].copy_from_slice(&p[y * pw..(y + 1) * pw]);
        }
        f
    }).collect()
}

fn bench(a: &Args) -> Res<()> {
    let n: usize = a.num("cameras", 4)?;
    let fps: f64 = a.num("fps", 10.0)?;
    let seconds: f64 = a.num("seconds", 20.0)?;
    let image = a.get("image").ok_or("--image PICTURE.pgm is needed")?;
    if a.flag("separate") {
        // One process per camera, each with a model of its own: the usual way.
        let me = std::env::current_exe()?;
        let mut children = Vec::new();
        for c in 0..n {
            let mut cmd = Command::new(&me);
            cmd.args(["bench", "--model", a.get("model").ok_or("--model")?, "--image", image, "--cameras", "1",
                      "--fps", &fps.to_string(), "--seconds", &seconds.to_string(), "--threads", "1",
                      "--look-every", &a.num::<u32>("look-every", 3)?.to_string(), "--seed", &c.to_string(), "--quiet", "1"]);
            children.push(cmd.stdout(Stdio::piped()).spawn()?);
        }
        std::thread::sleep(Duration::from_secs_f64(3.0 + 0.5 * n as f64)); // load and warm up
        let pids: Vec<u32> = children.iter().map(|c| c.id()).collect();
        let c0: f64 = pids.iter().map(|&p| usage(p).0).sum();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(seconds * 0.6));
        let rss: u64 = pids.iter().map(|&p| usage(p).1).sum();
        let cpu = (pids.iter().map(|&p| usage(p).0).sum::<f64>() - c0) / t0.elapsed().as_secs_f64();
        let mut rates = Vec::new();
        for c in children {
            let out = c.wait_with_output()?;
            rates.push(String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap_or(0.0));
        }
        let worst = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("{{\"cameras\":{n},\"fps_asked\":{fps},\"setup\":\"separate\",\"processes\":{n},\"cpu_cores\":{cpu:.2},\"memory_mb\":{},\"fps_per_camera\":{worst:.1}}}", rss >> 20);
        return Ok(());
    }
    let picture = read_pgm(image)?;
    let seed: usize = a.num("seed", 0)?;
    let videos: Vec<Vec<Vec<u8>>> = (0..n).map(|c| synthetic(&picture, seed + c, 30)).collect();
    let mut opts = detector_options(a)?;
    opts.threads = Some(a.num("threads", 1)?);
    opts.requests = a.num("requests", 1)?;
    let mut tracker: BoxTracker<Detector, usize> = BoxTracker::new(Detector::new(a.get("model").ok_or("--model")?, opts)?,
        TrackerOptions { look_every: LookEvery::All(a.num("look-every", 3)?), ..Default::default() })?;
    let frame = |c: usize, k: usize| GrayFrame::new(&videos[c][k % 30], 1280, 720);
    for c in 0..n {
        tracker.step(&[(c, frame(c, 0)?)])?; // warm up: the first call prepares the model's shapes
    }
    let me = std::process::id();
    let (c0, t0) = (usage(me).0, Instant::now());
    let mut rounds = 0usize;
    while t0.elapsed().as_secs_f64() < seconds {
        let frames: Vec<(usize, GrayFrame)> = (0..n).map(|c| Ok((c, frame(c, rounds)?))).collect::<Result<_, tamserve_boxes::Error>>()?;
        tracker.step(&frames)?;
        rounds += 1;
        let wait = rounds as f64 / fps - t0.elapsed().as_secs_f64();
        if wait > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(wait));
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let (c1, rss) = usage(me);
    if a.get("quiet").is_some() {
        println!("{:.2}", rounds as f64 / wall);
    } else {
        println!("{{\"cameras\":{n},\"fps_asked\":{fps},\"setup\":\"shared\",\"processes\":1,\"cpu_cores\":{:.2},\"memory_mb\":{},\"fps_per_camera\":{:.1}}}",
                 (c1 - c0) / wall, rss >> 20, rounds as f64 / wall);
    }
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = match argv.first().map(|s| s.as_str()) {
        Some(cmd @ ("fold" | "detect" | "run" | "bench" | "box" | "coordinator")) => {
            Args::parse(&argv[1..], &["separate"]).and_then(|a| match cmd {
                "fold" => fold(&a),
                "detect" => detect(&a),
                "run" => run(&a),
                "box" => run_box(&a),
                "coordinator" => coordinator(&a),
                _ => bench(&a),
            })
        }
        _ => Err(USAGE.into()),
    };
    if let Err(e) = result {
        eprintln!("tamserve-boxes: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cameras")]
fn run_box(a: &Args) -> Res<()> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use tamserve_boxes::coord::{self, CameraHealth, Channel, FromBox, FromCoordinator};
    use tamserve_boxes::{camera, config::Config, runner::Runner};

    let path = a.get("config").ok_or("box --config cameras.toml")?;
    let (cfg, cams) = Config::load(path)?;
    let token = coord::read_token(std::path::Path::new(a.get("token-file").ok_or("--token-file is needed")?))?;
    let addr = a.get("coordinator").ok_or("--coordinator HOST:PORT is needed")?.to_string();
    let name = a.get("name").ok_or("--name is needed: what the coordinator calls this box")?.to_string();
    let capacity: f64 = a.num("capacity", cams.iter().map(|c| c.weight).sum::<f64>())?;
    camera::init(cfg.decode.decoder)?;
    let tracker = BoxTracker::from_model(&cfg.detector.model, cfg.detector_options(), cfg.tracker_options(&cams))?;
    let mut runner = Runner::new(tracker, camera::Opening::from(&cfg.decode));
    let shared = runner.shared.clone();
    if let Some(l) = &cfg.http.listen {
        let base = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let at = tamserve_boxes::http::serve(l, cfg.http_token(base)?, shared.clone())?;
        eprintln!("tamserve-boxes box {name}: metrics on http://{at}/metrics");
    }
    let (assign_tx, assign_rx) = mpsc::channel::<Vec<String>>();
    let (report_tx, report_rx) = mpsc::sync_channel::<Vec<serde_json::Value>>(256);
    let known: Vec<String> = cams.iter().map(|c| c.name.clone()).collect();

    // The link to the coordinator, kept up with backoff. The box keeps serving its cameras
    // while the coordinator is away, so a coordinator outage does not stop the analytics.
    {
        let (name, shared) = (name.clone(), shared.clone());
        std::thread::Builder::new().name("coordinator link".into()).spawn(move || {
            let backoff = camera::Backoff { first: 1.0, max: 30.0 };
            let mut failures = 0u32;
            loop {
                let ch = Channel::connect(&addr, &token, &name).and_then(|mut ch| {
                    ch.send(&FromBox::Hello { protocol: coord::PROTOCOL.into(), capacity, known: known.clone() })?;
                    Ok(ch)
                });
                let mut ch = match ch {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("tamserve-boxes box {name}: cannot reach the coordinator: {e}");
                        std::thread::sleep(backoff.delay(failures));
                        failures = failures.saturating_add(1);
                        continue;
                    }
                };
                eprintln!("tamserve-boxes box {name}: joined the coordinator");
                failures = 0;
                let Ok(mut writer) = ch.split() else { continue };
                let _ = ch.set_read_timeout(Some(Duration::from_secs(10)));
                let (dead_tx, dead_rx) = mpsc::channel::<String>();
                let tx = assign_tx.clone();
                std::thread::spawn(move || loop {
                    match ch.recv::<FromCoordinator>() {
                        Ok(FromCoordinator::Assign { cameras }) => { let _ = tx.send(cameras); }
                        Ok(FromCoordinator::Ping) => {}
                        Ok(FromCoordinator::Refused { reason }) => { let _ = dead_tx.send(format!("refused: {reason}")); ch.shutdown(); break; }
                        Err(e) => { let _ = dead_tx.send(e.to_string()); ch.shutdown(); break; }
                    }
                });
                let mut last_us = shared.step_us_total.load(Ordering::Relaxed);
                let mut last_t = Instant::now();
                let why = 'session: loop {
                    let tick = Instant::now() + Duration::from_secs(1);
                    while Instant::now() < tick {
                        if let Ok(why) = dead_rx.try_recv() {
                            break 'session why;
                        }
                        match report_rx.recv_timeout(Duration::from_millis(100)) {
                            Ok(mut batch) => {
                                while let Ok(more) = report_rx.try_recv() {
                                    batch.extend(more);
                                }
                                if let Err(e) = writer.send(&FromBox::Reports { reports: batch }) {
                                    break 'session e.to_string();
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    }
                    let us = shared.step_us_total.load(Ordering::Relaxed);
                    let busy = (us - last_us) as f64 / 1e6 / last_t.elapsed().as_secs_f64();
                    (last_us, last_t) = (us, Instant::now());
                    let cameras = shared.cameras.lock().unwrap().iter().map(|(n, v)| CameraHealth {
                        name: n.clone(), state: v.health.state().as_str().into(), frame_age_seconds: v.health.frame_age(),
                        frames_decoded: v.health.decoded.load(Ordering::Relaxed), reconnects: v.health.reconnects.load(Ordering::Relaxed),
                    }).collect();
                    if let Err(e) = writer.send(&FromBox::Heartbeat { cameras, busy: busy.min(1.0) }) {
                        break 'session e.to_string();
                    }
                };
                writer.shutdown();
                eprintln!("tamserve-boxes box {name}: lost the coordinator ({why}); still serving its cameras, reconnecting");
            }
        })?;
    }

    let mut states = HashMap::new();
    loop {
        let mut wanted = None;
        while let Ok(w) = assign_rx.try_recv() {
            wanted = Some(w); // only the newest assignment matters
        }
        if let Some(w) = wanted {
            let set: Vec<_> = cams.iter().filter(|c| w.contains(&c.name)).cloned().collect();
            let (opened, closed) = runner.set_cameras(&set)?;
            if !opened.is_empty() || !closed.is_empty() {
                eprintln!("tamserve-boxes box {name}: now serving {:?} (opened {opened:?}, closed {closed:?})", runner.cameras());
            }
        }
        let reports = runner.step(Duration::from_millis(200))?;
        log_changes(&runner.shared, &mut states);
        if !reports.is_empty() {
            let batch = reports.iter().filter_map(|r| serde_json::to_value(r).ok()).collect();
            let _ = report_tx.try_send(batch); // a full queue drops results rather than stall the cameras
        }
    }
}

#[cfg(not(feature = "cameras"))]
fn run_box(_: &Args) -> Res<()> {
    Err("this build has no `cameras` feature".into())
}

fn coordinator(a: &Args) -> Res<()> {
    use tamserve_boxes::coord::{read_token, CoordConfig, Coordinator};
    let path = a.get("config").ok_or("coordinator --config coordinator.toml")?;
    let cfg: CoordConfig = toml::from_str(&std::fs::read_to_string(path)?).map_err(|e| format!("{path}: {e}"))?;
    let base = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    let abs = |p: &std::path::Path| if p.is_relative() { base.join(p) } else { p.to_path_buf() };
    let token = read_token(&abs(&cfg.token_file))?;
    let http = cfg.http.clone();
    let http_token = match &cfg.http_token_file {
        Some(t) => Some(String::from_utf8(read_token(&abs(t))?)?),
        None => None,
    };
    let n = cfg.cameras.len();
    let c = Coordinator::new(cfg, Box::new(std::io::stdout()))?;
    let at = c.serve(token)?;
    eprintln!("tamserve-boxes coordinator: {n} cameras; boxes connect to {at}");
    if let Some(l) = http {
        let h = tamserve_boxes::http::serve(&l, http_token, c.clone())?;
        eprintln!("tamserve-boxes coordinator: metrics on http://{h}/metrics, status on /status");
    }
    loop {
        std::thread::park();
    }
}
