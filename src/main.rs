//! tamserve-boxes: find and follow boxes on many cameras with YOLOX-Nano on gray frames,
//! on the CPU through OpenVINO.
//!
//!   tamserve-boxes fold yolox_nano.onnx yolox_nano_gray.onnx
//!   tamserve-boxes detect --model yolox_nano_gray.onnx picture.pgm ...
//!   tamserve-boxes run --model yolox_nano_gray.onnx --camera gate=rtsp://... --camera yard=rtsp://...
//!   tamserve-boxes bench --model yolox_nano_gray.onnx --image picture.pgm --cameras 8 --fps 25
//!
//! `run` reads each camera through ffmpeg, which hands over the gray picture (the Y
//! plane) with no colour conversion, keeps only each camera's newest frame, and prints
//! one JSON line per camera per round. OpenVINO's C library is found at run time: put
//! its folder on LD_LIBRARY_PATH or set OPENVINO_INSTALL_DIR.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tamserve_boxes::{BoxTracker, Detect, Detector, DetectorOptions, GrayFrame, LookEvery, TrackerOptions};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage:
  tamserve-boxes fold IN.onnx OUT.onnx
  tamserve-boxes detect --model M [--threads N] [--classes a,b] PICTURE.pgm ...
  tamserve-boxes run --model M --camera NAME=URL [--camera ...] [--threads N] [--look-every K]
                     [--look NAME=K ...] [--hold F] [--classes a,b] [--width W --height H] [--hwaccel vaapi]
                     [--realtime]   (read recordings at their own frame rate, as cameras would send them)
  tamserve-boxes bench --model M --image PICTURE.pgm [--cameras 4] [--fps 10] [--seconds 20]
                       [--threads 1] [--look-every 3] [--separate]";

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
    fn all(&self, k: &str) -> &[String] {
        self.opts.get(k).map(|v| v.as_slice()).unwrap_or(&[])
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

/// One camera's newest frame, refilled by its reader thread.
struct Slot {
    frame: Mutex<(Vec<u8>, u64, bool)>, // pixels, frame number, ended
    fresh: Arc<(Mutex<u64>, Condvar)>,   // shared: counts frames from every camera
}

fn probe(url: &str) -> Res<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", url])
        .output()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().lines().next().unwrap_or("").split(',');
    match (it.next().and_then(|w| w.parse().ok()), it.next().and_then(|h| h.parse().ok())) {
        (Some(w), Some(h)) => Ok((w, h)),
        _ => Err("could not read the camera's picture size (ffprobe)".into()), // not the address: it can hold a password
    }
}

fn spawn_reader(url: String, w: usize, h: usize, scale: bool, hwaccel: Option<String>, realtime: bool,
                slot: Arc<Slot>) -> Res<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error"]);
    if realtime {
        cmd.arg("-re"); // a recording read at its own frame rate, as a camera would send it
    }
    if url.starts_with("rtsp") {
        cmd.args(["-rtsp_transport", "tcp"]);
    }
    if let Some(hw) = &hwaccel {
        cmd.args(["-hwaccel", hw]); // decode on the integrated GPU; detection stays on the CPU
    }
    cmd.args(["-i", &url, "-an"]);
    if scale {
        cmd.args(["-vf", &format!("scale={w}:{h}")]);
    }
    cmd.args(["-pix_fmt", "gray", "-f", "rawvideo", "-"]).stdout(Stdio::piped()).stderr(Stdio::inherit());
    let mut child = cmd.spawn()?;
    let mut out = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; w * h];
        let mut n = 0u64;
        while out.read_exact(&mut buf).is_ok() {
            n += 1;
            {
                let mut f = slot.frame.lock().unwrap();
                std::mem::swap(&mut f.0, &mut buf); // keep only the newest: a loop that falls behind skips frames
                f.1 = n;
            }
            if buf.len() != w * h {
                buf.resize(w * h, 0);
            }
            let (m, cv) = &*slot.fresh;
            *m.lock().unwrap() += 1;
            cv.notify_all();
        }
        slot.frame.lock().unwrap().2 = true;
        let _ = child.kill();
        let (m, cv) = &*slot.fresh;
        *m.lock().unwrap() += 1;
        cv.notify_all();
    });
    Ok(())
}

fn tracker_options(a: &Args) -> Res<TrackerOptions<String>> {
    let every: u32 = a.num("look-every", 3)?;
    let mut own = HashMap::new();
    for spec in a.all("look") {
        let (n, k) = spec.split_once('=').ok_or("--look NAME=K")?;
        own.insert(n.to_string(), k.parse().map_err(|_| "--look NAME=K: K is not a number")?);
    }
    Ok(TrackerOptions {
        look_every: if own.is_empty() { LookEvery::All(every) } else { LookEvery::Each { cameras: own, default: every } },
        hold: a.num("hold", 30)?,
        ..Default::default()
    })
}

fn run(a: &Args) -> Res<()> {
    let det = Detector::new(a.get("model").ok_or("--model is needed")?, detector_options(a)?)?;
    let mut tracker = BoxTracker::new(det, tracker_options(a)?)?;
    let fresh = Arc::new((Mutex::new(0u64), Condvar::new()));
    let mut cams: Vec<(String, Arc<Slot>, usize, usize)> = Vec::new();
    for spec in a.all("camera") {
        let (name, url) = spec.split_once('=').ok_or("--camera NAME=URL")?;
        let (w, h, scale) = match (a.get("width"), a.get("height")) {
            (Some(w), Some(h)) => (w.parse()?, h.parse()?, true),
            _ => { let (w, h) = probe(url)?; (w, h, false) }
        };
        let slot = Arc::new(Slot { frame: Mutex::new((vec![0; w * h], 0, false)), fresh: fresh.clone() });
        spawn_reader(url.to_string(), w, h, scale, a.get("hwaccel").map(String::from), a.flag("realtime"), slot.clone())?;
        cams.push((name.to_string(), slot, w, h));
    }
    if cams.is_empty() {
        return Err("give at least one --camera NAME=URL".into());
    }
    let mut seen = vec![0u64; cams.len()];
    let mut copies: Vec<Vec<u8>> = cams.iter().map(|(_, _, w, h)| vec![0; w * h]).collect();
    let start = Instant::now();
    let stdout = std::io::stdout();
    loop {
        // Wait until some camera has a frame this loop has not seen, then take every camera's newest.
        {
            let (m, cv) = &*fresh;
            let mut g = m.lock().unwrap();
            let before = *g;
            while *g == before {
                g = cv.wait_timeout(g, Duration::from_secs(1)).unwrap().0;
                if cams.iter().all(|(_, s, _, _)| s.frame.lock().unwrap().2) {
                    return Ok(()); // every stream has ended
                }
            }
        }
        let mut batch = Vec::new();
        for (i, (name, slot, _, _)) in cams.iter().enumerate() {
            let f = slot.frame.lock().unwrap();
            if f.1 > seen[i] {
                seen[i] = f.1;
                copies[i].copy_from_slice(&f.0);
                batch.push(i);
            }
            let _ = name;
        }
        let frames: Vec<(String, GrayFrame)> = batch.iter()
            .map(|&i| Ok((cams[i].0.clone(), GrayFrame::new(&copies[i], cams[i].2, cams[i].3)?)))
            .collect::<Result<_, tamserve_boxes::Error>>()?;
        let found = tracker.step(&frames)?;
        let t = start.elapsed().as_secs_f64();
        let mut out = stdout.lock();
        for &i in &batch {
            let name = &cams[i].0;
            let objs: Vec<String> = found[name].iter().map(|(id, f)| {
                format!("{{\"id\":{id},\"label\":{},\"score\":{:.3},\"box\":[{:.1},{:.1},{:.1},{:.1}]}}",
                        json_str(&f.label), f.score, f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3])
            }).collect();
            writeln!(out, "{{\"t\":{t:.3},\"camera\":{},\"frame\":{},\"objects\":[{}]}}", json_str(name), seen[i], objs.join(","))?;
        }
        out.flush()?;
    }
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
        Some(cmd @ ("fold" | "detect" | "run" | "bench")) => {
            Args::parse(&argv[1..], &["separate", "realtime"]).and_then(|a| match cmd {
                "fold" => fold(&a),
                "detect" => detect(&a),
                "run" => run(&a),
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
