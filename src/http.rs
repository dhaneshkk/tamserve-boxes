//! A small HTTP endpoint on a box or the coordinator:
//!
//! - `GET /metrics`: Prometheus text, one series per camera and the loop's counters.
//! - `GET /status`: the same as JSON, with each camera's latest objects.
//! - `GET /v1/models` and `POST /v1/chat/completions`: OpenAI-compatible. The `model`
//!   is a camera's name (or `"all"`), and the answer's message is that camera's latest
//!   objects and health as JSON, so any OpenAI client can read the analytics.
//!
//! - `GET /live/<camera>.ts`: a camera with `live = true`, as MPEG-TS, from the box's own
//!   connection to it (see `restream`). At most 8 live viewers per box.
//! - `POST /safety/<camera>/drift`: a drift check's result (`crate::lens::DriftEstimate`
//!   as JSON) for a camera with safety rules, from a checker that runs OpenCV beside the
//!   box. Applied on the camera's next step: corrected, or every floor rule UNKNOWN.
//!
//! It binds where the config says (127.0.0.1 by default), reads each request with a 5 s
//! deadline and at most 1 MB of body, and with `token_file` set requires
//! `Authorization: Bearer <token>`, live streams included.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::{Error, Result};

/// What an endpoint serves: metrics text, and status as JSON with a `cameras` object.
pub trait Source: Send + Sync + 'static {
    fn metrics(&self) -> String;
    fn status(&self) -> Value;
    /// A camera's live stream, if it has one.
    /// A drift check's result for a camera's safety rules; false when the camera has none.
    fn drift(&self, _camera: &str, _est: crate::lens::DriftEstimate) -> bool {
        false
    }
    fn live(&self, _camera: &str) -> std::result::Result<Receiver<crate::restream::Chunk>, LiveRefused> {
        Err(LiveRefused::NoSuchStream)
    }
}

/// Why a live stream was not given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveRefused {
    NoSuchStream,
    Full,
}

const MOST_LIVE: usize = 8;

/// Serve `src` on `listen` in the background.
pub fn serve(listen: &str, token: Option<String>, src: Arc<dyn Source>) -> Result<std::net::SocketAddr> {
    let l = TcpListener::bind(listen).map_err(|e| Error::Invalid(format!("cannot listen on {listen}: {e}")))?;
    let addr = l.local_addr()?;
    let busy = Arc::new(AtomicUsize::new(0));
    let live = Arc::new(AtomicUsize::new(0));
    std::thread::Builder::new().name("http".into()).spawn(move || {
        for conn in l.incoming().flatten() {
            if busy.load(Ordering::Relaxed) >= 16 {
                continue; // dropped: too many at once
            }
            let (src, token, busy, live) = (src.clone(), token.clone(), busy.clone(), live.clone());
            busy.fetch_add(1, Ordering::Relaxed);
            std::thread::spawn(move || {
                let _ = handle(conn, token.as_deref(), &*src, &busy, &live);
                busy.fetch_sub(1, Ordering::Relaxed);
            });
        }
    })?;
    Ok(addr)
}

fn reply(mut s: &TcpStream, code: u16, kind: &str, body: &str) -> std::io::Result<()> {
    let reason = match code { 200 => "OK", 400 => "Bad Request", 401 => "Unauthorized", 404 => "Not Found",
                              503 => "Service Unavailable", _ => "Error" };
    write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
           body.len())
}

/// Send a live stream until the viewer leaves, falls behind, or the camera sends nothing
/// for 30 s. A live viewer does not count against the 16 requests at once.
fn stream_live(mut s: &TcpStream, rx: Receiver<crate::restream::Chunk>, busy: &AtomicUsize, live: &AtomicUsize) -> std::io::Result<()> {
    busy.fetch_sub(1, Ordering::Relaxed);
    let result = (|| {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: video/mp2t\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")?;
        loop {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(chunk) => s.write_all(&chunk)?,
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    })();
    live.fetch_sub(1, Ordering::Relaxed);
    busy.fetch_add(1, Ordering::Relaxed);
    result
}

fn handle(stream: TcpStream, token: Option<&str>, src: &dyn Source, busy: &AtomicUsize, live: &AtomicUsize) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut r = BufReader::new(&stream).take(1 << 20);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("").to_string());
    let (mut length, mut auth) = (0usize, None::<String>);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = v.trim().parse().unwrap_or(0).min(1 << 20),
                "authorization" => auth = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    if let Some(t) = token {
        let ok = auth.as_deref().and_then(|a| a.strip_prefix("Bearer ")).is_some_and(|a| ct_eq(a.as_bytes(), t.as_bytes()));
        if !ok {
            return reply(&stream, 401, "application/json", r#"{"error":{"message":"a bearer token is required"}}"#);
        }
    }
    let mut body = vec![0u8; length];
    r.read_exact(&mut body)?;
    let route = path.split('?').next().unwrap_or("");
    if let (true, Some(cam)) = (method == "GET", route.strip_prefix("/live/").and_then(|c| c.strip_suffix(".ts"))) {
        if live.fetch_add(1, Ordering::Relaxed) >= MOST_LIVE {
            live.fetch_sub(1, Ordering::Relaxed);
            return reply(&stream, 503, "application/json", r#"{"error":{"message":"too many live viewers on this box"}}"#);
        }
        return match src.live(cam) {
            Ok(rx) => stream_live(&stream, rx, busy, live),
            Err(why) => {
                live.fetch_sub(1, Ordering::Relaxed);
                match why {
                    LiveRefused::NoSuchStream => reply(&stream, 404, "application/json", r#"{"error":{"message":"no live stream for that camera"}}"#),
                    LiveRefused::Full => reply(&stream, 503, "application/json", r#"{"error":{"message":"too many viewers of that camera"}}"#),
                }
            }
        };
    }
    if let (true, Some(cam)) = (method == "POST", route.strip_prefix("/safety/").and_then(|c| c.strip_suffix("/drift"))) {
        return match serde_json::from_slice::<crate::lens::DriftEstimate>(&body) {
            Err(e) => reply(&stream, 400, "application/json", &json!({"error": {"message": e.to_string()}}).to_string()),
            Ok(est) => match src.drift(cam, est) {
                true => reply(&stream, 200, "application/json", r#"{"ok":true}"#),
                false => reply(&stream, 404, "application/json", &json!({"error": {"message": format!("no camera {cam:?} with safety rules")}}).to_string()),
            },
        };
    }
    match (method, route) {
        ("GET", "/metrics") => reply(&stream, 200, "text/plain; version=0.0.4", &src.metrics()),
        ("GET", "/status") => reply(&stream, 200, "application/json", &src.status().to_string()),
        ("GET", "/v1/models") => {
            let cams = src.status()["cameras"].as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
            let data: Vec<Value> = std::iter::once("all".to_string()).chain(cams)
                .map(|c| json!({"id": c, "object": "model", "owned_by": "tamserve-boxes"})).collect();
            reply(&stream, 200, "application/json", &json!({"object": "list", "data": data}).to_string())
        }
        ("POST", "/v1/chat/completions") => {
            let req: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => return reply(&stream, 400, "application/json", &json!({"error": {"message": e.to_string()}}).to_string()),
            };
            let model = req["model"].as_str().unwrap_or("all").to_string();
            let status = src.status();
            let content = if model == "all" {
                status["cameras"].clone()
            } else {
                match status["cameras"].get(&model) {
                    Some(c) => c.clone(),
                    None => return reply(&stream, 404, "application/json",
                                         &json!({"error": {"message": format!("no camera {model:?}"), "type": "invalid_request_error"}}).to_string()),
                }
            };
            let text = content.to_string();
            let created = crate::now_ms() / 1000;
            let answer = json!({
                "id": format!("chatcmpl-{created}"), "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            });
            reply(&stream, 200, "application/json", &answer.to_string())
        }
        _ => reply(&stream, 404, "application/json", r#"{"error":{"message":"not found"}}"#),
    }
}

/// Equal, in time that does not depend on where they differ.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// A Prometheus label value, escaped.
pub fn label(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Fake {
        got: Mutex<Vec<(String, crate::lens::DriftEstimate)>>,
    }

    impl Source for Fake {
        fn metrics(&self) -> String {
            String::new()
        }
        fn status(&self) -> Value {
            json!({"cameras": {}})
        }
        fn drift(&self, camera: &str, est: crate::lens::DriftEstimate) -> bool {
            if camera != "gate" {
                return false;
            }
            self.got.lock().unwrap().push((camera.to_string(), est));
            true
        }
    }

    fn post(addr: std::net::SocketAddr, path: &str, body: &str, token: Option<&str>) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        let auth = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
        write!(s, "POST {path} HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn a_drift_check_reaches_the_camera_and_needs_the_token() {
        let fake = Arc::new(Fake::default());
        let addr = serve("127.0.0.1:0", Some("t0k".into()), fake.clone()).unwrap();
        let est = r#"{"valid":true,"G":[[1,0,4],[0,1,0],[0,0,1]],"drift_px":4.2,"inliers":800,"ratio":0.9}"#;
        assert!(post(addr, "/safety/gate/drift", est, None).starts_with("HTTP/1.1 401"));
        assert!(post(addr, "/safety/gate/drift", est, Some("t0k")).starts_with("HTTP/1.1 200"));
        assert!(post(addr, "/safety/yard/drift", est, Some("t0k")).starts_with("HTTP/1.1 404"));
        assert!(post(addr, "/safety/gate/drift", "{not json", Some("t0k")).starts_with("HTTP/1.1 400"));
        let got = fake.got.lock().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].1.valid, got[0].1.drift_px, got[0].1.inliers), (true, Some(4.2), 800));
    }
}
