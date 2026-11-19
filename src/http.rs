//! A small HTTP endpoint on a box or the coordinator:
//!
//! - `GET /metrics`: Prometheus text, one series per camera and the loop's counters.
//! - `GET /status`: the same as JSON, with each camera's latest objects.
//! - `GET /v1/models` and `POST /v1/chat/completions`: OpenAI-compatible. The `model`
//!   is a camera's name (or `"all"`), and the answer's message is that camera's latest
//!   objects and health as JSON, so any OpenAI client can read the analytics.
//!
//! It binds where the config says (127.0.0.1 by default), reads each request with a 5 s
//! deadline and at most 1 MB of body, and with `token_file` set requires
//! `Authorization: Bearer <token>`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::{Error, Result};

/// What an endpoint serves: metrics text, and status as JSON with a `cameras` object.
pub trait Source: Send + Sync + 'static {
    fn metrics(&self) -> String;
    fn status(&self) -> Value;
}

/// Serve `src` on `listen` in the background.
pub fn serve(listen: &str, token: Option<String>, src: Arc<dyn Source>) -> Result<std::net::SocketAddr> {
    let l = TcpListener::bind(listen).map_err(|e| Error::Invalid(format!("cannot listen on {listen}: {e}")))?;
    let addr = l.local_addr()?;
    let busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    std::thread::Builder::new().name("http".into()).spawn(move || {
        for conn in l.incoming().flatten() {
            if busy.load(std::sync::atomic::Ordering::Relaxed) >= 16 {
                continue; // dropped: too many at once
            }
            let (src, token, busy) = (src.clone(), token.clone(), busy.clone());
            busy.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::thread::spawn(move || {
                let _ = handle(conn, token.as_deref(), &*src);
                busy.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            });
        }
    })?;
    Ok(addr)
}

fn reply(mut s: &TcpStream, code: u16, kind: &str, body: &str) -> std::io::Result<()> {
    let reason = match code { 200 => "OK", 400 => "Bad Request", 401 => "Unauthorized", 404 => "Not Found", _ => "Error" };
    write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
           body.len())
}

fn handle(stream: TcpStream, token: Option<&str>, src: &dyn Source) -> std::io::Result<()> {
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
    match (method, path.split('?').next().unwrap_or("")) {
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
