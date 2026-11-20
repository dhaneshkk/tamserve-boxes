//! The ONVIF lookup against a camera on localhost that answers as real ones do: a Profile T
//! camera with Media2, and an older one with Media1 only and no GetServices.

#![cfg(feature = "onvif")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use tamserve_boxes::onvif::{self, Options, Want};

const TT: &str = "http://www.onvif.org/ver10/schema";
const TDS: &str = "http://www.onvif.org/ver10/device/wsdl";
const TR2: &str = "http://www.onvif.org/ver20/media/wsdl";
const TRT: &str = "http://www.onvif.org/ver10/media/wsdl";

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    ProfileT,
    Media1Only,
    /// A Profile T camera whose streams are this file, so a real pipeline can open one.
    File(&'static str),
}

/// What the camera says its own address is: not one the test can reach, as behind NAT.
const REPORTED: &str = "192.168.1.64";

struct Camera {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

fn envelope(body: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tt="{TT}" xmlns:tds="{TDS}" xmlns:tr2="{TR2}" xmlns:trt="{TRT}"><s:Body>{body}</s:Body></s:Envelope>"#
    )
}

fn date_time() -> String {
    r#"<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:DateTimeType>NTP</tt:DateTimeType><tt:DaylightSavings>false</tt:DaylightSavings><tt:UTCDateTime><tt:Time><tt:Hour>10</tt:Hour><tt:Minute>0</tt:Minute><tt:Second>0</tt:Second></tt:Time><tt:Date><tt:Year>2026</tt:Year><tt:Month>11</tt:Month><tt:Day>20</tt:Day></tt:Date></tt:UTCDateTime></tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>"#.to_string()
}

fn services(host: &str, port: u16) -> String {
    let s = |ns: &str, path: &str| {
        format!("<tds:Service><tds:Namespace>{ns}</tds:Namespace><tds:XAddr>http://{host}:{port}{path}</tds:XAddr><tds:Version><tt:Major>2</tt:Major><tt:Minor>0</tt:Minor></tds:Version></tds:Service>")
    };
    format!("<tds:GetServicesResponse>{}{}{}</tds:GetServicesResponse>",
            s(TDS, "/onvif/device_service"), s(TRT, "/onvif/media_service"), s(TR2, "/onvif/media2_service"))
}

fn capabilities(port: u16) -> String {
    format!("<tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media><tt:XAddr>http://{REPORTED}:{port}/onvif/media_service</tt:XAddr><tt:StreamingCapabilities><tt:RTPMulticast>false</tt:RTPMulticast><tt:RTP_TCP>true</tt:RTP_TCP><tt:RTP_RTSP_TCP>true</tt:RTP_RTSP_TCP></tt:StreamingCapabilities></tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse>")
}

/// (token, name, encoding, width, height)
const PROFILES: [(&str, &str, &str, u32, u32); 3] = [
    ("Profile_101", "mainStream", "H265", 2560, 1440),
    ("Profile_102", "subStream", "H264", 640, 360),
    ("Profile_103", "thirdStream", "H264", 1280, 720),
];

fn media2_profiles() -> String {
    let p = PROFILES.iter().map(|(t, n, e, w, h)| format!(
        r#"<tr2:Profiles token="{t}" fixed="true"><tr2:Name>{n}</tr2:Name><tr2:Configurations><tr2:VideoEncoder token="VE_{t}" GovLength="50" Profile="Main"><tt:Name>VE_{t}</tt:Name><tt:UseCount>1</tt:UseCount><tt:Encoding>{e}</tt:Encoding><tt:Resolution><tt:Width>{w}</tt:Width><tt:Height>{h}</tt:Height></tt:Resolution><tt:RateControl ConstantBitRate="false"><tt:FrameRateLimit>25</tt:FrameRateLimit><tt:BitrateLimit>2048</tt:BitrateLimit></tt:RateControl><tt:Quality>3</tt:Quality></tr2:VideoEncoder></tr2:Configurations></tr2:Profiles>"#))
        .collect::<String>();
    // An audio-only profile, which has no video encoder and is skipped.
    format!(r#"<tr2:GetProfilesResponse>{p}<tr2:Profiles token="audio"><tr2:Name>audio</tr2:Name><tr2:Configurations></tr2:Configurations></tr2:Profiles></tr2:GetProfilesResponse>"#)
}

fn media1_profiles() -> String {
    // Media1 has no H.265: the camera offers the two H.264 streams.
    let p = PROFILES.iter().filter(|p| p.2 == "H264").map(|(t, n, e, w, h)| format!(
        r#"<trt:Profiles token="{t}" fixed="true"><tt:Name>{n}</tt:Name><tt:VideoEncoderConfiguration token="VEC_{t}"><tt:Name>VEC_{t}</tt:Name><tt:UseCount>1</tt:UseCount><tt:Encoding>{e}</tt:Encoding><tt:Resolution><tt:Width>{w}</tt:Width><tt:Height>{h}</tt:Height></tt:Resolution><tt:Quality>3</tt:Quality><tt:RateControl><tt:FrameRateLimit>15</tt:FrameRateLimit><tt:EncodingInterval>1</tt:EncodingInterval><tt:BitrateLimit>512</tt:BitrateLimit></tt:RateControl><tt:Multicast><tt:Address><tt:Type>IPv4</tt:Type><tt:IPv4Address>0.0.0.0</tt:IPv4Address></tt:Address><tt:Port>0</tt:Port><tt:TTL>0</tt:TTL><tt:AutoStart>false</tt:AutoStart></tt:Multicast><tt:SessionTimeout>PT60S</tt:SessionTimeout></tt:VideoEncoderConfiguration></trt:Profiles>"#))
        .collect::<String>();
    format!("<trt:GetProfilesResponse>{p}</trt:GetProfilesResponse>")
}

/// The token a GetStreamUri request asks for.
fn token(body: &str) -> String {
    let start = body.find("ProfileToken>").map(|i| i + "ProfileToken>".len()).unwrap_or(0);
    body[start..].split('<').next().unwrap_or("").to_string()
}

fn stream_uri(kind: Kind, body: &str, media2: bool) -> String {
    let t = token(body);
    // As cameras do: their own address, sometimes with credentials in it.
    let uri = match kind {
        Kind::File(path) => format!("file://{path}"),
        _ => format!("rtsp://admin:secret@{REPORTED}:554/Streaming/{t}?transportmode=unicast"),
    };
    if media2 {
        format!("<tr2:GetStreamUriResponse><tr2:Uri>{uri}</tr2:Uri></tr2:GetStreamUriResponse>")
    } else {
        format!("<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>{uri}</tt:Uri><tt:InvalidAfterConnect>false</tt:InvalidAfterConnect><tt:InvalidAfterReboot>false</tt:InvalidAfterReboot><tt:Timeout>PT0S</tt:Timeout></trt:MediaUri></trt:GetStreamUriResponse>")
    }
}

fn fault() -> String {
    envelope(r#"<s:Fault><s:Code><s:Value>s:Sender</s:Value></s:Code><s:Reason><s:Text xml:lang="en">Action not supported</s:Text></s:Reason></s:Fault>"#)
}

fn answer(kind: Kind, path: &str, body: &str, port: u16) -> (u16, String) {
    let ok = |b: String| (200, envelope(&b));
    if body.contains("GetSystemDateAndTime") {
        ok(date_time())
    } else if body.contains("GetServices") {
        match kind {
            Kind::Media1Only => (400, fault()),
            Kind::File(_) => ok(services("127.0.0.1", port)),
            Kind::ProfileT => ok(services(REPORTED, port)),
        }
    } else if body.contains("GetCapabilities") {
        ok(capabilities(port))
    } else if body.contains("GetProfiles") && path == "/onvif/media2_service" {
        ok(media2_profiles())
    } else if body.contains("GetProfiles") && path == "/onvif/media_service" {
        ok(media1_profiles())
    } else if body.contains("GetStreamUri") && path == "/onvif/media2_service" {
        ok(stream_uri(kind, body, true))
    } else if body.contains("GetStreamUri") && path == "/onvif/media_service" {
        ok(stream_uri(kind, body, false))
    } else {
        (400, fault())
    }
}

fn read_request(s: &mut TcpStream) -> Option<(String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let length: usize = head.lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
        .unwrap_or(0);
    while buf.len() < head_end + length {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
    Some((path, String::from_utf8_lossy(&buf[head_end..]).to_string()))
}

fn serve(kind: Kind) -> Camera {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
        for s in listener.incoming() {
            let Ok(mut s) = s else { continue };
            let Some((path, body)) = read_request(&mut s) else { continue };
            seen.lock().unwrap().push(format!("{path} {body}"));
            let (status, reply) = answer(kind, &path, &body, port);
            let reason = if status == 200 { "OK" } else { "Bad Request" };
            let _ = write!(s, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/soap+xml; charset=utf-8\r\n\
                              Content-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len());
        }
    });
    Camera { port, requests }
}

#[test]
fn profile_t_camera_lists_its_streams_through_media2() {
    let cam = serve(Kind::ProfileT);
    let device = onvif::device_url(&format!("127.0.0.1:{}", cam.port)).unwrap();
    let found = onvif::block_on(onvif::profiles(&device, Some("viewer"), Some("pw"), &Options::default())).unwrap().unwrap();

    // The audio profile is skipped; the rest come in the camera's order.
    let tokens: Vec<&str> = found.iter().map(|p| p.token.as_str()).collect();
    assert_eq!(tokens, ["Profile_101", "Profile_102", "Profile_103"]);
    assert_eq!((found[0].encoding.as_str(), found[0].width, found[0].height, found[0].fps), ("H265", 2560, 1440, Some(25.0)));
    // The reported address is replaced by the one the camera was reached at, and its
    // credentials are dropped: the RTSP login goes on the source element instead.
    assert_eq!(found[1].uri, "rtsp://127.0.0.1:554/Streaming/Profile_102?transportmode=unicast");
    // Media2 was used, after reaching the reported service address at the device's host.
    let log = cam.requests.lock().unwrap();
    assert!(log.iter().any(|r| r.starts_with("/onvif/media2_service") && r.contains("GetProfiles")));
    assert!(!log.iter().any(|r| r.starts_with("/onvif/media_service")));
}

#[test]
fn resolve_picks_the_gray_and_colour_streams() {
    let cam = serve(Kind::ProfileT);
    let device = format!("127.0.0.1:{}", cam.port);
    let opts = Options { min_width: Some(640), ..Default::default() };
    let gray = onvif::resolve(&device, &Want::Gray, None, None, &opts).unwrap();
    assert_eq!(gray, "rtsp://127.0.0.1:554/Streaming/Profile_102?transportmode=unicast");
    let colour = onvif::resolve(&device, &Want::Colour, None, None, &opts).unwrap();
    assert_eq!(colour, "rtsp://127.0.0.1:554/Streaming/Profile_101?transportmode=unicast");
    // A gray stream at least 1000 wide: the 720p one, not the sub-stream.
    let wide = Options { min_width: Some(1000), ..Default::default() };
    assert!(onvif::resolve(&device, &Want::Gray, None, None, &wide).unwrap().contains("Profile_103"));
    let named = onvif::resolve(&device, &Want::Profile("thirdStream".into()), None, None, &opts).unwrap();
    assert!(named.contains("Profile_103"));
    let missing = onvif::resolve(&device, &Want::Profile("nope".into()), None, None, &opts).unwrap_err();
    assert!(missing.to_string().contains("Profile_101 (mainStream)"), "{missing}");
}

#[test]
fn older_camera_falls_back_to_get_capabilities_and_media1() {
    let cam = serve(Kind::Media1Only);
    let device = format!("127.0.0.1:{}", cam.port);
    let opts = Options::default();
    let url = onvif::device_url(&device).unwrap();
    let found = onvif::block_on(onvif::profiles(&url, None, None, &opts)).unwrap().unwrap();
    let tokens: Vec<&str> = found.iter().map(|p| p.token.as_str()).collect();
    assert_eq!(tokens, ["Profile_102", "Profile_103"]);
    assert_eq!(found[0].fps, Some(15.0));
    assert_eq!(onvif::resolve(&device, &Want::Gray, None, None, &opts).unwrap(),
               "rtsp://127.0.0.1:554/Streaming/Profile_102?transportmode=unicast");
    let log = cam.requests.lock().unwrap();
    assert!(log.iter().any(|r| r.contains("GetCapabilities")));
}

#[test]
fn a_service_on_another_host_is_refused_without_use_device_host() {
    let cam = serve(Kind::ProfileT);
    let opts = Options { use_device_host: false, ..Default::default() };
    let err = onvif::resolve(&format!("127.0.0.1:{}", cam.port), &Want::Gray, Some("viewer"), Some("pw"), &opts).unwrap_err();
    assert!(err.to_string().contains("another host"), "{err}");
    // Nothing went to the other host's services: only the device service was asked.
    assert!(cam.requests.lock().unwrap().iter().all(|r| r.starts_with("/onvif/device_service")));
}

#[test]
fn a_camera_that_is_not_there_is_an_error_not_a_hang() {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port(); // closed again
    let opts = Options { timeout: std::time::Duration::from_secs(1), ..Default::default() };
    let started = std::time::Instant::now();
    assert!(onvif::resolve(&format!("127.0.0.1:{port}"), &Want::Gray, None, None, &opts).is_err());
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[test]
fn the_onvif_command_prints_camera_blocks_for_cameras_toml() {
    let cam = serve(Kind::ProfileT);
    let dir = std::env::temp_dir().join(format!("tsb-onvif-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pw = dir.join("pw");
    std::fs::write(&pw, "s3cret\n").unwrap();
    let run = || std::process::Command::new(env!("CARGO_BIN_EXE_tamserve-boxes"))
        .args(["onvif", "--device", &format!("127.0.0.1:{}", cam.port), "--user", "viewer",
               "--password-file", pw.to_str().unwrap(), "--min-width", "640"])
        .output().unwrap();

    // A password file others can read is refused, as in cameras.toml.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&pw, std::fs::Permissions::from_mode(0o644)).unwrap();
    let refused = run();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("chmod 600"));

    std::fs::set_permissions(&pw, std::fs::Permissions::from_mode(0o600)).unwrap();
    let out = run();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();

    assert!(text.contains("Profile_101 \"mainStream\"  H265 2560x1440 25 fps"), "{text}");
    assert!(text.contains("# colour: Profile_101 (2560x1440)"), "{text}");
    assert!(text.contains("stream = \"gray\"   # Profile_102 \"subStream\", H264 640x360"), "{text}");
    assert!(!text.contains("s3cret"), "{text}");

    // What it prints is a config that loads.
    let block = text.lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>().join("\n");
    let config = dir.join("cameras.toml");
    std::fs::write(&config, format!("[detector]\nmodel = \"m.onnx\"\n{block}\n")).unwrap();
    let (_, cams) = tamserve_boxes::config::Config::load(&config).unwrap();
    assert_eq!(cams[0].name, "cam-127-0-0-1");
    assert_eq!(cams[0].password.as_deref(), Some("s3cret"));
    assert_eq!(cams[0].onvif.as_ref().unwrap().device, format!("http://127.0.0.1:{}/onvif/device_service", cam.port));
}

/// The camera supervisor looks the stream up over ONVIF and opens it: here a file the
/// camera reports, so that GStreamer can read it without an RTSP server.
#[test]
fn a_camera_source_opens_the_stream_onvif_reports() {
    use gstreamer::prelude::*;
    use tamserve_boxes::camera::{self, Bell, Opening, State};
    use tamserve_boxes::config::{Camera, DecodeSection, OnvifCamera, OnvifSection};

    camera::init(tamserve_boxes::config::Decoder::Software).unwrap();
    let dir = std::env::temp_dir().join(format!("tsb-onvif-src-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let video = dir.join("cam.avi");
    let make = gstreamer::parse::launch(&format!(
        "videotestsrc num-buffers=20 ! video/x-raw,width=320,height=240,framerate=10/1 ! jpegenc ! avimux ! filesink location={}",
        video.display())).unwrap();
    make.set_state(gstreamer::State::Playing).unwrap();
    let bus = make.bus().unwrap();
    bus.timed_pop_filtered(gstreamer::ClockTime::from_seconds(20), &[gstreamer::MessageType::Eos, gstreamer::MessageType::Error]).unwrap();
    make.set_state(gstreamer::State::Null).unwrap();

    let path: &'static str = Box::leak(video.display().to_string().into_boxed_str());
    let cam = serve(Kind::File(path));
    let device = format!("127.0.0.1:{}", cam.port);
    let camera = Camera {
        name: "yard".into(), url: device.clone(), user: None, password: None, look_every: None, weight: 1.0,
        onvif: Some(OnvifCamera {
            device, stream: "gray".into(), min_width: None,
            // A file has no host to swap for the device's.
            section: OnvifSection { use_device_host: Some(false), ..Default::default() },
        }),
    };
    let bell = std::sync::Arc::new(Bell::default());
    let source = camera::Source::open(camera, &Opening::from(&DecodeSection::default()), bell).unwrap();

    let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while source.health.state() != State::Ended && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(source.health.state(), State::Ended, "{}", source.health.last_error());
    assert!(source.health.decoded.load(std::sync::atomic::Ordering::Relaxed) >= 20);
    let frame = source.take().unwrap();
    assert_eq!((frame.width, frame.height), (320, 240));
    assert!(cam.requests.lock().unwrap().iter().any(|r| r.contains("GetStreamUri")));
}
