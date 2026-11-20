//! Cameras found and opened through ONVIF (Profile T, with Media1 for older cameras):
//! discovery on the network, the camera's stream profiles, and the RTSP address of the
//! profile a camera's config asks for.
//!
//! A camera in the config can name its ONVIF device instead of an RTSP URL:
//!
//! ```toml
//! [[camera]]
//! name = "gate"
//! onvif = "10.0.0.21"          # or "http://10.0.0.21/onvif/device_service"
//! stream = "gray"              # "gray" (default), "colour", or a profile's token or name
//! user = "viewer"
//! password_file = "/etc/tamserve/secrets/gate"
//! ```
//!
//! The camera's supervisor asks the camera for its profiles each time it opens the stream,
//! so a camera that was replaced, reset or re-addressed by its own DHCP lease is followed
//! without editing the config.
//!
//! ONVIF does not say which profile is monochrome, so `stream` picks by name first and by
//! size after that (see [`pick`]): the detector reads the gray plane of any stream, so the
//! gray pick only has to be the cheapest stream that is big enough.

use std::net::Ipv4Addr;
use std::time::Duration;

use onvif_rs::discovery::{DiscoveryBuilder, DiscoveryMode};
use onvif_rs::soap::client::{AuthType, Client, ClientBuilder, Credentials};
use onvif_schema as schema;
use url::Url;

use crate::{Error, Result};

const DEVICE_SERVICE: &str = "onvif/device_service";
const MEDIA1: &str = "http://www.onvif.org/ver10/media/wsdl";
const MEDIA2: &str = "http://www.onvif.org/ver20/media/wsdl";

/// Which of a camera's profiles to stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Want {
    /// The stream for detection: a profile named gray, grey or mono if there is one, else
    /// the smallest stream at least `min_width` wide (or the biggest, if none is).
    Gray,
    /// The stream for people to look at: a profile named colour or color if there is one,
    /// else the biggest stream that is not named gray (usually the main stream).
    Colour,
    /// The profile with this token, or else this name (letter case ignored).
    Profile(String),
}

impl Want {
    pub fn parse(s: &str) -> Want {
        match s.to_ascii_lowercase().as_str() {
            "gray" | "grey" => Want::Gray,
            "colour" | "color" => Want::Colour,
            _ => Want::Profile(s.to_string()),
        }
    }
}

/// How to talk to cameras over ONVIF, from the config's `[onvif]`.
#[derive(Clone, Debug)]
pub struct Options {
    /// How long one ONVIF request may take.
    pub timeout: Duration,
    /// Use the address the camera was reached at in the RTSP and service URLs it reports,
    /// in place of the host it names. Cameras behind NAT, or with several interfaces,
    /// often report an address the box cannot reach.
    pub use_device_host: bool,
    /// Allow WS-Security UsernameToken logins over plain HTTP, for cameras that support
    /// nothing else. Its SHA-1 password digest can be cracked from one captured request.
    pub allow_username_token_over_http: bool,
    /// The width the detector wants at least, for [`Want::Gray`]: `[decode] width`.
    pub min_width: Option<u32>,
    /// A PEM file of certificates to trust for `https://` devices: a site CA or the
    /// cameras' own certificates.
    pub ca_file: Option<std::path::PathBuf>,
    /// Accept a trusted certificate that does not name the camera's address.
    pub accept_invalid_hostnames: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            timeout: Duration::from_secs(5),
            use_device_host: true,
            allow_username_token_over_http: false,
            min_width: None,
            ca_file: None,
            accept_invalid_hostnames: false,
        }
    }
}

impl From<&crate::config::OnvifCamera> for Options {
    fn from(o: &crate::config::OnvifCamera) -> Self {
        let d = Options::default();
        Options {
            timeout: o.section.timeout_seconds.map(Duration::from_secs_f64).unwrap_or(d.timeout),
            use_device_host: o.section.use_device_host.unwrap_or(d.use_device_host),
            allow_username_token_over_http: o.section.allow_username_token_over_http,
            min_width: o.min_width,
            ca_file: o.section.ca_file.clone(),
            accept_invalid_hostnames: o.section.accept_invalid_hostnames,
        }
    }
}

/// One of a camera's media profiles that carries video.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub token: String,
    pub name: String,
    /// As the camera names it: "H264", "H265", "JPEG", "MPEG4"...
    pub encoding: String,
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
    /// The RTSP address, without credentials.
    pub uri: String,
}

/// A device that answered WS-Discovery.
#[derive(Clone, Debug)]
pub struct Found {
    pub name: Option<String>,
    pub hardware: Option<String>,
    /// Its device service addresses, as it announced them.
    pub urls: Vec<String>,
}

/// The device service URL for `device`: a URL as given, or a host (with an optional port)
/// made into `http://host/onvif/device_service`.
pub fn device_url(device: &str) -> Result<Url> {
    let text = if device.contains("://") { device.to_string() } else { format!("http://{device}/{DEVICE_SERVICE}") };
    let url = Url::parse(&text).map_err(|e| Error::Invalid(format!("onvif = {device:?}: {e}")))?;
    if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
        return Err(Error::Invalid(format!("onvif = {device:?}: give a host, or an http:// or https:// URL")));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Error::Invalid(format!(
            "onvif = {device:?}: put the user and password in `user` and `password_file`, not in the URL")));
    }
    Ok(url)
}

/// `uri` as the box should use it: without credentials, and with `device`'s host in place
/// of the one the camera reported when `use_device_host` is set. The port and path stay.
pub fn clean_uri(uri: &str, device: &Url, use_device_host: bool) -> Result<Url> {
    let mut u = Url::parse(uri).map_err(|e| Error::Invalid(format!("the camera reported a bad URL: {e}")))?;
    let _ = u.set_username("");
    let _ = u.set_password(None);
    if use_device_host {
        if let Some(host) = device.host_str() {
            u.set_host(Some(host)).map_err(|e| Error::Invalid(format!("the camera reported a bad URL: {e}")))?;
        }
    }
    Ok(u)
}

fn is_named(p: &Profile, words: &[&str]) -> bool {
    let (n, t) = (p.name.to_ascii_lowercase(), p.token.to_ascii_lowercase());
    words.iter().any(|w| n.contains(w) || t.contains(w))
}

const GRAY: [&str; 3] = ["gray", "grey", "mono"];
const COLOUR: [&str; 2] = ["colour", "color"];

/// Encodings GStreamer decodes cheaply and that VA-API takes, ahead of the others.
fn usual(p: &Profile) -> bool {
    matches!(p.encoding.to_ascii_uppercase().as_str(), "H264" | "H265" | "HEVC")
}

fn pixels(p: &Profile) -> u64 {
    p.width as u64 * p.height as u64
}

/// The profile `want` asks for among `profiles`, or None if there is none.
///
/// For [`Want::Gray`] and [`Want::Colour`] a profile whose name or token says so comes
/// first. Otherwise H.264 and H.265 profiles come before JPEG and MPEG-4 ones, then size:
/// for gray, the smallest at least `min_width` wide, or the biggest if none is that wide;
/// for colour, the biggest. Equal profiles keep the camera's order.
pub fn pick<'a>(profiles: &'a [Profile], want: &Want, min_width: Option<u32>) -> Option<&'a Profile> {
    match want {
        Want::Profile(p) => profiles.iter().find(|x| x.token == *p)
            .or_else(|| profiles.iter().find(|x| x.name.eq_ignore_ascii_case(p))),
        Want::Gray => {
            let named: Vec<&Profile> = profiles.iter().filter(|p| is_named(p, &GRAY)).collect();
            let pool = if named.is_empty() { profiles.iter().collect() } else { named };
            let pool = preferred(pool);
            let wide = |p: &&Profile| min_width.is_none_or(|w| p.width >= w);
            pool.iter().copied().filter(wide).min_by_key(|p| pixels(p))
                .or_else(|| pool.iter().copied().rev().max_by_key(|p| pixels(p)))
        }
        Want::Colour => {
            let named: Vec<&Profile> = profiles.iter().filter(|p| is_named(p, &COLOUR) && !is_named(p, &GRAY)).collect();
            let pool = if named.is_empty() { profiles.iter().filter(|p| !is_named(p, &GRAY)).collect() } else { named };
            // max_by_key keeps the last of equals: reverse, so the camera's first wins.
            preferred(pool).into_iter().rev().max_by_key(|p| pixels(p))
        }
    }
}

/// The H.264/H.265 profiles of `pool` if it has any, else all of them.
fn preferred(pool: Vec<&Profile>) -> Vec<&Profile> {
    if pool.iter().any(|p| usual(p)) { pool.into_iter().filter(|p| usual(p)).collect() } else { pool }
}

/// Run `f` to completion on a runtime of its own, for callers outside async code (the
/// camera supervisors are plain threads).
pub fn block_on<T>(f: impl std::future::Future<Output = T>) -> Result<T> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(Error::Io)?;
    Ok(rt.block_on(f))
}

/// The devices that answer WS-Discovery within `wait`: by multicast on the local network,
/// or by unicast to every host of `subnet` (network, mask), for networks that drop multicast.
pub async fn discover(wait: Duration, subnet: Option<(Ipv4Addr, Ipv4Addr)>) -> Result<Vec<Found>> {
    use futures_util::StreamExt;
    let mut b = DiscoveryBuilder::default();
    b.duration(wait);
    if let Some((network, network_mask)) = subnet {
        b.discovery_mode(DiscoveryMode::Unicast { network, network_mask });
    }
    let devices = b.run().await.map_err(|e| Error::Invalid(format!("ONVIF discovery: {e}")))?;
    let mut found: Vec<Found> = devices
        .map(|d| Found { name: d.name, hardware: d.hardware, urls: d.urls.iter().map(|u| u.to_string()).collect() })
        .collect().await;
    found.sort_by(|a, b| a.urls.cmp(&b.urls));
    found.dedup_by(|a, b| a.urls == b.urls);
    Ok(found)
}

/// A client for one of the camera's services.
struct Session {
    device: Url,
    credentials: Option<Credentials>,
    options: Options,
    time_gap: Option<chrono::Duration>,
}

impl Session {
    fn client(&self, uri: &Url) -> Result<Client> {
        let mut b = ClientBuilder::new(uri)
            .credentials(self.credentials.clone())
            .auth_type(AuthType::Any)
            .timeout(self.options.timeout)
            .fix_time_gap(self.time_gap)
            .allow_username_token_over_http(self.options.allow_username_token_over_http)
            .max_redirections(2);
        if let Some(f) = &self.options.ca_file {
            let pem = std::fs::read(f)?;
            for c in reqwest::Certificate::from_pem_bundle(&pem)
                .map_err(|e| Error::Invalid(format!("{}: {e}", f.display())))? {
                b = b.add_root_certificate(c);
            }
        }
        b = b.danger_accept_invalid_hostnames(self.options.accept_invalid_hostnames);
        b.try_build().map_err(|e| Error::Invalid(format!("ONVIF client: {e}")))
    }

    /// A service address the camera reported, made safe to send the credentials to.
    fn service_url(&self, x_addr: &str) -> Result<Url> {
        let u = clean_uri(x_addr, &self.device, self.options.use_device_host)?;
        if u.host() != self.device.host() {
            return Err(Error::Invalid(format!(
                "the camera named a service on another host ({}); set use_device_host = true to use its own",
                u.host_str().unwrap_or(""))));
        }
        Ok(u)
    }
}

fn soap(what: &str) -> impl Fn(schema::transport::Error) -> Error + '_ {
    move |e| Error::Invalid(format!("ONVIF {what}: {e}"))
}

/// The video profiles of the camera at `device`, with their RTSP addresses: from Media2
/// (Profile T) when the camera has it, else from Media1.
pub async fn profiles(device: &Url, user: Option<&str>, password: Option<&str>, options: &Options) -> Result<Vec<Profile>> {
    let credentials = match (user, password) {
        (Some(u), Some(p)) => Some(Credentials { username: u.to_string(), password: p.to_string() }),
        (Some(u), None) => Some(Credentials { username: u.to_string(), password: String::new() }),
        _ => None,
    };
    let session = Session {
        device: device.clone(), credentials, options: options.clone(), time_gap: None,
    };
    profiles_with(session).await
}

async fn profiles_with(mut session: Session) -> Result<Vec<Profile>> {
    let device = session.client(&session.device.clone())?;

    // UsernameToken logins carry a timestamp the camera checks against its own clock, and
    // the box and a camera often disagree. Asking the time needs no login.
    if let Ok(t) = schema::devicemgmt::get_system_date_and_time(&device, &Default::default()).await {
        session.time_gap = time_gap(&t.system_date_and_time, chrono::Utc::now());
    }
    let device = session.client(&session.device.clone())?;

    let (mut media1, mut media2) = (None, None);
    match schema::devicemgmt::get_services(&device, &Default::default()).await {
        Ok(services) => {
            for s in &services.service {
                match s.namespace.as_str() {
                    MEDIA2 => media2 = Some(session.service_url(&s.x_addr)?),
                    MEDIA1 => media1 = Some(session.service_url(&s.x_addr)?),
                    _ => {}
                }
            }
        }
        // GetServices is newer than GetCapabilities; old cameras have only the latter.
        Err(_) => {
            let c = schema::devicemgmt::get_capabilities(&device, &Default::default()).await.map_err(soap("GetCapabilities"))?;
            if let Some(m) = c.capabilities.media.first() {
                media1 = Some(session.service_url(&m.x_addr)?);
            }
        }
    }
    let found = match (media2, media1) {
        (Some(m2), _) => media2_profiles(&session, &session.client(&m2)?).await?,
        (None, Some(m1)) => media1_profiles(&session, &session.client(&m1)?).await?,
        (None, None) => return Err(Error::Invalid("the camera has no ONVIF media service".into())),
    };
    if found.is_empty() {
        return Err(Error::Invalid("the camera has no video profiles".into()));
    }
    Ok(found)
}

/// How far the camera's clock is ahead of `now`, if by more than a few seconds.
fn time_gap(t: &schema::onvif::SystemDateTime, now: chrono::DateTime<chrono::Utc>) -> Option<chrono::Duration> {
    let utc = t.utc_date_time.as_ref()?;
    let camera = chrono::NaiveDate::from_ymd_opt(utc.date.year, utc.date.month as u32, utc.date.day as u32)?
        .and_hms_opt(utc.time.hour as u32, utc.time.minute as u32, utc.time.second as u32)?
        .and_utc();
    let gap = camera - now;
    (gap.num_seconds().abs() > 5).then_some(gap)
}

async fn media2_profiles(session: &Session, media: &Client) -> Result<Vec<Profile>> {
    let request = schema::media2::GetProfiles { token: None, _type: vec!["VideoEncoder".into()] };
    let r = schema::media2::get_profiles(media, &request).await.map_err(soap("Media2 GetProfiles"))?;
    let mut out = Vec::new();
    for p in &r.profiles {
        let Some(v) = p.configurations.as_ref().and_then(|c| c.video_encoder.as_ref()) else { continue };
        let request = schema::media2::GetStreamUri { protocol: "RTSP".into(), profile_token: schema::common::ReferenceToken(p.token.0.clone()) };
        let uri = schema::media2::get_stream_uri(media, &request).await.map_err(soap("Media2 GetStreamUri"))?.uri;
        out.push(Profile {
            token: p.token.0.clone(),
            name: p.name.0.clone(),
            encoding: v.encoding.clone(),
            width: v.resolution.width.max(0) as u32,
            height: v.resolution.height.max(0) as u32,
            fps: v.rate_control.as_ref().map(|r| r.frame_rate_limit),
            uri: clean_uri(&uri, &session.device, session.options.use_device_host)?.to_string(),
        });
    }
    Ok(out)
}

async fn media1_profiles(session: &Session, media: &Client) -> Result<Vec<Profile>> {
    use schema::onvif as tt;
    let r = schema::media::get_profiles(media, &Default::default()).await.map_err(soap("GetProfiles"))?;
    let mut out = Vec::new();
    for p in &r.profiles {
        let Some(v) = &p.video_encoder_configuration else { continue };
        let request = schema::media::GetStreamUri {
            profile_token: tt::ReferenceToken(p.token.0.clone()),
            stream_setup: tt::StreamSetup {
                stream: tt::StreamType::RtpUnicast,
                transport: tt::Transport { protocol: tt::TransportProtocol::Rtsp, tunnel: vec![] },
            },
        };
        let uri = schema::media::get_stream_uri(media, &request).await.map_err(soap("GetStreamUri"))?.media_uri.uri;
        let encoding = match &v.encoding {
            tt::VideoEncoding::Jpeg => "JPEG".to_string(),
            tt::VideoEncoding::Mpeg4 => "MPEG4".to_string(),
            tt::VideoEncoding::H264 => "H264".to_string(),
            tt::VideoEncoding::__Unknown__(s) => s.clone(),
        };
        out.push(Profile {
            token: p.token.0.clone(),
            name: p.name.0.clone(),
            encoding,
            width: v.resolution.width.max(0) as u32,
            height: v.resolution.height.max(0) as u32,
            fps: v.rate_control.as_ref().map(|r| r.frame_rate_limit as f64),
            uri: clean_uri(&uri, &session.device, session.options.use_device_host)?.to_string(),
        });
    }
    Ok(out)
}

/// The RTSP address to open for a camera configured with `onvif` and `stream`.
pub fn resolve(device: &str, want: &Want, user: Option<&str>, password: Option<&str>, options: &Options) -> Result<String> {
    let url = device_url(device)?;
    let found = block_on(profiles(&url, user, password, options))??;
    let p = pick(&found, want, options.min_width).ok_or_else(|| {
        let names: Vec<String> = found.iter().map(|p| format!("{} ({})", p.token, p.name)).collect();
        Error::Invalid(format!("the camera has no profile {want:?}; it has {}", names.join(", ")))
    })?;
    Ok(p.uri.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(token: &str, name: &str, encoding: &str, width: u32, height: u32) -> Profile {
        Profile { token: token.into(), name: name.into(), encoding: encoding.into(), width, height, fps: None,
                  uri: format!("rtsp://cam/{token}") }
    }

    fn hik() -> Vec<Profile> {
        // A usual camera: main stream, sub-stream, third stream.
        vec![p("Profile_101", "mainStream", "H265", 2560, 1440),
             p("Profile_102", "subStream", "H264", 640, 360),
             p("Profile_103", "thirdStream", "H264", 1280, 720)]
    }

    #[test]
    fn gray_is_the_smallest_wide_enough_stream() {
        assert_eq!(pick(&hik(), &Want::Gray, None).unwrap().token, "Profile_102");
        assert_eq!(pick(&hik(), &Want::Gray, Some(640)).unwrap().token, "Profile_102");
        assert_eq!(pick(&hik(), &Want::Gray, Some(1000)).unwrap().token, "Profile_103");
        // Nothing is that wide: the widest there is.
        assert_eq!(pick(&hik(), &Want::Gray, Some(4000)).unwrap().token, "Profile_101");
    }

    #[test]
    fn colour_is_the_main_stream() {
        assert_eq!(pick(&hik(), &Want::Colour, None).unwrap().token, "Profile_101");
    }

    #[test]
    fn named_gray_and_colour_profiles_win() {
        let mut v = hik();
        v.push(p("gray_1", "Gray 720p", "H264", 1280, 720));
        v.push(p("col", "Colour", "H264", 1920, 1080));
        assert_eq!(pick(&v, &Want::Gray, Some(640)).unwrap().token, "gray_1");
        assert_eq!(pick(&v, &Want::Colour, None).unwrap().token, "col");
        // A gray stream is never the colour pick, even when it is the biggest.
        let v = vec![p("mono", "Mono", "H264", 3840, 2160), p("a", "A", "H264", 1920, 1080)];
        assert_eq!(pick(&v, &Want::Colour, None).unwrap().token, "a");
    }

    #[test]
    fn h264_and_h265_come_before_jpeg() {
        let v = vec![p("mj", "MJPEG", "JPEG", 320, 240), p("s", "sub", "H264", 640, 480)];
        assert_eq!(pick(&v, &Want::Gray, None).unwrap().token, "s");
        let v = vec![p("mj", "MJPEG", "JPEG", 320, 240)];
        assert_eq!(pick(&v, &Want::Gray, None).unwrap().token, "mj");
    }

    #[test]
    fn equal_profiles_keep_the_camera_order() {
        let v = vec![p("a", "A", "H264", 640, 360), p("b", "B", "H264", 640, 360)];
        assert_eq!(pick(&v, &Want::Gray, None).unwrap().token, "a");
        assert_eq!(pick(&v, &Want::Colour, None).unwrap().token, "a");
    }

    #[test]
    fn a_profile_by_token_or_name() {
        assert_eq!(pick(&hik(), &Want::Profile("Profile_103".into()), None).unwrap().token, "Profile_103");
        assert_eq!(pick(&hik(), &Want::Profile("SUBSTREAM".into()), None).unwrap().token, "Profile_102");
        assert!(pick(&hik(), &Want::Profile("nope".into()), None).is_none());
        assert!(pick(&[], &Want::Gray, None).is_none());
        assert_eq!(Want::parse("Grey"), Want::Gray);
        assert_eq!(Want::parse("color"), Want::Colour);
        assert_eq!(Want::parse("Profile_101"), Want::Profile("Profile_101".into()));
    }

    #[test]
    fn device_urls() {
        assert_eq!(device_url("10.0.0.21").unwrap().as_str(), "http://10.0.0.21/onvif/device_service");
        assert_eq!(device_url("10.0.0.21:8080").unwrap().as_str(), "http://10.0.0.21:8080/onvif/device_service");
        assert_eq!(device_url("https://cam.site/onvif/device_service").unwrap().as_str(), "https://cam.site/onvif/device_service");
        assert!(device_url("http://admin:pw@10.0.0.21/").unwrap_err().to_string().contains("not in the URL"));
        assert!(device_url("rtsp://10.0.0.21/").is_err());
    }

    #[test]
    fn reported_uris_lose_credentials_and_get_the_device_host() {
        let device = device_url("10.20.0.5").unwrap();
        let u = clean_uri("rtsp://admin:pw@192.168.1.64:554/Streaming/Channels/102?transportmode=unicast", &device, true).unwrap();
        assert_eq!(u.as_str(), "rtsp://10.20.0.5:554/Streaming/Channels/102?transportmode=unicast");
        let u = clean_uri("rtsp://192.168.1.64/ch1", &device, false).unwrap();
        assert_eq!(u.as_str(), "rtsp://192.168.1.64/ch1");
    }

    #[test]
    fn clock_gaps_under_five_seconds_are_ignored() {
        use schema::onvif::{Date, DateTime, SystemDateTime, Time};
        let at = |h, m, s| SystemDateTime {
            utc_date_time: Some(DateTime { date: Date { year: 2026, month: 11, day: 20 }, time: Time { hour: h, minute: m, second: s } }),
            ..Default::default()
        };
        let now = chrono::NaiveDate::from_ymd_opt(2026, 11, 20).unwrap().and_hms_opt(10, 0, 0).unwrap().and_utc();
        assert_eq!(time_gap(&at(10, 0, 3), now), None);
        assert_eq!(time_gap(&at(10, 2, 0), now), Some(chrono::Duration::seconds(120)));
        assert_eq!(time_gap(&at(9, 59, 0), now), Some(chrono::Duration::seconds(-60)));
        assert_eq!(time_gap(&SystemDateTime::default(), now), None);
    }
}
