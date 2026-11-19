//! Lists the GStreamer elements tamserve-boxes uses, and whether each is installed.
use gstreamer::prelude::*;

fn main() {
    gstreamer::init().unwrap();
    println!("GStreamer {}", gstreamer::version_string());
    for e in ["uridecodebin", "rtspsrc", "videoconvert", "videoscale", "appsink", "avdec_h264", "avdec_h265",
              "openh264dec", "vah264dec", "vah265dec", "vapostproc", "h264parse", "x264enc"] {
        let f = gstreamer::ElementFactory::find(e);
        println!("{e:14} {}", f.map(|f| format!("rank {:?}", f.rank())).unwrap_or("missing".into()));
    }
}
