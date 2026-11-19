//! Lists a GStreamer element's properties: cargo run --example gst_props -- svtav1enc
use gstreamer::prelude::*;

fn main() {
    gstreamer::init().unwrap();
    for name in std::env::args().skip(1) {
        let Ok(e) = gstreamer::ElementFactory::make(&name).build() else { println!("{name}: missing"); continue };
        println!("== {name}");
        for p in e.list_properties() {
            let v = e.property_value(p.name());
            println!("  {:26} {:12} default {:?}  -- {}", p.name(), p.value_type().name(), v, p.blurb().unwrap_or(""));
        }
    }
}
