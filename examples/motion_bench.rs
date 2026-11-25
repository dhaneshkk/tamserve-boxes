//! The motion gate's cost per frame: its shrink the plain way, the same two-pass shrink
//! compiled for the x86-64 baseline only (SSE2, what a plain build gets), and with the
//! CPU's best SIMD picked at run time; then the whole gate per frame.
//!
//!   cargo run --release --example motion_bench [WIDTHxHEIGHT ...]
use std::hint::black_box;
use std::time::Instant;

use fearless_simd::Level;
use tamserve_boxes::motion::{shrink, shrink_reference, MotionDetector, MotionOptions};
use tamserve_boxes::GrayFrame;

fn per_frame(n: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..n / 10 { f(i) } // warm up
    let t = Instant::now();
    for i in 0..n { f(i) }
    t.elapsed().as_secs_f64() / n as f64 * 1e3
}

fn main() {
    let sizes: Vec<(usize, usize)> = std::env::args().skip(1)
        .filter_map(|a| a.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?))))
        .collect();
    let sizes = if sizes.is_empty() { vec![(704, 576), (1280, 720), (1920, 1080)] } else { sizes };
    println!("best SIMD level here: {:?}", Level::new());
    println!("{:>10} {:>10} {:>10} {:>10} {:>8} {:>12}", "frame", "plain ms", "sse2 ms", "best ms", "speedup", "gate ms");
    for (w, h) in sizes {
        let mut seed = 1u32;
        let px: Vec<u8> = (0..w * h).map(|_| { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 24) as u8 }).collect();
        let f = GrayFrame::new(&px, w, h).unwrap();
        let sh = 100.min(h);
        let sw = ((w * sh) as f64 / h as f64).round() as usize;
        let n = 400;
        let (mut out, mut sums) = (Vec::new(), Vec::new());
        let plain = per_frame(n, |_| { shrink_reference(black_box(&f), sw, sh, &mut out); black_box(&out); });
        let base = per_frame(n, |_| { shrink(Level::baseline(), black_box(&f), sw, sh, &mut out, &mut sums); black_box(&out); });
        let level = Level::new();
        let best = per_frame(n, |_| { shrink(level, black_box(&f), sw, sh, &mut out, &mut sums); black_box(&out); });
        let mut d = MotionDetector::new(MotionOptions::default());
        let gate = per_frame(n, |_| { black_box(d.next(black_box(&f))); });
        println!("{:>10} {plain:>10.3} {base:>10.3} {best:>10.3} {:>7.1}x {gate:>12.3}", format!("{w}x{h}"), plain / best);
    }
}
