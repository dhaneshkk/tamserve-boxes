//! Steady-state frames must not allocate.
mod common;
use common::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use vg360_scenegraph::*;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Ordering::Relaxed); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { ALLOCS.fetch_add(1, Ordering::Relaxed); System.realloc(p, l, n) }
}
#[global_allocator]
static A: Counting = Counting;

#[test]
fn steady_state_allocates_nothing() {
    let e = iogp_runtime::Engine::from_config(&fixture_config()).unwrap();
    let mut sg = builder("[]", Params::default(), &e);
    let p = projection();
    let mut dets: Vec<Det> = vec![];
    for i in 0..30u64 {
        let (x, y) = (-6.0 + (i % 6) as f64 * 2.0, 8.0 + (i / 6) as f64 * 1.5);
        dets.push(Det { track_id: i, class: if i % 5 == 0 { "vehicle" } else { "person" }, bbox: bbox(&p, x, y, 0.0, 0.5, 1.75), score: 0.9 });
    }
    dets.push(Det { track_id: 99, class: "suspended_load", bbox: bbox(&p, 0.0, 11.0, 5.0, 1.2, 0.8), score: 0.9 });
    let mut sink = NullSink::default();
    let step = |sg: &mut SceneBuilder, sink: &mut NullSink, dets: &mut Vec<Det>, k: usize| {
        for d in dets.iter_mut() { d.bbox[0] += 0.3; d.bbox[2] += 0.3; }
        sg.step(k as f64 * 0.1, dets, true, sink);
    };
    for k in 0..50 { step(&mut sg, &mut sink, &mut dets, k); }
    let before = ALLOCS.load(Ordering::Relaxed);
    for k in 50..550 { step(&mut sg, &mut sink, &mut dets, k); }
    let n = ALLOCS.load(Ordering::Relaxed) - before;
    assert!(sink.facts > 0);
    assert_eq!(n, 0, "{n} allocations in 500 steady-state frames");
}
