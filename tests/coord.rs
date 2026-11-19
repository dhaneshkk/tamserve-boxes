//! The coordinator over loopback, with boxes played by the test.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use tamserve_boxes::coord::{Channel, CoordCamera, CoordConfig, Coordinator, FromBox, FromCoordinator, PROTOCOL};

const TOKEN: &[u8] = b"a test token that is long enough for the rules";

fn coordinator(cams: &[(&str, f64)], dead: f64) -> (std::sync::Arc<Coordinator>, String) {
    let cfg = CoordConfig {
        listen: "127.0.0.1:0".into(), token_file: "unused".into(), dead_seconds: dead, http: None,
        http_token_file: None, rebalance_gap: 0.25,
        cameras: cams.iter().map(|(n, w)| CoordCamera { name: n.to_string(), weight: *w, boxes: None }).collect(),
    };
    let c = Coordinator::new(cfg, Box::new(std::io::sink())).unwrap();
    let at = c.serve(TOKEN.to_vec()).unwrap();
    (c, at.to_string())
}

/// A box: connects, says hello, and returns its channel.
fn join(addr: &str, name: &str, capacity: f64, known: &[&str]) -> Channel {
    let mut ch = Channel::connect(addr, TOKEN, name).unwrap();
    ch.send(&FromBox::Hello { protocol: PROTOCOL.into(), capacity, known: known.iter().map(|s| s.to_string()).collect() }).unwrap();
    ch.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    ch
}

/// The next assignment the box is sent (pings skipped).
fn assigned(ch: &mut Channel) -> Vec<String> {
    loop {
        match ch.recv::<FromCoordinator>().unwrap() {
            FromCoordinator::Assign { cameras } => return cameras,
            FromCoordinator::Ping => {}
            FromCoordinator::Refused { reason } => panic!("refused: {reason}"),
        }
    }
}

fn wait_for(c: &Coordinator, want: impl Fn(&std::collections::BTreeMap<String, String>) -> bool) -> std::collections::BTreeMap<String, String> {
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        let a = c.assignment();
        if want(&a) || Instant::now() > until {
            return a;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_wrong_token_is_turned_away_both_ways() {
    let (_c, addr) = coordinator(&[("a", 1.0)], 5.0);
    assert!(Channel::connect(&addr, b"another token that is long enough too!!", "x").is_err());
}

#[test]
fn a_peer_that_stalls_mid_handshake_is_cut_off() {
    let (_c, addr) = coordinator(&[("a", 1.0)], 5.0);
    let mut s = TcpStream::connect(&addr).unwrap();
    let mut hello = [0u8; 36];
    s.read_exact(&mut hello).unwrap();
    s.write_all(&[0, 5, b'h', b'a']).unwrap(); // announces a 5-byte name, sends 2, then stalls
    s.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
    let t = Instant::now();
    let mut buf = [0u8; 1];
    let n = s.read(&mut buf).unwrap_or(0); // the coordinator closes it
    assert_eq!(n, 0);
    assert!(t.elapsed() < Duration::from_secs(12), "cut off after {:?}", t.elapsed());
}

#[test]
fn a_changed_message_is_refused() {
    // A tiny man in the middle: passes the handshake through, then alters the box's first message.
    let real = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = real.local_addr().unwrap().to_string();
    let t = std::thread::spawn(move || {
        let (s, _) = real.accept().unwrap();
        let (mut ch, _) = Channel::accept(s, TOKEN).unwrap();
        ch.recv::<FromBox>().map(|_| ()).map_err(|e| e.to_string())
    });
    let mitm = TcpListener::bind("127.0.0.1:0").unwrap();
    let mitm_addr = mitm.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (mut boxside, _) = mitm.accept().unwrap();
        let mut up = TcpStream::connect(&addr).unwrap();
        let mut hello = [0u8; 36];
        up.read_exact(&mut hello).unwrap();
        boxside.write_all(&hello).unwrap();
        let mut proof = vec![0u8; 2 + 3 + 64];
        boxside.read_exact(&mut proof).unwrap();
        up.write_all(&proof).unwrap();
        let mut back = [0u8; 32];
        up.read_exact(&mut back).unwrap();
        boxside.write_all(&back).unwrap();
        let mut len = [0u8; 4];
        boxside.read_exact(&mut len).unwrap();
        let mut body = vec![0u8; u32::from_be_bytes(len) as usize + 32];
        boxside.read_exact(&mut body).unwrap();
        let i = body.iter().position(|&b| b == b'9').unwrap();
        body[i] = b'1'; // capacity 9 -> 1
        up.write_all(&len).unwrap();
        up.write_all(&body).unwrap();
    });
    let mut ch = Channel::connect(&mitm_addr, TOKEN, "box").unwrap();
    ch.send(&FromBox::Hello { protocol: PROTOCOL.into(), capacity: 9.0, known: vec![] }).unwrap();
    let got = t.join().unwrap();
    assert!(got.unwrap_err().contains("failed its check"));
}

#[test]
fn cameras_are_spread_by_capacity_and_move_off_a_box_that_goes_silent() {
    let cams = [("c1", 1.0), ("c2", 1.0), ("c3", 1.0), ("c4", 1.0), ("c5", 1.0), ("c6", 1.0)];
    let (c, addr) = coordinator(&cams, 1.5);
    let all = ["c1", "c2", "c3", "c4", "c5", "c6"];
    let mut a = join(&addr, "a", 4.0, &all);
    let first = assigned(&mut a);
    assert_eq!(first.len(), 6, "a lone box takes everything it knows, over capacity rather than leave cameras unserved");
    let mut b = join(&addr, "b", 2.0, &all);
    // b joins: cameras move one at a time until the two are about as full (4:2 for capacities 4 and 2).
    let plan = wait_for(&c, |m| m.values().filter(|b| *b == "b").count() == 2);
    assert_eq!(plan.values().filter(|x| *x == "a").count(), 4, "{plan:?}");
    // Keep b alive with heartbeats; let a go silent.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s2 = stop.clone();
    let keep = std::thread::spawn(move || {
        while !s2.load(std::sync::atomic::Ordering::Relaxed) {
            b.send(&FromBox::Heartbeat { cameras: vec![], busy: 0.1 }).unwrap();
            std::thread::sleep(Duration::from_millis(300));
        }
    });
    let t = Instant::now();
    let plan = wait_for(&c, |m| m.len() == 6 && m.values().all(|x| x == "b"));
    assert!(plan.values().all(|x| x == "b"), "{plan:?}");
    assert!(t.elapsed() < Duration::from_secs(4), "moved after {:?}", t.elapsed());
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    keep.join().unwrap();
    drop(a);
}

#[test]
fn a_box_only_gets_cameras_it_has_a_config_for() {
    let (c, addr) = coordinator(&[("dock", 1.0), ("gate", 1.0)], 5.0);
    let mut a = join(&addr, "a", 8.0, &["gate"]);
    assert_eq!(assigned(&mut a), vec!["gate".to_string()]);
    let plan = c.assignment();
    assert!(!plan.contains_key("dock"), "no box knows dock: it stays unassigned, and the metrics say so");
}

#[test]
fn a_box_that_reconnects_replaces_its_old_connection() {
    let (c, addr) = coordinator(&[("x", 1.0)], 5.0);
    let mut first = join(&addr, "a", 4.0, &["x"]);
    assert_eq!(assigned(&mut first), vec!["x".to_string()]);
    let mut again = join(&addr, "a", 4.0, &["x"]);
    assert_eq!(assigned(&mut again), vec!["x".to_string()]);
    // the old connection is told why it was dropped
    let mut refused = false;
    for _ in 0..5 {
        match first.recv::<FromCoordinator>() {
            Ok(FromCoordinator::Refused { .. }) | Err(_) => { refused = true; break; }
            _ => {}
        }
    }
    assert!(refused);
    assert_eq!(c.assignment().get("x").map(|s| s.as_str()), Some("a"));
}

#[test]
fn a_replayed_message_is_refused() {
    // The same genuine message, sent twice by someone on the path: the first passes, the second does not.
    let real = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = real.local_addr().unwrap().to_string();
    let t = std::thread::spawn(move || {
        let (s, _) = real.accept().unwrap();
        let (mut ch, _) = Channel::accept(s, TOKEN).unwrap();
        let first = ch.recv::<FromBox>().is_ok();
        let second = ch.recv::<FromBox>().map(|_| ()).map_err(|e| e.to_string());
        (first, second)
    });
    let mitm = TcpListener::bind("127.0.0.1:0").unwrap();
    let mitm_addr = mitm.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (mut boxside, _) = mitm.accept().unwrap();
        let mut up = TcpStream::connect(&addr).unwrap();
        let mut hello = [0u8; 36];
        up.read_exact(&mut hello).unwrap();
        boxside.write_all(&hello).unwrap();
        let mut proof = vec![0u8; 2 + 3 + 64];
        boxside.read_exact(&mut proof).unwrap();
        up.write_all(&proof).unwrap();
        let mut back = [0u8; 32];
        up.read_exact(&mut back).unwrap();
        boxside.write_all(&back).unwrap();
        let mut len = [0u8; 4];
        boxside.read_exact(&mut len).unwrap();
        let mut body = vec![0u8; u32::from_be_bytes(len) as usize + 32];
        boxside.read_exact(&mut body).unwrap();
        for _ in 0..2 {
            up.write_all(&len).unwrap();
            up.write_all(&body).unwrap();
        }
    });
    let mut ch = Channel::connect(&mitm_addr, TOKEN, "box").unwrap();
    ch.send(&FromBox::Hello { protocol: PROTOCOL.into(), capacity: 9.0, known: vec![] }).unwrap();
    let (first, second) = t.join().unwrap();
    assert!(first);
    assert!(second.unwrap_err().contains("failed its check"));
}
