//! One connection to a camera, many users of it: the detector, the recorder and any
//! number of live viewers all take the stream this box already reads, so the camera
//! serves one client however many watch. go2rtc's idea, inside this process.
//!
//! With `live = true` a camera's stream is muxed to MPEG-TS as it arrives, without
//! decoding or encoding it, and published on a [`Tap`]. `GET /live/<camera>.ts` on the
//! HTTP endpoint subscribes, so VLC, ffplay, mpv or mpegts.js play it, behind the same
//! bearer token as the rest. A viewer that falls behind by more than the tap's queue is
//! dropped rather than holding up the others; it can connect again.
//!
//! A viewer starts at once: the tap keeps the stream's latest PAT and PMT and every
//! packet since the latest keyframe (the packet `mpegtsmux` marks random-access), and a
//! new viewer is sent those first. Its player has a decodable picture straight away,
//! rather than waiting up to a picture group for the next keyframe.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

/// MPEG-TS chunks, shared.
pub type Chunk = Arc<[u8]>;

const PACKET: usize = 188;
/// The most the keyframe cache holds; a longer picture group is not cached.
const MOST_CACHED: usize = 8 << 20;

/// The start a new viewer needs: the tables, and the packets since the latest keyframe.
#[derive(Default)]
struct Start {
    pat: Option<Vec<u8>>,
    pmt_pid: Option<u16>,
    pmt: Option<Vec<u8>>,
    /// From the latest keyframe on; None while there is none, or it outgrew the cache.
    since_key: Option<Vec<u8>>,
}

fn pid(p: &[u8]) -> u16 {
    (((p[1] & 0x1f) as u16) << 8) | p[2] as u16
}

/// Where a packet's payload starts, if it has one.
fn payload(p: &[u8]) -> Option<usize> {
    match (p[3] >> 4) & 3 {
        1 => Some(4),
        3 => Some(5 + p[4] as usize).filter(|&o| o < PACKET),
        _ => None,
    }
}

/// The packet starts a keyframe: its adaptation field says random access.
fn random_access(p: &[u8]) -> bool {
    matches!((p[3] >> 4) & 3, 2 | 3) && p[4] > 0 && p[5] & 0x40 != 0
}

/// The PMT's PID, from a PAT packet: its first program that is not the network's.
fn pmt_pid(p: &[u8]) -> Option<u16> {
    if p[1] & 0x40 == 0 {
        return None; // not a section's start
    }
    let at = payload(p)?;
    let s = at + 1 + *p.get(at)? as usize; // past the pointer field
    let len = ((*p.get(s + 1)? as usize & 0x0f) << 8) | *p.get(s + 2)? as usize;
    let (mut i, end) = (s + 8, (s + 3 + len).saturating_sub(4).min(PACKET)); // the programs, before the CRC
    while i + 4 <= end {
        let (program, pid) = (((p[i] as u16) << 8) | p[i + 1] as u16, (((p[i + 2] & 0x1f) as u16) << 8) | p[i + 3] as u16);
        if program != 0 {
            return Some(pid);
        }
        i += 4;
    }
    None
}

impl Start {
    fn see(&mut self, chunk: &[u8]) {
        for p in chunk.as_chunks::<PACKET>().0 {
            if p[0] != 0x47 {
                continue;
            }
            match pid(p) {
                0 => {
                    if let Some(m) = pmt_pid(p) {
                        self.pmt_pid = Some(m);
                    }
                    self.pat = Some(p.to_vec());
                    continue;
                }
                id if Some(id) == self.pmt_pid => {
                    self.pmt = Some(p.to_vec());
                    continue;
                }
                _ => {}
            }
            if random_access(p) {
                self.since_key = Some(Vec::with_capacity(1 << 16));
            }
            if let Some(c) = &mut self.since_key {
                c.extend_from_slice(p);
                if c.len() > MOST_CACHED {
                    self.since_key = None;
                }
            }
        }
    }

    /// What a new viewer is sent first, if a keyframe has been seen.
    fn start(&self) -> Option<Vec<u8>> {
        let (pat, pmt, key) = (self.pat.as_ref()?, self.pmt.as_ref()?, self.since_key.as_ref()?);
        Some([pat.as_slice(), pmt, key].concat())
    }
}

/// A stream's chunks, fanned out to whoever is subscribed.
pub struct Tap {
    subscribers: Mutex<Vec<SyncSender<Chunk>>>,
    start: Mutex<Start>,
    /// Chunks a subscriber may be behind before it is dropped.
    pub queue: usize,
    /// The most subscribers at once.
    pub most: usize,
    pub bytes: AtomicU64,
    /// Viewers dropped for falling behind.
    pub dropped: AtomicU64,
}

impl Tap {
    pub fn new(queue: usize, most: usize) -> Tap {
        Tap { subscribers: Mutex::new(Vec::new()), start: Mutex::new(Start::default()), queue, most,
              bytes: AtomicU64::new(0), dropped: AtomicU64::new(0) }
    }

    /// A new viewer; None when there are `most` already. Its first chunk is the stream
    /// from the latest keyframe, with the tables before it, once there has been one.
    pub fn subscribe(&self) -> Option<Receiver<Chunk>> {
        let mut subs = self.subscribers.lock().unwrap();
        if subs.len() >= self.most {
            return None;
        }
        let (tx, rx) = sync_channel(self.queue);
        // Under the subscribers' lock, so no chunk is sent before this one or skipped.
        if let Some(first) = self.start.lock().unwrap().start() {
            let _ = tx.try_send(Arc::from(first));
        }
        subs.push(tx);
        Some(rx)
    }

    pub fn viewers(&self) -> usize {
        self.subscribers.lock().unwrap().len()
    }

    /// Hand a chunk (whole 188-byte packets) to every viewer. One whose queue is full, or
    /// who left, is dropped.
    pub fn publish(&self, chunk: &[u8]) {
        let mut subs = self.subscribers.lock().unwrap();
        self.start.lock().unwrap().see(chunk);
        if subs.is_empty() {
            return;
        }
        self.bytes.fetch_add(chunk.len() as u64, Ordering::Relaxed);
        let c: Chunk = Arc::from(chunk);
        subs.retain(|s| match s.try_send(c.clone()) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(TrySendError::Disconnected(_)) => false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_viewer_gets_every_chunk_in_order() {
        let tap = Tap::new(8, 4);
        let (a, b) = (tap.subscribe().unwrap(), tap.subscribe().unwrap());
        for i in 0..5u8 {
            tap.publish(&[i; 3]);
        }
        for rx in [a, b] {
            let got: Vec<u8> = rx.try_iter().map(|c| c[0]).collect();
            assert_eq!(got, [0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn a_slow_viewer_is_dropped_and_the_others_go_on() {
        let tap = Tap::new(2, 4);
        let slow = tap.subscribe().unwrap();
        let fast = tap.subscribe().unwrap();
        for i in 0..4u8 {
            tap.publish(&[i]);
            let _ = fast.try_recv();
        }
        assert_eq!(tap.viewers(), 1);
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 1);
        assert_eq!(slow.try_iter().count(), 2, "what it had queued before it was dropped");
    }

    /// A packet: `pid`, payload start, random access or not.
    fn packet(pid: u16, start: bool, key: bool, fill: u8) -> Vec<u8> {
        let mut p = vec![fill; PACKET];
        p[0] = 0x47;
        p[1] = (u8::from(start) << 6) | (pid >> 8) as u8;
        p[2] = pid as u8;
        if key {
            p[3] = 0x30; // adaptation field and payload
            p[4] = 1;
            p[5] = 0x40;
        } else {
            p[3] = 0x10;
        }
        p
    }

    fn pat(pmt: u16) -> Vec<u8> {
        let mut p = packet(0, true, false, 0xff);
        // pointer 0; table 0, length 13; ts id 1; version; section 0/0; program 1 -> pmt; CRC
        let section = [0x00, 0x00, 0xb0, 0x0d, 0x00, 0x01, 0xc1, 0x00, 0x00, 0x00, 0x01, 0xe0 | (pmt >> 8) as u8, pmt as u8, 0, 0, 0, 0];
        p[4..4 + section.len()].copy_from_slice(&section);
        p
    }

    #[test]
    fn a_late_viewer_starts_at_the_latest_keyframe_with_the_tables() {
        let tap = Tap::new(64, 4);
        let (pat, pmt) = (pat(32), packet(32, true, false, 7));
        let stream = [pat.clone(), pmt.clone(), packet(65, true, true, 1), packet(65, false, false, 2),
                      pat.clone(), packet(65, true, true, 3), packet(65, false, false, 4)].concat();
        tap.publish(&stream[..3 * PACKET]);
        tap.publish(&stream[3 * PACKET..]);
        let late = tap.subscribe().unwrap();
        let first = late.try_recv().unwrap();
        let want = [pat, pmt, packet(65, true, true, 3), packet(65, false, false, 4)].concat();
        assert_eq!(&first[..], &want[..], "the tables, then the second keyframe and what followed it");
        tap.publish(&packet(65, false, false, 5));
        assert_eq!(late.try_recv().unwrap()[10], 5);
    }

    #[test]
    fn no_keyframe_yet_means_no_start() {
        let tap = Tap::new(8, 4);
        tap.publish(&[pat(32), packet(32, true, false, 7), packet(65, false, false, 2)].concat());
        assert!(tap.subscribe().unwrap().try_recv().is_err());
    }

    #[test]
    fn a_viewer_who_leaves_is_forgotten_and_the_limit_holds() {
        let tap = Tap::new(2, 1);
        let a = tap.subscribe().unwrap();
        assert!(tap.subscribe().is_none());
        drop(a);
        tap.publish(&[1]);
        assert_eq!(tap.viewers(), 0);
        assert!(tap.subscribe().is_some());
    }
}
