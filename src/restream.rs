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
//! A viewer that joins mid-picture-group sees the stream from the next keyframe on, as
//! players do with a broadcast.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

/// MPEG-TS chunks, shared.
pub type Chunk = Arc<[u8]>;

/// A stream's chunks, fanned out to whoever is subscribed.
pub struct Tap {
    subscribers: Mutex<Vec<SyncSender<Chunk>>>,
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
        Tap { subscribers: Mutex::new(Vec::new()), queue, most, bytes: AtomicU64::new(0), dropped: AtomicU64::new(0) }
    }

    /// A new viewer; None when there are `most` already.
    pub fn subscribe(&self) -> Option<Receiver<Chunk>> {
        let mut subs = self.subscribers.lock().unwrap();
        if subs.len() >= self.most {
            return None;
        }
        let (tx, rx) = sync_channel(self.queue);
        subs.push(tx);
        Some(rx)
    }

    pub fn viewers(&self) -> usize {
        self.subscribers.lock().unwrap().len()
    }

    /// Hand a chunk to every viewer. One whose queue is full, or who left, is dropped.
    pub fn publish(&self, chunk: &[u8]) {
        let mut subs = self.subscribers.lock().unwrap();
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
