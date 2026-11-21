//! Day and night, and the switch between them.
//!
//! At dusk a camera moves its IR cut filter out of the light path and turns its IR
//! LEDs on. From then on its sensor sees only brightness: the colour stream carries a
//! gray picture, and anything that draws on "the colour frame" has nothing to draw on.
//! The switch itself is a jump in exposure that wakes the motion gate, makes the
//! detector see things that are not there for a second or two, and can lose what the
//! tracker follows.
//!
//! [`LightMeter`] follows each camera:
//! - **Mode.** When the camera's frames come with colour planes (NV12, I420 and the
//!   like: the Y plane the detector reads is one of them), how far the chroma is from
//!   neutral says which mode the camera is in. A camera in IR mode sends neutral
//!   chroma; a colour picture does not. The mode changes only after `confirm` frames in
//!   a row agree, and only past two thresholds with a gap between them, so a dim or
//!   dull colour scene does not flicker between the two. A stream with no colour planes
//!   (GRAY8, a mono profile) reports `Unknown`.
//! - **Switches.** A mode change, motion's `lighting` reset (most of the picture changed
//!   at once), or the mean brightness jumping by more than `jump` from one frame to the
//!   next starts a **settle**: for `settle_ms` the runner does not look at that camera,
//!   and reports what it followed before the switch, so a person in a hazard zone does
//!   not vanish and reappear as a new id.
//!
//! The mode decision is proved in `proofs/Light.lean`: values between the thresholds
//! never change it, it changes only after `confirm` agreeing frames in a row, and
//! `confirm` such frames always change it.

use crate::GrayFrame;
use serde::Serialize;

/// Which way the camera is seeing.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// No colour planes to tell by, or not yet decided.
    #[default]
    Unknown,
    Day,
    /// IR: the picture is gray whatever stream it comes on.
    Night,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Unknown => "unknown",
            Mode::Day => "day",
            Mode::Night => "night",
        }
    }
    /// For a gauge: 0 unknown, 1 day, 2 night.
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// How day, night and switches are found. `Default` suits common IR domes and bullets.
#[derive(Clone, Debug, PartialEq)]
pub struct LightOptions {
    /// Chroma (mean distance from neutral, 0 to 128) below which a frame looks like IR.
    pub night_below: f32,
    /// Chroma above which a frame looks like colour. Between the two: no evidence.
    pub day_above: f32,
    /// Frames in a row that must agree before the mode changes.
    pub confirm: u32,
    /// How long after a switch the camera is not looked at, in ms. 0: never.
    pub settle_ms: u64,
    /// A change in mean brightness (0 to 255) from one frame to the next that counts
    /// as a switch. 0: not used.
    pub jump: f32,
}

impl Default for LightOptions {
    fn default() -> Self {
        LightOptions { night_below: 2.0, day_above: 4.0, confirm: 5, settle_ms: 2000, jump: 40.0 }
    }
}

impl LightOptions {
    pub fn check(&self) -> crate::Result<()> {
        let bad = if !(self.night_below >= 0.0 && self.night_below < self.day_above && self.day_above <= 128.0) {
            "light.night_below must be at least 0 and below light.day_above, which is at most 128"
        } else if self.confirm == 0 {
            "light.confirm must be at least 1"
        } else if !(self.jump >= 0.0 && self.jump <= 255.0) {
            "light.jump must be from 0 to 255"
        } else {
            return Ok(());
        };
        Err(crate::Error::Invalid(bad.into()))
    }
}

/// What one frame said.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Light {
    pub mode: Mode,
    /// The camera is within `settle_ms` of a switch: not looked at.
    pub settling: bool,
    /// This frame started a settle.
    pub switched: bool,
}

/// One frame's chroma as evidence: night, day, or neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    Night,
    Day,
    Neither,
}

/// The mode decision alone, with no clock: what `proofs/Light.lean` is about.
/// `run` counts frames in a row that pointed to `pending`, a mode other than `mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Decision {
    pub mode: Mode,
    pub pending: Mode,
    pub run: u32,
}

impl Decision {
    /// One frame's evidence. Returns true when the mode changed.
    pub fn next(&mut self, e: Evidence, confirm: u32) -> bool {
        let want = match e {
            Evidence::Neither => {
                self.run = 0;
                return false;
            }
            Evidence::Night => Mode::Night,
            Evidence::Day => Mode::Day,
        };
        if want == self.mode {
            self.run = 0;
            return false;
        }
        if want == self.pending {
            self.run += 1;
        } else {
            // Evidence the other way: a new run.
            self.pending = want;
            self.run = 1;
        }
        if self.run >= confirm {
            self.mode = want;
            self.run = 0;
            return true;
        }
        false
    }
}

/// One camera's day, night and switches. See the module.
#[derive(Clone, Debug)]
pub struct LightMeter {
    pub opts: LightOptions,
    decision: Decision,
    last_luma: Option<f32>,
    settle_until_ms: u64,
    /// Settles started.
    pub switches: u64,
}

impl LightMeter {
    pub fn new(opts: LightOptions) -> LightMeter {
        LightMeter { opts, decision: Decision::default(), last_luma: None, settle_until_ms: 0, switches: 0 }
    }

    pub fn mode(&self) -> Mode {
        self.decision.mode
    }

    /// A new connection: the mode stands (the camera's light did not change), but its
    /// brightness is not compared with the old connection's.
    pub fn reconnected(&mut self) {
        self.last_luma = None;
    }

    /// One frame, seen at `now_ms`. `chroma`: its [`chroma`] score, if it had colour
    /// planes. `luma`: its [`mean_luma`]. `lighting`: motion reset on it.
    pub fn next(&mut self, now_ms: u64, chroma: Option<f32>, luma: f32, lighting: bool) -> Light {
        let o = &self.opts;
        let e = match chroma {
            Some(c) if c < o.night_below => Evidence::Night,
            Some(c) if c > o.day_above => Evidence::Day,
            _ => Evidence::Neither,
        };
        let was = self.decision.mode;
        let changed = self.decision.next(e, o.confirm);
        // Unknown -> a mode is the first decision, not a switch of the light.
        let mode_switch = changed && was != Mode::Unknown;

        let jumped = o.jump > 0.0 && self.last_luma.is_some_and(|l| (luma - l).abs() > o.jump);
        self.last_luma = Some(luma);

        let switched = o.settle_ms > 0 && (mode_switch || lighting || jumped);
        if switched {
            self.switches += 1;
            self.settle_until_ms = self.settle_until_ms.max(now_ms.saturating_add(o.settle_ms));
        }
        Light { mode: self.decision.mode, settling: now_ms < self.settle_until_ms, switched }
    }
}

/// Mean brightness of a frame, from every 8th pixel of every 8th row.
pub fn mean_luma(f: &GrayFrame) -> f32 {
    let (mut sum, mut n) = (0u64, 0u64);
    for r in (0..f.height).step_by(8) {
        let row = &f.pixels[r * f.stride..r * f.stride + f.width];
        for &p in row.iter().step_by(8) {
            sum += p as u64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum as f32 / n as f32
    }
}

/// The colour planes of a decoded frame.
pub enum Chroma<'a> {
    /// NV12 or NV21: one plane of interleaved pairs, `width` pairs a row.
    Interleaved { plane: &'a [u8], stride: usize, width: usize, height: usize },
    /// I420 or YV12: two planes, `width` by `height` each.
    Planar { u: &'a [u8], u_stride: usize, v: &'a [u8], v_stride: usize, width: usize, height: usize },
}

/// How far a frame's colour is from neutral gray: the mean of |c - 128| over both
/// chroma channels, 0 (no colour, as in IR mode) to 128. Sampled on every 4th
/// chroma row and every 4th chroma sample, which is every 8th row and column of the picture.
pub fn chroma(c: &Chroma) -> f32 {
    let (mut sum, mut n) = (0u64, 0u64);
    let mut add = |b: u8| {
        sum += (b as i32 - 128).unsigned_abs() as u64;
        n += 1;
    };
    match *c {
        Chroma::Interleaved { plane, stride, width, height } => {
            for r in (0..height).step_by(4) {
                let row = &plane[r * stride..r * stride + 2 * width];
                for pair in row.chunks_exact(2).step_by(4) {
                    add(pair[0]);
                    add(pair[1]);
                }
            }
        }
        Chroma::Planar { u, u_stride, v, v_stride, width, height } => {
            for r in (0..height).step_by(4) {
                for x in (0..width).step_by(4) {
                    add(u[r * u_stride + x]);
                    add(v[r * v_stride + x]);
                }
            }
        }
    }
    if n == 0 {
        0.0
    } else {
        sum as f32 / n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> LightMeter {
        LightMeter::new(LightOptions::default())
    }

    #[test]
    fn decides_after_confirm_frames() {
        let mut m = meter();
        for i in 0..4 {
            assert_eq!(m.next(i * 40, Some(10.0), 100.0, false).mode, Mode::Unknown);
        }
        let l = m.next(160, Some(10.0), 100.0, false);
        assert_eq!(l.mode, Mode::Day);
        // The first decision is not a switch of the light.
        assert!(!l.switched && !l.settling);
    }

    #[test]
    fn dusk_switch_settles_then_clears() {
        let mut m = meter();
        let mut t = 0;
        for _ in 0..5 {
            m.next(t, Some(10.0), 100.0, false);
            t += 40;
        }
        assert_eq!(m.mode(), Mode::Day);
        // IR on: chroma goes neutral; brightness about the same.
        let mut switched_at = None;
        for _ in 0..5 {
            let l = m.next(t, Some(0.5), 100.0, false);
            if l.switched {
                switched_at = Some(t);
            }
            t += 40;
        }
        let s = switched_at.expect("a switch");
        assert_eq!(m.mode(), Mode::Night);
        assert!(m.next(s + 1999, Some(0.5), 100.0, false).settling);
        assert!(!m.next(s + 2000, Some(0.5), 100.0, false).settling);
        assert_eq!(m.switches, 1);
    }

    #[test]
    fn between_thresholds_never_flips() {
        let mut m = meter();
        for i in 0..5 {
            m.next(i, Some(10.0), 100.0, false);
        }
        for i in 0..1000 {
            let l = m.next(100 + i, Some(3.0), 100.0, false);
            assert_eq!(l.mode, Mode::Day);
            assert!(!l.switched);
        }
    }

    #[test]
    fn mixed_evidence_does_not_add_up() {
        // Day; 4 night-looking frames, 1 day frame, 4 night frames: never 5 in a row.
        let mut m = meter();
        for i in 0..5 {
            m.next(i, Some(10.0), 100.0, false);
        }
        for c in [0.5, 0.5, 0.5, 0.5, 10.0, 0.5, 0.5, 0.5, 0.5] {
            m.next(10, Some(c), 100.0, false);
        }
        assert_eq!(m.mode(), Mode::Day);
        // A day frame ends the night run; a frame between the thresholds also breaks it.
        for c in [10.0, 0.5, 0.5, 0.5, 0.5, 3.0, 0.5, 0.5, 0.5, 0.5] {
            m.next(10, Some(c), 100.0, false);
        }
        assert_eq!(m.mode(), Mode::Day);
        // One more night frame makes 5 in a row.
        assert_eq!(m.next(10, Some(0.5), 100.0, false).mode, Mode::Night);
    }

    #[test]
    fn gray_stream_is_unknown_but_still_settles() {
        let mut m = meter();
        assert!(!m.next(0, None, 60.0, false).settling);
        assert!(!m.next(40, None, 70.0, false).settling);
        // IR LEDs on: brightness jumps.
        let l = m.next(80, None, 150.0, false);
        assert_eq!(l.mode, Mode::Unknown);
        assert!(l.switched && l.settling);
        // Motion's lighting reset also starts one, and extends it.
        let l = m.next(1500, None, 150.0, true);
        assert!(l.switched);
        assert!(m.next(3400, None, 150.0, false).settling);
        assert!(!m.next(3500, None, 150.0, false).settling);
    }

    #[test]
    fn reconnect_is_not_a_jump() {
        let mut m = meter();
        m.next(0, None, 30.0, false);
        m.reconnected();
        assert!(!m.next(40, None, 200.0, false).switched);
    }

    #[test]
    fn settle_zero_disables_settling() {
        let mut m = LightMeter::new(LightOptions { settle_ms: 0, ..Default::default() });
        m.next(0, None, 0.0, false);
        let l = m.next(40, None, 255.0, true);
        assert!(!l.switched && !l.settling);
    }

    #[test]
    fn options_are_checked() {
        assert!(LightOptions::default().check().is_ok());
        assert!(LightOptions { night_below: 5.0, day_above: 4.0, ..Default::default() }.check().is_err());
        assert!(LightOptions { confirm: 0, ..Default::default() }.check().is_err());
        assert!(LightOptions { jump: 300.0, ..Default::default() }.check().is_err());
    }

    #[test]
    fn chroma_of_neutral_and_coloured_planes() {
        let (w, h) = (16, 8); // chroma plane size
        // NV12, stride padded to 40 bytes.
        let mut nv = vec![128u8; 40 * h];
        assert_eq!(chroma(&Chroma::Interleaved { plane: &nv, stride: 40, width: w, height: h }), 0.0);
        for r in 0..h {
            for x in 0..w {
                nv[r * 40 + 2 * x] = 128 + 20; // U
                nv[r * 40 + 2 * x + 1] = 128 - 10; // V
            }
        }
        assert_eq!(chroma(&Chroma::Interleaved { plane: &nv, stride: 40, width: w, height: h }), 15.0);
        // I420.
        let u = vec![100u8; w * h];
        let v = vec![128u8; w * h];
        assert_eq!(chroma(&Chroma::Planar { u: &u, u_stride: w, v: &v, v_stride: w, width: w, height: h }), 14.0);
    }

    #[test]
    fn mean_luma_respects_stride() {
        let mut px = vec![255u8; 32 * 16];
        for r in 0..16 {
            for x in 0..24 {
                px[r * 32 + x] = 100;
            }
        }
        let f = GrayFrame::with_stride(&px, 24, 16, 32).unwrap();
        assert_eq!(mean_luma(&f), 100.0);
    }
}
