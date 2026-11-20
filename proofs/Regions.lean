/-!
# The squares of `src/regions.rs` stay inside the frame

`place c side extent` is where a `side` square centred on `c` starts along one axis of a
frame `extent` pixels long: `min (max (c - side/2) 0) (extent - side)`. In `Nat`,
`c - side / 2` already stops at 0, which is the `max`. (The Rust rounds the centre to a
whole pixel first; from there it is this.) Core Lean 4.31 and `omega` only.

What is proved:
1. A square no bigger than the frame starts and ends inside it.
2. So a find inside the square, moved by the square's start, is inside the frame:
   `join` never places a box off the frame.
3. A square holds the centre it was placed on, whenever that centre is inside the frame
   by at least half a side from each end: what a region was made for is in it.
-/

def place (c side extent : Nat) : Nat := min (c - side / 2) (extent - side)

/-- 1. Inside the frame. -/
theorem place_inside (c side extent : Nat) (h : side ≤ extent) :
    place c side extent + side ≤ extent := by
  unfold place
  omega

/-- 2. A find at `[a, b]` within the square, moved onto the frame, is within the frame. -/
theorem find_inside (c side extent a b : Nat) (h : side ≤ extent) (hab : a ≤ b) (hb : b ≤ side) :
    place c side extent + a ≤ place c side extent + b ∧ place c side extent + b ≤ extent := by
  have := place_inside c side extent h
  omega

/-- 3. The centre is in the square when it is half a side from each end of the frame. -/
theorem centre_inside (c side extent : Nat) (hlo : side / 2 ≤ c) (hhi : c + side / 2 ≤ extent)
    (h : side ≤ extent) :
    place c side extent ≤ c ∧ c ≤ place c side extent + side := by
  unfold place
  omega
