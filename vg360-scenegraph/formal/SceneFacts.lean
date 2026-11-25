/-!
# Conservative fact emission (vg360-scenegraph `policy.rs`)

Not machine-checked in the build session (no Lean toolchain there): run `lean formal/SceneFacts.lean`.

Units: millimetres, as `Int`, so every proof is linear integer arithmetic (`omega`, Lean core only,
no Mathlib).

`m` is the true margin of a relation (for `near`: R − d; for `in`: signed distance inside the
polygon; for `under`: drop-zone radius − distance), `mHat` the margin computed from the estimated
floor position, `s` the slack k·σ. The single modelling assumption is `Within`: the estimate is off
by at most the slack. That bound is the k-sigma claim of the Kalman filter and is *not* proved here;
everything after it is.

Result: with the polarity chosen by `Policy::from_conditions`, uncertainty never removes an alert
that the true positions would raise.
-/

namespace SceneFacts

inductive Emit where
  | possible
  | definite

/-- `policy::decide`: whether the fact is emitted. -/
def Emits : Emit → Int → Int → Prop
  | .possible, mHat, s => 0 ≤ mHat + s
  | .definite, mHat, s => 0 ≤ mHat - s

/-- The k-sigma assumption. -/
def Within (mHat m s : Int) : Prop := mHat - m ≤ s ∧ m - mHat ≤ s

/-- "Possibly" never misses a true fact. -/
theorem possible_complete {mHat m s : Int} (h : Within mHat m s) (ht : 0 ≤ m) :
    Emits .possible mHat s := by
  unfold Within at h
  obtain ⟨_, h2⟩ := h
  show 0 ≤ mHat + s
  omega

/-- "Definitely" never asserts a false fact. -/
theorem definite_sound {mHat m s : Int} (h : Within mHat m s) (he : Emits .definite mHat s) :
    0 ≤ m := by
  unfold Within at h
  obtain ⟨h1, _⟩ := h
  have : 0 ≤ mHat - s := he
  omega

/-- `Policy::from_conditions`: used positively at all → possible; only negated → definite. -/
def choose : (usedPos usedNeg : Bool) → Emit
  | false, true => .definite
  | _, _ => .possible

/-- A positive condition (`a worker is near a vehicle`) that truly holds is still satisfied. -/
theorem positive_condition_kept {mHat m s : Int} (usedNeg : Bool)
    (h : Within mHat m s) (ht : 0 ≤ m) : Emits (choose true usedNeg) mHat s := by
  cases usedNeg <;> exact possible_complete h ht

/-- A negated condition (`... is not in a walkway`) that truly holds is still satisfied:
    the fact is not emitted, so its absence lets the rule fire. -/
theorem negated_condition_kept {mHat m s : Int}
    (h : Within mHat m s) (ht : ¬ 0 ≤ m) : ¬ Emits (choose false true) mHat s :=
  fun he => ht (definite_sound h he)

/-- `near` specialised: true distance d ≤ R, estimate within s of d ⇒ the fact is emitted. -/
theorem near_complete {R d dHat s : Int} (h : dHat - d ≤ s ∧ d - dHat ≤ s) (ht : d ≤ R) :
    Emits .possible (R - dHat) s := by
  obtain ⟨_, h2⟩ := h
  show 0 ≤ R - dHat + s
  omega

/-- Monotonicity in the slack: a coasting track's growing covariance can only add `possible`
    facts, never remove them. -/
theorem possible_monotone_in_slack {mHat s s' : Int} (hs : s ≤ s') (he : Emits .possible mHat s) :
    Emits .possible mHat s' := by
  have : 0 ≤ mHat + s := he
  show 0 ≤ mHat + s'
  omega

/-- Mixed use is the one case with a gap: a negated rule over a `possible` fact can be silenced by
    uncertainty. Witness: true margin −1 (false), estimate −1, slack 2. `Policy` warns about it. -/
theorem mixed_use_can_silence_negated :
    ∃ mHat m s : Int, Within mHat m s ∧ ¬ 0 ≤ m ∧ Emits (choose true true) mHat s :=
  ⟨-1, -1, 2, by unfold Within; omega, by omega, show (0 : Int) ≤ -1 + 2 by omega⟩

end SceneFacts
