/-!
# The safety rules of `src/grammar.rs`, `src/lens.rs` and `src/datalog.rs`

Core Lean 4.31 only (`Nat`, `omega`, `decide`), as the other proofs here.

What is proved:
1. **No flicker.** The leaky bucket (`step_s`) moves by at most one per frame and stays in
   `0..s_max`, so going from OFF (`s ≤ lo`) to ON (`s ≥ hi`), or back, takes at least `hi - lo`
   frames (`no_fast_rise`, `no_fast_fall`); and evidence that persists does raise the alarm
   (`persistent_fires`).
2. **PPE is never guessed.** An occluded person is never reported compliant, a conformal set
   holding both labels is never collapsed to a verdict, and "compliant" means exactly
   "visible, compliant in the set, violation not in it" (`occluded_never_compliant`,
   `compliant_iff`, `ambiguous_is_unknown`).
3. **Grid and ray.** A clamped index is inside the grid (`clamp_in`); the line-of-fire test is
   symmetric and monotone in its radius (`adist_symm`, `lof_radius_mono`).
4. **Breach proofs check out.** A path accepted by `validPath`, the check `valid_path` runs on
   every Datalog breach derivation, moves between 4-neighbours and never stands on a barrier
   cell (`valid_step`, `valid_head_free`, `valid_all_free`).
5. **A lost calibration decides nothing.** While the drift check cannot confirm the view, no
   floor rule says "clear" or "alarm" (`lost_never_clear`, `lost_never_alarm`); once it is
   trusted again, verdicts pass through unchanged (`trusted_passes`).
6. **Fusion never hides or invents.** Filling a PPE gap from remembered evidence keeps the
   current verdict when there is one (`fuse_keeps_violation`, `fuse_keeps_compliant`), lets
   stale evidence expire (`fuse_stale_is_unknown`), and a filled "compliant" always has a
   decided witness inside the window (`fuse_compliant_has_witness`).
-/

def stepS (S : Nat) : Bool → Nat → Nat
  | true,  s => if s < S then s + 1 else S
  | false, s => s - 1

def run (S : Nat) : List Bool → Nat → Nat
  | [],      s => s
  | p :: ps, s => run S ps (stepS S p s)

theorem step_bound (S s : Nat) (p : Bool) (h : s ≤ S) : stepS S p s ≤ S := by
  cases p with
  | false => show s - 1 ≤ S; omega
  | true  => show (if s < S then s + 1 else S) ≤ S; split <;> omega

theorem step_up (S s : Nat) (p : Bool) : stepS S p s ≤ s + 1 := by
  cases p with
  | false => show s - 1 ≤ s + 1; omega
  | true  => show (if s < S then s + 1 else S) ≤ s + 1; split <;> omega

theorem step_down (S s : Nat) (p : Bool) (h : s ≤ S) : s ≤ stepS S p s + 1 := by
  cases p with
  | false => show s ≤ s - 1 + 1; omega
  | true  => show s ≤ (if s < S then s + 1 else S) + 1; split <;> omega

theorem run_bound (S : Nat) : ∀ (ps : List Bool) (s : Nat), s ≤ S → run S ps s ≤ S
  | [],      _, h => h
  | p :: ps, s, h => run_bound S ps (stepS S p s) (step_bound S s p h)

theorem run_up (S : Nat) : ∀ (ps : List Bool) (s : Nat), run S ps s ≤ s + ps.length
  | [],      s => by show s ≤ s + 0; omega
  | p :: ps, s => by
      have h1 := step_up S s p
      have h2 := run_up S ps (stepS S p s)
      show run S ps (stepS S p s) ≤ s + (ps.length + 1)
      omega

theorem run_down (S : Nat) : ∀ (ps : List Bool) (s : Nat), s ≤ S → s ≤ run S ps s + ps.length
  | [],      s, _ => by show s ≤ s + 0; omega
  | p :: ps, s, h => by
      have h1 := step_down S s p h
      have h2 := run_down S ps (stepS S p s) (step_bound S s p h)
      show s ≤ run S ps (stepS S p s) + (ps.length + 1)
      omega

theorem no_fast_rise (S LO HI s : Nat) (ps : List Bool)
    (hlo : s ≤ LO) (hhi : HI ≤ run S ps s) : HI - LO ≤ ps.length := by
  have := run_up S ps s; omega

theorem no_fast_fall (S LO HI s : Nat) (ps : List Bool) (hs : s ≤ S)
    (hhi : HI ≤ s) (hlo : run S ps s ≤ LO) : HI - LO ≤ ps.length := by
  have := run_down S ps s hs; omega

theorem persistent_fires (S : Nat) : ∀ (n s : Nat), s + n ≤ S → run S (List.replicate n true) s = s + n
  | 0,     _, _ => rfl
  | n + 1, s, h => by
      have hlt : s < S := by omega
      have hstep : stepS S true s = s + 1 := by
        show (if s < S then s + 1 else S) = s + 1
        rw [if_pos hlt]
      show run S (List.replicate n true) (stepS S true s) = s + (n + 1)
      rw [hstep, persistent_fires S n (s + 1) (by omega)]
      omega

-- (2) 3-valued PPE decision: an occluded person is NEVER reported compliant; an ambiguous
--     conformal set {ok, violation} is never collapsed to a verdict.
inductive PPE where
  | compliant | violation | unknown
  deriving DecidableEq

def decidePPE : Bool → Bool → Bool → PPE
  | true,  _,     _     => .unknown
  | false, true,  false => .compliant
  | false, false, true  => .violation
  | false, _,     _     => .unknown

theorem occluded_never_compliant (a b : Bool) : decidePPE true a b ≠ PPE.compliant := by
  cases a <;> cases b <;> decide

theorem compliant_iff (o a b : Bool) :
    decidePPE o a b = PPE.compliant ↔ (o = false ∧ a = true ∧ b = false) := by
  cases o <;> cases a <;> cases b <;> decide

theorem ambiguous_is_unknown (o : Bool) : decidePPE o true true = PPE.unknown := by
  cases o <;> rfl

-- (3) Grid indexing never leaves the array; line-of-fire trigger is symmetric and monotone in radius.
def clampIdx (W x : Nat) : Nat := if x < W then x else W - 1

theorem clamp_in (W x : Nat) (hW : 0 < W) : clampIdx W x < W := by
  show (if x < W then x else W - 1) < W
  split <;> omega

def adist (a b : Nat) : Nat := (a - b) + (b - a)

theorem adist_symm (a b : Nat) : adist a b = adist b a := by
  show (a - b) + (b - a) = (b - a) + (a - b); omega

theorem lof_radius_mono (a b r r' : Nat) (h : adist a b ≤ r) (hr : r ≤ r') : adist a b ≤ r' := by
  omega

-- (4) Proof-carrying breach alerts. The Datalog derivation of a perimeter breach is a cell path;
--     the runtime re-checks it with validPath. Checker-true => every step is a 4-neighbour move and
--     the head cell is free, so the certificate cannot cross a barrier diagonally (4/8 duality).
def adj4 (a b : Nat × Nat) : Bool :=
  (a.1 == b.1 && (a.2 + 1 == b.2 || b.2 + 1 == a.2)) ||
  (a.2 == b.2 && (a.1 + 1 == b.1 || b.1 + 1 == a.1))

def validPath (blocked : Nat × Nat → Bool) : List (Nat × Nat) → Bool
  | []            => true
  | [c]           => !blocked c
  | c :: d :: rest => !blocked c && adj4 c d && validPath blocked (d :: rest)

theorem valid_step (b : Nat × Nat → Bool) (c d : Nat × Nat) (rest : List (Nat × Nat))
    (h : validPath b (c :: d :: rest) = true) :
    adj4 c d = true ∧ validPath b (d :: rest) = true := by
  have h' : (!b c && adj4 c d && validPath b (d :: rest)) = true := h
  revert h'
  cases b c <;> cases adj4 c d <;> cases validPath b (d :: rest) <;> decide

theorem valid_head_free (b : Nat × Nat → Bool) (c : Nat × Nat) (rest : List (Nat × Nat))
    (h : validPath b (c :: rest) = true) : b c = false := by
  cases rest with
  | nil =>
      have h' : (!b c) = true := h
      revert h'; cases b c <;> decide
  | cons d r =>
      have h' : (!b c && adj4 c d && validPath b (d :: r)) = true := h
      revert h'; cases b c <;> cases adj4 c d <;> cases validPath b (d :: r) <;> decide

theorem valid_all_free (b : Nat × Nat → Bool) :
    ∀ (p : List (Nat × Nat)), validPath b p = true → ∀ x, x ∈ p → b x = false
  | [], _, x, hx => nomatch hx
  | [c], h, x, hx => by
      cases hx with
      | head => exact valid_head_free b c [] h
      | tail _ h2 => exact nomatch h2
  | c :: d :: rest, h, x, hx => by
      cases hx with
      | head => exact valid_head_free b c (d :: rest) h
      | tail _ h2 => exact valid_all_free b (d :: rest) (valid_step b c d rest h).2 x h2

-- (5) Calibration gate. When the drift monitor cannot confirm the camera still sees the calibrated view,
--     no rule may say "clear" (or "alarm"): every geometric verdict is UNKNOWN until calibration is trusted.
inductive Calib where
  | ok | corrected | unknown
  deriving DecidableEq

inductive Verdict where
  | alarm | clear | unknown
  deriving DecidableEq

def gate : Calib → Bool → Verdict
  | .unknown, _     => .unknown
  | _,        true  => .alarm
  | _,        false => .clear

theorem lost_never_clear (b : Bool) : gate Calib.unknown b ≠ Verdict.clear := by
  cases b <;> decide

theorem lost_never_alarm (b : Bool) : gate Calib.unknown b ≠ Verdict.alarm := by
  cases b <;> decide

theorem trusted_passes (c : Calib) (b : Bool) (h : c ≠ Calib.unknown) :
    gate c b = (if b then Verdict.alarm else Verdict.clear) := by
  cases c with
  | unknown   => exact absurd rfl h
  | ok        => cases b <;> rfl
  | corrected => cases b <;> rfl

-- (6) PPE fusion. The current verdict of the camera watching the person always stands; remembered evidence
--     (its own last clear look, or the entry camera) only fills an UNKNOWN, and only while fresh. So a filled
--     "compliant" always has a decided witness no older than the window, and a current violation is never hidden.
def fuse (main mem : PPE) (age window : Nat) : PPE :=
  match main with
  | .compliant => .compliant
  | .violation => .violation
  | .unknown   => if age ≤ window then mem else .unknown

theorem fuse_keeps_violation (mem : PPE) (age w : Nat) : fuse PPE.violation mem age w = PPE.violation := rfl

theorem fuse_keeps_compliant (mem : PPE) (age w : Nat) : fuse PPE.compliant mem age w = PPE.compliant := rfl

theorem fuse_stale_is_unknown (mem : PPE) (age w : Nat) (h : w < age) : fuse PPE.unknown mem age w = PPE.unknown := by
  show (if age ≤ w then mem else PPE.unknown) = PPE.unknown
  rw [if_neg (by omega)]

theorem fuse_compliant_has_witness (main mem : PPE) (age w : Nat) (h : fuse main mem age w = PPE.compliant) :
    main = PPE.compliant ∨ (main = PPE.unknown ∧ mem = PPE.compliant ∧ age ≤ w) := by
  cases main with
  | compliant => exact Or.inl rfl
  | violation => cases h
  | unknown =>
      have h' : (if age ≤ w then mem else PPE.unknown) = PPE.compliant := h
      by_cases ha : age ≤ w
      · rw [if_pos ha] at h'; exact Or.inr ⟨rfl, h', ha⟩
      · rw [if_neg ha] at h'; cases h'

#print axioms no_fast_rise
#print axioms fuse_compliant_has_witness
#print axioms lost_never_clear
#print axioms valid_all_free
#print axioms persistent_fires
#print axioms occluded_never_compliant
