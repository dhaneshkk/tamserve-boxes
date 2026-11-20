/-!
# The retention rules of `src/retention.rs`

Times are milliseconds as `Nat`. Core Lean 4 only (no Mathlib): `induction`, `simp` on
core lemmas, and `omega`. Check with `lean proofs/Retention.lean` (Lean 4.31).

What is proved:
1. A segment is kept at least `all` (or `longest`, past the activity log) after its end.
2. A segment that meets an activity, widened by `pre` before and `post` after, is kept
   at least that activity's days after its end: no age sweep deletes it sooner.
3. A held segment (pinned, or its camera's newest) is never deleted, for age or space.
4. Deleting for space takes a prefix of the candidates sorted by keep-until, so every
   segment deleted for space expires no later than every candidate left.
-/

/-- One interval of activity: from `lo` to `hi`, kept `days` (the longest of its motion
and its labels' retentions). -/
structure Act where
  lo : Nat
  hi : Nat
  days : Nat

/-- The activity, widened by `pre` before and `post` after, meets the segment `[s, e]`.
As in the Rust, `a.lo - pre` saturates at 0. -/
def meets (pre post s e : Nat) (a : Act) : Bool :=
  decide (a.lo - pre ≤ e) && decide (s ≤ a.hi + post)

/-- How long after its end the segment `[s, e]` is kept: `base`, raised to the days of
every activity it meets. -/
def keepFor (pre post s e base : Nat) : List Act → Nat
  | [] => base
  | a :: as =>
    if meets pre post s e a then max a.days (keepFor pre post s e base as)
    else keepFor pre post s e base as

def keepUntil (pre post s e base : Nat) (acts : List Act) : Nat :=
  e + keepFor pre post s e base acts

/-- `base` is `all`, or `longest` for a segment that ends past what the log covers. -/
def base (covered e all longest : Nat) : Nat :=
  if e > covered then longest else all

theorem base_le_keepFor (pre post s e b : Nat) (acts : List Act) :
    b ≤ keepFor pre post s e b acts := by
  induction acts with
  | nil => simp [keepFor]
  | cons a as ih =>
    simp only [keepFor]
    split <;> omega

theorem met_le_keepFor (pre post s e b : Nat) (acts : List Act) (a : Act)
    (ha : a ∈ acts) (hm : meets pre post s e a = true) :
    a.days ≤ keepFor pre post s e b acts := by
  induction acts with
  | nil => simp at ha
  | cons x xs ih =>
    simp only [keepFor]
    cases List.mem_cons.mp ha with
    | inl h =>
      subst h
      simp [hm]
      omega
    | inr h =>
      have := ih h
      split <;> omega

/-- 1. Every segment is kept `all` after its end, and `longest` past the log. -/
theorem kept_all (pre post s e covered all longest : Nat) (acts : List Act)
    (hl : all ≤ longest) :
    e + all ≤ keepUntil pre post s e (base covered e all longest) acts ∧
    (e > covered → e + longest ≤ keepUntil pre post s e (base covered e all longest) acts) := by
  unfold keepUntil base
  by_cases hc : e > covered
  · have h := base_le_keepFor pre post s e longest acts
    simp only [hc, if_true]
    exact ⟨by omega, fun _ => by omega⟩
  · have h := base_le_keepFor pre post s e all acts
    simp only [hc, if_false]
    exact ⟨by omega, fun h' => h'.elim⟩

/-- The age rule: a segment goes once `now` is past its keep-until, unless held. -/
def expired (held : Bool) (now keep : Nat) : Bool :=
  !held && decide (now > keep)

/-- 2. No age sweep deletes a segment before the days of an activity it meets are up. -/
theorem event_outlives_age_sweep (pre post s e b now : Nat) (acts : List Act) (a : Act)
    (held : Bool) (ha : a ∈ acts) (hm : meets pre post s e a = true) (hnow : now ≤ e + a.days) :
    expired held now (keepUntil pre post s e b acts) = false := by
  have := met_le_keepFor pre post s e b acts a ha hm
  unfold expired keepUntil
  have : ¬ (now > e + keepFor pre post s e b acts) := by omega
  simp [this]

/-- One candidate for the sweep: its keep-until, and whether it is held. -/
structure Cand where
  keep : Nat
  held : Bool

/-- Deleting for space: the candidates not held and not expired, in keep-until order
(`sorted`), of which a prefix of `n` goes. -/
def spaceGone (sorted : List Cand) (n : Nat) : List Cand :=
  (sorted.filter (fun c => !c.held)).take n

/-- 3a. A held segment never goes for age. -/
theorem held_not_expired (now keep : Nat) : expired true now keep = false := by
  simp [expired]

/-- 3b. A held segment never goes for space. -/
theorem held_not_space (sorted : List Cand) (n : Nat) (c : Cand) (hc : c ∈ spaceGone sorted n) :
    c.held = false := by
  unfold spaceGone at hc
  have := (List.mem_filter.mp (List.mem_of_mem_take hc)).2
  simpa using this

/-- 4. In a list sorted by keep-until, each of the first `n` expires no later than each
of the rest: so space is freed from what would go soonest anyway. -/
theorem space_takes_soonest (l : List Cand) (hs : l.Pairwise (fun x y => x.keep ≤ y.keep)) (n : Nat) :
    ∀ x ∈ l.take n, ∀ y ∈ l.drop n, x.keep ≤ y.keep := by
  have h : (l.take n ++ l.drop n).Pairwise (fun x y => x.keep ≤ y.keep) := by
    rw [List.take_append_drop]; exact hs
  intro x hx y hy
  exact (List.pairwise_append.mp h).2.2 x hx y hy

/-- 4, for the sweep: a candidate deleted for space expires no later than any candidate
not held that is kept. (The filter keeps the order, so it stays sorted.) -/
theorem space_gone_first (sorted : List Cand) (hs : sorted.Pairwise (fun x y => x.keep ≤ y.keep)) (n : Nat) :
    ∀ x ∈ spaceGone sorted n, ∀ y ∈ (sorted.filter (fun c => !c.held)).drop n, x.keep ≤ y.keep :=
  space_takes_soonest _ (hs.filter _) n
