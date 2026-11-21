/-!
# Day and night in `src/light.rs` don't flicker, and do follow the light

`Decision::next` as it is written in Rust: one frame's evidence (night, day, or neither,
from where its chroma falls against `night_below` and `day_above`) moves `run`, the
count of frames in a row that pointed to `pending`, a mode other than `mode`; the mode
changes when `run` reaches `confirm` (`k` here). Rust's `u32` run never passes `k`, so
`Nat` is exact. Core Lean 4.31 and `omega` only.

What is proved:
1. Chroma between the two thresholds is no evidence, and no evidence never changes the mode.
2. The mode never changes on fewer than `k` frames: from a run of `r`, any `n` frames with
   `r + n < k` leave it as it was, whatever they say. From a fresh run, `k - 1` frames of
   anything are not enough.
3. `k` frames in a row that say the same, from any state, always end in that mode.
-/

inductive Mode | unknown | day | night
  deriving DecidableEq

inductive Ev | night | day | neither
  deriving DecidableEq

structure St where
  mode : Mode
  pending : Mode
  run : Nat

/-- `night_below = lo`, `day_above = hi`, chroma `c`, all in the same units. -/
def classify (lo hi c : Nat) : Ev :=
  if c < lo then .night else if hi < c then .day else .neither

/-- The run after a frame for `w`: one longer if it was for `w` already, else a new run. -/
def bump (s : St) (w : Mode) : Nat := if s.pending = w then s.run + 1 else 1

def toward (k : Nat) (s : St) (w : Mode) : St :=
  if s.mode = w then { s with run := 0 }
  else if k ≤ bump s w then ⟨w, w, 0⟩
  else ⟨s.mode, w, bump s w⟩

def step (k : Nat) (s : St) : Ev → St
  | .neither => { s with run := 0 }
  | .night => toward k s .night
  | .day => toward k s .day

def runs (k : Nat) : St → List Ev → St
  | s, [] => s
  | s, e :: es => runs k (step k s e) es

/-! ## 1. Between the thresholds -/

theorem between_is_neither (lo hi c : Nat) (h1 : lo ≤ c) (h2 : c ≤ hi) : classify lo hi c = .neither := by
  have h3 : ¬ c < lo := by omega
  have h4 : ¬ hi < c := by omega
  unfold classify
  rw [if_neg h3, if_neg h4]

theorem neither_keeps_mode (k : Nat) (s : St) : (step k s .neither).mode = s.mode := rfl

/-! ## 2. Fewer than `k` frames change nothing -/

theorem bump_le (s : St) (w : Mode) : bump s w ≤ s.run + 1 := by
  unfold bump
  split <;> omega

theorem toward_small (k : Nat) (s : St) (w : Mode) (h : s.run + 1 < k) :
    (toward k s w).mode = s.mode ∧ (toward k s w).run ≤ s.run + 1 := by
  have hb := bump_le s w
  unfold toward
  by_cases hm : s.mode = w
  · rw [if_pos hm]
    exact ⟨rfl, Nat.zero_le _⟩
  · rw [if_neg hm]
    by_cases hk : k ≤ bump s w
    · exfalso
      omega
    · rw [if_neg hk]
      exact ⟨rfl, hb⟩

theorem step_small (k : Nat) (s : St) (e : Ev) (h : s.run + 1 < k) :
    (step k s e).mode = s.mode ∧ (step k s e).run ≤ s.run + 1 := by
  cases e with
  | neither => exact ⟨rfl, Nat.zero_le _⟩
  | night => exact toward_small k s .night h
  | day => exact toward_small k s .day h

theorem no_change_before_confirm (k : Nat) :
    ∀ (es : List Ev) (s : St), s.run + es.length < k → (runs k s es).mode = s.mode
  | [], _, _ => rfl
  | e :: es, s, h => by
    simp only [List.length_cons] at h
    obtain ⟨hm, hr⟩ := step_small k s e (by omega)
    have h2 : (step k s e).run + es.length < k := by omega
    show (runs k (step k s e) es).mode = s.mode
    rw [no_change_before_confirm k es (step k s e) h2, hm]

/-- From a fresh run, `k - 1` frames of anything keep the mode. -/
theorem fresh_needs_k (k : Nat) (s : St) (h0 : s.run = 0) (es : List Ev) (hl : es.length + 1 ≤ k) :
    (runs k s es).mode = s.mode :=
  no_change_before_confirm k es s (by omega)

/-! ## 3. `k` agreeing frames always change it -/

/-- How far `s` has got towards mode `w`: all the way (`k`), its run, or nothing. -/
def prog (k : Nat) (w : Mode) (s : St) : Nat :=
  if s.mode = w then k else if s.pending = w then s.run else 0

theorem toward_prog (k : Nat) (s : St) (w : Mode) (hk : 0 < k) :
    (k ≤ prog k w (toward k s w) ∨ prog k w s + 1 ≤ prog k w (toward k s w)) ∧ (toward k s w).run < k := by
  unfold toward
  by_cases hm : s.mode = w
  · rw [if_pos hm]
    have h : prog k w { s with run := 0 } = k := by
      simp [prog, hm]
    refine ⟨Or.inl (by omega), ?_⟩
    show 0 < k
    exact hk
  · rw [if_neg hm]
    by_cases hb : k ≤ bump s w
    · rw [if_pos hb]
      have h : prog k w ⟨w, w, 0⟩ = k := by
        simp [prog]
      refine ⟨Or.inl (by omega), ?_⟩
      show 0 < k
      exact hk
    · rw [if_neg hb]
      have h1 : prog k w ⟨s.mode, w, bump s w⟩ = bump s w := by
        simp [prog, hm]
      refine ⟨Or.inr ?_, ?_⟩
      · rw [h1]
        by_cases hp : s.pending = w <;> simp [prog, bump, hm, hp]
      · show bump s w < k
        omega

theorem repeat_progress (k : Nat) (w : Mode) (e : Ev) (he : ∀ s, step k s e = toward k s w) (hk : 0 < k) :
    ∀ (n : Nat) (s : St),
      (k ≤ prog k w (runs k s (List.replicate (n + 1) e)) ∨
        prog k w s + (n + 1) ≤ prog k w (runs k s (List.replicate (n + 1) e))) ∧
      (runs k s (List.replicate (n + 1) e)).run < k
  | 0, s => by
    show (k ≤ prog k w (step k s e) ∨ prog k w s + (0 + 1) ≤ prog k w (step k s e)) ∧ (step k s e).run < k
    rw [he]
    have := toward_prog k s w hk
    omega
  | n + 1, s => by
    have ih := repeat_progress k w e he hk n (step k s e)
    have h1 := toward_prog k s w hk
    rw [← he] at h1
    show (k ≤ prog k w (runs k (step k s e) (List.replicate (n + 1) e)) ∨
        prog k w s + (n + 1 + 1) ≤ prog k w (runs k (step k s e) (List.replicate (n + 1) e))) ∧
      (runs k (step k s e) (List.replicate (n + 1) e)).run < k
    omega

theorem prog_lt (k : Nat) (w : Mode) (f : St) (hk : 0 < k) (hm : ¬ f.mode = w) (hr : f.run < k) :
    prog k w f < k := by
  unfold prog
  rw [if_neg hm]
  split <;> omega

theorem confirm_frames_switch (k : Nat) (w : Mode) (e : Ev) (he : ∀ s, step k s e = toward k s w)
    (hk : 0 < k) (s : St) : (runs k s (List.replicate k e)).mode = w := by
  obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
  have h := repeat_progress (n + 1) w e he hk n s
  generalize runs (n + 1) s (List.replicate (n + 1) e) = f at h ⊢
  by_cases hm : f.mode = w
  · exact hm
  · have := prog_lt (n + 1) w f hk hm h.2
    exfalso
    omega

/-- Night evidence `k` frames in a row ends in night; day evidence, in day. -/
theorem k_night_frames (k : Nat) (hk : 0 < k) (s : St) :
    (runs k s (List.replicate k .night)).mode = .night :=
  confirm_frames_switch k .night .night (fun _ => rfl) hk s

theorem k_day_frames (k : Nat) (hk : 0 < k) (s : St) :
    (runs k s (List.replicate k .day)).mode = .day :=
  confirm_frames_switch k .day .day (fun _ => rfl) hk s
