import Mathlib.Data.Real.Archimedean
import Mathlib.Data.Rat.Floor
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
Formal companion to `_posts/2026-09-05-quantization-unit-tests-proof.md`.

Run from this directory with Lean 4.24.0:
  MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
  lake exe cache get Mathlib.Data.Real.Archimedean Mathlib.Data.Rat.Floor \
    Mathlib.Algebra.Order.Floor.Ring Mathlib.Algebra.BigOperators.Fin \
    Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith \
    Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
  lake build
  lake env lean QuantizationProof.lean

The code domains contain only legal codes. All arithmetic below is exact;
Int is unbounded, and Real is mathematical real arithmetic. No claim is made
about IEEE-754 evaluation, fixed-width overflow, or a particular device kernel.
The stability lemma uses explicit open rounding cells, rather than formalizing
the article's distance-to-a-set notation. Its error bounds are hypotheses, not
computed bounds for an IEEE-754 program. The file also proves the reference-test
equivalence and the conditional one-code difference bound. The final commands
report axioms.
-/

namespace QuantizationProof

/-- Equation (4). The input round trip and output injectivity are explicit. -/
theorem correct_iff_fake
    {X Y Cx Cy : Type*}
    (Qx : X → Cx) (Dx : Cx → X)
    (Qy : Y → Cy) (Dy : Cy → Y)
    (f : X → Y) (K : Cx → Cy)
    (input_roundtrip : ∀ q, Qx (Dx q) = q)
    (output_injective : Function.Injective Dy) :
    (∀ q, K q = Qy (f (Dx q))) ↔
    (∀ x, Dy (K (Qx x)) = Dy (Qy (f (Dx (Qx x))))) := by
  constructor
  · intro h x
    rw [h (Qx x)]
  · intro h q
    apply output_injective
    simpa only [input_roundtrip] using h (Dx q)

def clip (lo hi q : ℤ) : ℤ := min hi (max lo q)

theorem clip_of_legal {lo hi q : ℤ} (h : lo ≤ q ∧ q ≤ hi) :
    clip lo hi q = q := by
  simp only [clip, max_eq_right h.1, min_eq_right h.2]

/-- Fixed affine parameters, including a rounding rule that preserves integers. -/
structure Affine where
  scale : ℝ
  scale_pos : 0 < scale
  zeroPoint : ℤ
  qmin : ℤ
  qmax : ℤ
  range_nonempty : qmin ≤ qmax
  rounding : ℝ → ℤ
  rounding_int : ∀ n : ℤ, rounding (n : ℝ) = n

namespace Affine

def Code (a : Affine) := {q : ℤ // a.qmin ≤ q ∧ q ≤ a.qmax}

noncomputable def encode (a : Affine) (x : ℝ) : a.Code :=
  ⟨clip a.qmin a.qmax (a.rounding (x / a.scale + a.zeroPoint)),
    le_min a.range_nonempty (le_max_left _ _), min_le_left _ _⟩

noncomputable def decode (a : Affine) (q : a.Code) : ℝ :=
  a.scale * ((q.val : ℝ) - a.zeroPoint)

/-- Equation (1), including saturation and the legal-code restriction. -/
theorem encode_decode (a : Affine) (q : a.Code) :
    a.encode (a.decode q) = q := by
  apply Subtype.ext
  change clip a.qmin a.qmax
    (a.rounding (a.scale * ((q.val : ℝ) - a.zeroPoint) / a.scale + a.zeroPoint))
      = q.val
  have cancel_scale :
      a.scale * ((q.val : ℝ) - a.zeroPoint) / a.scale + a.zeroPoint = (q.val : ℝ) := by
    field_simp [ne_of_gt a.scale_pos]
    ring
  rw [cancel_scale, a.rounding_int]
  exact clip_of_legal q.property

theorem decode_injective (a : Affine) : Function.Injective a.decode := by
  intro u v h
  apply Subtype.ext
  have hsub : (u.val : ℝ) - a.zeroPoint = (v.val : ℝ) - a.zeroPoint :=
    mul_left_cancel₀ (ne_of_gt a.scale_pos) h
  have hcast : (u.val : ℝ) = (v.val : ℝ) := by linarith
  exact_mod_cast hcast

/-- Equation (5). -/
theorem decoded_difference (a : Affine) (u v : a.Code) :
    a.decode u - a.decode v = a.scale * ((u.val : ℝ) - v.val) := by
  simp only [decode]
  ring

/-- Fake quantization is idempotent in exact arithmetic. -/
theorem fake_idempotent (a : Affine) (x : ℝ) :
    a.decode (a.encode (a.decode (a.encode x))) = a.decode (a.encode x) := by
  rw [a.encode_decode]

/-- Equation (4) instantiated with the actual affine quantization functions. -/
theorem correct_iff_fake_affine (a b : Affine) (f : ℝ → ℝ) (K : a.Code → b.Code) :
    (∀ q, K q = b.encode (f (a.decode q))) ↔
    (∀ x, b.decode (K (a.encode x)) =
      b.decode (b.encode (f (a.decode (a.encode x))))) :=
  correct_iff_fake a.encode a.decode b.encode b.decode f K
    a.encode_decode b.decode_injective

def reluCode (a : Affine) (hz : a.qmin ≤ a.zeroPoint ∧ a.zeroPoint ≤ a.qmax)
    (q : a.Code) : a.Code :=
  ⟨max q.val a.zeroPoint,
    le_trans q.property.1 (le_max_left _ _), max_le q.property.2 hz.2⟩

theorem decode_relu (a : Affine)
    (hz : a.qmin ≤ a.zeroPoint ∧ a.zeroPoint ≤ a.qmax) (q : a.Code) :
    a.decode (a.reluCode hz q) = max (a.decode q) 0 := by
  change a.scale * (((max q.val a.zeroPoint : ℤ) : ℝ) - a.zeroPoint) =
    max (a.decode q) 0
  by_cases h : a.zeroPoint ≤ q.val
  · have hreal : (a.zeroPoint : ℝ) ≤ q.val := by exact_mod_cast h
    have hn : 0 ≤ a.decode q :=
      mul_nonneg (le_of_lt a.scale_pos) (sub_nonneg.mpr hreal)
    rw [max_eq_left h, max_eq_left hn]
    rfl
  · have hq : q.val ≤ a.zeroPoint := le_of_lt (lt_of_not_ge h)
    have hreal : (q.val : ℝ) ≤ a.zeroPoint := by exact_mod_cast hq
    have hn : a.decode q ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (le_of_lt a.scale_pos) (sub_nonpos.mpr hreal)
    rw [max_eq_right hq, max_eq_right hn]
    simp

/-- Equation (9), for every legal input and every admissible parameter choice. -/
theorem relu_correct (a : Affine)
    (hz : a.qmin ≤ a.zeroPoint ∧ a.zeroPoint ≤ a.qmax) (q : a.Code) :
    a.encode (max (a.decode q) 0) = a.reluCode hz q := by
  rw [← a.decode_relu hz q, a.encode_decode]

end Affine

open scoped BigOperators

/-- Equation (6), for arbitrary dot-product length. Integers do not overflow. -/
theorem centered_dot_expansion {p : ℕ} (A B : Fin p → ℤ) (zx zw : ℤ) :
    (∑ k, (A k - zx) * (B k - zw)) =
      (∑ k, A k * B k) - zw * (∑ k, A k) - zx * (∑ k, B k) + p * zx * zw := by
  calc
    _ = ∑ k, (A k * B k - zw * A k - zx * B k + zx * zw) := by
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib,
        ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul]
      ring

/-- The decoded dot product with bias, before output quantization. -/
theorem decoded_dot {p : ℕ} (A B : Fin p → ℤ) (zx zw c zb : ℤ) (sx sw sb : ℝ) :
    (∑ k, (sx * ((A k : ℝ) - zx)) * (sw * ((B k : ℝ) - zw))) +
        sb * ((c : ℝ) - zb) =
      sx * sw * ((∑ k, (A k - zx) * (B k - zw) : ℤ) : ℝ) +
        sb * ((c : ℝ) - zb) := by
  simp only [Int.cast_sum, Int.cast_mul, Int.cast_sub, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- The algebra inside Equation (7); applying identical rounding/clipping
to both sides preserves equality. -/
theorem requantize_argument (sx sw sb sy : ℝ) (t c zb zy : ℤ) :
    (sx * sw * t + sb * ((c : ℝ) - zb)) / sy + zy =
      (sx * sw / sy) * t + (sb / sy) * ((c : ℝ) - zb) + zy := by
  ring

/-- Equation (7), including the output quantizer, for an arbitrary matrix entry. -/
theorem matmul_reference_entry (y : Affine) {p : ℕ}
    (A B : Fin p → ℤ) (zx zw c zb : ℤ) (sx sw sb : ℝ) :
    (y.encode ((∑ k, (sx * ((A k : ℝ) - zx)) * (sw * ((B k : ℝ) - zw))) +
      sb * ((c : ℝ) - zb))).val =
    clip y.qmin y.qmax (y.rounding
      ((sx * sw / y.scale) * ((∑ k, (A k - zx) * (B k - zw) : ℤ) : ℝ) +
        (sb / y.scale) * ((c : ℝ) - zb) + y.zeroPoint)) := by
  rw [decoded_dot]
  change clip y.qmin y.qmax (y.rounding _) = _
  rw [requantize_argument]

/-- Equation (10) is an algebraic decomposition, with no error bound implied. -/
theorem error_decomposition (impl fake original : ℝ) :
    impl - original = (impl - fake) + (fake - original) := by
  ring

/-- A precise rounding-cell version of the sufficient condition in (13).
`nearest` leaves the tie rule arbitrary because every argument is strictly
inside a cell. Saturation may subsequently make further outputs equal. -/
theorem rounding_stable_in_cell (R : ℝ → ℤ)
    (nearest : ∀ (n : ℤ) (u : ℝ),
      (n : ℝ) - 1 / 2 < u → u < (n : ℝ) + 1 / 2 → R u = n)
    (n : ℤ) (t dref dimpl : ℝ)
    (href : |dref| < min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t))
    (himpl : |dimpl| < min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t)) :
    R (t + dref) = R (t + dimpl) := by
  have same (d : ℝ)
      (hd : |d| < min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t)) :
      R (t + d) = n := by
    have hlo := (lt_min_iff.mp hd).1
    have hhi := (lt_min_iff.mp hd).2
    have ha := (abs_lt.mp hlo).1
    have hb := (abs_lt.mp hhi).2
    apply nearest n (t + d) <;> linarith
  rw [same dref href, same dimpl himpl]

/-- A certified error bound below the cell margin preserves the ideal code. -/
theorem quantized_eq_of_error_bound (R : ℝ → ℤ)
    (nearest : ∀ (n : ℤ) (u : ℝ),
      (n : ℝ) - 1 / 2 < u → u < (n : ℝ) + 1 / 2 → R u = n)
    (lo hi n : ℤ) (t d ε : ℝ)
    (herror : |d| ≤ ε)
    (hmargin : ε < min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t)) :
    clip lo hi (R (t + d)) = clip lo hi (R t) := by
  have hd := lt_of_le_of_lt herror hmargin
  have hzero : |(0 : ℝ)| <
      min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t) := by
    rw [abs_zero]
    exact lt_of_le_of_lt (abs_nonneg d) hd
  have hr := rounding_stable_in_cell R nearest n t d 0 hd hzero
  simpa only [add_zero] using congrArg (clip lo hi) hr

/-- Equation (14), with the cell margin in place of distance-to-a-set notation. -/
theorem quantized_paths_eq_of_error_bounds (R : ℝ → ℤ)
    (nearest : ∀ (n : ℤ) (u : ℝ),
      (n : ℝ) - 1 / 2 < u → u < (n : ℝ) + 1 / 2 → R u = n)
    (lo hi n : ℤ) (t dref dimpl εref εimpl : ℝ)
    (href : |dref| ≤ εref) (himpl : |dimpl| ≤ εimpl)
    (hmargin : max εref εimpl <
      min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t)) :
    clip lo hi (R (t + dref)) = clip lo hi (R t) ∧
      clip lo hi (R (t + dimpl)) = clip lo hi (R t) := by
  exact ⟨quantized_eq_of_error_bound R nearest lo hi n t dref εref href
      (lt_of_le_of_lt (le_max_left _ _) hmargin),
    quantized_eq_of_error_bound R nearest lo hi n t dimpl εimpl himpl
      (lt_of_le_of_lt (le_max_right _ _) hmargin)⟩

/-- Equation (15) on a certified cell. No correctness assumption on `actual`. -/
theorem reference_test_iff_in_cell (R : ℝ → ℤ)
    (nearest : ∀ (n : ℤ) (u : ℝ),
      (n : ℝ) - 1 / 2 < u → u < (n : ℝ) + 1 / 2 → R u = n)
    (lo hi n actual : ℤ) (t dref εref : ℝ)
    (href : |dref| ≤ εref)
    (hmargin : εref < min (t - ((n : ℝ) - 1 / 2)) (((n : ℝ) + 1 / 2) - t)) :
    (actual = clip lo hi (R (t + dref))) ↔ (actual = clip lo hi (R t)) := by
  rw [quantized_eq_of_error_bound R nearest lo hi n t dref εref href hmargin]

/-- Clipping cannot enlarge an integer-code difference. -/
theorem clip_nonexpansive (lo hi a b : ℤ) :
    |clip lo hi a - clip lo hi b| ≤ |a - b| := by
  obtain ⟨hneg, hpos⟩ := abs_le.mp (le_refl |a - b|)
  have hnonneg := abs_nonneg (a - b)
  simp only [clip, min_def, max_def, abs_le]
  split_ifs <;> constructor <;> omega

/-- Equation (16). The hypothesis covers nearest rounding with any tie rule. -/
theorem quantized_difference_le_one (R : ℝ → ℤ)
    (nearest_error : ∀ x : ℝ, |(R x : ℝ) - x| ≤ 1 / 2)
    (lo hi : ℤ) (a b : ℝ) (hab : |a - b| < 1) :
    |clip lo hi (R a) - clip lo hi (R b)| ≤ 1 := by
  have ha := abs_le.mp (nearest_error a)
  have hb := abs_le.mp (nearest_error b)
  have hd := abs_lt.mp hab
  have hreal : |(((R a - R b : ℤ) : ℝ))| < 2 := by
    rw [Int.cast_sub, abs_lt]
    constructor <;> linarith [ha.1, ha.2, hb.1, hb.2, hd.1, hd.2]
  have hint : |R a - R b| < (2 : ℤ) := by exact_mod_cast hreal
  have hone : |R a - R b| ≤ (1 : ℤ) := by omega
  exact le_trans (clip_nonexpansive lo hi (R a) (R b)) hone

/-- Explicit ties-to-even: Mathlib's `round` uses a different tie convention. -/
noncomputable def nearestEven (x : ℝ) : ℤ :=
  let n := Int.floor x
  let frac := x - (n : ℝ)
  if frac < 1 / 2 then n
  else if 1 / 2 < frac then n + 1
  else if n % 2 = 0 then n else n + 1

theorem nearestEven_int (n : ℤ) : nearestEven (n : ℝ) = n := by
  norm_num [nearestEven]

/-- The explicit ties-to-even function satisfies the rounding-cell hypothesis. -/
theorem nearestEven_in_cell (n : ℤ) (u : ℝ)
    (hlo : (n : ℝ) - 1 / 2 < u) (hhi : u < (n : ℝ) + 1 / 2) :
    nearestEven u = n := by
  by_cases h : (n : ℝ) ≤ u
  · have hf : Int.floor u = n := Int.floor_eq_iff.mpr ⟨h, by linarith⟩
    have hfrac : u - (n : ℝ) < 1 / 2 := by linarith
    simp only [nearestEven, hf, if_pos hfrac]
  · have hf : Int.floor u = n - 1 := by
      apply Int.floor_eq_iff.mpr
      constructor <;> norm_num only [Int.cast_sub, Int.cast_one] <;> linarith
    have hfrac : 1 / 2 < u - ((n - 1 : ℤ) : ℝ) := by
      norm_num only [Int.cast_sub, Int.cast_one]
      linarith
    simp only [nearestEven, hf, if_neg (not_lt.mpr hfrac.le), if_pos hfrac]
    omega

/-- The concrete ties-to-even function meets the one-code theorem's hypothesis. -/
theorem nearestEven_error_le_half (x : ℝ) : |(nearestEven x : ℝ) - x| ≤ 1 / 2 := by
  have hlo := Int.floor_le x
  have hhi := Int.lt_floor_add_one x
  simp only [nearestEven]
  split_ifs <;> simp only [Int.cast_add, Int.cast_one, abs_le] <;>
    constructor <;> linarith

/-- A pre-rounding distance of exactly one can produce a two-code difference. -/
theorem one_unit_endpoint_can_differ_by_two :
    |nearestEven (3 / 2) - nearestEven (1 / 2)| = 2 := by
  norm_num [nearestEven]

/-- A concrete family of affine quantizers satisfying every structure field. -/
noncomputable def nearestEvenAffine (s : ℝ) (hs : 0 < s)
    (z lo hi : ℤ) (hrange : lo ≤ hi) : Affine where
  scale := s
  scale_pos := hs
  zeroPoint := z
  qmin := lo
  qmax := hi
  range_nonempty := hrange
  rounding := nearestEven
  rounding_int := nearestEven_int

theorem ties_even_not_translation_invariant :
    nearestEven ((1 / 2 : ℝ) + 1) ≠ nearestEven (1 / 2) + 1 := by
  norm_num [nearestEven]

/-- Exact integer-rounding facts used in (11)-(12). This does not by itself
formalize the IEEE-754 addition that produces the second argument. -/
theorem exact_sum_and_half_round_differ :
    nearestEven ((1 / 2 : ℝ) + 1 / 2 ^ 26) = 1 ∧ nearestEven (1 / 2) = 0 := by
  norm_num [nearestEven]

#print axioms correct_iff_fake
#print axioms Affine.encode_decode
#print axioms Affine.correct_iff_fake_affine
#print axioms Affine.relu_correct
#print axioms centered_dot_expansion
#print axioms decoded_dot
#print axioms requantize_argument
#print axioms matmul_reference_entry
#print axioms rounding_stable_in_cell
#print axioms quantized_eq_of_error_bound
#print axioms quantized_paths_eq_of_error_bounds
#print axioms reference_test_iff_in_cell
#print axioms clip_nonexpansive
#print axioms quantized_difference_le_one
#print axioms nearestEven_int
#print axioms nearestEven_in_cell
#print axioms nearestEven_error_le_half
#print axioms one_unit_endpoint_can_differ_by_two
#print axioms ties_even_not_translation_invariant
#print axioms exact_sum_and_half_round_differ

end QuantizationProof
