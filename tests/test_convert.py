"""Tests for the conversion kernel: perm, convert, reindex.

These are the tests the whole library exists for. Two carry the most weight:

  * `test_letter_is_preserved_*` compares CHARACTERS, not indices. Index equality is precisely
    the check that cannot catch the bug class this library addresses.
  * `test_table_covers_the_whole_source_domain` pins the length-of-src.size rule, which makes
    the original "length-20 table over a 21-valued domain" defect structurally impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from abcdefghijk import (
  Alphabet,
  IncompatibleAlphabetError,
  MaskedPerm,
  MissingSpecialError,
  Policy,
  SpecialKind,
  UnmappableSymbolError,
  convert,
  known,
  perm,
  reindex,
)

LOSSLESS_PAIRS = [
  (known.MPNN_20, known.AF_20),
  (known.AF_20, known.MPNN_20),
  (known.MPNN_20, known.ESM_C),
  (known.MPNN_GAP_21, known.MPNN_GAPFIRST_21),
  (known.MPNN_GAPFIRST_21, known.MPNN_GAP_21),
  (known.AF_X_21, known.MPNN_X_21),
]
"""Pairs where `src -> dst` loses nothing. Directional: see `INVERTIBLE_PAIRS`."""

INVERTIBLE_PAIRS = [p for p in LOSSLESS_PAIRS if known.ESM_C not in p]
"""Pairs that are lossless in BOTH directions.

`MPNN_20 -> ESM_C` is injective but not surjective: ESM declares eight specials with no
counterpart in a bare q=20 alphabet, so the return trip is lossy and `Policy.RAISE` refuses it.
Losslessness is a property of a direction, not of a pair, and conflating the two is how a
"round-trips fine" claim gets made about a conversion that only round-trips one way.
"""


@pytest.mark.parametrize(("src", "dst"), LOSSLESS_PAIRS)
def test_letter_is_preserved_across_lossless_conversions(src: Alphabet, dst: Alphabet) -> None:
  """The invariant. Decode both sides and compare characters."""
  table = perm(src, dst, policy=Policy.RAISE)
  codes = src.encode(src.symbols)
  assert dst.decode(table[codes]) == src.decode(codes)


@pytest.mark.parametrize(("src", "dst"), INVERTIBLE_PAIRS)
def test_round_trip_is_the_identity_on_residues(src: Alphabet, dst: Alphabet) -> None:
  there = perm(src, dst, policy=Policy.RAISE)
  back = perm(dst, src, policy=Policy.RAISE)
  codes = src.encode(src.symbols)
  assert np.array_equal(back[there[codes]], codes)


@pytest.mark.parametrize(("src", "dst"), LOSSLESS_PAIRS)
def test_table_covers_the_whole_source_domain(src: Alphabet, dst: Alphabet) -> None:
  """Length is src.size, never src.n_symbols.

  A table shorter than the domain is what let a JAX gather clamp the gap index to Valine.
  """
  assert perm(src, dst, policy=Policy.RAISE).shape == (src.size,)


def test_policy_is_required() -> None:
  with pytest.raises(TypeError):
    perm(known.MPNN_20, known.AF_20)  # type: ignore[call-arg]


def test_raise_policy_rejects_an_unmappable_symbol() -> None:
  """MPNN_GAP_21 -> MPNN_20 drops the gap; RAISE refuses to guess."""
  with pytest.raises(UnmappableSymbolError, match="gap"):
    perm(known.MPNN_GAP_21, known.MPNN_20, policy=Policy.RAISE)


def test_gap_policy_needs_a_gap_in_the_destination() -> None:
  with pytest.raises(MissingSpecialError, match="gap"):
    perm(known.MPNN_GAP_21, known.MPNN_X_21, policy=Policy.GAP)


def test_unknown_policy_routes_the_gap_to_the_destination_unknown() -> None:
  table = perm(known.MPNN_GAP_21, known.MPNN_X_21, policy=Policy.UNKNOWN)
  gap_index = known.MPNN_GAP_21.specials[SpecialKind.GAP]
  assert int(table[gap_index]) == known.MPNN_X_21.specials[SpecialKind.UNKNOWN]


def test_mask_policy_returns_a_distinct_type_not_an_index_array() -> None:
  """The fix for the in-band `-1` sentinel.

  `-1` is a valid numpy index selecting the last element, so a dropped symbol would silently
  become whatever sits at the end of the destination axis. A different type cannot be used as
  an index array by accident.
  """
  result = perm(known.MPNN_GAP_21, known.MPNN_20, policy=Policy.MASK)
  assert isinstance(result, MaskedPerm)
  assert not isinstance(result, np.ndarray)
  gap_index = known.MPNN_GAP_21.specials[SpecialKind.GAP]
  assert not result.valid[gap_index]
  assert result.valid[:20].all()


def test_per_special_policy_does_not_collapse_every_sentinel_onto_one_index() -> None:
  """ESM declares eight specials. A single uniform policy sends them all to one index.

  That is the many-meanings-one-index defect this library condemns, so the policy is keyed
  per SpecialKind.
  """
  table = perm(
    known.ESM_C,
    known.MPNN_GAP_X_STOP_22,
    policy={
      SpecialKind.GAP: Policy.GAP,
      SpecialKind.UNKNOWN: Policy.UNKNOWN,
      SpecialKind.STOP: Policy.UNKNOWN,
      SpecialKind.BOS: Policy.MASK,
      SpecialKind.EOS: Policy.MASK,
      SpecialKind.PAD: Policy.MASK,
      SpecialKind.MASK: Policy.MASK,
      SpecialKind.CHAIN_BREAK: Policy.MASK,
      None: Policy.MASK,
    },
  )
  assert isinstance(table, MaskedPerm)
  gap_out = int(table.table[known.ESM_C.specials[SpecialKind.GAP]])
  unk_out = int(table.table[known.ESM_C.specials[SpecialKind.UNKNOWN]])
  assert gap_out == known.MPNN_GAP_X_STOP_22.specials[SpecialKind.GAP]
  assert unk_out == known.MPNN_GAP_X_STOP_22.specials[SpecialKind.UNKNOWN]
  assert gap_out != unk_out


def test_an_integer_policy_pins_an_explicit_destination_index() -> None:
  table = perm(known.MPNN_GAP_21, known.MPNN_20, policy={SpecialKind.GAP: 0, None: Policy.RAISE})
  assert int(table[known.MPNN_GAP_21.specials[SpecialKind.GAP]]) == 0


def test_the_esm_return_trip_is_refused_under_raise() -> None:
  """The asymmetry behind `INVERTIBLE_PAIRS`, asserted rather than left as a comment."""
  with pytest.raises(UnmappableSymbolError):
    perm(known.ESM_C, known.MPNN_20, policy=Policy.RAISE)


def test_dtype_that_cannot_hold_the_destination_raises_rather_than_wrapping() -> None:
  """No shipped alphabet triggers this, which is why the destination here is synthetic.

  Every real declaration fits in `int8` (`ESM_C` is the widest at 33). The check is defensive:
  it exists so that a future wide alphabet fails loudly instead of wrapping into a valid-looking
  index.
  """
  wide = Alphabet(
    symbols=known.MPNN_20.symbols + "".join(chr(0x100 + i) for i in range(110)),
    name="synthetic_130",
    citation="test fixture; no real alphabet is this wide",
  )
  assert wide.size == 130
  with pytest.raises(ValueError, match="dtype"):
    perm(known.MPNN_20, wide, policy=Policy.RAISE, dtype=np.int8)


def test_a_non_integer_dtype_is_refused() -> None:
  with pytest.raises(ValueError, match="dtype"):
    perm(known.MPNN_20, known.AF_20, policy=Policy.RAISE, dtype=np.float32)


@pytest.mark.parametrize("dst", [known.AF_20, known.ESM_C])
def test_dtype_is_honoured_when_it_fits(dst: Alphabet) -> None:
  """`ESM_C` is included deliberately: size 33 does fit in int8, and the test says so."""
  table = perm(known.MPNN_20, dst, policy=Policy.RAISE, dtype=np.int8)
  assert table.dtype == np.int8


def test_a_conflated_source_index_resolving_two_ways_is_refused() -> None:
  """`MPNN_GAP_X_STOP_22` puts UNKNOWN and STOP both at 21; ESM separates them (24 and 29).

  One source index cannot carry two destinations. Picking either silently would be the shipped
  bug's exact shape, so the conflation surfaces here instead of being resolved by dict order.
  """
  with pytest.raises(UnmappableSymbolError, match="conflates"):
    perm(known.MPNN_GAP_X_STOP_22, known.ESM_C, policy=Policy.RAISE)


def test_convert_rejects_a_code_outside_the_source_domain() -> None:
  with pytest.raises(ValueError, match=r"\[0, 20\)"):
    convert(np.array([0, 20]), known.MPNN_20, known.AF_20, policy=Policy.RAISE)


def test_incompatible_alphabets_raise_regardless_of_policy() -> None:
  for policy in (Policy.RAISE, Policy.UNKNOWN, Policy.MASK):
    with pytest.raises(IncompatibleAlphabetError):
      perm(known.MPNN_20, known.DNA_4, policy=policy)


# --- convert: the data-dependent form ------------------------------------------------------


def test_convert_raises_only_when_the_bad_value_is_actually_present() -> None:
  """`perm` cannot express this: it is a pure function of two alphabets.

  This is the mechanism `asr/dca_alphabet.py` implements and the kernel previously lacked.
  """
  gapless = known.MPNN_GAP_21.encode("ACDE")
  assert convert(gapless, known.MPNN_GAP_21, known.MPNN_20, policy=Policy.RAISE) is not None

  with_gap = np.append(gapless, known.MPNN_GAP_21.specials[SpecialKind.GAP])
  with pytest.raises(UnmappableSymbolError):
    convert(with_gap, known.MPNN_GAP_21, known.MPNN_20, policy=Policy.RAISE)


def test_convert_preserves_letters() -> None:
  codes = known.MPNN_20.encode("ACDEFGHIKLMNPQRSTVWY")
  out = convert(codes, known.MPNN_20, known.AF_20, policy=Policy.RAISE)
  assert known.AF_20.decode(out) == "ACDEFGHIKLMNPQRSTVWY"


def test_convert_masks_rather_than_emitting_a_usable_index() -> None:
  codes = np.array([known.MPNN_GAP_21.specials[SpecialKind.GAP], 0])
  out = convert(codes, known.MPNN_GAP_21, known.MPNN_20, policy=Policy.MASK)
  assert isinstance(out, np.ma.MaskedArray)
  assert bool(out.mask[0])
  assert not bool(out.mask[1])


# --- reindex: the output-axis form ---------------------------------------------------------


def test_reindex_moves_an_axis_into_the_destination_ordering() -> None:
  """A per-symbol posterior in MPNN order, reindexed to AlphaFold order."""
  posterior = np.eye(known.MPNN_20.size)
  out = reindex(posterior, known.MPNN_20, known.AF_20, policy=Policy.RAISE, axes=-1)
  for i, symbol in enumerate(known.MPNN_20.symbols):
    landed = int(np.argmax(out[i]))
    assert known.AF_20.symbols[landed] == symbol


def test_reindex_requires_axes_to_be_named() -> None:
  posterior = np.eye(known.MPNN_20.size)
  with pytest.raises(TypeError):
    reindex(posterior, known.MPNN_20, known.AF_20, policy=Policy.RAISE)  # type: ignore[call-arg]


def test_reindex_handles_both_axes_of_a_matrix() -> None:
  matrix = np.arange(20 * 20).reshape(20, 20).astype(float)
  out = reindex(matrix, known.MPNN_20, known.AF_20, policy=Policy.RAISE, axes=(0, 1))
  for i, si in enumerate(known.MPNN_20.symbols):
    for j, sj in enumerate(known.MPNN_20.symbols):
      assert out[known.AF_20.index_of(si), known.AF_20.index_of(sj)] == matrix[i, j]


def test_reindex_rejects_a_mismatched_axis_length() -> None:
  """The padded-vocabulary hazard, made an error instead of a silent truncation."""
  wrong = np.zeros((5, 7))
  with pytest.raises(ValueError, match="axis"):
    reindex(wrong, known.MPNN_20, known.AF_20, policy=Policy.RAISE, axes=-1)


def test_reindex_refuses_when_two_source_positions_land_on_one_destination() -> None:
  """A scatter that overwrites is data loss wearing a successful return value.

  Routing the gap to X while X already receives the source's own X sends two distinct rows onto
  one destination row. `perm` and `convert` tolerate this -- many-to-one is a legitimate
  relabelling -- but reindex would silently discard whichever row is written first.
  """
  data = np.arange(3 * 22, dtype=float).reshape(3, 22)
  policy = {
    SpecialKind.GAP: Policy.UNKNOWN,
    SpecialKind.STOP: Policy.UNKNOWN,
    None: Policy.RAISE,
  }
  assert perm(known.MPNN_GAP_X_STOP_22, known.MPNN_X_21, policy=policy).tolist()[20:] == [20, 20]
  with pytest.raises(UnmappableSymbolError, match="overwrite"):
    reindex(data, known.MPNN_GAP_X_STOP_22, known.MPNN_X_21, policy=policy, axes=-1)


def test_reindex_accepts_a_padded_axis_when_padded_size_is_declared() -> None:
  """ESM output heads are 64 wide while the vocabulary is 33."""
  padded = np.zeros((2, known.ESM_C.padded_size))
  out = reindex(
    padded,
    known.ESM_C,
    known.MPNN_20,
    policy={None: Policy.MASK, **dict.fromkeys(SpecialKind, Policy.MASK)},
    axes=-1,
  )
  assert out.shape == (2, known.MPNN_20.size)
