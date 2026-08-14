"""Tests for the Alphabet value type and its declaration-time invariants.

Every test here maps to a numbered row of the test matrix in
`.praxia/docs/specs/260814_alphabet-api-surface.md` §8.
"""

from __future__ import annotations

import pytest

from abcdefghijk import Alphabet, AlphabetDeclarationError, SpecialKind


def test_symbols_and_offset_place_residues_in_the_index_space() -> None:
  """T-core: offset shifts where the residue run begins. This is the ESM shape.

  Note the four specials below the offset. An earlier version of this test omitted them and
  the totality invariant rejected the declaration -- correctly. Leaving indices 0-3 meaning
  nothing is exactly the hole the invariant exists to close, and it caught it here first.
  """
  esm = Alphabet(
    symbols="LAGV",
    name="toy_esm",
    citation="toy",
    offset=4,
    declared_size=8,
    specials={
      SpecialKind.BOS: 0,
      SpecialKind.PAD: 1,
      SpecialKind.EOS: 2,
      SpecialKind.UNKNOWN: 3,
    },
  )
  assert esm.index_of("L") == 4
  assert esm.index_of("V") == 7
  assert esm.n_symbols == 4


def test_equality_ignores_name_and_citation() -> None:
  """T4a: two declarations of one index space ARE one alphabet, however labelled.

  This is what makes the alias problem detectable instead of invisible.
  """
  a = Alphabet(symbols="ACDE", name="canonical", citation="paper A")
  b = Alphabet(symbols="ACDE", name="mpnn", citation="paper B")
  assert a == b
  assert hash(a) == hash(b)


def test_equality_distinguishes_different_orderings() -> None:
  """T4b: the two base orderings must not compare equal."""
  mpnn = Alphabet(symbols="ACDE", name="mpnn", citation="x")
  other = Alphabet(symbols="ADCE", name="af", citation="x")
  assert mpnn != other


def test_declaration_rejects_duplicate_symbols() -> None:
  a_dup = "ACCE"
  with pytest.raises(AlphabetDeclarationError, match="duplicate"):
    Alphabet(symbols=a_dup, name="bad", citation="x")


def test_declaration_rejects_special_colliding_with_a_residue() -> None:
  with pytest.raises(AlphabetDeclarationError, match="collides"):
    Alphabet(symbols="ACDE", name="bad", citation="x", specials={SpecialKind.GAP: 2})


def test_declaration_requires_totality() -> None:
  """T9: every index must be a residue, a special, or explicitly unclaimed.

  This is the structural fix for the length-20-table-over-a-21-valued-domain half
  of the shipped bug. Index 4 here is nothing at all, so the declaration is invalid.
  """
  with pytest.raises(AlphabetDeclarationError, match="unaccounted"):
    Alphabet(
      symbols="ACDE",
      name="bad",
      citation="x",
      declared_size=6,
      specials={SpecialKind.GAP: 5},
    )


def test_declaration_accepts_explicit_unclaimed() -> None:
  """The same shape as above becomes valid once index 4 is declared unclaimed."""
  a = Alphabet(
    symbols="ACDE",
    name="ok",
    citation="x",
    declared_size=6,
    specials={SpecialKind.GAP: 5},
    unclaimed=frozenset({4}),
  )
  assert a.size == 6


def test_declaration_requires_a_citation() -> None:
  """A declaration without provenance is how `PROXIDE_ORDER` happened."""
  with pytest.raises(AlphabetDeclarationError, match="citation"):
    Alphabet(symbols="ACDE", name="bad", citation="")


def test_size_defaults_to_the_smallest_that_fits() -> None:
  a = Alphabet(symbols="ACDE", name="x", citation="x", specials={SpecialKind.GAP: 4})
  assert a.size == 5


def test_conflated_specials_reports_two_kinds_at_one_index() -> None:
  """T10: correction C3 -- proteinsmc's STOP == UNKNOWN == 21 must be visible, not hidden."""
  a = Alphabet(
    symbols="ACDE",
    name="x",
    citation="x",
    specials={SpecialKind.UNKNOWN: 4, SpecialKind.STOP: 4},
  )
  assert a.conflated_specials == frozenset({frozenset({SpecialKind.UNKNOWN, SpecialKind.STOP})})
  assert any("conflat" in w.lower() for w in a.lint())


def test_lint_is_clean_for_an_unremarkable_alphabet() -> None:
  a = Alphabet(symbols="ACDE", name="x", citation="x")
  assert a.lint() == []


def test_encode_decode_round_trip() -> None:
  a = Alphabet(
    symbols="ACDE",
    name="x",
    citation="x",
    offset=2,
    declared_size=6,
    specials={SpecialKind.BOS: 0, SpecialKind.EOS: 1},
  )
  codes = a.encode("CAD")
  assert a.decode(codes) == "CAD"
  assert codes.tolist() == [3, 2, 4]
