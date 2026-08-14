"""Tests for the alphabet relation classifier.

`Relation` is what lets a caller assert that a conversion is free and safe (an EXTENSION) and
be forced to think when it is a PERMUTATION -- the case that silently corrupted data.
"""

from __future__ import annotations

from alphex import Alphabet, RelationKind, SpecialKind, known, relation


def test_identical_declarations_are_identity() -> None:
  assert relation(known.MPNN_20, known.MPNN_20).kind is RelationKind.IDENTITY


def test_adding_a_sentinel_is_an_extension() -> None:
  """q=20 -> q=21 with X at 20 preserves every residue's index."""
  rel = relation(known.MPNN_20, known.MPNN_X_21)
  assert rel.kind is RelationKind.EXTENSION
  assert not rel.lossy
  assert rel.moved == frozenset()


def test_moving_the_gap_to_the_front_is_a_permutation() -> None:
  """gap-last -> gap-first shifts every residue by one. The dangerous case."""
  rel = relation(known.MPNN_GAP_21, known.MPNN_GAPFIRST_21)
  assert rel.kind is RelationKind.PERMUTATION
  assert rel.moved == frozenset(known.MPNN_20.symbols)


def test_the_two_base_orderings_are_a_permutation_moving_seventeen_symbols() -> None:
  rel = relation(known.MPNN_20, known.AF_20)
  assert rel.kind is RelationKind.PERMUTATION
  assert len(rel.moved) == 17
  assert rel.moved == frozenset(set(known.MPNN_20.symbols) - {"A", "S", "T"})


def test_dropping_the_gap_is_lossy() -> None:
  """q=21 gap-last -> q=20 keeps every residue index but loses the gap."""
  rel = relation(known.MPNN_GAP_21, known.MPNN_20)
  assert rel.lossy
  assert rel.moved == frozenset()


def test_esm_is_a_permutation_from_both_base_orderings() -> None:
  for base in (known.MPNN_20, known.AF_20):
    assert relation(base, known.ESM_C).kind is RelationKind.PERMUTATION


def test_disjoint_symbol_sets_are_incompatible() -> None:
  assert relation(known.MPNN_20, known.DNA_4).kind is RelationKind.INCOMPATIBLE


def test_a_moved_special_prevents_extension() -> None:
  """Two alphabets sharing residue indices but disagreeing on where the gap lives.

  Nothing residue-level moves, so a symbols-only classifier would call this an EXTENSION.
  It is not: the gap moved, and a caller trusting EXTENSION would misplace it.
  """
  a = Alphabet(symbols="ACDE", name="a", citation="x", specials={SpecialKind.GAP: 4})
  b = Alphabet(
    symbols="ACDE",
    name="b",
    citation="x",
    specials={SpecialKind.GAP: 5, SpecialKind.UNKNOWN: 4},
  )
  rel = relation(a, b)
  assert rel.kind is RelationKind.PERMUTATION
  assert rel.moved == frozenset()  # `moved` names residue characters; a gap is not one
