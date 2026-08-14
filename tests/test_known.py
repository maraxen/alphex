"""Tests for the shipped declarations.

These are the tests that would have caught architecture-review finding #1: a `known.py` row
declaring an AlphaFold-ordered index space under a ProteinMPNN name.
"""

from __future__ import annotations

import pytest

from alphex import Alphabet, SpecialKind, known

MPNN = "ACDEFGHIKLMNPQRSTVWY"
AF = "ARNDCQEGHILKMFPSTWYV"
ESM = "LAGVSERTIDPKQNFYMHWC"


def test_the_two_base_protein_orderings_are_what_they_claim() -> None:
  """T4: literal assertion of both base orderings, so a silent edit fails the suite."""
  assert known.MPNN_20.symbols == MPNN
  assert known.AF_20.symbols == AF
  assert known.MPNN_20 != known.AF_20


def test_esm_is_a_distinct_third_ordering() -> None:
  """T5: correction C1. Locked so a future edit cannot quietly collapse it into one of the two."""
  assert known.ESM_C.symbols == ESM
  assert known.ESM_C.symbols not in (MPNN, AF)
  assert sorted(known.ESM_C.symbols) == sorted(MPNN)
  assert [c for i, c in enumerate(ESM) if MPNN[i] == c] == ["W"]
  assert [c for i, c in enumerate(ESM) if AF[i] == c] == ["K"]


def test_esm_residues_are_contiguous_at_offset_four() -> None:
  """Correction C2: offset, not an arbitrary learned vocabulary."""
  assert known.ESM_C.offset == 4
  assert known.ESM_C.index_of("L") == 4
  assert known.ESM_C.index_of("C") == 23
  assert known.ESM_C.padded_size == 64


@pytest.mark.parametrize(
  ("alphabet", "expected_base"),
  [
    (known.MPNN_20, MPNN),
    (known.MPNN_X_21, MPNN),
    (known.MPNN_GAP_21, MPNN),
    (known.MPNN_GAPFIRST_21, MPNN),
    (known.MPNN_GAP_X_STOP_22, MPNN),
    (known.MPNN_X_GAP_22, MPNN),
    (known.AF_20, AF),
    (known.AF_X_21, AF),
    (known.AF_GAP_21, AF),
    (known.AF_GAP_X_STOP_22, AF),
  ],
)
def test_every_declaration_carries_the_base_ordering_its_name_claims(
  alphabet: Alphabet, expected_base: str,
) -> None:
  """The direct guard against review finding #1 -- a name asserting the wrong ordering.

  My spec's `MPNN_X_STOP_22` row declared an AF-ordered space under an MPNN name, because it was
  named after the constant `PROTEINMPNN_X_INT` rather than after the ordering that constant
  indexes. That is the shipped bug's causal step, and this parametrisation is what makes it
  impossible to repeat silently.
  """
  assert alphabet.symbols == expected_base


def test_proteinsmc_conflation_is_declared_not_hidden() -> None:
  """T10: correction C3. Both 22-wide proteinsmc spaces share STOP and UNKNOWN at index 21."""
  for a in (known.MPNN_GAP_X_STOP_22, known.AF_GAP_X_STOP_22):
    assert a.conflated_specials == frozenset(
      {frozenset({SpecialKind.UNKNOWN, SpecialKind.STOP})},
    )
    assert a.lint(), f"{a.name} should lint dirty"


def test_proxide_q22_sentinels_are_in_the_reverse_order_to_proteinsmc() -> None:
  """A real, easily-missed hazard: two q=22 spaces whose sentinels are swapped."""
  assert known.MPNN_X_GAP_22.specials[SpecialKind.UNKNOWN] == 20
  assert known.MPNN_X_GAP_22.specials[SpecialKind.GAP] == 21
  assert known.MPNN_GAP_X_STOP_22.specials[SpecialKind.GAP] == 20
  assert known.MPNN_GAP_X_STOP_22.specials[SpecialKind.UNKNOWN] == 21


def test_every_declaration_is_valid_and_cited() -> None:
  """Construction validates, so this asserts the whole module loaded, plus real citations."""
  for a in known.ALL:
    assert isinstance(a, Alphabet)
    assert len(a.citation) > 10, f"{a.name} citation too thin: {a.citation!r}"


def test_no_two_declarations_share_an_index_space_under_different_names() -> None:
  """The synonymy half of F1, enforced inside the library that exists to remove it.

  Adding an alias constant would recreate the problem here, so this fails if anyone does.
  """
  seen: dict[Alphabet, str] = {}
  for a in known.ALL:
    if a in seen:
      msg = f"{a.name!r} and {seen[a]!r} declare the same index space"
      raise AssertionError(msg)
    seen[a] = a.name


def test_nucleotide_alphabet_exists() -> None:
  """Correction C4: the domain is not protein-only."""
  assert known.DNA_4.symbols == "ACGT"
  assert known.DNA_4.size == 4
