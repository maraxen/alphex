"""Tests for the shared CLI/MCP surface.

`_surface` exists so the two agent-facing surfaces cannot answer the same question differently.
These tests cover it directly rather than through either surface: `alphex.cli` needs the `cli`
extra and `alphex.mcp` needs `mcp` (plus Python 3.13 for cisternal), and the base test run must
work without either. What matters is that the answers are right and that the module imports
nothing beyond the base install -- both asserted below.
"""

from __future__ import annotations

import sys

import pytest

from alphex import IncompatibleAlphabetError, Policy, SpecialKind, _surface, known


def test_surface_imports_nothing_beyond_the_base_install() -> None:
  """The base install is numpy-only, and that is the reason this package exists separately.

  `_surface` is imported by both agent surfaces, so a stray import here would leak an optional
  dependency into every consumer that installs neither extra.
  """
  leaked = [m for m in ("cyclopts", "fastmcp", "cisternal", "jax") if m in sys.modules]
  assert leaked == []


def test_catalog_lists_every_shipped_declaration() -> None:
  assert [row["name"] for row in _surface.catalog()] == [a.name for a in known.ALL]


def test_lookup_reports_the_valid_names_on_a_miss() -> None:
  """A wrong guess needs the actual set, not a diagnosis."""
  with pytest.raises(KeyError, match="MPNN_GAP_21"):
    _surface.lookup("CANONICAL")


def test_describe_carries_the_citation() -> None:
  """Provenance must survive the trip to an agent; it is the point of the declaration."""
  described = _surface.describe("AF_20")
  assert described["symbols"] == known.AF_20.symbols
  assert "AlphaFold" in described["citation"]
  assert described["size"] == 20


def test_classify_reports_the_dangerous_case() -> None:
  result = _surface.classify("MPNN_20", "AF_20")
  assert result["kind"] == "permutation"
  assert result["n_moved"] == 17
  assert not result["lossy"]


def test_table_checks_letters_not_indices() -> None:
  """`letters_preserved` is the whole point of exposing this over a raw table."""
  result = _surface.table("MPNN_GAP_21", "ESM_C", "raise")
  assert result["letters_preserved"]
  assert result["table"][20] == known.ESM_C.specials[SpecialKind.GAP]
  assert len(result["table"]) == known.MPNN_GAP_21.size
  assert not result["masked"]


def test_table_reports_masking_as_a_flag_not_a_usable_index() -> None:
  result = _surface.table("MPNN_GAP_21", "MPNN_20", "mask")
  assert result["masked"]
  assert result["valid"] is not None
  assert result["valid"][known.MPNN_GAP_21.specials[SpecialKind.GAP]] is False
  assert all(result["valid"][:20])


def test_table_refuses_an_incompatible_pair() -> None:
  with pytest.raises(IncompatibleAlphabetError):
    _surface.table("MPNN_20", "DNA_4", "raise")


def test_lint_reports_the_two_conflated_declarations_and_only_those() -> None:
  """These warn forever, on purpose: proteinsmc really does put STOP and UNKNOWN on index 21.

  Pinning the exact set means a NEW conflation shows up as a test failure rather than blending
  into expected noise.
  """
  assert sorted(f["name"] for f in _surface.lint()) == [
    "AF_GAP_X_STOP_22",
    "MPNN_GAP_X_STOP_22",
  ]


def test_every_policy_name_is_accepted_by_the_surface() -> None:
  """`POLICIES` is what both surfaces validate against; it must track the enum.

  The pair here has no specials on either side, so every policy is a no-op and all four are
  exercised. A pair with an unmappable special would make three of the four raise, which is
  correct behaviour but tests the policies rather than the plumbing.
  """
  assert set(_surface.POLICIES) == {p.value for p in Policy}
  for policy in _surface.POLICIES:
    assert _surface.table("MPNN_20", "AF_20", policy)["letters_preserved"]
