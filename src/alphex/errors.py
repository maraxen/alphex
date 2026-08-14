"""The exception hierarchy.

This module imports nothing, deliberately, so every other module can raise from it without
introducing a cycle. See `.praxia/docs/specs/260814_alphabet-api-surface.md` §2.
"""

from __future__ import annotations


class AlphabetError(Exception):
  """Base for every error this library raises."""


class AlphabetDeclarationError(AlphabetError):
  """An Alphabet's own fields are inconsistent.

  Duplicate symbols, a special index outside the index space, a special colliding with a
  residue, a missing citation, or an index that is neither residue, special, nor explicitly
  unclaimed. The last of those is the structural fix for the length-20-table-over-a-21-valued
  domain half of the bug this library exists to prevent.
  """


class UnmappableSymbolError(AlphabetError):
  """A source symbol has no counterpart in the destination, under Policy.RAISE."""


class MissingSpecialError(AlphabetError):
  """A policy named a SpecialKind the destination does not declare."""


class IncompatibleAlphabetError(AlphabetError):
  """The two alphabets share no symbols; no meaningful conversion exists."""


class AliasCollisionError(AlphabetError):
  """Raised in BOTH directions.

  HOMONYMY (primary, malignant) -- one name denotes two different index spaces. This is the
  failure that actually shipped: `restypes` denotes AlphaFold order in proxide and proteinsmc,
  was read as ProteinMPNN by a caller, and the resulting table was labelled MPNN.

  SYNONYMY (secondary, largely inert) -- two names denote one index space, e.g.
  `LG_ORDER == AF_20`. Worth flagging to keep the census honest, but nobody has been harmed
  by it.
  """


class UnsupportedFeatureError(AlphabetError):
  """Base for features this library refuses to model.

  Every subclass carries a `remedy` naming what a caller would register or do instead, so a
  refusal is an extension point rather than a dead end.
  """

  remedy: str = ""

  def __init__(self, message: str, remedy: str = "") -> None:
    """Record the message and the remedy that makes the refusal actionable."""
    super().__init__(message if not remedy else f"{message} Remedy: {remedy}")
    self.remedy = remedy or type(self).remedy


class DegenerateSymbolError(UnsupportedFeatureError):
  """IUPAC ambiguity codes (B, Z, J and the nucleotide set) are set-valued."""

  remedy = (
    "declare an alphabet with an `aliases` mapping (v0.2), or resolve the ambiguity "
    "before conversion; this library maps symbols to single indices and will not guess"
  )


class NonStandardResidueError(UnsupportedFeatureError):
  """U (selenocysteine) and O (pyrrolysine) are real residues outside the declared alphabets."""

  remedy = "register an alphabet whose `symbols` includes them"


class ReducedAlphabetError(UnsupportedFeatureError):
  """Many-to-one symbol morphisms (Dayhoff groups, hydrophobicity classes) are lossy."""

  remedy = "perform the grouping in the caller; this library only permutes, never merges"


class MultiCharTokenError(UnsupportedFeatureError):
  """Codons and any k>1 token."""

  remedy = "use a tokenizer; this library maps single symbols to indices"
