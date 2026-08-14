"""Amino-acid alphabet orderings and letter-preserving conversions between them.

This package exists because a silent data-corruption bug shipped: a token table built from the
AlphaFold residue ordering while named for the ProteinMPNN ordering, permuting every residue
that is not a fixed point of the permutation (only A, S and T are). Being shape-valid, it
raised nothing.

The unit of value here is the **declaration**, not the encoding. See
`.praxia/docs/specs/260814_alphabet-contract.md` for the contract and
`260814_alphabet-api-surface.md` for the shape.

v0.1 is Phase 0 (decision D4): the `Alphabet` value type and the shipped declarations, for
**dev-dependency-only conformance testing** in each consumer repo, plus the Phase 1 conversion
kernel (`relation`, `perm`, `convert`, `reindex`). No asset layer and no plugin registry (D5)
yet.
"""

from alphex import known
from alphex.alphabet import Alphabet, SpecialKind
from alphex.convert import MaskedPerm, Policy, PolicySpec, convert, perm, reindex
from alphex.errors import (
  AliasCollisionError,
  AlphabetDeclarationError,
  AlphabetError,
  DegenerateSymbolError,
  IncompatibleAlphabetError,
  MissingSpecialError,
  MultiCharTokenError,
  NonStandardResidueError,
  ReducedAlphabetError,
  UnmappableSymbolError,
  UnsupportedFeatureError,
)
from alphex.relation import Relation, RelationKind, relation

__all__ = [
  "AliasCollisionError",
  "Alphabet",
  "AlphabetDeclarationError",
  "AlphabetError",
  "DegenerateSymbolError",
  "IncompatibleAlphabetError",
  "MaskedPerm",
  "MissingSpecialError",
  "MultiCharTokenError",
  "NonStandardResidueError",
  "Policy",
  "PolicySpec",
  "ReducedAlphabetError",
  "Relation",
  "RelationKind",
  "SpecialKind",
  "UnmappableSymbolError",
  "UnsupportedFeatureError",
  "convert",
  "known",
  "perm",
  "reindex",
  "relation",
]
