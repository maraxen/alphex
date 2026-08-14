"""The `Alphabet` value type and its declaration-time invariants.

See `.praxia/docs/specs/260814_alphabet-api-surface.md` §3.2. The design in one line: an
alphabet is a bijection between a symbol set and a contiguous integer range, plus named
exceptions at declared indices.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

import numpy as np

from abcdefghijk.errors import AlphabetDeclarationError

if TYPE_CHECKING:
  from collections.abc import Mapping


class SpecialKind(enum.Enum):
  """A non-residue index, by meaning.

  `UNKNOWN` and `STOP` are deliberately distinct: proteinsmc sets
  `STOP_INT = UNKNOWN_AA_INT = PROTEINMPNN_X_INT = 21`, one index for two meanings, and the
  point of typing them separately is that the conflation becomes reportable rather than
  invisible.
  """

  UNKNOWN = "unknown"
  GAP = "gap"
  STOP = "stop"
  MASK = "mask"
  BOS = "bos"
  EOS = "eos"
  PAD = "pad"
  CHAIN_BREAK = "chain_break"


@dataclasses.dataclass(frozen=True)
class Alphabet:
  """An ordered symbol set occupying a known integer index space.

  `symbols` is the contiguous run of *residue* symbols; `offset` is where that run starts in
  the index space. ESM's 20 canonical residues sit contiguously at indices 4-23, so they are
  expressed as `symbols=<20 chars>, offset=4` rather than as a 33-character string with
  placeholders -- which would reintroduce the "is this position a residue?" ambiguity the
  sentinel half of the shipped bug lived in.

  Equality and hashing ignore `name` and `citation`: two declarations of the same index space
  ARE the same alphabet however they are labelled. That is what makes the alias problem
  detectable instead of invisible.
  """

  symbols: str
  name: str
  citation: str
  offset: int = 0
  specials: Mapping[SpecialKind, int] = dataclasses.field(default_factory=dict)
  unclaimed: frozenset[int] = frozenset()

  declared_size: int | None = None
  """Width of the index space, when the source states one. `None` infers the smallest that fits.

  Read `size` instead: it is always an `int`. The two are separate names because they are
  separate facts -- what the source wrote, and how wide the alphabet actually is. ESM is the one
  shipped declaration that states a width (33) exceeding what its own indices imply.
  """

  padded_size: int | None = None
  """Width of a model's output axis when it exceeds `size`. `None` when there is no padding."""

  @property
  def size(self) -> int:
    """Width of the index space. Always an `int`; never `None`, whatever was declared."""
    if self.declared_size is not None:
      return self.declared_size
    return max([*self._residue_indices, *self.specials.values(), *self.unclaimed]) + 1

  @property
  def _residue_indices(self) -> range:
    return range(self.offset, self.offset + len(self.symbols))

  def __post_init__(self) -> None:
    """Validate the declaration. Every failure here is a bug caught before it can propagate."""
    if not self.symbols:
      raise AlphabetDeclarationError("symbols must be non-empty")
    if any(len(c) != 1 for c in self.symbols):
      msg = f"symbols must be single characters, got {self.symbols!r}"
      raise AlphabetDeclarationError(msg)
    if len(set(self.symbols)) != len(self.symbols):
      dupes = sorted({c for c in self.symbols if self.symbols.count(c) > 1})
      msg = f"duplicate symbols in {self.symbols!r}: {dupes}"
      raise AlphabetDeclarationError(msg)
    if not self.citation.strip():
      msg = f"alphabet {self.name!r} has no citation; a declaration without provenance is "
      msg += "how an ordering loses track of where it came from"
      raise AlphabetDeclarationError(msg)

    residue_indices = frozenset(self._residue_indices)
    special_indices = frozenset(self.specials.values())
    resolved = self.size

    if collide := residue_indices & special_indices:
      msg = f"special index collides with a residue index: {sorted(collide)}"
      raise AlphabetDeclarationError(msg)
    out_of_range = {
      i for i in residue_indices | special_indices | self.unclaimed if not 0 <= i < resolved
    }
    if out_of_range:
      msg = f"index outside [0, {resolved}): {sorted(out_of_range)}"
      raise AlphabetDeclarationError(msg)

    accounted = residue_indices | special_indices | self.unclaimed
    if unaccounted := set(range(resolved)) - accounted:
      msg = (
        f"alphabet {self.name!r} leaves index(es) {sorted(unaccounted)} unaccounted for. "
        f"Every index in [0, {resolved}) must be a residue, a special, or explicitly "
        f"unclaimed -- an index that means nothing is how a lookup table silently clamps."
      )
      raise AlphabetDeclarationError(msg)

  # --- identity -------------------------------------------------------------------

  def _identity(self) -> tuple[object, ...]:
    return (self.symbols, self.offset, frozenset(self.specials.items()), self.size)

  def __eq__(self, other: object) -> bool:
    """Compare index spaces, not labels."""
    if not isinstance(other, Alphabet):
      return NotImplemented
    return self._identity() == other._identity()

  def __hash__(self) -> int:
    """Hash the index space, so `name` cannot split one alphabet into two keys."""
    return hash(self._identity())

  # --- accessors ------------------------------------------------------------------

  @property
  def n_symbols(self) -> int:
    """Number of residue symbols, excluding specials."""
    return len(self.symbols)

  @property
  def conflated_specials(self) -> frozenset[frozenset[SpecialKind]]:
    """Groups of SpecialKinds sharing one index.

    Non-empty means two meanings are indistinguishable at runtime.
    """
    by_index: dict[int, set[SpecialKind]] = {}
    for kind, idx in self.specials.items():
      by_index.setdefault(idx, set()).add(kind)
    return frozenset(frozenset(g) for g in by_index.values() if len(g) > 1)

  def index_of(self, symbol: str) -> int:
    """Absolute index of a residue symbol."""
    return self.offset + self.symbols.index(symbol)

  def encode(self, text: str) -> np.ndarray:
    """Residue characters to absolute indices."""
    return np.array([self.index_of(c) for c in text], dtype=np.int32)

  def decode(self, codes: np.ndarray) -> str:
    """Absolute indices to residue characters."""
    return "".join(self.symbols[int(i) - self.offset] for i in np.asarray(codes).ravel())

  def lint(self) -> list[str]:
    """Return non-fatal warnings. Never raises: a dirty declaration must stay loadable.

    Refusing to load a declaration that lints dirty would recreate the pressure to declare
    something untrue, which is worse than declaring something imperfect.
    """
    warnings: list[str] = []
    for group in sorted(self.conflated_specials, key=lambda g: sorted(k.value for k in g)):
      kinds = ", ".join(sorted(k.value for k in group))
      idx = self.specials[next(iter(group))]
      warnings.append(
        f"conflated specials at index {idx}: {kinds} are indistinguishable at runtime",
      )
    return warnings
