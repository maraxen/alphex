"""How two alphabets stand to one another, computed before any table is built.

See `.praxia/docs/specs/260814_alphabet-api-surface.md` §3.3. The point of this type is that a
caller can assert `rel.kind is EXTENSION and not rel.lossy` -- a conversion that is free and
safe -- and is forced to think when the answer is `PERMUTATION`, which is the case that silently
corrupted data.

`kind` and `lossy` are orthogonal, deliberately: `q21(gap@20) -> q20` preserves every residue
index (so it is EXTENSION-shaped) while dropping the gap entirely (so it is lossy). Folding
those into one enum would multiply the cases without adding information.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from alphex.alphabet import Alphabet


class RelationKind(enum.Enum):
  """The shape of the index mapping between two alphabets."""

  IDENTITY = "identity"
  """Same symbols at the same indices with the same specials; the table is `arange`."""

  EXTENSION = "extension"
  """Every residue and every surviving special keeps its index. Check `lossy` separately."""

  PERMUTATION = "permutation"
  """At least one residue or special sits at a different index. The dangerous case."""

  INCOMPATIBLE = "incompatible"
  """Some source residue has no counterpart in the destination. No policy can fix this."""


@dataclasses.dataclass(frozen=True)
class Relation:
  """The classification, plus the two facts a caller needs to act on it."""

  kind: RelationKind

  lossy: bool
  """A source *special* has no destination counterpart, so its meaning is unrepresentable.

  Unclaimed source indices do not count: they already denote nothing, so nothing is lost.
  """

  moved: frozenset[str]
  """Residue characters whose absolute index differs. Empty for INCOMPATIBLE.

  Residues only. A special that moves is reflected in `kind` (it forces PERMUTATION) but has no
  character to name here -- `SpecialKind.GAP` is not a symbol of the alphabet. Assert on `kind`
  when specials matter; `moved` answers "which letters shifted".
  """


def relation(src: Alphabet, dst: Alphabet) -> Relation:
  """Classify the mapping from `src` to `dst`. Pure; builds no table."""
  if frozenset(src.symbols) - frozenset(dst.symbols):
    return Relation(kind=RelationKind.INCOMPATIBLE, lossy=True, moved=frozenset())

  moved = frozenset(c for c in src.symbols if src.index_of(c) != dst.index_of(c))
  dropped = {k for k in src.specials if k not in dst.specials}
  shifted = {k for k, i in src.specials.items() if k in dst.specials and dst.specials[k] != i}

  if src == dst:
    kind = RelationKind.IDENTITY
  elif moved or shifted:
    kind = RelationKind.PERMUTATION
  else:
    kind = RelationKind.EXTENSION

  return Relation(kind=kind, lossy=bool(dropped), moved=moved)
