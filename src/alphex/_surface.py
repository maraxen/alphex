"""Shared logic behind the CLI and the MCP server.

Both surfaces answer the same questions and must not drift, so the answers live here once and
each surface only handles its own I/O. Everything below is plain data in, plain data out -- no
printing, no MCP types, and nothing imported that is not already a hard dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphex import known
from alphex.convert import MaskedPerm, Policy, perm
from alphex.relation import relation

if TYPE_CHECKING:
  import numpy as np

  from alphex.alphabet import Alphabet

POLICIES = tuple(p.value for p in Policy)


def names() -> list[str]:
  """Every shipped declaration name, in declaration order."""
  return [a.name for a in known.ALL]


def lookup(name: str) -> Alphabet:
  """Resolve a declaration by name.

  Raises `KeyError` listing the valid names rather than a bare miss: an agent or a shell user
  who guessed a name needs the actual set, not a diagnosis.
  """
  for alphabet in known.ALL:
    if alphabet.name == name:
      return alphabet
  msg = f"unknown alphabet {name!r}; shipped declarations are: {', '.join(names())}"
  raise KeyError(msg)


def describe(name: str) -> dict[str, Any]:
  """Everything declared about one alphabet, including its provenance and any lint warnings."""
  a = lookup(name)
  return {
    "name": a.name,
    "symbols": a.symbols,
    "n_symbols": a.n_symbols,
    "offset": a.offset,
    "size": a.size,
    "padded_size": a.padded_size,
    "specials": {kind.value: index for kind, index in sorted(a.specials.items(), key=_by_value)},
    "unclaimed": sorted(a.unclaimed),
    "citation": a.citation,
    "warnings": a.lint(),
  }


def _by_value(item: tuple[Any, int]) -> int:
  return item[1]


def catalog() -> list[dict[str, Any]]:
  """One row per shipped declaration, for a listing."""
  return [
    {
      "name": a.name,
      "symbols": a.symbols,
      "offset": a.offset,
      "size": a.size,
      "specials": {kind.value: index for kind, index in sorted(a.specials.items(), key=_by_value)},
      "warnings": len(a.lint()),
    }
    for a in known.ALL
  ]


def classify(src: str, dst: str) -> dict[str, Any]:
  """How two declarations stand to one another. Cheap, and the right thing to ask first."""
  rel = relation(lookup(src), lookup(dst))
  return {
    "src": src,
    "dst": dst,
    "kind": rel.kind.value,
    "lossy": rel.lossy,
    "moved": sorted(rel.moved),
    "n_moved": len(rel.moved),
  }


def table(src: str, dst: str, policy: str) -> dict[str, Any]:
  """Build a permutation table and report it alongside the letter-level check.

  `policy` is a single uniform value here, which is a deliberate limit of these surfaces rather
  than of the library: the per-`SpecialKind` mapping form exists so that ESM's eight specials
  cannot be collapsed onto one index by accident, and squeezing a mapping through a CLI flag or
  an MCP string argument would make that easy to do carelessly. Anything needing per-kind policy
  should call `perm` directly.
  """
  source, destination = lookup(src), lookup(dst)
  chosen = Policy(policy)
  built = perm(source, destination, policy=chosen)

  masked = isinstance(built, MaskedPerm)
  raw: np.ndarray = built.table if isinstance(built, MaskedPerm) else built
  valid = [bool(v) for v in built.valid] if isinstance(built, MaskedPerm) else None

  return {
    "src": src,
    "dst": dst,
    "policy": chosen.value,
    "masked": masked,
    "table": [int(v) for v in raw],
    "valid": valid,
    "letters_preserved": _letters_preserved(source, destination, raw),
    "relation": classify(src, dst),
  }


def _letters_preserved(src: Alphabet, dst: Alphabet, raw: np.ndarray) -> bool:
  """Check the invariant on the residues: same character, whatever the index."""
  return all(dst.symbols[int(raw[src.index_of(c)]) - dst.offset] == c for c in src.symbols)


def lint() -> list[dict[str, Any]]:
  """Report declarations carrying warnings. Empty means every shipped declaration is clean."""
  return [
    {"name": a.name, "warnings": warnings} for a in known.ALL if (warnings := a.lint())
  ]
