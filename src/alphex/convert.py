"""The conversion kernel: `perm` (table), `convert` (data), `reindex` (output axis).

See `.praxia/docs/specs/260814_alphabet-api-surface.md` §3.4-3.5. Three entry points because
three genuinely different things get converted, and applying the wrong one silently scrambles
letter identity:

  * `perm`   -- build the lookup table. A pure function of two alphabets.
  * `convert` -- relabel *codes*. Raises only when a bad value is actually present.
  * `reindex` -- move a *distribution or matrix axis* that is indexed by an alphabet.

`policy` is keyword-only with no default everywhere. Generalising `asr/dca_alphabet.py`'s stated
philosophy: the failure mode being guarded against is a silent default, not an unavailable
conversion.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Union

import numpy as np

from alphex.alphabet import SpecialKind
from alphex.errors import (
  IncompatibleAlphabetError,
  MissingSpecialError,
  UnmappableSymbolError,
)
from alphex.relation import RelationKind, relation

if TYPE_CHECKING:
  from collections.abc import Mapping

  from alphex.alphabet import Alphabet


class Policy(enum.Enum):
  """What to do with a source index the destination cannot represent."""

  RAISE = "raise"
  """Refuse. `perm` raises immediately; `convert` raises only if the value is present."""

  UNKNOWN = "unknown"
  """Route to the destination's UNKNOWN. `MissingSpecialError` if it declares none."""

  GAP = "gap"
  """Route to the destination's GAP. `MissingSpecialError` if it declares none."""

  MASK = "mask"
  """Mark the position invalid rather than giving it any index at all."""


PolicySpec = Union[Policy, "Mapping[SpecialKind | None, Policy | int]"]
"""A bare `Policy` applies uniformly; a mapping keys per `SpecialKind`.

The `None` key is the fallback, used for two things: a source index that denotes nothing (an
`unclaimed` index, or padding beyond `size` on an axis of length `padded_size`), and any
`SpecialKind` the mapping does not name explicitly. An `int` value pins a destination index.

A mapping is verbose at an ESM call site, which is the correct amount of friction by this
library's own premise: a single `Policy` would send all eight of ESM's specials to one
destination index, silently -- the many-meanings-one-index defect this library condemns.
"""

_OK = 0
_MASKED = 1
_ERROR = 2


@dataclasses.dataclass(frozen=True, eq=False)
class MaskedPerm:
  """A permutation table with holes. Returned whenever any position resolves under MASK.

  A distinct type, not an `ndarray` with an in-band sentinel, because every in-band choice is a
  valid index somewhere: `-1` selects the last element of the destination axis, and any
  non-negative value is an index by definition. A caller cannot pass this where an index array
  is expected without noticing.

  `eq=False` keeps the dataclass from generating `__eq__`/`__hash__` over its ndarray fields,
  which would raise on comparison and be unhashable. Identity semantics are correct here.
  """

  table: np.ndarray
  """Destination index per source index. Values where `valid` is False are **meaningless**.

  They are set out of range on purpose, so a numpy gather with them raises `IndexError` rather
  than returning an element. That is a backstop, not the protection: JAX *clamps* out-of-range
  gathers instead of raising, which is the original bug. The protection is the type.
  """

  valid: np.ndarray
  """Bool, same shape as `table`."""


def _policy_for(policy: PolicySpec, kind: SpecialKind | None) -> Policy | int:
  """Resolve the policy governing one source special (or `None` for a meaningless index)."""
  if isinstance(policy, Policy):
    return policy
  if kind in policy:
    return policy[kind]
  if None in policy:
    return policy[None]
  return Policy.RAISE


def _special_index(dst: Alphabet, kind: SpecialKind, requested: Policy) -> int:
  """Look up a routing target in `dst`, or explain why it cannot be routed there."""
  if kind not in dst.specials:
    msg = (
      f"policy {requested.value!r} routes to the destination's {kind.value}, but alphabet "
      f"{dst.name!r} declares no {kind.value}. Choose a destination that has one, pin an "
      f"explicit index with an int policy, or use Policy.MASK."
    )
    raise MissingSpecialError(msg)
  return dst.specials[kind]


def _resolve_one(
  src_index: int,
  kind: SpecialKind | None,
  dst: Alphabet,
  policy: PolicySpec,
) -> tuple[int, int, str]:
  """Resolve one non-residue source index to `(target, status, reason)`.

  A special whose kind the destination also declares maps straight across; `policy` governs only
  what the destination genuinely cannot represent.
  """
  if kind is not None and kind in dst.specials:
    return (dst.specials[kind], _OK, "")

  chosen = _policy_for(policy, kind)
  what = f"{kind.value} at source index {src_index}" if kind else f"source index {src_index}"

  if isinstance(chosen, int):
    if not 0 <= chosen < dst.size:
      msg = f"policy pins index {chosen} for {what}, outside [0, {dst.size}) of {dst.name!r}"
      raise ValueError(msg)
    return (chosen, _OK, "")
  if chosen is Policy.UNKNOWN:
    return (_special_index(dst, SpecialKind.UNKNOWN, chosen), _OK, "")
  if chosen is Policy.GAP:
    return (_special_index(dst, SpecialKind.GAP, chosen), _OK, "")
  if chosen is Policy.MASK:
    return (0, _MASKED, "")
  return (0, _ERROR, f"{what} has no counterpart in {dst.name!r} and policy is RAISE")


def _resolve(
  src: Alphabet,
  dst: Alphabet,
  policy: PolicySpec,
  length: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
  """Build the raw table and per-index status over `[0, length)`.

  `length` is `src.size` for codes, or `src.padded_size` for a padded output axis; indices at or
  beyond `src.size` are padding and denote nothing, so they resolve under the `None` policy
  exactly like an `unclaimed` index.
  """
  if relation(src, dst).kind is RelationKind.INCOMPATIBLE:
    absent = sorted(frozenset(src.symbols) - frozenset(dst.symbols))
    msg = (
      f"{src.name!r} -> {dst.name!r}: residues {absent} have no counterpart in the destination. "
      f"This library permutes index spaces; it never merges or drops residues, so no policy "
      f"makes this conversion meaningful."
    )
    raise IncompatibleAlphabetError(msg)

  table = np.zeros(length, dtype=np.int64)
  status = np.zeros(length, dtype=np.int8)
  reasons: dict[int, str] = {}

  for c in src.symbols:
    table[src.index_of(c)] = dst.index_of(c)

  by_index: dict[int, list[SpecialKind]] = {}
  for kind, i in src.specials.items():
    by_index.setdefault(i, []).append(kind)

  residues = range(src.offset, src.offset + src.n_symbols)
  for i in range(length):
    if i in residues:
      continue
    kinds = by_index.get(i)
    outcomes = {_resolve_one(i, k, dst, policy) for k in (kinds or [None])}
    if len(outcomes) > 1:
      shared = ", ".join(sorted(k.value for k in kinds or []))
      msg = (
        f"source index {i} of {src.name!r} conflates {shared}, and they resolve differently in "
        f"{dst.name!r}. One index cannot carry two destinations: pin one with an int policy."
      )
      raise UnmappableSymbolError(msg)
    target, state, reason = outcomes.pop()
    table[i], status[i] = target, state
    if reason:
      reasons[i] = reason

  return table, status, reasons


def _check_dtype(dtype: np.dtype | type, dst: Alphabet) -> None:
  """Refuse a dtype that cannot hold every destination index plus the out-of-range filler."""
  resolved = np.dtype(dtype)
  if resolved.kind not in "iu":
    msg = f"dtype must be an integer type, got {resolved}"
    raise ValueError(msg)
  largest = int(np.iinfo(resolved.name).max)
  if largest < dst.size:
    msg = (
      f"dtype {resolved} holds at most {largest}, too small for {dst.name!r} "
      f"whose indices run to {dst.size - 1}. A narrower dtype would wrap silently."
    )
    raise ValueError(msg)


def _fill_masked(table: np.ndarray, status: np.ndarray, dst: Alphabet) -> np.ndarray:
  """Overwrite masked entries with an out-of-range value. See `MaskedPerm.table`."""
  filled = table.copy()
  filled[status == _MASKED] = dst.size
  return filled


def perm(
  src: Alphabet,
  dst: Alphabet,
  *,
  policy: PolicySpec,
  dtype: np.dtype | type = np.int32,
) -> np.ndarray | MaskedPerm:
  """Build the lookup table from source codes to destination codes.

  Shape is `(src.size,)`, never `(src.n_symbols,)`. That is the shipped bug's second half made
  structurally impossible: a length-20 table over a 21-valued domain, where a JAX gather clamped
  the gap index to Valine.

  Returns a `MaskedPerm` rather than an `ndarray` when any position resolves under
  `Policy.MASK`.
  """
  _check_dtype(dtype, dst)
  table, status, reasons = _resolve(src, dst, policy, src.size)
  if (status == _ERROR).any():
    raise UnmappableSymbolError(reasons[int(np.argmax(status == _ERROR))])

  out = _fill_masked(table, status, dst).astype(dtype)
  masked = status == _MASKED
  if masked.any():
    return MaskedPerm(table=out, valid=~masked)
  return out


def convert(
  codes: np.ndarray,
  src: Alphabet,
  dst: Alphabet,
  *,
  policy: PolicySpec,
) -> np.ndarray | np.ma.MaskedArray:
  """Relabel an array of source codes into destination codes.

  Unlike `perm`, this raises only when an unmappable value is **actually present**, which is the
  mechanism `asr/dca_alphabet.py` implements and the reason `perm` alone is insufficient.

  The *return type* still depends only on `(src, dst, policy)`, never on the data: if the table
  has any masked entry the result is a `MaskedArray` whether or not one was hit. Only the
  raising is data-dependent.
  """
  values = np.asarray(codes)
  if values.size and (int(values.min()) < 0 or int(values.max()) >= src.size):
    msg = (
      f"codes must lie in [0, {src.size}) for {src.name!r}, got "
      f"[{int(values.min())}, {int(values.max())}]"
    )
    raise ValueError(msg)

  table, status, reasons = _resolve(src, dst, policy, src.size)
  hit = status[values]
  if (hit == _ERROR).any():
    offending = int(values.ravel()[int(np.argmax((hit == _ERROR).ravel()))])
    raise UnmappableSymbolError(reasons[offending])

  out = _fill_masked(table, status, dst)[values].astype(np.int32)
  if (status == _MASKED).any():
    return np.ma.MaskedArray(out, mask=hit == _MASKED)
  return out


def _axis_length(data: np.ndarray, axis: int, src: Alphabet) -> int:
  """Validate one axis against the source's declared width, padded or not."""
  n = data.shape[axis]
  if n == src.size or (src.padded_size is not None and n == src.padded_size):
    return n
  expected = str(src.size)
  if src.padded_size is not None:
    expected += f" or {src.padded_size} (padded)"
  msg = f"axis {axis} has length {n}; {src.name!r} requires {expected}"
  raise ValueError(msg)


def reindex(  # noqa: PLR0913 -- six parameters, none removable; see the docstring for each
  data: np.ndarray,
  src: Alphabet,
  dst: Alphabet,
  *,
  policy: PolicySpec,
  axes: int | tuple[int, ...],
  fill: float = 0.0,
) -> np.ndarray:
  """Move one or more axes indexed by `src` into `dst`'s ordering.

  The output-axis counterpart of `convert`: use this for a posterior, a frequency vector, or a
  substitution matrix, whose axis *is* the alphabet rather than holding codes. Applying the
  wrong one of the two silently scrambles letter identity, which is why both are exported side
  by side.

  `axes` is required. A rate matrix reindexed on one axis and not the other is silently
  transposed nonsense, and this library exists because an index operation was applied without
  the caller declaring what they meant.

  Destination positions no source position lands on are filled with `fill`. The default of `0`
  is right for a probability vector and **wrong for anything in log space** -- pass `-np.inf`
  there rather than letting a zero mean "certain".
  """
  values = np.asarray(data)
  wanted = (axes,) if isinstance(axes, int) else tuple(axes)
  normalised = tuple(ax % values.ndim for ax in wanted)
  lengths = [_axis_length(values, ax, src) for ax in normalised]

  out = values
  for axis, length in zip(normalised, lengths, strict=True):
    table, status, reasons = _resolve(src, dst, policy, length)
    if (status == _ERROR).any():
      raise UnmappableSymbolError(reasons[int(np.argmax(status == _ERROR))])

    keep = np.flatnonzero(status == _OK)
    targets = table[keep]
    if len(np.unique(targets)) != len(targets):
      msg = (
        f"axis {axis}: several source positions of {src.name!r} land on one index of "
        f"{dst.name!r}, which would overwrite data rather than move it. Mask the duplicates "
        f"or aggregate them in the caller."
      )
      raise UnmappableSymbolError(msg)

    shape = list(out.shape)
    shape[axis] = dst.size
    moved = np.full(shape, fill, dtype=out.dtype)
    index: list[slice | np.ndarray] = [slice(None)] * out.ndim
    index[axis] = targets
    moved[tuple(index)] = np.take(out, keep, axis=axis)
    out = moved

  return out
