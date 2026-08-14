"""FastMCP server for alphex. Install with the `mcp` extra: `uv add "alphex[mcp]"`.

Mirrors the CLI surface, and exists for one reason: an agent about to write a permutation table
by hand should be able to ask for it instead. Every silent-corruption instance this library was
built from began with someone hand-rolling a table from an ordering they had not checked.

`alphex_relation` is the one to reach for first. It is cheap, it takes no policy, and it answers
whether a conversion is free (`extension`, not lossy) or dangerous (`permutation`) before any
table exists.

Nothing here is imported by `alphex/__init__.py`, so the base install stays numpy-only.
"""

from __future__ import annotations

from typing import Any

from alphex import _surface

try:
  # ty: ignore[unresolved-import] -- cisternal requires Python >=3.13 and the `mcp` extra
  # carries a marker for it, so on a 3.11/3.12 checkout (this package's own floor is 3.11)
  # it is legitimately absent. The `except` below is the handling; the type checker cannot
  # see that and would otherwise fail every run on the interpreters we most want to support.
  import cisternal
  from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
  msg = (
    "the alphex MCP server needs the 'mcp' extra: `uv add \"alphex[mcp]\"` "
    "(or `pip install 'alphex[mcp]'`). The base install is numpy-only on purpose. "
    "If the extra IS installed and cisternal is the missing one, this interpreter is "
    "older than 3.13 -- cisternal requires >=3.13, so the extra carries a marker and "
    "skips it below that. The library and the `alphex` CLI have no such floor."
  )
  raise SystemExit(msg) from exc

app: FastMCP = FastMCP("alphex")


@cisternal.tool(registry="alphex", name="list_alphabets")
async def alphex_list_alphabets() -> list[dict[str, Any]]:
  """List every shipped alphabet declaration: name, symbols, offset, size, specials."""
  return _surface.catalog()


@cisternal.tool(registry="alphex", name="show_alphabet")
async def alphex_show_alphabet(name: str) -> dict[str, Any]:
  """Show one declaration in full, including its citation and any lint warnings.

  Args:
    name: a shipped declaration name, e.g. `MPNN_GAP_21`. Call `list_alphabets` for the set.

  """
  return _surface.describe(name)


@cisternal.tool(registry="alphex", name="relation")
async def alphex_relation(src: str, dst: str) -> dict[str, Any]:
  """Classify how two declarations relate, before building any table.

  Returns `kind` (identity, extension, permutation, incompatible), whether the conversion is
  `lossy`, and which residue letters `moved`. A `permutation` result is the case that silently
  corrupts data: the conversion is legal and the numbers will look fine either way.

  Args:
    src: name of the alphabet the data is currently indexed by.
    dst: name of the alphabet it should end up in.

  """
  return _surface.classify(src, dst)


@cisternal.tool(registry="alphex", name="perm")
async def alphex_perm(src: str, dst: str, policy: str = "raise") -> dict[str, Any]:
  """Build the permutation table from `src` to `dst`, with the letter-preservation check.

  The returned `table` is indexed by a source code and gives the destination code for the same
  symbol. Its length is the whole source domain, sentinels included -- never just the residues.

  Note `letters_preserved` in the response: that is the invariant that matters, and it compares
  characters rather than indices, which index equality cannot do.

  Args:
    src: name of the source alphabet.
    dst: name of the destination alphabet.
    policy: one of raise, unknown, gap, mask -- what to do with a symbol the destination cannot
      represent. Applied uniformly. A conversion needing a different policy per special kind
      (ESM declares eight) must call the library directly rather than through this tool.

  """
  if policy not in _surface.POLICIES:
    msg = f"policy must be one of {', '.join(_surface.POLICIES)}, got {policy!r}"
    raise ValueError(msg)
  return _surface.table(src, dst, policy)


@cisternal.tool(registry="alphex", name="lint")
async def alphex_lint() -> list[dict[str, Any]]:
  """Report shipped declarations whose conventions are worth knowing about.

  A warning is not a defect. proteinsmc really does put STOP and UNKNOWN on one index; declaring
  that is what makes it visible and therefore fixable.
  """
  return _surface.lint()


def mcp_server() -> None:
  """Entry point for the MCP server (stdio transport). Console script: `alphex-mcp`.

  The registry name must match the one on every `@cisternal.tool` above; `wire` snapshots a
  single named partition, so a mismatch registers nothing and raises nothing.

  `wire` can also mount the same callables onto a Cyclopts app, which this package does not use:
  `alphex.cli` is hand-written so it can format tables for a terminal and offer `--json-out`,
  which a generic passthrough cannot. Both surfaces call `alphex._surface`, so they answer
  identically without sharing a registration path.
  """
  cisternal.wire(app, registry="alphex")
  app.run()


if __name__ == "__main__":
  mcp_server()
