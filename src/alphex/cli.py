"""`alphex` command line. Install with the `cli` extra: `uv add "alphex[cli]"`.

Deliberately small. This is a library first; the CLI exists so a person or an agent can answer
"which ordering is this?" and "what does the table look like?" without writing a script -- the
two questions whose absence let a mislabelled table ship.

Nothing here is imported by `alphex/__init__.py`, so the base install stays numpy-only.
"""

from __future__ import annotations

import json
import sys

from alphex import _surface

try:
  from cyclopts import App
except ImportError as exc:  # pragma: no cover - exercised only without the extra
  msg = (
    "the alphex CLI needs the 'cli' extra: `uv add \"alphex[cli]\"` "
    "(or `pip install 'alphex[cli]'`). The base install is numpy-only on purpose."
  )
  raise SystemExit(msg) from exc

app = App(
  name="alphex",
  help="Biological sequence alphabet orderings and conversions between them.",
)


def _emit(payload: object, *, as_json: bool) -> None:
  if as_json:
    print(json.dumps(payload, indent=2))
    return
  print(_render(payload))


def _render(payload: object) -> str:
  """Plain-text rendering. No `rich` dependency -- this output is meant to be piped."""
  if isinstance(payload, list):
    return "\n".join(_render(row) for row in payload)
  if isinstance(payload, dict):
    width = max((len(k) for k in payload), default=0)
    return "\n".join(f"{k:<{width}}  {_scalar(v)}" for k, v in payload.items())
  return str(payload)


def _scalar(value: object) -> str:
  if isinstance(value, dict):
    return ", ".join(f"{k}={v}" for k, v in value.items()) or "-"
  if isinstance(value, list):
    return ", ".join(str(v) for v in value) if value else "-"
  return str(value)


@app.command
def list_(*, json_out: bool = False) -> None:
  """List every shipped alphabet declaration."""
  rows = _surface.catalog()
  if json_out:
    _emit(rows, as_json=True)
    return
  width = max(len(r["name"]) for r in rows)
  for r in rows:
    specials = ", ".join(f"{k}@{v}" for k, v in r["specials"].items()) or "-"
    flag = "  (lints)" if r["warnings"] else ""
    print(f"{r['name']:<{width}}  size={r['size']:<3} offset={r['offset']}  {specials}{flag}")


@app.command
def show(name: str, *, json_out: bool = False) -> None:
  """Show one declaration in full, including its citation."""
  _emit(_surface.describe(name), as_json=json_out)


@app.command
def relation(src: str, dst: str, *, json_out: bool = False) -> None:
  """Classify the mapping between two declarations before converting anything."""
  _emit(_surface.classify(src, dst), as_json=json_out)


@app.command
def perm(src: str, dst: str, *, policy: str = "raise", json_out: bool = False) -> None:
  """Build the permutation table from `src` to `dst`.

  `policy` is one of raise, unknown, gap, mask -- applied uniformly. Conversions needing a
  different policy per special kind must call the library directly; see `alphex._surface.table`.
  """
  if policy not in _surface.POLICIES:
    raise SystemExit(f"policy must be one of {', '.join(_surface.POLICIES)}, got {policy!r}")
  _emit(_surface.table(src, dst, policy), as_json=json_out)


@app.command
def lint(*, strict: bool = False, json_out: bool = False) -> None:
  """Report declarations whose conventions are worth knowing about.

  Exits 0 by default even when there are findings. A warning here is not a defect to fix: two
  shipped declarations conflate STOP and UNKNOWN because proteinsmc really does put both on
  index 21, and declaring that is what makes it visible. Those two will warn forever, so a
  non-zero default would be permanent noise and would train people to ignore the command.

  Pass `--strict` to exit 1 on any finding -- useful in a repo that has decided its own
  declarations must stay clean.
  """
  findings = _surface.lint()
  if not findings:
    if json_out:
      _emit([], as_json=True)
    else:
      print("no warnings")
    return
  _emit(findings, as_json=json_out)
  if strict:
    sys.exit(1)


def main() -> None:
  """Console-script entry point."""
  app()


if __name__ == "__main__":
  main()
