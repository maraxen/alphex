"""Inventory every amino-acid / nucleotide alphabet declaration in the ecosystem, via AST.

WHY THIS EXISTS

The first census was a regex for 18-25-character string literals under `src/`. It produced the
7-alphabet / 29-site figure in `.praxia/docs/specs/260814_alphabet-contract.md` §1, and that
figure is wrong in both directions:

  * A regex over string literals cannot see a `list` (missed ESM's vocabulary -> correction C1)
    or a `dict` (missed proxide's HHblits maps and `restypes_with_x_and_gap` -> correction C5).
  * It counted a docstring mention as a declaration (`aminx/.../ddg_stability.py:77`).

Two of the four refusals in §8.6 rested on counts of zero that C5 showed to be false. Since the
abstraction ceiling admits an axis on an *instance count*, the count is load-bearing and a regex
is not good enough to produce it. This script walks the AST instead, so `str`, `list` and `dict`
declarations are all visible and comments/docstrings are structurally excluded.

WHAT COUNTS AS A DECLARATION

An assignment whose value resolves to an ordered sequence of residue-or-sentinel symbols:

  * a `str` literal of >= 4 single-character symbols drawn from the symbol set
  * a `list`/`tuple` of >= 4 single-character `str` literals
  * a `dict` whose keys are >= 4 single-character `str` literals (an ordering by insertion, e.g.
    proxide's `HHBLITS_AA_TO_ID`), or whose *values* are (e.g. `ID_TO_HHBLITS_AA`)

Nucleotide alphabets (`ACGT`) are included: correction C4 established the domain is not
protein-only. Codon tables are reported separately -- their keys are 3-character, so they are a
k=3 token map, not an alphabet (`MultiCharTokenError` territory).

KNOWN LIMITATION -- read before treating any count as final

This walks *literals*. A declaration built by computation is still invisible, and at least one
real one is: `proxide/src/proxide/chem/residues.py:753`

    restypes_with_x_and_gap = [*restypes, "X", "-"]

is an AlphaFold-ordered q=22 alphabet (X@20, gap@21) that this script does NOT report, because
the list contains a starred expression rather than 22 string constants. Evaluating such
declarations would mean importing the modules, which is exactly what a numpy-only tool
cannot do for a jax/Rust-backed package.

So the counts below are a **lower bound**, tighter than the regex's but still not a total. Treat
"0 instances" from this script as "none found by literal analysis", never as "none exist" -- that
inference is what invalidated two refusals in contract §8.6 the first time.

USAGE

    uv run python scripts/census_alphabets.py --repo-root ~/projects --json outputs/census.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("census")

REPOS = ("proxide", "proteinsmc", "aminx", "asr", "prolix", "tev_design", "trex")

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
SENTINELS = set("X-*.|")
DEGENERATE = set("BZJ")          # IUPAC ambiguity: Asx, Glx, Xle
NONSTANDARD = set("UO")          # selenocysteine, pyrrolysine
NUCLEOTIDES = set("ACGTU")
SYMBOLS = AA20 | SENTINELS | DEGENERATE | NONSTANDARD | NUCLEOTIDES

MIN_SYMBOLS = 4
"""Four is the smallest real alphabet in the ecosystem (ACGT)."""

CODON_LEN = 3

MIN_CANONICAL_AA = 15
"""A protein alphabet must contain at least this many of the 20 canonical residues.

Without this floor the scan reports chemical-element key sets -- proxide's `CNOS`
(`chem/residues.py:391`), `CNOSP`, `HCNOSP`, `COHNSP` -- as alphabets, because element
symbols are drawn from the same single-character space as residues. Worse, it reports them
as NONSTANDARD, since oxygen collides with pyrrolysine (`O`) and sulfur-adjacent sets pick up
`U`. That inflated "sites with nonstandard UO" from 1 to 8 on the first run. The count feeds
the §8.6 abstraction ceiling, so a false positive here directly corrupts a design decision.
"""

NUCLEOTIDE_ONLY = set("ACGTU")
"""A declaration may also qualify as nucleotide-only, which legitimately has just 4-5 symbols."""


def _is_alphabet(order: str) -> bool:
  """Distinguish a residue/nucleotide alphabet from an incidental single-char key set."""
  letters = set(order)
  if letters <= NUCLEOTIDE_ONLY:
    return True
  return len(letters & AA20) >= MIN_CANONICAL_AA


@dataclass(frozen=True)
class Declaration:
  """One alphabet declaration found in source."""

  repo: str
  path: str
  line: int
  symbol: str          # the assigned name
  container: str       # str | list | tuple | dict-keys | dict-values
  order: str           # the symbols, in declaration order
  n: int
  has_degenerate: bool
  has_nonstandard: bool
  nucleotide: bool


def _symbols_from_str(node: ast.AST) -> str | None:
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    s = node.value
    if len(s) >= MIN_SYMBOLS and all(c in SYMBOLS for c in s):
      return s
  return None


def _symbols_from_seq(node: ast.AST) -> str | None:
  if not isinstance(node, (ast.List, ast.Tuple)):
    return None
  out = []
  for el in node.elts:
    if not (isinstance(el, ast.Constant) and isinstance(el.value, str) and len(el.value) == 1):
      return None
    if el.value not in SYMBOLS:
      return None
    out.append(el.value)
  return "".join(out) if len(out) >= MIN_SYMBOLS else None


def _single_char_strs(nodes: list[ast.AST]) -> str | None:
  out = []
  for el in nodes:
    if not (isinstance(el, ast.Constant) and isinstance(el.value, str) and len(el.value) == 1):
      return None
    if el.value not in SYMBOLS:
      return None
    out.append(el.value)
  return "".join(out) if len(out) >= MIN_SYMBOLS else None


def _symbols_from_dict(node: ast.AST) -> tuple[str, str] | None:
  """Return (container_kind, order) for a dict keyed or valued by single symbols."""
  if not isinstance(node, ast.Dict):
    return None
  keys = [k for k in node.keys if k is not None]
  by_key = _single_char_strs(keys)
  if by_key:
    return ("dict-keys", by_key)
  by_val = _single_char_strs(list(node.values))
  if by_val:
    return ("dict-values", by_val)
  return None


def _is_codon_table(node: ast.AST) -> bool:
  """Report whether this is a genuine k=3 nucleotide token map.

  Keys must be 3 characters drawn from ACGT(U). Without that restriction this also matches
  proxide's residue-name tables (`chi_angles_atoms`, `residue_atoms`, ...), whose keys are
  3-letter residue names like `ALA` -- 5 false positives on the first run.
  """
  if not isinstance(node, ast.Dict):
    return False
  keys = [k for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
  if len(keys) < MIN_SYMBOLS:
    return False
  return all(
    len(k.value) == CODON_LEN and set(k.value) <= NUCLEOTIDE_ONLY for k in keys
  )


def scan_file(path: Path, repo: str) -> tuple[list[Declaration], list[tuple[str, int, str]]]:
  """Return (declarations, codon_tables) for one file."""
  try:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
  except SyntaxError:
    logger.debug("unparseable, skipped: %s", path)
    return [], []

  decls: list[Declaration] = []
  codons: list[tuple[str, int, str]] = []

  for node in ast.walk(tree):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
      continue
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [t.id for t in targets if isinstance(t, ast.Name)]
    if not names or node.value is None:
      continue
    name = names[0]

    if _is_codon_table(node.value):
      codons.append((str(path), node.lineno, name))
      continue

    order, container = None, None
    if (s := _symbols_from_str(node.value)) is not None:
      order, container = s, "str"
    elif (s := _symbols_from_seq(node.value)) is not None:
      order, container = s, type(node.value).__name__.lower()
    elif (pair := _symbols_from_dict(node.value)) is not None:
      container, order = pair

    if order is None or not _is_alphabet(order):
      continue

    letters = set(order)
    decls.append(
      Declaration(
        repo=repo,
        path=str(path),
        line=node.lineno,
        symbol=name,
        container=container,
        order=order,
        n=len(order),
        has_degenerate=bool(letters & DEGENERATE),
        has_nonstandard=bool(letters & NONSTANDARD),
        nucleotide=letters <= NUCLEOTIDES,
      ),
    )
  return decls, codons


def main() -> int:
  """Walk each repo's src/ and report every declaration."""
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--repo-root", type=Path, default=Path.home() / "projects")
  ap.add_argument("--json", type=Path, default=None, help="write the full inventory here")
  ap.add_argument("--log-level", default="INFO")
  args = ap.parse_args()
  logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

  all_decls: list[Declaration] = []
  all_codons: list[tuple[str, int, str]] = []

  for repo in REPOS:
    src = args.repo_root / repo / "src"
    if not src.is_dir():
      logger.info("%-12s no src/, skipped", repo)
      continue
    n_before = len(all_decls)
    for py in sorted(src.rglob("*.py")):
      # skip vendored trees and build artefacts: they are copies, not declarations
      if any(p in {"vendor", "build", "__pycache__", ".venv"} for p in py.parts):
        continue
      d, c = scan_file(py, repo)
      all_decls.extend(d)
      all_codons.extend(c)
    logger.info("%-12s %3d declarations", repo, len(all_decls) - n_before)

  print(f"\n{'=' * 78}\nDISTINCT ORDERS (the thing the census is actually counting)\n{'=' * 78}")
  by_order: dict[str, list[Declaration]] = {}
  for d in all_decls:
    by_order.setdefault(d.order, []).append(d)

  for order, group in sorted(by_order.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    flags = []
    if group[0].has_degenerate:
      flags.append("DEGENERATE")
    if group[0].has_nonstandard:
      flags.append("NONSTANDARD")
    if group[0].nucleotide:
      flags.append("nucleotide")
    tail = "  " + ",".join(flags) if flags else ""
    print(f"\n  {order!r}  (q={len(order)}, {len(group)} site(s)){tail}")
    for d in group:
      rel = d.path.split("/src/", 1)[-1]
      print(f"      {d.repo:11s} {rel}:{d.line}  {d.symbol}  [{d.container}]")

  print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
  print(f"  distinct orders          : {len(by_order)}")
  print(f"  declaration sites        : {len(all_decls)}")
  print(f"  sites with degenerate BZJ: {sum(d.has_degenerate for d in all_decls)}")
  print(f"  sites with nonstandard UO: {sum(d.has_nonstandard for d in all_decls)}")
  print(f"  k=3 codon tables (not alphabets): {len(all_codons)}")
  for p, ln, nm in all_codons:
    print(f"      {p.split('/src/', 1)[-1]}:{ln}  {nm}")

  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
      json.dumps(
        {
          "declarations": [asdict(d) for d in all_decls],
          "codon_tables": [{"path": p, "line": ln, "symbol": nm} for p, ln, nm in all_codons],
          "distinct_orders": {k: len(v) for k, v in by_order.items()},
        },
        indent=2,
      ),
    )
    print(f"\nwrote {args.json}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
