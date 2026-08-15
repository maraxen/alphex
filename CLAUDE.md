# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A zero-to-one-dependency library of **amino-acid alphabet orderings and letter-preserving
conversions between them**. Nothing else belongs here — no scoring, no I/O, no structure
handling, no model code.

**The name is now load-bearing.** It was a placeholder (`abcdefghijk`) through 2026-08-14,
under a rule forbidding entry points, a CLI, or env-var prefixes built on it. That rule is
**spent**: renamed to `alphex` and published to PyPI as `0.1.0a1` on 2026-08-15, with four
repos (asr, proteinsmc, aminx, proxide) depending on it by name from the registry. Renaming
now means a new PyPI project and a coordinated bump across all four — treat it as a breaking
change, not a tidy-up.

The one thing still *deliberately* decoupled from the distribution name is the entry-point
group namespace, `alphabet_contract.v1.*`. It names the contract rather than the package
implementing it, and that decoupling is what made the 2026-08-14 rename free. Do not
"simplify" it to match the package name.

**Status: released, alpha.** The `Alphabet` value type, the shipped declarations, and the
conversion kernel (`relation`, `perm`, `convert`, `reindex`) are implemented and tested.
Not implemented: aliases for degenerate and non-standard residues (`B`, `Z`, `J`, `U`, `O`),
the substitution/rate-matrix asset layer, and the plugin registry — all v0.2, all recorded
in the specs. The base install is numpy-only and must stay that way; the CLI and MCP
surfaces live behind the `cli` / `mcp` extras and are imported by neither `__init__.py` nor
each other's module.

## Why it exists

Four repos in this ecosystem (proteinsmc, aminx, asr, proxide) each carry their own
amino-acid ordering constants, and at least one silent-corruption bug has already shipped
from the mismatch: a token table in proteinsmc was built from the AlphaFold ordering while
named for ProteinMPNN, so every caller honouring the documented contract had its residues
permuted. Only A, S and T are fixed points. It was shape-valid and therefore silent.

The same table was length 20 against a 21-valued domain, so a JAX gather clamped the gap
index to Valine.

See `proteinsmc/.praxia/docs/research/260813_alphabet-provenance-trace.md` and
`asr/.praxia/docs/research/260813_g2-esm-metric-delta.md` for the full history.

## The census (established 2026-08-14, verified by execution)

The ecosystem does **not** have four or five orderings. It has **two base orderings under
five names**, plus gap-placement variants:

| ordering | known aliases |
|---|---|
| `ARNDCQEGHILKMFPSTWYV` | proxide `AF_ALPHABET[:20]`, asr `LG_ORDER`, proteinsmc `restypes` |
| `ACDEFGHIKLMNPQRSTVWY` | proxide `MPNN_ALPHABET[:20]`, asr `CANONICAL_Q20`, `PROTEINMPNN_RESTYPES` |

asr's "LG order" is byte-identical to AlphaFold order. That alias collision is the hazard
this package exists to remove.

Sentinel/gap placement varies independently of the base ordering:

| convention | shape | where |
|---|---|---|
| X at 20 | q=21 | proxide `MPNN_ALPHABET`, `AF_ALPHABET` |
| gap at 20 | q=21 | asr `CANONICAL_ALPHABET` |
| gap at 0 | q=21 | asr Potts order (`pdz_utils.py`) |
| X at 21, stop at 21 | q=22 | proteinsmc `PROTEINMPNN_X_INT` |

## The design contract

**Orderings are named values, not function names.** Write `perm(AF, MPNN)`, never
`af_to_mpnn`. Two reasons:

1. Pairwise functions need *n²* of them, and n is already at least 4 with gap variants.
2. A pairwise function name does not force the caller to *declare* the ordering their data
   is in. Declaration is the property that would have made the original bug impossible:
   the broken table could not have been built without naming its source alphabet, and
   naming it would have exposed the mismatch immediately.

**Every conversion is letter-preserving.** The amino acid at each position is invariant;
only the integer index changes. Any function that can change which residue a position
denotes is out of scope.

**Every mapping is total over its declared domain.** No out-of-range index may reach a
gather and clamp. A sentinel outside the source alphabet resolves explicitly — to a
declared unknown value — and the table is sized to cover the full domain including
sentinels. The original bug's second half was a length-20 table serving a 21-valued domain.

**Prior art to lift from, not reinvent:** `asr/src/asr/alphabet_reconcile.py` already
implements this contract well — `perm_int_between(from_order, to_order)`,
`reindex_posterior` for the output axis, `trim_gap` for q=21. It is better designed than
proxide's pairwise version and is stranded in a leaf consumer. Start there.

Note that input-side and output-side conversion are **both** required and are different
operations: feeding canonical-indexed integers to an LG-order model needs the input
relabelled *and* the output posterior reindexed. Doing only one silently scrambles letter
identity.

## The dependency contract

**numpy is the ceiling, not the floor.** Do not add jax as a runtime dependency.

This package exists because the alternative homes could not offer a cheap install.
proxide is the strongest candidate on ownership grounds — it already holds both alphabets
and both permutations in `chem/conversion.py`, it is the shared upstream for aminx, asr,
prolix and tev_design, and it has no dependency cycles. It was still rejected, for two
reasons that are worth not relitigating:

1. **Extras cannot subtract.** proxide's *base* install pulls jax, jax-md, hydride, grain
   and array_record on top of a maturin-built Rust extension. Optional-dependency extras
   only add to a base; no extras or dependency-group scheme can make
   `import proxide.chem.alphabet` cheap. Making a subset cheap requires the subset to be
   its own distribution.
2. **proteinsmc has no proxide edge**, and the ecosystem partition plan exists to *reduce*
   proteinsmc's inbound edges. Adding one to fix an alphabet bug trades the wrong way.

If jnp-backed tables are wanted, return plain integer arrays and let the consumer call
`jnp.asarray`, or gate it behind the `jax` extra with a lazy import and an actionable
`ImportError`. Never import an optional dependency at module import time.

## Intended consumers

proteinsmc, aminx, asr, proxide, prolix, tev_design. proxide should eventually adopt this
and re-export `af_to_mpnn`/`mpnn_to_af` as thin shims, so the four repos already depending
on proxide keep their existing path.

## Commands

```bash
uv sync --group dev        # install with dev tooling
uv run pytest              # tests
uv run ruff check src/ --fix
uv run ty check src/alphex
```

Use `uv run python`, never bare `python`.

## Testing expectation

This package's whole value is that it cannot silently permute. Test accordingly:

- **Round-trip property**: `perm(B, A)[perm(A, B)[x]] == x` for every ordering pair.
- **Letter preservation**: decode both sides and assert the character sequence is
  identical — not just that indices changed.
- **Totality**: every index in the declared domain, sentinels included, maps to something
  explicit. Assert no gather clamps.
- **Alias identity**: assert the known aliases really are equal (AF == LG), so a future
  divergence in the census fails a test instead of becoming a fifth convention.

Property-based tests via hypothesis are a better fit than examples here.
