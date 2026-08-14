# abcdefghijk

> **Placeholder name.** Chosen at scaffold time to avoid bikeshedding; expected to be
> renamed before any consumer depends on this package.

Amino-acid alphabet orderings and letter-preserving conversions between them.

**Status: scaffold.** No API is implemented yet.

## Why

Several libraries in this ecosystem each carry their own amino-acid ordering constants, and
a silent-corruption bug has already shipped from the mismatch — a token table built from
the AlphaFold ordering while named for ProteinMPNN, permuting every residue that is not a
fixed point of the permutation (only A, S and T are). Being shape-valid, it raised nothing.

This package holds one declaration of each ordering and one way to convert between them.

## Scope

In scope: alphabet orderings, sentinel/gap placement conventions, letter-preserving
index conversions, posterior-axis reindexing.

Out of scope: scoring, structure I/O, model code, anything needing jax at runtime.

## Design

Orderings are **named values, not function names** — `perm(AF, MPNN)` rather than
`af_to_mpnn` — so that callers must declare the ordering their data is in. That declaration
is what makes the original class of bug impossible to write.

numpy is the only runtime dependency, deliberately. See `CLAUDE.md` for the full contract
and for why this is a separate distribution rather than a submodule of an existing library.
