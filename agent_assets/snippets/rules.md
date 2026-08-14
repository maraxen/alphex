# Alphabet rules

Applies in any repo that declares an amino-acid or nucleotide ordering.

**Do not hand-roll a permutation table.** No `[to.index(c) for c in frm]`, no
`{aa: i for i, aa in enumerate(ALPHABET)}` used as a conversion, no `jnp.arange(21)` standing in
for "these two orderings probably match". Use `alphex.perm` / `convert` / `reindex` against a
declaration in `alphex.known`. Three orderings are in play — ProteinMPNN, AlphaFold and ESM —
and the first two share only three fixed points (`A`, `S`, `T`), so a mislabelled table permutes
17 of 20 residues while staying shape-valid and raising nothing.

**Classify before converting.** `relation(src, dst)` returns `PERMUTATION` for exactly the cases
that corrupt data silently. `EXTENSION and not lossy` is the only combination that is free.

**Direction: to relabel a gathered axis (`out = data[..., T]`) from A to B, `T = perm(B, A)`.**
The arguments run opposite to the data flow. Three separate defects in this ecosystem were the
tables swapped at their use sites — invisible every time, because the tables involved were the
identity, and armed for the first real permutation.

**A sentinel policy is never a default.** `policy` is keyword-only with no default. When the
source declares more than one special, pass a mapping keyed per `SpecialKind` — a single value
sends every special to one destination index, silently. Use `Policy.MASK` rather than an in-band
`-1` or `size`: both are valid indices somewhere, and JAX clamps rather than raising.

**Assert on letters, not indices.** `dst.decode(table[codes]) == src.decode(codes)`. Index
equality cannot distinguish a correct table from a broken one — both round-trip within
themselves. When replacing existing tables, also pin their pre-migration values literally so a
changed number fails a test rather than moving a published result.

**JAX: build tables at import, gather at runtime.** `alphex` is numpy and is not traceable.
`jnp.asarray(perm(...))` once at module level; never call `convert`/`reindex` inside `jit`.
