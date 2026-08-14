---
name: using-alphex
description: Converting between biological sequence alphabet orderings without silently permuting residues — declarations, perm/convert/reindex, gather-vs-scatter direction, sentinel policy
triggers: [alphex, alphabet, ordering, residue order, ProteinMPNN, AlphaFold, restypes, ESM vocab, permutation table, remap, reindex, gap, canonical, LG order, one-hot, posterior, substitution matrix]
---

# using-alphex

`alphex` holds one declaration of each biological sequence alphabet ordering, and one way to
convert between them. Use it any time you are about to write `[order.index(c) for c in other]`,
`data[..., SOME_TABLE]`, or a dict from letters to integers.

## Why this needs a skill at all

Three amino-acid orderings are in common use:

| ordering | letters |
|---|---|
| ProteinMPNN | `ACDEFGHIKLMNPQRSTVWY` |
| AlphaFold / LG / IQ-TREE | `ARNDCQEGHILKMFPSTWYV` |
| ESM3 / ESM-C | `LAGVSERTIDPKQNFYMHWC` (at offset 4) |

ProteinMPNN and AlphaFold order share **exactly three fixed points** — `A`, `S`, `T`. Build a
table from one and label it with the other and 17 of 20 residues are permuted. The result is
shape-valid, dtype-valid, and wrong. **Nothing raises.** That bug has shipped in this ecosystem,
and a census found the three orderings declared at 30 sites across four repos under five
different names (`CANONICAL_Q20`, `MPNN_ALPHABET`, `LG_ORDER`, `AA_ORDER`, `restypes`).

The failure mode is not that conversion is hard. It is that **the wrong answer looks exactly
like the right one**, so nothing downstream can catch it.

## The first move is always `relation`

Before building any table, ask how the two alphabets stand to one another:

```python
from alphex import known, relation

rel = relation(known.MPNN_20, known.AF_20)
rel.kind        # RelationKind.PERMUTATION
rel.moved       # 17 letters
rel.lossy       # False
```

- `IDENTITY` / `EXTENSION` **and not lossy** → the conversion is free and safe.
- `PERMUTATION` → **stop and think.** This is the case that corrupts data silently.
- `INCOMPATIBLE` → a source residue has no destination counterpart; no policy fixes it.

Note `lossy` is orthogonal to `kind`: q21-with-gap → q20 is `EXTENSION`-shaped on the residues
and still drops the gap entirely. Check both.

## Pick the right one of the three converters

They are not interchangeable, and using the wrong one scrambles letter identity with no signal:

| you have | use |
|---|---|
| integer **codes** (a sequence) | `convert(codes, src, dst, *, policy)` |
| a **table** you will apply yourself | `perm(src, dst, *, policy)` |
| an **axis** indexed by the alphabet (posterior, logits, substitution matrix) | `reindex(data, src, dst, *, policy, axes=)` |

A rate or substitution matrix has **two** alphabet axes. Reindexing one and not the other is
silently transposed nonsense — pass `axes=(0, 1)`.

## The direction rule, which is the part everyone gets wrong

If you apply a table yourself as a **gather** — `out = data[..., T]`, so `out[k] == data[T[k]]` —
then to relabel that axis from alphabet A to alphabet B:

```
T = perm(B, A)          # arguments RUN OPPOSITE to the data flow
```

Because `T[k]` must be *A's* index for whatever symbol *B* puts at position `k`.

This is unintuitive exactly once and mechanical thereafter, and getting it backwards is the most
common defect found in this ecosystem — three separate instances, all invisible because the
tables involved happened to be the identity, all armed to corrupt the moment a real permutation
appeared. **When two tables are inverses, swapping them changes nothing until it changes
everything.**

If you can, avoid the question: `reindex` takes the alphabets and works out the direction.

## Never let a sentinel be decided by a default

`policy` is keyword-only with no default, everywhere. That is deliberate — the failure being
guarded against is a silent default, not an unavailable conversion.

```python
perm(known.MPNN_GAP_21, known.MPNN_X_21, policy=Policy.UNKNOWN)   # gap -> X, stated
```

- `Policy.RAISE` — refuse. `perm` raises immediately; `convert` raises only if the bad value is
  actually **present** in the data.
- `Policy.UNKNOWN` / `Policy.GAP` — route to the destination's own; errors if it declares none.
- `Policy.MASK` — returns a `MaskedPerm`, a **distinct type**, not an index array with a
  sentinel. Every in-band marker is a valid index somewhere: `-1` selects the last element of
  the destination axis, and an out-of-range value is *clamped* rather than rejected by JAX.

**Pass a mapping, not one value, when the source has several specials.** ESM declares eight. A
single uniform policy sends all eight to one index, silently — which is the many-meanings-one-
index defect this library exists to condemn:

```python
perm(known.ESM_C, known.MPNN_GAP_X_STOP_22, policy={
  SpecialKind.GAP: Policy.GAP,
  SpecialKind.UNKNOWN: Policy.UNKNOWN,
  SpecialKind.BOS: Policy.MASK,
  ...
  None: Policy.MASK,       # fallback: unclaimed indices, padding, unnamed kinds
})
```

## Test by letters, never by indices

```python
assert dst.decode(table[codes]) == src.decode(codes)      # compares CHARACTERS
```

Index equality cannot catch this bug class: the broken table and the correct one both round-trip
perfectly *within themselves*. Only decoding both sides and comparing characters distinguishes
them. When migrating existing code, additionally pin the pre-migration values literally, so a
changed number fails a test instead of moving a result.

## JAX consumers

`alphex` returns numpy and takes no JAX dependency. Build the table once at module import and
convert:

```python
TABLE = jnp.asarray(perm(CANONICAL, MPNN, policy={SpecialKind.GAP: Policy.UNKNOWN}))
```

`reindex` and `convert` are numpy and are **not** traceable — do not call them inside `jit`.
Build tables at import time; gather with them at runtime.

## Declaring an alphabet the library does not ship

```python
Alphabet(symbols="ACDE", name="...", citation="...", offset=0, specials={...}, unclaimed=...)
```

Two invariants will reject a sloppy declaration, both on purpose:

- **Totality** — every index in `[0, size)` must be a residue, a named special, or explicitly
  `unclaimed`. An index that means nothing is how a lookup table silently clamps.
- **Citation required** — an ordering without a recorded source is how one gets relabelled.

Equality ignores `name` and `citation`: two declarations of the same index space *are* the same
alphabet however they are labelled. That is what makes an alias collision detectable.

## Quick reference

```bash
alphex list                              # every shipped declaration
alphex show MPNN_GAP_21                  # symbols, specials, citation, warnings
alphex relation MPNN_20 AF_20            # classify before converting
alphex perm MPNN_GAP_21 ESM_C            # the table, plus a letter-preservation check
alphex lint                              # declarations with conflated sentinels
```

MCP tools mirror these: `list_alphabets`, `show_alphabet`, `relation`, `perm`, `lint`. Reach for
`relation` first — it is cheap, needs no policy, and answers whether you have a problem.
