# alphex

Biological sequence alphabet orderings, and letter-preserving conversions between them.

```bash
uv add alphex
```

## Why

Three different amino-acid orderings are in common use, under at least five different names:

| ordering | letters | used by |
|---|---|---|
| ProteinMPNN | `ACDEFGHIKLMNPQRSTVWY` | ProteinMPNN, and most "canonical" alphabets |
| AlphaFold | `ARNDCQEGHILKMFPSTWYV` | AlphaFold `restypes`, LG/JTT substitution models, IQ-TREE |
| ESM | `LAGVSERTIDPKQNFYMHWC` | ESM3 / ESM-C, at offset 4 |

ProteinMPNN and AlphaFold order have **exactly three fixed points** — `A`, `S` and `T`. Build a
lookup table from one and label it with the other, and 17 of 20 residues are silently permuted.
The result is shape-valid, dtype-valid, and wrong; nothing raises. That bug has shipped.

An ecosystem census that motivated this package found those three orderings declared at **30
sites across four repositories**, in six different sentinel conventions (no sentinel; `X` at 20;
gap at 20; gap at 0 with residues shifted to 1; gap at 20 with `X` at 21; `X` at 20 with gap at
21). Every site was an independent opportunity to get it wrong.

`alphex` holds one declaration of each, and one way to convert between them.

## What it does

```python
from alphex import Policy, SpecialKind, known, perm

# A table from ProteinMPNN order (gap at 20) into ESM's vocabulary.
table = perm(known.MPNN_GAP_21, known.ESM_C, policy=Policy.RAISE)

table.shape        # (21,) -- the whole source domain, never just the 20 residues
int(table[0])      # 5     -- 'A'
int(table[20])     # 30    -- the gap, which ESM declares
```

Four entry points, because four genuinely different things get converted and using the wrong
one scrambles letter identity silently:

| | |
|---|---|
| `relation(src, dst)` | classify first: `IDENTITY`, `EXTENSION`, `PERMUTATION`, `INCOMPATIBLE`, plus `lossy` and which letters moved |
| `perm(src, dst, *, policy)` | build the lookup table |
| `convert(codes, src, dst, *, policy)` | relabel sequence codes |
| `reindex(data, src, dst, *, policy, axes)` | move a posterior or substitution-matrix axis |

## The design commitments

**Letters, not indices.** The invariant is that the amino acid at each position is unchanged and
only its integer moves. Index equality cannot catch this bug class — the broken table and the
correct one both round-trip perfectly within themselves.

**Every index means something.** A declaration must account for every index in `[0, size)` as a
residue, a named special, or explicitly `unclaimed`. A table shorter than its domain is how a
JAX gather clamped a gap index onto valine.

**No silent defaults.** `policy` is keyword-only and has no default, anywhere. It is keyed per
`SpecialKind`, so ESM's eight specials cannot collapse onto one destination index without you
writing that down.

**No in-band sentinels.** `Policy.MASK` returns a `MaskedPerm`, a distinct type — because every
in-band marker is a valid index somewhere. `-1` selects the last element of the destination
axis; `size` is clamped rather than rejected by JAX.

**Declarations carry provenance.** `citation` is required and must be non-empty. An ordering
without a source is how one gets relabelled.

**Names are not identity.** Two `Alphabet`s compare equal when their index spaces match,
whatever they are called. That is what makes an alias collision detectable instead of invisible.

## Dependencies

`numpy`, and nothing else. That ceiling is the reason this is its own distribution rather than a
module inside a larger library: extras can only *add* to a base install, never subtract, so the
only way to make "just the alphabets" cheap for a consumer is for it to ship separately.

JAX consumers pass the result through `jnp.asarray` — the table is a small constant.

## Status

Alpha. The value type, the shipped declarations and the conversion kernel are implemented and
tested. Not yet implemented: aliases for degenerate and non-standard residues (`B`, `Z`, `J`,
`U`, `O`), the substitution/rate-matrix asset layer, and the plugin registry for third-party
alphabets. The entry-point group names (`alphabet_contract.v1.*`) are reserved and deliberately
independent of this distribution's name.

## Provenance

The orderings are facts about published tools, cited per declaration in `known.py`: ProteinMPNN
(Dauparas et al. 2022, MIT), AlphaFold `residue_constants` (Apache-2.0), ESM3/ESM-C
`SEQUENCE_VOCAB` (MIT). No code from those projects is included or derived from.

MIT licensed.
