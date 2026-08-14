---
title: Alphabet library API surface (v0.1)
description: 'Full library shape: types, signatures, semantics, error hierarchy, shipped declarations, registry contract, and the test matrix that makes the invariants enforceable'
status: draft
task_id: 260814_alphabet-contract
date: '260814'
backlog_ids: ''
adversarial_review: ''
---
# Alphabet library API surface (v0.1)

Implements the contract in `260814_alphabet-contract.md` (that doc's §2 contract, §3 asset
design, §5 decisions D1-D3, and §8 feature census / abstraction ceiling). Read it first; this
doc does not restate the reasoning, only the shape.

**Nothing here is implemented.** This is the shape to agree before writing code.

---

## 0. The naming problem, resolved

`abcdefghijk` is a placeholder, and `CLAUDE.md` forbids making it load-bearing. Entry-point
**group names would do exactly that** — a downstream repo's `pyproject.toml` would hardcode the
distribution name, and renaming would silently break every registration (entry points fail by
returning nothing, not by erroring).

**Resolution: the group namespace is decoupled from the distribution name.**

```
alphabet_contract.v1.alphabets
alphabet_contract.v1.assets
```

`alphabet_contract` names the *contract*, not the package that happens to implement it, and `v1`
gives room to change the contract without breaking old registrations. Renaming the distribution
is then a metadata change with zero downstream effect. These two strings are frozen on first
release and must not be changed casually.

---

## 1. Module layout

```
src/abcdefghijk/
  __init__.py     public re-exports only; no logic
  errors.py       the exception hierarchy (imported by everything, imports nothing)
  alphabet.py     SpecialKind, Alphabet, RelationKind, Relation, relation(), Policy, perm()
  known.py        the shipped Alphabet declarations (constants only)
  asset.py        AssetKind, Asset (validates on construction)
  registry.py     entry-point discovery, alias-collision guard
  data/           ships empty, per D1
```

Dependency direction is strictly downward: `errors` ← `alphabet` ← {`known`, `asset`} ←
`registry`. No cycles, and `errors.py` must stay import-free so any module can raise.

`known.py` holds constants only — no functions. It ships **alphabet declarations** (the core
value), never data assets (D1).

---

## 2. `errors.py`

```python
class AlphabetError(Exception):
  """Base for every error this library raises."""

# --- conversion ---
class UnmappableSymbolError(AlphabetError):
  """A symbol in the source alphabet has no counterpart in the destination,
  and Policy.RAISE was requested."""

class MissingSpecialError(AlphabetError):
  """A policy named a special kind (UNKNOWN/GAP) the destination does not declare."""

class IncompatibleAlphabetError(AlphabetError):
  """The two alphabets share no symbols; no meaningful conversion exists."""

# --- declaration / registry ---
class AliasCollisionError(AlphabetError):
  """Two declarations describe the same index space under different names.
  This is finding F1 made enforceable."""

class AlphabetDeclarationError(AlphabetError):
  """An Alphabet's own fields are inconsistent (duplicate symbols, a special
  index outside the index space, an index that is neither symbol, special,
  nor explicitly unclaimed)."""

class AssetInvariantError(AlphabetError):
  """An Asset's data violates the invariants its kind requires."""

# --- deliberate refusals (contract doc §8.6) ---
class UnsupportedFeatureError(AlphabetError):
  """Base for features this library refuses to model. Each subclass's message
  MUST name what a caller would register or do instead, so the refusal is an
  extension point rather than a dead end."""

class DegenerateSymbolError(UnsupportedFeatureError): ...
class NonStandardResidueError(UnsupportedFeatureError): ...
class ReducedAlphabetError(UnsupportedFeatureError): ...
class MultiCharTokenError(UnsupportedFeatureError): ...
```

Every `UnsupportedFeatureError` subclass carries a required `remedy: str`. Enforced by a test,
not by convention.

---

## 3. `alphabet.py`

### 3.1 `SpecialKind`

```python
class SpecialKind(enum.Enum):
  UNKNOWN     = "unknown"      # X — a residue is present, identity undetermined
  GAP         = "gap"          # - — no residue at this position (alignment)
  STOP        = "stop"         # * — translation stop
  MASK        = "mask"         # a masked position (model input)
  BOS         = "bos"
  EOS         = "eos"
  PAD         = "pad"
  CHAIN_BREAK = "chain_break"  # | — ESM multimer separator
```

`UNKNOWN` and `STOP` are distinct kinds. This is the fix for correction C3: proteinsmc sets
`STOP_INT = UNKNOWN_AA_INT = 21`, and the library must be able to *express* that (so proteinsmc
can declare what it truly has) while making it visible.

### 3.2 `Alphabet`

```python
@dataclasses.dataclass(frozen=True)
class Alphabet:
  symbols:   str                            # contiguous run of residue symbols
  name:      str
  provenance: str                           # citation / upstream source, required
  offset:    int = 0                        # index of symbols[0] in the index space
  specials:  Mapping[SpecialKind, int] = <empty frozen mapping>
  unclaimed: frozenset[int] = frozenset()   # indices this declaration deliberately does not describe
  size:      int | None = None              # index-space size; defaults to the smallest that fits
  padded_size: int | None = None            # metadata only, never affects behaviour
```

**Why `symbols` + `offset` rather than one full string.** Correction C2: ESM's 20 residues are
contiguous at offset 4, so `symbols="LAGVSERTIDPKQNFYMHWC", offset=4` describes it exactly. A
single string would force placeholder characters for the specials and re-introduce the
"is this position a residue?" ambiguity that the sentinel bug lived in.

**Validated in `__post_init__`**, raising `AlphabetDeclarationError`:

- `symbols` non-empty, all single characters, **no duplicates**
- every `specials` value in `[0, size)`
- no special index collides with a residue index (`offset <= i < offset+len(symbols)`)
- **totality**: every index in `[0, size)` is exactly one of — a residue, a special, or in
  `unclaimed`. This is contract §2.3's totality invariant enforced at declaration, which is
  the point at which the original length-20-table bug was introducible.
- `provenance` non-empty (a declaration without a citation is how `PROXIDE_ORDER` happened)

Two `SpecialKind`s **may** share an index (proteinsmc's reality). Not an error; surfaced by:

```python
@property
def conflated_specials(self) -> frozenset[frozenset[SpecialKind]]:
  """Groups of special kinds sharing one index. Non-empty means two meanings
  are indistinguishable at runtime (correction C3)."""
```

Other members:

```python
@property
def n_symbols(self) -> int              # len(symbols)
def index_of(self, symbol: str) -> int  # absolute index; KeyError if absent
def symbol_at(self, index: int) -> str  # residue char, or the SpecialKind's value
def decode(self, codes: np.ndarray) -> str
def encode(self, text: str) -> np.ndarray
def lint(self) -> list[str]             # non-fatal warnings, incl. conflated specials
def __eq__ / __hash__                   # by (symbols, offset, specials, size) — NOT by name
```

**Equality ignores `name` and `provenance`.** Two declarations with the same index space *are*
the same alphabet however they are labelled. This is what makes F1's alias problem detectable
rather than invisible, and it is what the registry's collision guard tests.

### 3.3 `RelationKind` and `Relation`

```python
class RelationKind(enum.Enum):
  IDENTITY     = "identity"      # same symbols at same indices; perm is arange
  EXTENSION    = "extension"     # dst preserves every shared symbol's index
  PERMUTATION  = "permutation"   # same symbol set, indices differ
  INCOMPATIBLE = "incompatible"  # no shared symbols

@dataclasses.dataclass(frozen=True)
class Relation:
  kind:  RelationKind
  lossy: bool               # some src symbol has no dst counterpart (narrowing)
  moved: frozenset[str]     # symbols whose absolute index differs

def relation(src: Alphabet, dst: Alphabet) -> Relation: ...
```

`kind` and `lossy` are orthogonal: `q21(gap@20) -> q20` is `EXTENSION`-shaped on the shared
prefix but `lossy=True`. Keeping them separate avoids a combinatorial enum.

This is the type contract doc §8.7 most wanted. It lets a caller or test assert
`relation(a, b).kind is RelationKind.EXTENSION and not relation(a, b).lossy` — a conversion that
is free and safe — and be forced to think when the answer is `PERMUTATION`, which is the case
that silently corrupted data.

### 3.4 `Policy` and `perm`

```python
class Policy(enum.Enum):
  RAISE   = "raise"    # UnmappableSymbolError on any symbol dst lacks
  UNKNOWN = "unknown"  # map to dst's UNKNOWN; MissingSpecialError if undeclared
  GAP     = "gap"      # map to dst's GAP; MissingSpecialError if undeclared
  DROP    = "drop"     # map to DROP_SENTINEL; caller must mask

DROP_SENTINEL: int = -1

def perm(src: Alphabet, dst: Alphabet, *, policy: Policy,
         dtype: np.dtype = np.int32) -> np.ndarray: ...
```

- `policy` is **keyword-only and has no default.** Contract §2.3. Generalises
  `dca_alphabet.py`'s stated philosophy: *"the failure mode this guards against is a silent
  default, not an unavailable conversion."*
- Returns `np.ndarray`, `dtype=np.int32`, **shape `(src.size,)`** — indexed by a source code,
  giving the destination code for the same symbol.
- **Total by construction.** Length is `src.size`, not `src.n_symbols`, so no legal source
  index can fall outside the table. This is precisely the shipped bug's second half — a
  length-20 table over a 21-valued domain, where a JAX gather clamped the gap to Valine — made
  structurally impossible.
- Specials map to the destination's **same kind**, when declared; otherwise per `policy`.
- `unclaimed` source indices resolve per `policy` (they denote nothing this declaration knows).
- Raises `IncompatibleAlphabetError` when `relation().kind is INCOMPATIBLE`, regardless of
  policy — a policy chooses how to handle *missing* symbols, not whether the conversion is
  meaningful at all.
- **numpy out, deliberately.** No jax dependency (contract §4). JAX consumers call
  `jnp.asarray(perm(...))`; the result is a small constant table, so this costs nothing.

```python
def reindex(data: np.ndarray, src: Alphabet, dst: Alphabet, *, policy: Policy,
            axes: int | tuple[int, ...]) -> np.ndarray: ...
```

The **output-axis** counterpart, for posteriors and matrices whose axis is indexed by an
alphabet rather than holding codes. Contract §2.4: input relabel and output reindex are distinct
operations and usually both needed; omitting either silently scrambles letter identity. Both are
exported side by side so the pairing is visible at the call site.

`axes` is **required** — see Q1 in §10. A rate matrix reindexed on one axis and not the other is
silently transposed nonsense, and this library exists because an index operation was applied
without the caller declaring what they meant.

`Asset.reindexed()` is the one place `axes` may be omitted, because `AssetKind` determines it:
`SIMILARITY`/`RATE` reindex both axes, `FREQUENCY`/`PROPERTY` their only one. The default is
derived from a declaration rather than assumed, which is the distinction that matters.

---

## 4. `known.py` — shipped declarations

One declaration per row, each with `provenance`. These replace the 29 scattered sites.

| constant | symbols | offset | specials | provenance |
|---|---|---|---|---|
| `MPNN_20` | `ACDEFGHIKLMNPQRSTVWY` | 0 | — | ProteinMPNN (Dauparas 2022), MIT |
| `MPNN_X_21` | `ACDEFGHIKLMNPQRSTVWY` | 0 | `UNKNOWN:20` | proxide `chem/conversion.py:16` |
| `MPNN_GAP_21` | `ACDEFGHIKLMNPQRSTVWY` | 0 | `GAP:20` | asr `alphabet.py:8` |
| `MPNN_GAPFIRST_21` | `ACDEFGHIKLMNPQRSTVWY` | 1 | `GAP:0` | asr `pdz_utils.py:8` (Potts/DCA) |
| `MPNN_X_STOP_22` | `ACDEFGHIKLMNPQRSTVWY` | 0 | `UNKNOWN:21, STOP:21` | proteinsmc `constants.py:115-117` |
| `AF_20` | `ARNDCQEGHILKMFPSTWYV` | 0 | — | AlphaFold `residue_constants`, Apache-2.0 |
| `AF_X_21` | `ARNDCQEGHILKMFPSTWYV` | 0 | `UNKNOWN:20` | proxide `chem/conversion.py:17` |
| `AF_GAP_21` | `ARNDCQEGHILKMFPSTWYV` | 0 | `GAP:20` | asr `blosum_utils.py:32` (`PROXIDE_ORDER`) |
| `ESM_C` | `LAGVSERTIDPKQNFYMHWC` | 4 | `BOS:0, PAD:1, EOS:2, UNKNOWN:24, STOP:29, GAP:30, CHAIN_BREAK:31, MASK:32` | ESM3/ESM-C `SEQUENCE_VOCAB`, MIT |
| `DNA_4` | `ACGT` | 0 | — | proteinsmc `constants.py:40` |

`MPNN_X_STOP_22` deliberately declares the conflation (C3) rather than hiding it, so
`.conflated_specials` reports `{{UNKNOWN, STOP}}` and `lint()` warns. Declaring reality is what
lets it be fixed; refusing to declare it would leave proteinsmc outside the system.

`ESM_C.size = 33`, `padded_size = 64`, and `unclaimed = {3, 25, 26, 27, 28}` — index 3 is the
tokenizer-level `<unk>` (distinct from residue-level `X` at 24, and pure tokenizer machinery we
do not model per §8.5) and 25-28 are `B U Z O`, refused per §8.6. Every index is then accounted
for, so totality holds.

**Aliases are *not* shipped as separate constants.** `LG_ORDER`, `CANONICAL_Q20`, `restypes`,
`AA_ORDER` all resolve to `AF_20` or `MPNN_20`. Adding an alias constant would recreate F1
inside the library that exists to remove it.

---

## 5. `asset.py`

```python
class AssetKind(enum.Enum):
  SIMILARITY = "similarity"   # BLOSUM, PAM — scores
  RATE       = "rate"         # JTT, LG, WAG — continuous-time Markov generator
  FREQUENCY  = "frequency"    # equilibrium frequencies pi
  PROPERTY   = "property"     # per-symbol scalars (hydrophobicity, volume)

@dataclasses.dataclass(frozen=True)
class Asset:
  data:       np.ndarray
  alphabet:   Alphabet
  kind:       AssetKind
  name:       str
  provenance: str            # required
  sha256:     str | None = None
  rtol:       float = 1e-6
```

Validated in `__post_init__` (contract §8.6: at registration, not first use), raising
`AssetInvariantError`:

| kind | invariants |
|---|---|
| `SIMILARITY` | 2-D, shape `(size, size)`, symmetric within `rtol` |
| `RATE` | 2-D, shape `(size, size)`, **rows sum to 0** within `rtol`, off-diagonals `>= 0` |
| `FREQUENCY` | 1-D, length `size`, all `>= 0`, **sums to 1** within `rtol` |
| `PROPERTY` | 1-D, length `size` |

`size` is `alphabet.size` — the full index space, not `n_symbols` — because asr's BLOSUM62 is
21x21 over a 21-index alphabet including gap.

```python
def detailed_balance(rate: Asset, freq: Asset, *, rtol: float = 1e-6) -> bool: ...
```

Checks `pi_i Q_ij == pi_j Q_ji`. **Separate from `__post_init__`**, because it needs two assets
and a rate matrix is independently valid without its frequencies to hand.

```python
def reindexed(self, dst: Alphabet, *, policy: Policy) -> "Asset": ...
```
Returns the asset with its axes reindexed to `dst` — one axis for 1-D kinds, both for 2-D — and
re-validates. This is what makes asr's runtime `PROXIDE_ORDER -> ASR_ORDER` remap a declared
operation instead of hand-rolled index arithmetic.

**Ships no data (D1).** `data/` is empty. asr keeps ownership of BLOSUM62/JTT/LG and registers
them; §6 is the mechanism.

---

## 6. `registry.py`

```python
ALPHABET_GROUP = "alphabet_contract.v1.alphabets"
ASSET_GROUP    = "alphabet_contract.v1.assets"

def alphabets(*, refresh: bool = False) -> Mapping[str, Alphabet]: ...
def assets(*, refresh: bool = False) -> Mapping[str, Asset]: ...
def register_alphabet(alphabet: Alphabet) -> None: ...   # in-process, for tests and notebooks
def register_asset(asset: Asset) -> None: ...
```

- Discovery via `importlib.metadata.entry_points(group=...)` — stdlib, zero dependencies.
- **Lazy and cached.** Nothing is loaded until first call; `refresh=True` re-scans.
- Result includes `known.py`'s declarations plus everything registered.
- **Alias-collision guard**: registering an `Alphabet` that `==` an existing one under a
  different `name` raises `AliasCollisionError`, naming both. This is F1 enforced — the check
  that turns "a sixth name for an existing ordering" from a silent convention into a failure.
  Re-registering an identical name *and* index space is a no-op, so repeated imports are safe.
- A plugin whose entry point raises on load propagates the error rather than being skipped
  silently. A broken registration must be loud.

Downstream registration, e.g. asr — **assets only**:

```toml
[project.entry-points."alphabet_contract.v1.assets"]
jtt      = "asr.jtt_model:JTT_ASSET"
lg       = "asr.lg_model:LG_ASSET"
blosum62 = "asr.blosum_utils:BLOSUM62_ASSET"
```

**asr registers no alphabets, and that is the collision guard working as designed.** Its Potts
ordering (`-ACDEFGHIKLMNPQRSTVWY`, `pdz_utils.py:8`) *is* `known.MPNN_GAPFIRST_21` — same
symbols, same offset, same specials — so registering it as `potts_gap_first` would raise
`AliasCollisionError`, correctly: that is a sixth name for an ordering that already has one,
which is exactly F1. asr imports `MPNN_GAPFIRST_21` instead.

Since `known.py` ships all seven census orderings plus ESM and DNA, the expected steady state is
that **no consumer registers an alphabet at all** — the group exists for genuinely new orderings
(a new model's vocabulary, a non-protein symbol set), not for renaming existing ones. If a
consumer finds itself wanting to register an alphabet, the first question is whether `known.py`
already has it under another name.

---

## 7. `__init__.py`

Re-exports and nothing else:

```
Alphabet, SpecialKind, Relation, RelationKind, relation, Policy, perm, reindex,
Asset, AssetKind, detailed_balance,
registry (module), known (module),
AlphabetError and every subclass
```

---

## 8. Test matrix

Tests are as much the deliverable as the code — every contract invariant maps to a named test.
`hypothesis` for the property tests, over pairs drawn from `known.py` plus generated alphabets.

| # | test | asserts | why |
|---|---|---|---|
| T1 | `test_round_trip` | `perm(b,a)[perm(a,b)[x]] == x` for every compatible pair, all `x` | conversion is invertible where lossless |
| T2 | `test_letter_preserved` | `b.decode(perm(a,b)[codes]) == a.decode(codes)` — **compares characters** | contract §2.3; index equality is *not* the check that catches the bug |
| T3 | `test_totality` | `len(perm(a,b)) == a.size` for every pair; no index unmapped | the length-20-over-21-domain bug, made impossible |
| T4 | `test_alias_identity` | `MPNN_20.symbols == "ACDEFGHIKLMNPQRSTVWY"`, `AF_20.symbols == "ARNDCQEGHILKMFPSTWYV"`, and `MPNN_20 != AF_20` | **F1 enforced**: a sixth alias fails the suite instead of becoming a convention |
| T5 | `test_esm_is_third_ordering` | `ESM_C.symbols` equals neither, is a permutation of the 20, and shares exactly 1 fixed point with each | correction C1, locked so a future edit cannot quietly collapse it |
| T6 | `test_relation_classification` | `MPNN_20 -> MPNN_X_21` is `EXTENSION, not lossy`; `MPNN_GAP_21 -> MPNN_GAPFIRST_21` is `PERMUTATION`; `MPNN_GAP_21 -> MPNN_20` is `lossy` | the safety distinction is checkable |
| T7 | `test_policy_required` | `perm(a, b)` without `policy` is a `TypeError` | no silent default, ever |
| T8 | `test_policy_semantics` | each `Policy` member behaves as specified; `MissingSpecialError` when the destination lacks the kind | refusals are correct, not incidental |
| T9 | `test_declaration_totality` | an `Alphabet` with an index that is neither symbol, special, nor unclaimed raises | totality at declaration time |
| T10 | `test_conflated_specials` | `MPNN_X_STOP_22.conflated_specials == {{UNKNOWN, STOP}}` and `lint()` warns | correction C3 visible, not hidden |
| T11 | `test_asset_invariants` | each `AssetKind`'s violations raise `AssetInvariantError`: asymmetric SIMILARITY, RATE rows not summing to 0, FREQUENCY not summing to 1, wrong length | §8.4 — the invariants a bare `(data, alphabet)` pair would have discarded |
| T12 | `test_detailed_balance` | true for a constructed reversible pair, false for a perturbed one | verifies the check itself, per the BATHOS synthetic-invariant rule |
| T13 | `test_alias_collision` | registering an `==` alphabet under a new name raises `AliasCollisionError`; same name + space is a no-op | F1 at the registry boundary |
| T14 | `test_refusals` | each `UnsupportedFeatureError` subclass raises where specified and its `remedy` is non-empty | refusals are extension points, not dead ends |
| T15 | `test_no_jax_import` | importing the package does not import `jax` | the dependency contract, mechanically |
| T16 | `test_asr_parity` | for every conversion asr's `alphabet_reconcile.perm_int_between` performs, this library agrees | the cherry-pick is behaviour-preserving, not a rewrite (D2's real gate) |

T16 is the one that decides whether D2's migration is safe. It runs against asr's existing
implementation as the reference, in the spirit of the ESM-C port's parity test — a migration is
graded, not assumed.

---

## 9. Explicitly out of scope for v0.1

| excluded | reason |
|---|---|
| biotite bridge | D3 — defer until a caller exists; `pyproject.toml`'s `jax` extra models the lazy shape |
| any shipped matrix data | D1 — asr keeps ownership and registers |
| degenerate/ambiguity symbols | §8.6, 0 instances — `DegenerateSymbolError` |
| nonstandard residues (U, O) | §8.6, 0 uses — `NonStandardResidueError` |
| reduced alphabets / morphisms | §8.6, 1 instance and it is a subset predicate — `ReducedAlphabetError` |
| codons / multi-char tokens | §8.6, 1 instance — `MultiCharTokenError` |
| case / soft-masking | 0 instances |
| sequence objects, alignment, I/O, scoring | contract §2.5 |
| jax anywhere at runtime | contract §4 |
| the asset **sidecar file format** (contract §3.2) | moot while `data/` ships empty (D1). `Asset.provenance` + `Asset.sha256` carry the same information in-process. Specify the on-disk sidecar only if this library ever ships data. |
| `importlib.resources` loading | same reason — nothing to load. The rule that it must be used *instead of* `__file__` arithmetic (finding F6) still stands for whenever that changes. |

---

## 10. Shape questions, resolved

All three are local and cheap to reverse; resolved rather than left open so implementation is
unblocked.

**Q1 — `reindex` on 2-D data: axes must be named explicitly.**

```python
def reindex(data, src, dst, *, policy: Policy, axes: int | tuple[int, ...]) -> np.ndarray
```

`axes` is required, not defaulted to `-1` or to "all of them". A substitution matrix needs both
axes; a posterior needs its last only; a rate matrix reindexed on one axis and not the other is
silently transposed nonsense. The whole library exists because an index operation was applied
without the caller declaring what they meant, so a convenient default here would contradict its
premise. Consistent with `policy` having no default.

**Q2 — `perm` returns `int32`, with an explicit `dtype` parameter.**

```python
def perm(src, dst, *, policy: Policy, dtype: np.dtype = np.int32) -> np.ndarray
```

`int32` by default because the table must hold destination indices up to at least 63 (ESM padded)
and `int8` would be a silent-overflow hazard on any future larger vocabulary. The `dtype`
parameter exists because the census's sequences are `int8`, so `table[codes]` yields `int32` and
a caller round-tripping into an `int8` buffer would otherwise need a cast they might forget.
Passing `dtype=np.int8` is safe for every alphabet in `known.py` — all have `size <= 64` — and
the constructor validates that `dst.size - 1` fits the requested dtype, raising rather than
wrapping.

**Q3 — `MPNN_X_STOP_22` ships, with `lint()` warning.**

The alternative — keeping `known.py` free of defective declarations — sounds principled and is
worse in practice. proteinsmc's `STOP_INT = UNKNOWN_AA_INT = 21` is a fact about a live consumer.
If the library refuses to express it, proteinsmc cannot declare its alphabet at all, and stays
outside the system that exists to make its convention explicit. Worse, it would have to keep its
own declaration, which is an eighth site.

The library's job is to make conventions **declarable and inspectable**, not to certify them as
good. So it ships, `conflated_specials` reports `{{UNKNOWN, STOP}}`, and `lint()` warns — which is
strictly more visibility than the status quo, where the conflation is three consecutive
assignments in `constants.py:115-117` that nothing flags.

Corollary for `lint()`: it returns warnings and never raises. A declaration that lints dirty is
still a valid declaration. Refusing to load it would recreate the pressure to declare something
untrue.
