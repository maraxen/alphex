---
title: 'Alphabet library: abstraction, contract, assets, and acquisition decisions'
description: Census of 7 declared AA alphabets across 29 sites in 5 repos; the contract lifted from asr; asset storage + entry-point plugin design; six-way acquisition verdicts incl. biotite
status: draft
task_id: 260814_alphabet-contract
date: '260814'
backlog_ids: ''
adversarial_review: ''
---
# Alphabet library: abstraction, contract, assets, and acquisition decisions

**Method.** Every claim below is from reading or executing the named source. Alphabet strings
were extracted by regex across `src/` in proxide, proteinsmc, aminx, asr, prolix; biotite's
behaviour was verified by running it, not from documentation.

---

## 1. The census

**7 distinct declared alphabets across 29 declaration sites in 5 repos.** They are **2 base
orderings** crossed with **4 sentinel conventions** — the multiplicity is almost entirely
sentinel placement, not disagreement about residue order.

| # | string | q | base | sentinel | declared at |
|---|---|---|---|---|---|
| 1 | `ACDEFGHIKLMNPQRSTVWY` | 20 | MPNN | none | proteinsmc `constants.py:294`; asr `alphabet_reconcile.py:13,30`, `dca_alphabet.py:55` |
| 2 | `ACDEFGHIKLMNPQRSTVWYX` | 21 | MPNN | X@20 | proxide `chem/conversion.py:16`, `io/parsing/mappings.py:23`; aminx `potts/model.py:40`, `ebm/ddg_stability.py:77`, `utils/aa_convert.py:16`; asr `alphabet.py:14`, `dca_alphabet.py:46` |
| 3 | `ACDEFGHIKLMNPQRSTVWY-` | 21 | MPNN | gap@20 | asr `alphabet.py:8`, `alphabet_reconcile.py:32`, `mtt_training_pipeline.py:37` |
| 4 | `-ACDEFGHIKLMNPQRSTVWY` | 21 | MPNN | gap@0 | asr `alphabet_reconcile.py:33`, `pdz_utils.py:8`, `baselines/paml.py:19`, `dca_alphabet.py:43`, `blosum_utils.py:33` |
| 5 | `ARNDCQEGHILKMFPSTWYV` | 20 | AF/LG | none | asr `alphabet_reconcile.py:31`, `lg_model.py:17`, `jtt_model.py:47`, `felsenstein_map.py:35`, `iqtree_runner.py:22,141` |
| 6 | `ARNDCQEGHILKMFPSTWYVX` | 21 | AF/LG | X@20 | proxide `chem/conversion.py:17`, `io/parsing/mappings.py:24`; aminx `utils/aa_convert.py:17` |
| 7 | `ARNDCQEGHILKMFPSTWYV-` | 21 | AF/LG | gap@20 | asr `blosum_utils.py:32` (named `PROXIDE_ORDER`) |

Plus, outside this table, proteinsmc's `PROTEINMPNN_X_INT = 21` q=22 convention (X *and* stop
at 21) and the ESM `SEQUENCE_VOCAB` (33 tokens, residues non-contiguous, gap at 30).

### What the census establishes

**F1 — The alias problem is worse than the ordering problem.** Only two residue orderings
exist. But they travel under at least five names: base MPNN is `MPNN_ALPHABET`,
`CANONICAL_ALPHABET`, `CANONICAL_Q20`, `PROTEINMPNN_RESTYPES`; base AF is `AF_ALPHABET`,
`LG_ORDER`, `AA_ORDER`, `restypes`. Verified by execution: asr's `LG_ORDER` is byte-identical
to proxide's `AF_ALPHABET[:20]`, and `CANONICAL_Q20` to `MPNN_ALPHABET[:20]`. Two names for one
thing is how a table gets built from one and labelled the other, which is the shipped bug.

**F2 — Sentinel placement is the real variance, and it is where the damage happened.** Four
conventions (none / X@20 / gap@20 / gap@0), and the proteinsmc bug's second half was exactly a
sentinel failure: a length-20 table serving a 21-valued domain, so a JAX gather clamped the gap
index to Valine. Any contract that models only the 20 residues and treats sentinels as an
afterthought reproduces this.

**F3 — proxide declares the same pair twice, internally.** `chem/conversion.py:16-17` and
`io/parsing/mappings.py:23-24` are independent declarations of alphabets 2 and 6. Even inside
one repo there is no single source.

**F4 — There is already an undeclared cross-repo asset dependency, done by copy.** asr's
`blosum_utils.py:32` names its ordering `PROXIDE_ORDER` — the BLOSUM62 matrix was copied from
proxide in AF+gap-last order and is remapped at runtime to asr's gap-first order. Nothing
records the provenance beyond that variable name.

**F5 — asr has already solved most of this, better than any external option, and it is
stranded.** Three modules, arrived at independently:

- `alphabet_reconcile.py` — `perm_int_between(from_order, to_order)` (generic, letter-preserving),
  `reindex_posterior` (output axis), `trim_gap` (q=21→20). Its docstring already states the
  invariant we want and names the failure mode: *"~17/20 positions differ between CANONICAL_Q20
  and LG_ORDER"* with no error signal, and that input relabel and output reindex are **both**
  required — omitting either silently scrambles letter identity.
- `dca_alphabet.py` — an explicit `unknown_policy` for X with **no permissive default**:
  *"Making the caller name the policy is the point: the failure mode this guards against is a
  silent default, not an unavailable conversion."* This is the design philosophy, already written.
- `jtt_model.py` — embedded matrix constants plus the source file checked in at
  `tests/fixtures/jones.dat` with its sha256, plus a test that re-parses the source and asserts
  equality against the constants. Provenance and verification, done properly.

**F6 — asset location in this ecosystem is currently accidental.** `jtt_model.py` says its
source file lives under `tests/` *"only because this repo gitignores every directory named `data`
for large scientific inputs."* And proxide resolves assets by `Path(__file__).parent.parent /
"assets"` (`chem/gaff2.py:558,601`, `physics/force_fields/loader.py:342,366`) with a comment
still naming a long-dead package (`"Assuming module path structure: src/priox/assets"`). Neither
is a design; both are sediment.

---

## 2. The abstraction and contract

### 2.1 An alphabet is a value, not a string and not a module constant

```
Alphabet = (symbols: str, name: str, provenance: str)
```

Immutable, hashable, comparable by symbols. Two alphabets with the same symbols are the same
alphabet regardless of name — that is what makes F1's alias problem detectable instead of
invisible, and it should be asserted in tests so a sixth alias fails the suite rather than
becoming a seventh convention.

Rejected: the bare-string approach (asr, proxide, aminx all use it today). A string cannot carry
its own name or provenance, so a caller cannot be made to declare which one they hold, and the
mismatch stays silent.

### 2.2 Conversion is by declared pair, never by function name

`perm(src: Alphabet, dst: Alphabet) -> ndarray` — never `af_to_mpnn`. Two reasons, both from
the census:

1. *n* orderings need *n²* named functions; n is already 7.
2. A pairwise name lets the caller convert without ever declaring what they hold. Declaration is
   the property that makes the shipped bug unwritable: the broken table could not have been
   constructed without naming its source alphabet, and naming it would have surfaced the
   mismatch at construction.

### 2.3 Three invariants, each testable

- **Letter-preserving.** The residue at each position is invariant; only the integer changes.
  Test by decoding both sides and comparing characters — not by checking that indices changed.
- **Total over the declared domain.** Every index the source alphabet can produce, sentinels
  included, maps to something explicit. No out-of-range index may reach a gather and clamp.
  This is F2 made structural.
- **Sentinel policy is named by the caller, never defaulted.** Lifted from `dca_alphabet.py`.
  Converting gap@20 → an alphabet with no gap is not an error and not a silent drop; it is a
  question (`error` | `unknown` | `gap` | `drop`) the caller answers. A default here is the bug.

### 2.4 Both axes, or neither

Input relabel and output reindex are distinct operations and usually both needed: feeding
canonical-indexed integers to an AF-ordered model requires relabelling the input *and*
reindexing the output posterior. `alphabet_reconcile.py` already documents this; the API should
make doing only one awkward rather than natural.

### 2.5 Out of scope

Sequence objects, alignment, I/O, scoring, structure. If it needs jax at runtime it does not
belong here.

---

## 3. Assets: native storage and plugin

An asset here is any array whose axes are indexed by an alphabet — substitution matrices
(BLOSUM62, JTT, LG), exchangeabilities, equilibrium frequencies, codon tables.

### 3.1 An asset is a (data, alphabet) pair — never a bare array

This is the single most important rule, and it is the one F4 violates. A BLOSUM62 matrix without
its ordering is not data, it is a trap. This is also exactly biotite's `SubstitutionMatrix`
contract (it carries `alphabet1`/`alphabet2` and will not construct without them), which is
strong evidence the shape is right.

### 3.2 Native assets: `importlib.resources`, not `__file__` arithmetic

Ship under `src/<pkg>/data/`, load via `importlib.resources.files("<pkg>.data")`. Do **not**
copy proxide's `Path(__file__).parent.parent / "assets"` pattern — it breaks under zipimport,
is silently wrong after any module move, and has already rotted once (F6's stale `priox` path).

Each asset is a pair of files: the data, and a sidecar declaring `alphabet`, `source`,
`sha256`, and `citation`. A matrix whose sidecar's sha256 does not match fails loudly at load.
`jtt_model.py`'s discipline — checked-in source + sha256 + a test that re-derives the constants
from it — is the standard to generalise, not to leave in one module.

### 3.3 Plugin assets: entry points

```toml
[project.entry-points."<pkg>.alphabets"]
potts_gap_first = "asr.alphabets:POTTS_Q21"

[project.entry-points."<pkg>.matrices"]
jtt = "asr.jtt_model:JTT_ASSET"
```

Discovered via `importlib.metadata.entry_points()` — stdlib, zero dependencies, the standard
Python mechanism. A downstream repo registers its own orderings and matrices without a PR to
this library, and registration is declarative and inspectable.

Two guards, both from the census: registration **fails** on a symbols-collision with a
different name (F1 — that is how aliases are born), and every plugin asset must satisfy §2.3's
invariants at registration time, not at first use.

### 3.4 Why not a plugin *format*

No custom loader protocol, no plugin API version. Entry points resolve to objects satisfying
the same `Alphabet`/`Asset` contracts the core defines. Anything richer is scope the census does
not justify.

---

## 4. Acquisition decisions

Adjudicated against `cisternal:customizing-tools` `references/decision-matrix.md`
(jury-adjudicated 2026-08-13, 3 lenses). All four recorded as `customize_tools.decision` events
in cisternal telemetry, 2026-08-14.

### biotite 1.5.0 → **derive inspiration** (confidence 0.82)

Verified by execution: `Alphabet` (`encode`/`decode`/`extends`/`get_symbols`),
`AlphabetMapper` (letter-preserving — confirmed on a canonical→AF round trip),
`LetterAlphabet` (expresses all 7 of our variants), `SubstitutionMatrix` (carries its
alphabets). `ProteinSequence.alphabet`'s first 20 symbols **are** the MPNN/canonical ordering.
License BSD-3-Clause, actively maintained, widely adopted.

This is the closest external analog and its design is the pre-validated shape §2 adopts. But
`importlib.metadata.requires('biotite')` gives `biotraj, msgpack, networkx, numpy, packaging,
requests` plus Cython extensions. For a library whose job is to return a permutation array,
that is disproportionate — and `requests` in a numerical leaf's transitive closure is precisely
the risk the matrix's `Avoid when` flags. It is also *not* currently a base dependency anywhere
in the ecosystem: it sits in proxide's `trajectories` extra only.

Matrix basis: `derive_inspiration` → *"You plan to write your own implementation regardless
(legal posture or fit-to-your-callers reasons) and want the design pre-validated against a
proven shape before you start."* Exact match. Guarded against the `Avoid when` (Chesterton's
Fence): the invariants in §2.3 are the load-bearing behaviour, and each is a named test.

### biotite → **accept dependency, as an optional extra** (confidence 0.75)

A `to_biotite`/`from_biotite` bridge behind a `biotite` extra, lazily imported with an
actionable `ImportError`. Consumers who already have it get interop; the zero-dep core is
untouched. This is the "extras may only add" rule applied in the one direction it works.

### asr's three modules → **cherry-pick** (confidence 0.90)

`alphabet_reconcile.py`'s contract, `dca_alphabet.py`'s explicit-policy pattern, and
`jtt_model.py`'s provenance discipline.

The matrix rates cherry-pick weakest (2.3/5) on license/IP grounds and on the
Debian-OpenSSL-PRNG context-loss failure. **Neither objection applies here**: this is
first-party code under the same ownership, so there is no attribution or copyleft exposure and
no metadata to drop; and the context is not lost because the modules document their own
invariants and the person who wrote them is the person adopting them. Its remaining valid
warning — *"the extracted code will need upstream tracking with no resync tooling"* — is
answered by moving asr to import the library rather than keeping a second copy.

### AlphaFold `residue_constants` (Apache-2.0), ProteinMPNN (MIT), ESM (MIT) → **derive lessons** (confidence 0.85)

These are the authoritative provenance for orderings 1/2/5/6 and the ESM vocab. A 20-character
residue ordering is a scientific fact rather than creative expression, so what is taken is
citation, not code: each declared `Alphabet` carries `provenance` naming its upstream source so
a future reader can check it. This is what F4's bare `PROXIDE_ORDER` variable name should have
been.

### Not adopted

- **biopython** `Bio.Align.substitution_matrices` — same idea as biotite, heavier install,
  quirkier `Array` API. No advantage over biotite, which is already in the ecosystem's vocabulary.
- **scikit-bio** — BSD-3 and has alphabets, but a heavier install for no capability biotite
  lacks.
- **pyhmmer** `easel.Alphabet` — narrow (HMMER interop), Cython. Not a general alphabet layer.

*(These three were assessed on install weight and API fit relative to biotite, not
independently benchmarked — if one is proposed later it deserves its own pass.)*

---

## 5. Scope decisions (user-decided 2026-08-14)

**D1 — Own the alphabets and the asset *contract*; ship no assets.** The core provides
`Alphabet`, `perm()`, the sentinel policies, the `Asset = (data, alphabet)` contract, and
entry-point registration. `data/` ships empty. This library never becomes the owner of
BLOSUM62/JTT/LG.

Consequence: F4 gets a correct home without a custody transfer. asr keeps `jtt_model.py` and
`blosum_utils.py` as the data's owner and *registers* them:

```toml
[project.entry-points."<pkg>.matrices"]
jtt = "asr.jtt_model:JTT_ASSET"
```

The `PROXIDE_ORDER` provenance gap closes because registration requires a declared alphabet —
the thing that variable name was standing in for. `jtt_model.py`'s sha256 + parity-test
discipline stays where it already works rather than being generalised speculatively.

**D2 — asr migrates first, then reassess.** asr is both the source of the contract (F5) and
where 7 of the 29 declaration sites live, including all four sentinel conventions and the only
instance of ordering 7. It is the hardest case, so it is the right proving ground: a contract
that cannot express asr's needs is wrong, and better to learn that against one repo than four.

Explicitly *not* now: proxide's internal duplication (F3), proteinsmc, aminx. proteinsmc and
aminx are mid-partition and should not take a new edge while that is in flight. F3 is a
two-line dedup that can happen any time and does not gate this.

Accepted risk: until asr migrates, this library is an eighth declaration site, so the census
gets marginally worse before it gets better. D2 is what bounds how long that lasts.

**D3 — Defer the biotite bridge.** Design so a bridge is cheap; write no biotite code. Nothing
in the ecosystem currently converts to or from a `biotite.sequence.Alphabet`, so building it now
is speculative surface. §4's `accept_dependency` verdict stands as a pre-adjudicated decision to
be *executed when a caller appears*, not withdrawn — the `[project.optional-dependencies] jax`
slot in `pyproject.toml` already models the lazy-import shape a `biotite` extra would follow.

## 6. What implementation follows from this

Three modules, numpy-only, no shipped data:

| module | contents |
|---|---|
| `alphabet.py` | `Alphabet` value (symbols, name, provenance); `perm(src, dst)`; sentinel policy enum |
| `asset.py` | `Asset = (data, alphabet)` contract; validation against §2.3 |
| `registry.py` | `importlib.metadata` entry-point discovery; symbols-collision guard |

Tests are the deliverable as much as the code, per §2.3: round-trip property, letter
preservation by character comparison, totality over sentinels, and an alias-identity assertion
so a sixth name for an existing ordering fails the suite (F1 made enforceable).

## 7. Not yet done

No code written. The library at `abcdefghijk` remains a scaffold (`6627a79` + this doc).
Nothing in proxide, proteinsmc, aminx, or asr has been modified by this analysis.
