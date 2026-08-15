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

**F1 — The alias problem is worse than the ordering problem.** ~~Only two residue orderings
exist.~~ **CORRECTED in §8.1: there are three** — the ESM vocabulary contributes a third,
which this section missed by only regexing `src/` for 18-25-char literals (ESM's is a Python
list, not a string). The alias point stands and is unaffected: the orderings travel under at
least five names. Base MPNN is `MPNN_ALPHABET`, `CANONICAL_ALPHABET`, `CANONICAL_Q20`,
`PROTEINMPNN_RESTYPES`; base AF is `AF_ALPHABET`, `LG_ORDER`, `AA_ORDER`, `restypes`. Verified
by execution: asr's `LG_ORDER` is byte-identical to proxide's `AF_ALPHABET[:20]`, and
`CANONICAL_Q20` to `MPNN_ALPHABET[:20]`. Two names for one thing is how a table gets built from
one and labelled the other, which is the shipped bug.

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

> **D4 completed and partly superseded, 2026-08-15.** Its Phase 0 shipped as written and did its
> job. Its "never to `dependencies`" clause was always scoped to Phase 0 and is now spent for
> **asr**, which took a genuine runtime edge on 2026-08-14 and builds its permutation tables with
> `perm()`. The other three (proteinsmc, aminx, proxide) remain dev-only, exactly as D4 intended
> — nothing under their `src/` imports the library, so the ecosystem partition is still
> unaffected. All four now resolve `alphex>=0.1.0a1` from PyPI rather than a path source, which
> also removed asr's requirement to `myxcel push alphex` before every cluster run.

**D4 — (2026-08-14, supersedes D2's sequencing) Phase 0 is a dev-dependency-only conformance
pass across all five repos, before any runtime migration.**

Ship `errors.py`, `alphabet.py` and `known.py` and nothing else. Every repo adds the library to
its **dev group** — never to `dependencies` — plus one test asserting its local constants equal
the corresponding declaration. No import in any `src/`, so no runtime edge exists and the
ecosystem partition is unaffected: a dev-only test dependency is not a coupling edge in any sense
the partition constrains.

Why this dominates "asr first":

- **It retires the eighth-site risk immediately** rather than bounding its duration. D2 accepted
  that this library would be an extra declaration site until asr migrated. A duplicate that is
  *mechanically checked against its source* is harmless; the risk was never duplication, it was
  *unchecked* duplication.
- **It covers five repos at once** instead of one, at lower risk than migrating one.
- **It would have caught architecture-review finding #1 mechanically** — my `known.py` row
  declaring an AF-ordered space under a ProteinMPNN name. T4 as originally written asserted a
  hardcoded string against itself, which is a tautology, not a conformance check.
- **It reopens proteinsmc.** D2 excluded it while the partition is in flight, but the partition's
  concern is heavyweight inbound coupling (jax / proxide / Rust / maturin) and this edge carries
  none of it — numpy only, no transitive closure, no build step, and in Phase 0 not even a
  runtime import. Excluding it would leave the one repo where a bug actually shipped as the last
  one fixed. **Phase 0 includes proteinsmc.**

**D5 — (2026-08-14) `registry.py` is cut from v0.1.**

Reasons, in descending force:

1. **It inverts the dependency at runtime.** `registry.assets()` would resolve
   `asr.blosum_utils:BLOSUM62_ASSET`, importing `asr.blosum_utils`, which imports `jax.numpy` at
   line 1. The numpy-only library would become, on first `assets()` call, a jax-importing loader
   of its own consumers — runtime graph `asr → lib → asr`. T15 ("importing the package does not
   import jax") would still pass, because it tests the easy half.
2. **By this spec's own admission it buys nothing.** §6 stated the expected steady state is that
   *no consumer registers an alphabet at all*, and the asset group had exactly one expected
   registrant with three objects, all reachable by direct import. Entry points buy
   enumeration-without-knowing-names; with one owner and six repos, nobody needs to enumerate.
3. **It violated `CLAUDE.md`'s categorical instruction** — "no entry points" while the name is a
   placeholder. The decoupled group namespace mitigated the rename hazard specifically, but the
   instruction was not conditional.
4. **Entry points require an installed distribution**, and proteinsmc currently cannot be
   imported normally (unresolvable dependency; `uv run --no-project` workaround). A discovery
   mechanism keyed on installed metadata is fragile in exactly this ecosystem.

Consequence: the `Asset` layer loses its delivery mechanism, so **`asset.py` is also deferred**
past v0.1. D1 said own the contract and ship no data; with no registry there is no consumer for
the contract yet, and the honest move is to add it when a second registrant appears — at which
point the distribution will have its real name and the group-name question resolves itself.

### D4 status: DELIVERED 2026-08-14

Library at `alphex` — `errors.py`, `alphabet.py`, `known.py`, 30 tests, ruff and ty clean.
Conformance suites landed in all four repos that hold declarations:

| repo | tests | commit | notes |
|---|---|---|---|
| proteinsmc | 9 | `b558797` (pushed) | on worktree branch `wt-20260813-170029` |
| asr | 9 | `8cc2fa5` | includes 2 tripwires on live defects |
| aminx | 4 | `108b206b` | includes the proxide-duplication equality |
| proxide | 6 | `c201c68` | includes the degenerate/nonstandard pin |

prolix and tev_design were skipped: the AST census found **zero** declarations in either.

**The mechanism was verified to detect drift, not merely to pass.** One character of the
library's ProteinMPNN declaration was changed and proteinsmc's suite failed with both orderings
printed; then reverted. A conformance test that has never failed proves nothing.

**Two live defects found and pinned, not fixed** — `asr/alphabet.py:15-16` interchanges gap and
unknown in both directions, and `:22` asserts an identity permutation between gap-first Potts
and gap-last canonical. Both are this library's own bug class, in production source. They are
asserted as-is so they are visible and cannot drift further; fixing them changes numbers and
needs its own pass with its own evidence.

### Phase 1 status: defects fixed 2026-08-14; runtime migration BLOCKED on the kernel

asr commit `2058572`. Investigation before fixing changed the diagnosis, and **two of the three
recorded defects were not what §D4's status note claimed**:

| recorded | actual |
|---|---|
| `POTTS_TO_CANONICAL` identity is wrong | **Confirmed.** Fixed to `[1..20, 0]`. Blast radius **zero** — `remap_from_potts` had no callers, so no published number moved. Latent, not live. |
| gap/X conflation is a live defect whose fix changes numbers | **Wrong on both counts.** Both maps are the **identity**, since `CANONICAL_ALPHABET[:20] == MPNN_ALPHABET[:20]`. Only index 20 differs, and in *meaning* not position. ProteinMPNN has no gap state, so no reindexing can fix it. Remedy is documentation, now in `asr/alphabet.py`. |
| *(not previously recorded)* | **New:** `mtt_training_pipeline.py:663` applied `CANONICAL_TO_MPNN` to logits *returned by* the MPNN scorer — the wrong direction. Invisible because both maps are the identity, but armed: correcting either map would have made it silently wrong. Now `remap_from_mpnn`. |

The lesson is the one the contract already argues for elsewhere: a claim about an ordering that
has not been traced to its consumers is not evidence. The tripwires were right that something
was wrong; they were not evidence about *what*.

**Blocker CLEARED 2026-08-14: the kernel is built.** `relation.py` and `convert.py` implement
`relation`, `perm`, `convert` and `reindex` per `260814_alphabet-api-surface.md` §3.3-3.5, with
the architecture-review corrections folded in (`PolicySpec` per `SpecialKind`, `MaskedPerm`
instead of an in-band `-1`, the `reindex` axis-length precondition). 49 kernel tests; 79 in the
library; the 28 conformance tests across the four consumer repos still pass unchanged.

Implementing it forced five corrections to the spec and added two error cases it had not named —
see `260814_alphabet-api-surface.md` §11. The load-bearing one for this document is **C6**: an
`INCOMPATIBLE` relation is now "some source residue is absent from the destination", not "no
shared symbols", because `MPNN_20` and `DNA_4` share `A/C/G/T` and the original criterion would
have let a protein→DNA conversion through to a policy.

**Remaining for Phase 1:** asr still uses its own hand-rolled arrays — now correct, documented,
and pinned by `tests/test_alphabet_maps.py` and the inverted conformance tests. Replacing them
with `perm(...)` calls is the next step, and is no longer blocked; it is now a question of
whether asr should take a runtime dependency (D4 currently says dev-only).

**D2 — asr migrates first, then reassess.** *(Sequencing superseded by D4; the reasoning below
still governs Phase 1, the runtime migration.)* asr is both the source of the contract (F5) and
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

No code written. The library at `alphex` remains a scaffold (`6627a79` + this doc).
Nothing in proxide, proteinsmc, aminx, or asr has been modified by this analysis.

---

# 8. Addendum: systematic feature census, and the tokenizer question

Added 2026-08-14 in response to: *are we reinventing the wheel, is this really a discrete
tokenizer, and what is the right balance of abstraction and grounding?*

## 8.1 Corrections to §1

**C1 — There are THREE base orderings, not two.** The ESM vocabulary's 20 canonical residues
are ordered `LAGVSERTIDPKQNFYMHWC` (roughly by corpus frequency), a genuine third permutation:
verified by execution, it shares exactly **one** fixed point with MPNN (`W`) and **one** with AF
(`K`). §1 missed it because the extraction regex matched string literals in `src/`, and the ESM
vocabulary is a Python list of per-token strings. A census whose method cannot see a list is not
a complete census — worth remembering before treating the 29-site figure as exhaustive.

**C2 — ESM's canonical 20 are contiguous, at an offset of 4.** I had described this vocabulary
as having non-contiguous residues; that is wrong. Indices 4-23 are the 20 canonical residues in
ESM order, with special tokens at 0-3. This is the difference between "arbitrary learned
vocabulary" (not tractable to model) and "ordering + offset" (trivial to model), so the error
mattered: it made the ESM path look like it needed a tokenizer when it needs an integer.

**C3 — Sentinels are not one kind, and proteinsmc conflates two of them.** The ESM vocabulary
distinguishes at least six: BOS/pad/EOS/unk (0-3), degenerate-and-nonstandard (24-28: `X B U Z
O`), `.` (29), gap (30), chain-break (31), mask (32). Against that, proteinsmc sets
`STOP_INT = UNKNOWN_AA_INT = PROTEINMPNN_X_INT = 21` (`constants.py:115-117`) — one integer for
"stop codon" and "unknown residue". That is a latent defect of the same family as the one already
fixed: two distinct meanings sharing an index, with no error signal when they are confused.

**C4 — The domain is not protein-only.** proteinsmc declares a nucleotide alphabet
(`NUCLEOTIDES_CHAR = ACGT`, `constants.py:40`). Same abstraction, different symbol set. Nothing
in the contract should assume 20 residues.

## 8.2 The feature census

Every axis along which real alphabets in and around this ecosystem vary, with the count of
independent instances actually observed:

| axis | instances observed | verdict |
|---|---|---|
| residue ordering | **3** (MPNN, AF, ESM) | **core** |
| sentinel placement | **4+** (none, X@20, gap@20, gap@0, ESM's layout) | **core** |
| sentinel *kind*, typed | **6+** in ESM; conflated in proteinsmc (C3) | **core** |
| index offset | **1** (ESM, +4) | **core** — sole instance, but it is the bug site |
| extension vs permutation | biotite models it; MPNN→MPNN+X is extension, MPNN→gap-first is not | **core** — cheap, and it *is* the safety distinction |
| cardinality / padding | ESM 33 symbols padded to 64 | **metadata only**, no behaviour |
| non-protein symbol sets | 1 (ACGT) | **core by construction** — just don't assume 20 |
| similarity matrices | asr BLOSUM62 | **asset**, registered not owned (D1) |
| rate matrices | asr JTT, LG | **asset** — *different invariants*, see 8.4 |
| degenerate/ambiguity symbols | ~~0 in ecosystem~~ **AT LEAST 1, LIVE — count invalid, see C5** | ~~declare unrepresentable~~ **RE-ADJUDICATE** |
| nonstandard residues (U, O) | ~~0 uses~~ **LIVE in proxide — count invalid, see C5** | ~~declare unrepresentable~~ **RE-ADJUDICATE** |
| reduced alphabets / morphisms | **1** (aminx `_HYDROPHOBIC_LETTERS`, and it is a subset predicate, not a morphism) | **declare unrepresentable** |
| multi-character tokens (codons) | **1** (proteinsmc `CODON_TO_RES_CHAR`, 3 nt → 1 aa) | **declare unrepresentable** |
| case / soft-masking | **0** | **out of scope** |

## 8.3 Are we reinventing the wheel? Partly — and the part that is not is the point

**What biotite already covers,** verified by execution: `Alphabet` (encode/decode/get_symbols),
`AlphabetMapper` (letter-preserving conversion), `LetterAlphabet` (arbitrary orderings),
`extends` (subsumption — `ProteinSequence.alphabet.extends(the 20)` is True, the converse
False), full IUPAC ambiguity for nucleotides (`A C G T R Y W S M K H B V D N`), and **91**
bundled similarity matrices. On these axes we are reinventing, and §4's `derive_inspiration`
verdict is the acknowledgement.

**What nothing external covers:**

1. **Cross-convention index reconciliation as a first-class concern.** Every library assumes it
   *owns* the vocabulary. biotite's `AlphabetMapper` converts between two alphabets you already
   hold as objects; it has nothing to say about a bare `int8` array arriving from another library
   under an undeclared convention. That undeclared arrival is our entire bug class.
2. **Rate matrices.** biotite ships 91 similarity matrices and **zero** rate matrices — no JTT,
   LG, or WAG — and its `phylo` module offers only distance methods (NJ, UPGMA), no substitution
   models. asr's JTT and LG have no external home. (Real gap, but per D1 in the asset layer we
   deliberately do not own.)

So: the *data structure* is a wheel; the *reconciliation discipline* is not.

## 8.4 Assets need a kind, because their invariants differ

§3.1 said an asset is `(data, alphabet)`. That is insufficient — it must be
`(data, alphabet, kind)`, because what makes an asset *valid* depends on kind:

| kind | invariants that are checkable |
|---|---|
| `SIMILARITY` (BLOSUM, PAM) | square, symmetric, alphabet-length axes |
| `RATE` (JTT, LG, WAG) | rows sum to 0; `Q = S·diag(π)` with `S` symmetric; detailed balance `π_i Q_ij = π_j Q_ji` |
| `FREQUENCY` (π) | non-negative, sums to 1, length = alphabet |
| `PROPERTY` (hydrophobicity) | length = alphabet, no algebraic constraint |

Treating a rate matrix as merely "a matrix with an alphabet" discards every invariant that would
catch a transposition or a mis-scaling. This is the same lesson as the alphabet itself: the type
is what makes the error checkable.

## 8.5 Is this a discrete tokenizer? No — and the difference is exactly the contract

The reframe is close enough to be worth taking seriously, and rejecting precisely.

A tokenizer's contract is **`decode(encode(s)) == s`**. Its integer IDs are *private*: nobody
outside cares that ESM assigns `L → 4`, only that the round trip holds. The vocabulary is owned
by one model, and ownership is what makes the arbitrary assignment safe.

Our integers are **public**, and they cross library boundaries *as array indices into data
ordered by a different convention*. Integer 1 denotes `C` under MPNN and `R` under AF. The bug
was not a failed round trip — it was two libraries agreeing on the integer and disagreeing on
the letter.

**This is why the tokenizer contract is not merely different but insufficient:** proteinsmc's
broken table round-tripped perfectly *within itself*. So did the correct one. Round-trip fidelity
cannot distinguish them. Only a cross-convention invariant — letter preservation under
conversion — can, and no tokenizer models that, because a tokenizer has no notion of a rival
convention for the same symbols.

A tokenizer also cannot have our bug at all, structurally, because it owns both ends. Our
problem is the *absence* of single ownership. So:

> This is not a tokenizer. It is an **index-convention reconciliation layer**. The unit of value
> is the *declaration*, not the encoding.

**Where the tokenizer framing does earn its keep:** ESM's vocabulary genuinely *is* a tokenizer
vocabulary — specials, offset, mask token, padding to 64, frequency ordering. The library must be
able to **describe** such a vocabulary as data (ordering + offset + typed specials, per C2/C3)
without acquiring any tokenization *behaviour*. Model the vocabulary, not the tokenizer. That
distinction is the whole answer to the balance question.

## 8.6 The abstraction ceiling, as a checkable rule

The failure mode to avoid is named in §4's own citation set — the second-system effect, which
the decision matrix flags specifically against `derive_inspiration`: freed from the original's
constraints, a successor design over-embellishes. A general "discrete symbol algebra" is exactly
that trap, and it is attractive right now.

**Rule.** An abstraction is admitted only if either:

- **(a)** the census shows **≥2 independent instances of variance** along that axis, or
- **(b)** it is an axis along which a **shipped bug** has actually occurred.

Everything else is **declared explicitly unrepresentable, with a named error** — never modelled
generically, and never silently accepted.

Applying it to 8.2: ordering, sentinel placement, and sentinel kind enter under (a). Offset
enters under (b) alone — one instance, but the ESM path is where the bug lived. Extension-vs-
permutation enters under (b), since it is the distinction between a free conversion and the one
that corrupted data. Degenerate symbols, nonstandard residues, reduced alphabets, and codons are
each **1 or 0 instances with no bug history**, so all four are refused — and each gets a named
error rather than silence:

```
DegenerateSymbolError   -- B/Z/J and IUPAC nucleotide ambiguity codes are set-valued;
                           this library maps symbols to single indices and will not guess.
NonStandardResidueError -- U (Sec), O (Pyl) are real residues outside every declared
                           alphabet here; register an alphabet that includes them.
ReducedAlphabetError    -- many-to-one symbol morphisms are lossy; out of scope.
MultiCharTokenError     -- codons and any k>1 token; that is a tokenizer's job, not this.
```

This is the `dca_alphabet.py` philosophy generalised: *"the failure mode this guards against is a
silent default, not an unavailable conversion."* Refusing loudly is a feature. It also keeps the
door open — each error names what a caller would have to register to proceed, so the refusals are
extension points rather than dead ends.

## 8.7 Revised core surface

```
Alphabet(symbols, offset=0, specials={}, name, provenance)   # specials: SpecialKind -> int
Relation(src, dst) -> IDENTITY | EXTENSION | PERMUTATION | INCOMPATIBLE
perm(src, dst, policy) -> ndarray                            # policy required, never defaulted
Asset(data, alphabet, kind)                                  # kind gates the invariants
registry: entry-point discovery, collision-guarded
```

`Relation` is the addition this addendum most wants: it lets a caller (or a test) assert
*"this conversion is an EXTENSION"* — a claim that is free and safe — and be forced to think
when it is a `PERMUTATION`, which is the case that silently corrupted data.

## 8.9 C5 — the refusal counts are wrong, and the census method is the reason
*(added 2026-08-14 after architecture review, finding #6; verified by direct read)*

Two of the four refusals in §8.6 rest on instance counts of **0**. Both are false, in the repo
this document names as the shared upstream. `proxide/src/proxide/chem/residues.py`:

- `:695-725` `HHBLITS_AA_TO_ID` maps `B → 2`, `Z → 3` — **degenerate/ambiguity symbols, live**.
- `:728-751` `ID_TO_HHBLITS_AA` carries `1: "C",  # Also U.` and `20: "X",  # Includes J and O.`
  — **nonstandard residues U and O, live, with a stated resolution policy** (`U → C` is
  precisely a shape-valid silent remap of the kind this document is about).
- `:753` `restypes_with_x_and_gap = [*restypes, "X", "-"]` — an **eighth alphabet**: AF-ordered,
  q=22, `X@20`, `gap@21`. Absent from §1's seven-row table. Note its sentinel order is the
  *reverse* of proteinsmc's 22-wide space (`GAP@20, UNKNOWN@21`).
- `:754-756` `MAP_HHBLITS_AATYPE_TO_OUR_AATYPE` — a **live cross-convention conversion table**,
  exactly the class of object this library claims to own.

So **F3 also undercounts**: proxide has *three* alphabet declaration sites, not two, and the
third is the AlphaFold-derived origin the other two copy from.

**The cause is the census method, and C1 already named it without acting on it.** The extraction
was a regex for 18-25-character string literals in `src/`. C1 admitted that cannot see a Python
*list* and corrected one row — but never re-ran the method over **dicts**, and both now-falsified
refusal rows are dict-declared. The 29-site figure is wrong in the other direction too:
`aminx/ebm/ddg_stability.py:77` is a docstring mention, not a declaration.

**Consequence for the §8.6 rule.** The rule is sound; its *application* is not, and the flaw is
structural rather than clerical. A rule whose predicate is an instance count is only as good as
the count — and the counts on the refusal side (`0, 0, 1, 1`) were the least-evidenced numbers in
the document while being the most load-bearing.

Criterion (b) — "a shipped bug occurred along it" — is also **survivor-biased toward axes that
are already instrumented**. `offset` was admitted on one instance because a bug was *noticed*
there; degenerate symbols were refused on a miscount because nothing along that axis has failed
*loudly* yet — and it structurally cannot, since `U → C` is silent by construction. As written,
the rule would refuse to model live production code on the grounds that the code has not yet been
caught.

**Amendment adopted — criterion (c):** *an axis is admitted if refusing it makes an existing
first-party declaration inexpressible.* Under (c), `restypes_with_x_and_gap` and the HHblits
maps force degenerate and nonstandard symbols to be reconsidered.

### C5.1 — AST census run, 2026-08-14. Authoritative counts.

`scripts/census_alphabets.py`, output at `outputs/census.json`. Supersedes §1's regex figures.

| | regex census (§1) | AST census |
|---|---|---|
| distinct orders | 7 | **10** |
| declaration sites | 29 | **30** |
| sites with degenerate `B/Z/J` | 0 | **1** |
| sites with nonstandard `U/O` | 0 | **1** |
| k=3 codon tables | 1 | **2** |

Both the degenerate and the nonstandard site are `proxide/chem/residues.py:697`
(`HHBLITS_AA_TO_ID`). The AST pass also found three orders the regex missed entirely —
`ID_TO_HHBLITS_AA` (`:728`, dict-*values*, MPNN-ordered q=22 with `X@20, gap@21`),
`restype_1to3` (`:653`, dict-keys), and `iqtree_runner.py`'s DNA declarations — and it dropped
the docstring false positive at `ddg_stability.py:77`.

Writing the script surfaced two of its own false-positive classes, both of which had corrupted
the first run and both of which matter because these counts gate a design decision:

- **Chemical element symbols read as alphabets.** `CNOS`, `CNOSP`, `HCNOSP`, `COHNSP` (proxide's
  van der Waals tables) are single-character key sets drawn from the same space as residues, and
  they flagged as NONSTANDARD because oxygen collides with pyrrolysine (`O`). That inflated the
  nonstandard count from 1 to **8**. Fixed with a floor of 15 canonical residues, or
  nucleotide-only.
- **3-letter residue names read as codons.** `chi_angles_atoms`, `residue_atoms` and three others
  are keyed by `ALA`/`ARG`, not by codons — 5 false positives. Fixed by requiring codon keys to
  be drawn from `ACGT(U)`.

**Still a lower bound.** The AST pass walks *literals*, so a computed declaration remains
invisible — notably `proxide/chem/residues.py:753`,
`restypes_with_x_and_gap = [*restypes, "X", "-"]`, an AF-ordered q=22 alphabet (`X@20, gap@21`)
that the script does not report because the list holds a starred expression. Evaluating it would
require importing a jax/Rust-backed package, which a numpy-only tool cannot do. Read "0 from this
script" as "none found by literal analysis", never as "none exist" — that inference is what
invalidated two refusals in the first place.

### C5.2 — Re-adjudication under the amended rule

Degenerate and nonstandard symbols sit at **1 instance each**, so criterion (a) (≥2) still
refuses them and criterion (b) (a shipped bug) does not apply. **Criterion (c) admits both**:
`HHBLITS_AA_TO_ID` maps `B→2, Z→3, J→20, U→1, O→20`, and refusing degenerate/nonstandard symbols
makes that existing first-party declaration inexpressible.

**Verdict: admit, as a read-only `aliases: Mapping[str, int]` on `Alphabet`** — many-to-one
symbol→index, excluded from `decode` (so `decode` stays a function), with the invariant that
every alias target is an already-declared index. Deferred to **v0.2**: Phase 0 (D4) is
conformance-only over the 10 orders the census found, none of which needs alias support, and
proxide's HHblits sites are simply not covered by Phase 0's tests. Recorded as a known gap
rather than silently omitted.

`ReducedAlphabetError` and `MultiCharTokenError` survive: `_HYDROPHOBIC_LETTERS`
(`ddg_stability.py:146`) is a subset predicate, and the two codon tables are genuine k=3 maps.

**Original requirement, now satisfied:** re-run the census with an **AST pass** over dict and list
literals, not a regex over strings. Until then, treat every count in §8.2 as a lower bound and
§8.6's refusals as provisional. `ReducedAlphabetError` and `MultiCharTokenError` survive review
(both verified: `_HYDROPHOBIC_LETTERS` at `ddg_stability.py:146` is a subset predicate, and
`CODON_TO_RES_CHAR` at `proteinsmc/utils/constants.py:46-111` is a genuine k=3 table).

## 8.8 What is still not decided

The four refusals in 8.6 are refusals *for now*, justified by instance count. If asr's DCA work
starts needing reduced alphabets, or a nucleotide path needs IUPAC ambiguity, the rule in 8.6 is
the thing to re-run — not the conclusion to defend.
