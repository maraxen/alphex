"""The shipped alphabet declarations. Constants only -- no functions.

Every row below is derived from the **ordering literal** in the cited source, never from the
name of a sentinel constant. That rule exists because violating it is what produced
architecture-review finding #1: a row declaring an AlphaFold-ordered index space under a
ProteinMPNN name, because it was named after `PROTEINMPNN_X_INT` rather than after the ordering
that constant actually indexes. That is the shipped bug's causal step, committed inside the
spec written to prevent it.

Sites are from `scripts/census_alphabets.py` (AST pass, 2026-08-14): 10 distinct orders across
30 declaration sites in proxide, proteinsmc, aminx and asr.

**No aliases are shipped.** `LG_ORDER`, `CANONICAL_Q20`, `AA_ORDER` and `restypes` all resolve
to `AF_20` or `MPNN_20`. Adding an alias constant would recreate, inside this module, the exact
problem the module exists to remove.
"""

from __future__ import annotations

from abcdefghijk.alphabet import Alphabet, SpecialKind

_MPNN = "ACDEFGHIKLMNPQRSTVWY"
"""ProteinMPNN ordering. Sites: proteinsmc `constants.py:294`, asr `alphabet_reconcile.py:30`,
asr `dca_alphabet.py:55`, and (with X appended) proxide `chem/conversion.py:16`,
proxide `io/parsing/mappings.py:23`, aminx `potts/model.py:40`, aminx `utils/aa_convert.py:16`,
asr `alphabet.py:14`, asr `dca_alphabet.py:46`."""

_AF = "ARNDCQEGHILKMFPSTWYV"
"""AlphaFold ordering, identical to what asr calls LG order and IQ-TREE order. Sites:
proxide `chem/residues.py:623`, proxide `chem/residues.py:653`, proteinsmc `constants.py:10`,
asr `alphabet_reconcile.py:31`, asr `jtt_model.py:47`, asr `lg_model.py:17`, and (with X
appended) proxide `chem/conversion.py:17`, proxide `io/parsing/mappings.py:24`,
aminx `utils/aa_convert.py:17`."""

_ESM = "LAGVSERTIDPKQNFYMHWC"
"""ESM3/ESM-C ordering, roughly by corpus frequency. A genuine third permutation: exactly one
fixed point with ProteinMPNN (W) and one with AlphaFold (K)."""


# --- ProteinMPNN base -------------------------------------------------------------------

MPNN_20 = Alphabet(
  symbols=_MPNN,
  name="MPNN_20",
  citation="ProteinMPNN (Dauparas et al. 2022), MIT; proteinsmc constants.py:294",
)

MPNN_X_21 = Alphabet(
  symbols=_MPNN,
  name="MPNN_X_21",
  citation="proxide chem/conversion.py:16 MPNN_ALPHABET; also aminx, asr (6 sites)",
  specials={SpecialKind.UNKNOWN: 20},
)

MPNN_GAP_21 = Alphabet(
  symbols=_MPNN,
  name="MPNN_GAP_21",
  citation="asr alphabet.py:8 CANONICAL_ALPHABET; asr alphabet_reconcile.py:32 CANONICAL_Q21",
  specials={SpecialKind.GAP: 20},
)

MPNN_GAPFIRST_21 = Alphabet(
  symbols=_MPNN,
  name="MPNN_GAPFIRST_21",
  citation="asr pdz_utils.py:8 AA_ALPHABET (Potts/DCA); also alphabet_reconcile.py:33, "
  "blosum_utils.py:33, dca_alphabet.py:43",
  offset=1,
  specials={SpecialKind.GAP: 0},
)

MPNN_GAP_X_STOP_22 = Alphabet(
  symbols=_MPNN,
  name="MPNN_GAP_X_STOP_22",
  citation="domain of proteinsmc PROTEINMPNN_TO_ESM_AA_MAP_JAX, built from "
  "constants.py:294 over a table of length PROTEINMPNN_X_INT + 1",
  specials={SpecialKind.GAP: 20, SpecialKind.UNKNOWN: 21, SpecialKind.STOP: 21},
)

MPNN_X_GAP_22 = Alphabet(
  symbols=_MPNN,
  name="MPNN_X_GAP_22",
  citation="proxide chem/residues.py:728 ID_TO_HHBLITS_AA values, in id order",
  specials={SpecialKind.UNKNOWN: 20, SpecialKind.GAP: 21},
)


# --- AlphaFold base ---------------------------------------------------------------------

AF_20 = Alphabet(
  symbols=_AF,
  name="AF_20",
  citation="AlphaFold residue_constants restypes, Apache-2.0; proxide chem/residues.py:623, "
  "proteinsmc constants.py:10; == asr LG_ORDER and IQ-TREE AA_ORDER",
)

AF_X_21 = Alphabet(
  symbols=_AF,
  name="AF_X_21",
  citation="proxide chem/conversion.py:17 AF_ALPHABET; also io/parsing/mappings.py:24, "
  "aminx utils/aa_convert.py:17",
  specials={SpecialKind.UNKNOWN: 20},
)

AF_GAP_21 = Alphabet(
  symbols=_AF,
  name="AF_GAP_21",
  citation="asr blosum_utils.py:32 PROXIDE_ORDER -- the ordering of the BLOSUM62 matrix asr "
  "copied from proxide, remapped at runtime to gap-first",
  specials={SpecialKind.GAP: 20},
)

AF_GAP_X_STOP_22 = Alphabet(
  symbols=_AF,
  name="AF_GAP_X_STOP_22",
  citation="domain of proteinsmc ALPHAFOLD_TO_ESM_AA_MAP_JAX, built from constants.py:10-31; "
  "the space AA_CHAR_TO_INT_MAP (:113) and CODON_INT_TO_RES_INT_JAX (:121) inhabit",
  specials={SpecialKind.GAP: 20, SpecialKind.UNKNOWN: 21, SpecialKind.STOP: 21},
)


# --- ESM ---------------------------------------------------------------------------------

ESM_C = Alphabet(
  symbols=_ESM,
  name="ESM_C",
  citation="ESM3/ESM-C SEQUENCE_VOCAB, MIT; proteinsmc constants.py:223-257",
  offset=4,
  specials={
    SpecialKind.BOS: 0,
    SpecialKind.PAD: 1,
    SpecialKind.EOS: 2,
    SpecialKind.STOP: 29,
    SpecialKind.GAP: 30,
    SpecialKind.CHAIN_BREAK: 31,
    SpecialKind.MASK: 32,
    SpecialKind.UNKNOWN: 24,
  },
  # 3 is the tokenizer-level <unk>, distinct from the residue-level X at 24 and not modelled
  # here (we describe the vocabulary, not the tokenizer). 25-28 are B, U, Z, O -- degenerate
  # and nonstandard symbols, refused in v0.1 and admitted in v0.2 as `aliases`.
  unclaimed=frozenset({3, 25, 26, 27, 28}),
  size=33,
  padded_size=64,
)


# --- nucleotide ---------------------------------------------------------------------------

DNA_4 = Alphabet(
  symbols="ACGT",
  name="DNA_4",
  citation="proteinsmc constants.py:40 NUCLEOTIDES_CHAR; asr iqtree_runner.py:47 DNA_ALPHABET",
)


ALL: tuple[Alphabet, ...] = (
  MPNN_20,
  MPNN_X_21,
  MPNN_GAP_21,
  MPNN_GAPFIRST_21,
  MPNN_GAP_X_STOP_22,
  MPNN_X_GAP_22,
  AF_20,
  AF_X_21,
  AF_GAP_21,
  AF_GAP_X_STOP_22,
  ESM_C,
  DNA_4,
)
"""Every shipped declaration. Note there is no `restypes_with_x_and_gap` row: that AF-ordered
q=22 alphabet (proxide `chem/residues.py:753`) is built by a starred expression, so the AST
census cannot see it and it has not been verified by literal analysis. Recorded as a known gap
rather than guessed at."""
