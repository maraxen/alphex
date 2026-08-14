"""Amino-acid alphabet orderings and letter-preserving conversions between them.

Scaffold only -- no API is implemented yet. See CLAUDE.md for the design contract that
implementation must follow, in particular:

  * orderings are named values, not function names (`perm(AF, MPNN)`, not `af_to_mpnn`)
  * every conversion is letter-preserving
  * every mapping is total over its declared domain, sentinels included
  * numpy is the only runtime dependency

`asr/src/asr/alphabet_reconcile.py` already implements this contract well and is the
intended starting point rather than a blank page.
"""
