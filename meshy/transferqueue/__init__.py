"""TransferQueue data-plane client layer for Meshy.

This package is Meshy's *client-side* view of TransferQueue (TQ). It never
starts controller / storage components itself -- those run as standalone
processes (see :mod:`meshy.transferqueue.launch`) and are discovered through
the run-level bootstrap store (``store://`` ref) or, for an externally managed
TQ, an endpoints file named by ``XRL_TQ_ENDPOINTS``.

Modules:

* :mod:`~meshy.transferqueue.client` -- import shim, ``connect`` / ``build_config``
  and endpoint discovery (``resolve_endpoints_file``).
* :mod:`~meshy.transferqueue.adapter` -- converts between Meshy's per-sample
  ``TensorDict`` and the batched, column-oriented ``TensorDict`` TQ stores
  (``FIELD_KINDS`` column contract).
* :mod:`~meshy.transferqueue.control` -- the ``gen_gate`` control partition
  (one pulse per weight version).
* :mod:`~meshy.transferqueue.colocation` -- the colocation request ledger
  transport used by the ring scheduler.
* :mod:`~meshy.transferqueue.spec` -- derives TQ capacity from the recipe's
  typed configs.
* :mod:`~meshy.transferqueue.launch` -- spawns / reaps the standalone controller
  and storage processes.

Everything is imported lazily so a recipe module (or the launcher) can import
this package without pulling in torch / tensordict / zmq.
"""

from __future__ import annotations

from meshy.transferqueue.client import build_config, connect, import_tq
from meshy.transferqueue.colocation import RequestHandle, RequestLedgerTransport, RequestRecord, TQRequestLedgerTransport

__all__ = ["build_config", "connect", "import_tq", "RequestHandle", "RequestLedgerTransport", "RequestRecord", "TQRequestLedgerTransport"]
