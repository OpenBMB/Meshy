"""Compatibility import for the Meshy TransferQueue Worker."""

from meshy.worker.tq import TQInput, TQOutput, TQWorker, TQWorkerError

__all__ = ["TQInput", "TQOutput", "TQWorker", "TQWorkerError"]
