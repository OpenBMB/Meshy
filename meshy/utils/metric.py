"""Per-sample rollout quality indicators shared by the rollout and trainer sides.

These are deliberately tiny, dependency-free functions so the rollout worker
can stamp every sample before it enters TransferQueue and the trainer can
aggregate the stamps into TensorBoard scalars without re-deriving them.
"""

from __future__ import annotations

import zlib

#: ``has_repetition`` only looks at responses longer than this many characters.
REPETITION_MIN_CHARS = 10_000
#: Compression ratio above which the tail of a response counts as degenerate.
REPETITION_COMPRESSION_RATIO = 10.0


def compression_ratio(text: str) -> float:
    """``len(raw) / len(zlib(raw))`` for the UTF-8 encoding of ``text``."""
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    compressed = zlib.compress(raw, 6)
    return len(raw) / max(len(compressed), 1)


def has_repetition(text: str) -> bool:
    """Return whether ``text`` ends in degenerate repetition.

    Same rule as the Miles baseline: a response longer than 10k characters
    whose last 10k characters compress more than 10x is flagged. Natural
    language compresses ~2-4x, so this only fires on genuine loops.
    """
    if len(text) <= REPETITION_MIN_CHARS:
        return False
    return compression_ratio(text[-REPETITION_MIN_CHARS:]) > REPETITION_COMPRESSION_RATIO


__all__ = ["compression_ratio", "has_repetition"]
