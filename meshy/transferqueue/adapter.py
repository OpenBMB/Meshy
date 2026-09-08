"""Glue between Meshy's per-sample ``TensorDict`` and TQ's batched layout.

Meshy represents one rollout sample as a *batch-less* ``TensorDict`` whose
fields are variable-length tensors (``tokens`` ``[L]`` and ``logprobs`` ``[L]``),
per-sample scalars (``advantage``) or arbitrary Python objects.
TransferQueue stores and returns a *batched* ``TensorDict`` with a leading batch
dim. This module converts both ways; the column -> storage-form mapping lives in
:data:`FIELD_KINDS` so producers and consumers cannot drift apart.

Two properties of the TQ round-trip drive the design here:

**The form you put in is not necessarily the form you get back.** ``put`` unbinds
every column into per-sample values (``_select_by_positions``), and ``get_data``
re-packs them with a fallback chain: all-0-dim -> ``torch.stack``; all tensors ->
jagged nested, else strided nested, else ``NonTensorStack``
(``simple_storage_manager._pack_field_values``). :func:`td_to_samples` therefore
accepts all of those shapes rather than assuming nested.

**Slicing a packed column yields a view of the whole batch's storage.** Returning
those views verbatim makes all B samples alias one buffer, so pickling the sample
list (e.g. ``broadcast_object_list`` inside ``split_batch_to_local``) serializes
that buffer once *per sample* -- an O(B^2) blow-up, so every value
leaving this module is ``.clone()``d onto its own compact storage.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

# How each known column is materialised inside a batched TensorDict.
#   "nested" -> variable-length tensor ([L] or [L, K]), stored as torch.nested
#   "scalar" -> one value per sample, stored as a dense [B, 1] tensor
#   "object" -> arbitrary Python object, stored as NonTensorStack
#
# Unknown columns fall back to "object" via :func:`field_kind`.
FIELD_KINDS: dict[str, str] = {
    # GRPO / shared
    "tokens": "nested",  # [L] int64
    "logprobs": "nested",  # [L] float32
    "mask_assistant": "nested",  # [L] float32
    "advantage": "scalar",
    "reward": "scalar",
    # Rollout quality stamps (0/1 flags) consumed by the trainer's metrics:
    # response hit max_new_tokens / tail is degenerate repetition / generation
    # resumed on a newer weight version after a colocate abort.
    "truncated": "scalar",
    "repetition": "scalar",
    "mixed_version": "scalar",
    # Staleness tag: the weight version the sample was generated against (the
    # newest gen-gate the rollout had observed when the rollout started; a
    # documented lower bound when generation overlaps a sync). Also the payload
    # column of the gen-gate control partitions (see
    # :mod:`meshy.transferqueue.control`).
    "weight_version": "scalar",  # [B, 1] int64
    # Student Top-K OPD: the Teacher's ``[L, K]`` candidates per position
    # (see ``meshy.config.OPD_TEACHER_FIELDS``).
    "teacher_topk_ids": "nested",  # [L, K] int64
    "teacher_topk_logprobs": "nested",  # [L, K] float32
}

_SCALAR_DTYPE: dict[str, torch.dtype] = {
    "reward": torch.float32,
    "advantage": torch.float32,
    "weight_version": torch.int64,
    "truncated": torch.int64,
    "repetition": torch.int64,
    "mixed_version": torch.int64,
}


def field_kind(field: str) -> str:
    """Return the storage kind for *field* (defaults to ``"object"``)."""
    return FIELD_KINDS.get(field, "object")


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def samples_to_td(samples: list[TensorDict], fields: Iterable[str]) -> TensorDict:
    """Pack per-sample TensorDicts into one batched TensorDict ready for ``put``.

    Only *fields* are included, so a producer writes exactly the columns it owns.
    """
    fields = list(fields)
    batch = len(samples)
    if batch == 0:
        raise ValueError("samples_to_td received an empty sample list")

    out: dict[str, Any] = {}
    for field in fields:
        kind = field_kind(field)
        values = [s[field] for s in samples]

        if kind == "nested":
            tensors = [_as_tensor(v) for v in values]
            ndims = {t.dim() for t in tensors}
            if len(ndims) != 1:
                raise ValueError(
                    f"column {field!r}: all samples must have the same rank, got {sorted(ndims)}"
                )
            if tensors[0].dim() >= 2:
                # jagged nested requires every non-ragged dim to agree; catching
                # a K mismatch here beats a confusing failure inside TQ.
                trailing = {tuple(t.shape[1:]) for t in tensors}
                if len(trailing) != 1:
                    raise ValueError(
                        f"column {field!r}: trailing dims must match across samples, got {sorted(trailing)}"
                    )
            out[field] = torch.nested.nested_tensor(tensors, layout=torch.jagged)
        elif kind == "scalar":
            # Cast per-value on the tensor (not through float()) so integer
            # scalar columns (weight_version) round-trip exactly.
            dtype = _SCALAR_DTYPE.get(field, torch.float32)
            out[field] = torch.stack(
                [_as_tensor(v).reshape(-1)[0].to(dtype) for v in values]
            ).reshape(batch, 1)
        else:  # object
            out[field] = NonTensorStack(*[NonTensorData(v) for v in values])

    return TensorDict(out, batch_size=[batch])


def _explode_column(value: Any, field: str, batch: int) -> list[Any]:
    """Split one batched column into ``batch`` independent per-sample values.

    Handles every shape ``get_data`` can hand back (nested / dense / non-tensor),
    and clones each result so no sample keeps a reference to the batch buffer
    (see the module docstring).
    """
    if isinstance(value, NonTensorData):
        return [value.data] * batch
    if isinstance(value, NonTensorStack):
        return [value[i] for i in range(batch)]
    if isinstance(value, list):
        return list(value)

    if isinstance(value, torch.Tensor):
        # One unbind for the whole column, not one per sample: unbinding inside a
        # per-sample loop would be O(B^2).
        parts = value.unbind() if value.is_nested else [value[i] for i in range(batch)]
        if field_kind(field) == "scalar":
            return [p.reshape(-1)[0].clone() for p in parts]
        return [p.clone() for p in parts]

    return [value[i] for i in range(batch)]


def td_to_samples(
    td: TensorDict, fields: Iterable[str] | None = None
) -> list[TensorDict]:
    """Unpack a batched TensorDict from TQ into per-sample TensorDicts.

    Inverse of :func:`samples_to_td`. ``scalar`` columns come back as 0-dim
    tensors, ``nested`` columns as their original ``[L]`` / ``[L, K]`` shape, and
    ``object`` columns as the original Python objects -- matching what
    ``build_micro_batch`` and the trainer expect.
    """
    fields = list(td.keys()) if fields is None else list(fields)
    batch = int(td.batch_size[0])

    columns = {f: _explode_column(td.get(f), f, batch) for f in fields}
    return [
        TensorDict({f: columns[f][i] for f in fields}, batch_size=[])
        for i in range(batch)
    ]
