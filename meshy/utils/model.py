"""Model-path resolution shared by the services.

Lets the Rollout Service (which builds a tokenizer) and the Training Service
(which builds the forge config) resolve a HuggingFace model id / local path the
same way.
"""

from __future__ import annotations

import os


def resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace model ID or local path to a local directory.

    If *model_path* already points to an existing local directory it is returned
    as-is. Otherwise it is treated as a HuggingFace Hub model ID and downloaded /
    resolved from the local cache via ``snapshot_download``.
    """
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)
