# Copyright (c) Meshy project.
"""HF<->torchtitan state-dict adapter for Qwen2.5-Math / DeepSeek-R1-Distill-Qwen.

These models share the standard Qwen2.5 dense layout (Qwen3 building
blocks + QKV bias, no QK-Norm), so the HF<->torchtitan key mapping is
identical to ``qwen2_5``. This file just re-exports the existing
adapter so that ``ModelSpec(state_dict_adapter=...)`` resolves to the
same class without the caller having to reach across packages.
"""

from ..qwen2_5.state_dict_adapter import Qwen2D5StateDictAdapter

__all__ = ["Qwen2D5StateDictAdapter"]
