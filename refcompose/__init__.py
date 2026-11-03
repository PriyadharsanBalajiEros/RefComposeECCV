"""
RefCompose: Multi-Reference Image Generation via LoRA-Conditioned Diffusion.

Reference:
    Sai Sri Teja Kuppa, Parth Shinde, Priyadharsan Balaji S,
    Jinka Harshavardhan, and Sriprabha Ramanarayanan.
    "RefCompose: Multi-Reference Image Generation via LoRA-Conditioned Diffusion"
    arXiv:2609.32389v1 (2026).
"""

__version__ = "1.0.0"

from refcompose.config import (
    RefComposeConfig,
    load_config,
)
from refcompose.prompts.template import (
    ScenePromptSlots,
    PromptTemplateBuilder,
)
from refcompose.canvas.composer import (
    BoundingBox,
    ReferenceCanvasComposer,
)
from refcompose.models.dsa_lora import (
    DSALoRALinear,
    DSAConditionInjectionProcessor,
    inject_dsa_lora,
    clear_kv_cache,
    save_dsa_lora_weights,
    load_dsa_lora_weights,
)

__all__ = [
    "__version__",
    "RefComposeConfig",
    "load_config",
    "ScenePromptSlots",
    "PromptTemplateBuilder",
    "BoundingBox",
    "ReferenceCanvasComposer",
    "DSALoRALinear",
    "DSAConditionInjectionProcessor",
    "inject_dsa_lora",
    "clear_kv_cache",
    "save_dsa_lora_weights",
    "load_dsa_lora_weights",
]

