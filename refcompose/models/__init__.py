"""Neural model wrappers for RefCompose: DSA LoRA, FLUX.2 Dev, Grounding DINO, DA3, and Qwen3-VL."""

from refcompose.models.depth_estimator import DepthEstimator
from refcompose.models.detector import GroundingDINOLocalizer
from refcompose.models.dsa_lora import (
    DSAConditionInjectionProcessor,
    DSALoRALinear,
    LoRABranch,
    build_dsa_causal_attention_mask,
    clear_kv_cache,
    get_dsa_lora_parameters,
    inject_dsa_lora,
    load_dsa_lora_weights,
    save_dsa_lora_weights,
    set_dsa_trainable_branches,
)
from refcompose.models.flux2_wrapper import (
    Flux2BackboneWrapper,
    StandaloneMMDiT,
    StandaloneVAE,
    pack_latents,
    prepare_latent_image_ids,
    unpack_latents,
)
from refcompose.models.vlm_annotator import Qwen3VLAnnotator

__all__ = [
    "DepthEstimator",
    "GroundingDINOLocalizer",
    "DSAConditionInjectionProcessor",
    "DSALoRALinear",
    "LoRABranch",
    "build_dsa_causal_attention_mask",
    "clear_kv_cache",
    "get_dsa_lora_parameters",
    "inject_dsa_lora",
    "load_dsa_lora_weights",
    "save_dsa_lora_weights",
    "set_dsa_trainable_branches",
    "Flux2BackboneWrapper",
    "StandaloneMMDiT",
    "StandaloneVAE",
    "pack_latents",
    "prepare_latent_image_ids",
    "unpack_latents",
    "Qwen3VLAnnotator",
]

