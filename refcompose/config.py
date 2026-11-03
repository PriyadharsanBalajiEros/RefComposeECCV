"""Configuration dataclasses and hierarchical YAML loader for RefCompose."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ModelPathsConfig:
    flux2_repo_id: str = "black-forest-labs/FLUX.2-dev"
    grounding_dino_repo_id: str = "IDEA-Research/grounding-dino-base"
    depth_anything3_repo_id: str = "depth-anything/da3-large"
    depth_fallback_repo_id: str = "depth-anything/Depth-Anything-V2-Large-hf"
    qwen3_vl_repo_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    dino_repo_id: str = "facebook/dino-vitb16"
    clip_repo_id: str = "openai/clip-vit-large-patch14"
    pickscore_repo_id: str = "yuvalkirstain/PickScore_v1"
    pickscore_clip_repo_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = False

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(self.dtype.lower(), torch.bfloat16)


@dataclass
class ResolutionConfig:
    width: int = 1280
    height: int = 720
    vae_scale_factor: int = 8
    patch_size: int = 2

    @property
    def latent_height(self) -> int:
        return self.height // self.vae_scale_factor

    @property
    def latent_width(self) -> int:
        return self.width // self.vae_scale_factor

    @property
    def packed_height(self) -> int:
        return self.latent_height // self.patch_size

    @property
    def packed_width(self) -> int:
        return self.latent_width // self.patch_size

    @property
    def num_image_tokens(self) -> int:
        return self.packed_height * self.packed_width


@dataclass
class DSALoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    inject_double_blocks: bool = True
    inject_single_blocks: bool = False
    target_projections: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )
    # Paper Section 4.3: subject (appearance) LoRA loaded before depth (structure) LoRA
    stream_order: List[str] = field(
        default_factory=lambda: ["appearance", "structure"]
    )
    use_causal_mask: bool = True
    use_kv_cache: bool = True
    structure_scale: float = 1.0
    appearance_scale: float = 1.0
    # Optional ablation mode (Paper p. 12: merging depth + canvas in a single LoRA stream)
    merged_single_stream_ablation: bool = False


@dataclass
class SamplingConfig:
    num_inference_steps: int = 40
    layout_inference_steps: int = 40
    guidance_scale: float = 4.0
    seed: int = 42


@dataclass
class DetectionConfig:
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    iou_nms_threshold: float = 0.50
    fallback_layout_on_miss: bool = True


@dataclass
class CanvasConfig:
    background_value: float = 0.0
    interpolation: str = "bicubic"
    feather_border_px: int = 0


@dataclass
class CurriculumConfig:
    stage1_end_step: int = 10000
    stage2_end_step: int = 30000
    stage3_end_step: int = 50000


@dataclass
class AugmentationConfig:
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    saturation_jitter: float = 0.15
    hue_jitter: float = 0.04
    max_rotation_deg: float = 5.0
    max_shear_deg: float = 3.0
    scale_range: Tuple[float, float] = (0.95, 1.05)
    max_box_translate_frac: float = 0.05
    perturbation_prob: float = 0.85


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/checkpoints/refcompose_dsa_lora"
    logging_dir: str = "./outputs/logs"
    seed: int = 42
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    max_train_steps: int = 50000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 1000
    timestep_sampling: str = "uniform"
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    save_every_steps: int = 2500
    log_every_steps: int = 50
    val_every_steps: int = 2500
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    cinematic_manifest: str = "./data/processed/cinematic_manifest.jsonl"
    synthetic_manifest: str = "./data/processed/synthetic_manifest.jsonl"
    cinematic_ratio: float = 0.50
    use_cached_latents: bool = False


@dataclass
class CurationConfig:
    raw_cinematic_dir: str = "./data/raw/cinematic_frames"
    synthetic_output_dir: str = "./data/raw/synthetic_flux2"
    processed_output_dir: str = "./data/processed"
    target_cinematic_samples: int = 50000
    target_synthetic_samples: int = 50000
    min_detections_exclusive: int = 4
    canonical_resolution: Tuple[int, int] = (1024, 1024)
    canonical_steps: int = 28
    canonical_guidance_scale: float = 3.5
    canonical_prompt_template: str = (
        "A studio product/subject photograph of {object_description} in a neutral front-facing pose, "
        "centered on a plain neutral light-gray studio background, uniform soft diffused lighting, "
        "no harsh shadows, preserving exact colors, textures, materials, and identity details."
    )
    dino_similarity_threshold: float = 0.90


@dataclass
class EvaluationConfig:
    benchmark_name: str = "DenseLayout"
    denselayout_dataset_id: str = "FireRedTeam/DenseLayout"
    layoutsam_eval_dataset_id: str = "HuiZhang0812/LayoutSAM-eval"
    local_benchmark_dir: str = "./data/benchmarks/denselayout_test"
    canonical_cache_dir: str = "./data/benchmarks/denselayout_canonical_cache"
    output_dir: str = "./outputs/eval_denselayout"
    num_eval_samples: int = 5000
    split: str = "test"
    baseline_max_refs: int = 5
    scalable_max_refs: int = 15
    canonicalize_gt_crops: bool = True
    region_eval_mode: str = "hybrid"
    methods: List[str] = field(
        default_factory=lambda: [
            "refcompose_n5",
            "refcompose_n_gt_5",
            "creatilayout",
            "instanceassemble",
            "uno",
            "xverse",
        ]
    )
    checkpoints: Dict[str, str] = field(
        default_factory=lambda: {
            "refcompose_lora_dir": "./outputs/checkpoints/refcompose_dsa_lora/final",
            "creatilayout_repo_id": "HuiZhang0812/CreatiLayout",
            "instanceassemble_repo_id": "FireRedTeam/InstanceAssemble",
            "uno_repo_id": "bytedance-research/UNO",
            "xverse_repo_id": "ByteDance/XVerse",
        }
    )


@dataclass
class MemoryBenchmarkConfig:
    output_dir: str = "./outputs/benchmark_memory"
    batch_size: int = 1
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 40
    warmup_runs: int = 1
    timed_runs: int = 3
    reference_counts: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 12, 15])
    table1_reference_counts: List[int] = field(default_factory=lambda: [2, 4, 6])
    methods: List[str] = field(
        default_factory=lambda: [
            "creatilayout",
            "instanceassemble",
            "uno",
            "xverse",
            "refcompose",
        ]
    )


@dataclass
class RefComposeConfig:
    model: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    lora: DSALoRAConfig = field(default_factory=DSALoRAConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    memory_benchmark: MemoryBenchmarkConfig = field(default_factory=MemoryBenchmarkConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    """Fallback indentation-based YAML parser used only when `pyyaml` is not yet installed."""
    import ast

    def _parse_scalar(val: str) -> Any:
        v = val.strip()
        if v in ("true", "True"):
            return True
        if v in ("false", "False"):
            return False
        if v in ("null", "None", "~", ""):
            return None
        try:
            return ast.literal_eval(v)
        except Exception:
            return v.strip("\"'")

    lines = []
    for raw_line in text.splitlines():
        stripped_comment = raw_line.split("#", 1)[0].rstrip()
        if stripped_comment.strip():
            indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
            lines.append((indent, stripped_comment.strip()))

    def _parse_block(idx: int, min_indent: int) -> Tuple[Any, int]:
        if idx >= len(lines):
            return {}, idx
        if lines[idx][1].startswith("- "):
            arr: List[Any] = []
            list_indent = lines[idx][0]
            while idx < len(lines) and lines[idx][0] == list_indent and lines[idx][1].startswith("- "):
                arr.append(_parse_scalar(lines[idx][1][2:]))
                idx += 1
            return arr, idx

        obj: Dict[str, Any] = {}
        while idx < len(lines) and lines[idx][0] >= min_indent:
            cur_indent, content = lines[idx]
            if cur_indent > min_indent:
                break
            if content.startswith("- "):
                break
            if ":" not in content:
                idx += 1
                continue
            key, rest = content.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            idx += 1
            if rest in (">-", "|", ">"):
                folded: List[str] = []
                while idx < len(lines) and lines[idx][0] > cur_indent:
                    folded.append(lines[idx][1])
                    idx += 1
                obj[key] = " ".join(folded)
            elif rest == "":
                if idx < len(lines) and lines[idx][0] > cur_indent:
                    child, idx = _parse_block(idx, lines[idx][0])
                    obj[key] = child
                else:
                    obj[key] = None
            else:
                obj[key] = _parse_scalar(rest)
        return obj, idx

    parsed, _ = _parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {}


def _load_yaml_recursive(path: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        raw = yaml.safe_load(text) or {}
    else:
        raw = _parse_minimal_yaml(text)

    base_rel = raw.pop("base_config", None)
    if base_rel:
        candidate = (path.parent / base_rel).resolve()
        if not candidate.exists() and project_root is not None:
            candidate = (project_root / base_rel).resolve()
        if candidate.exists():
            parent_dict = _load_yaml_recursive(candidate, project_root=project_root)
            raw = _deep_merge(parent_dict, raw)
    return raw


def _populate_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    if not data:
        return cls()
    kwargs: Dict[str, Any] = {}
    valid_fields = cls.__dataclass_fields__
    for name, field_def in valid_fields.items():
        if name not in data:
            continue
        val = data[name]
        if name == "curriculum" and isinstance(val, dict):
            kwargs[name] = _populate_dataclass(CurriculumConfig, val)
        elif name == "augmentations" and isinstance(val, dict):
            if "scale_range" in val and isinstance(val["scale_range"], list):
                val = {**val, "scale_range": tuple(val["scale_range"])}
            kwargs[name] = _populate_dataclass(AugmentationConfig, val)
        elif name == "canonical_resolution" and isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RefComposeConfig:
    """Load a RefComposeConfig from YAML (resolving `base_config`) plus optional dict overrides."""
    raw: Dict[str, Any] = {}
    if config_path is not None:
        cfg_file = Path(config_path).resolve()
        project_root = Path(__file__).resolve().parent.parent
        raw = _load_yaml_recursive(cfg_file, project_root=project_root)

    if overrides:
        raw = _deep_merge(raw, overrides)

    return RefComposeConfig(
        model=_populate_dataclass(ModelPathsConfig, raw.get("model", {})),
        resolution=_populate_dataclass(ResolutionConfig, raw.get("resolution", {})),
        lora=_populate_dataclass(DSALoRAConfig, raw.get("lora", {})),
        sampling=_populate_dataclass(SamplingConfig, raw.get("sampling", {})),
        detection=_populate_dataclass(DetectionConfig, raw.get("detection", {})),
        canvas=_populate_dataclass(CanvasConfig, raw.get("canvas", {})),
        training=_populate_dataclass(TrainingConfig, raw.get("training", {})),
        data=_populate_dataclass(DataConfig, raw.get("data", {})),
        curation=_populate_dataclass(CurationConfig, raw.get("curation", {})),
        evaluation=_populate_dataclass(EvaluationConfig, raw.get("evaluation", {})),
        memory_benchmark=_populate_dataclass(MemoryBenchmarkConfig, raw.get("memory_benchmark", {})),
    )
