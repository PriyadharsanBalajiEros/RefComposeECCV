"""
End-to-End 3-Stage RefCompose Compositional Pipeline (Paper Section 3.1-3.4 & Figure 2).

Supports:
  - Standard multi-reference inference for arbitrary N (N=1..15+) at constant token budget.
  - Multi-reference flexibility (Paper Figure 5): swap distinct reference sets R_1, R_2 on shared layout L & depth D_s.
  - Depth-controlled orientation (Paper Figure 6): hold references fixed while editing depth map D_s.
  - Ablation modes (Paper Section 4.6 & Figure 7):
      * "full":               Both reference canvas z_C and depth prior z_{D_s} via decoupled LoRA streams
      * "wo_canvas":          Remove reference canvas (black canvas C=0, appearance_scale=0.0)
      * "wo_depth":           Remove depth map prior (structure_scale=0.0)
      * "merged_single_lora": Route merged depth + canvas through a single combined LoRA stream
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
from PIL import Image
import torch

from refcompose.canvas.composer import BoundingBox, ReferenceCanvasComposer
from refcompose.config import RefComposeConfig
from refcompose.models.depth_estimator import DepthEstimator
from refcompose.models.detector import GroundingDINOLocalizer
from refcompose.models.dsa_lora import iter_dsa_processors, load_dsa_lora_weights
from refcompose.models.flux2_wrapper import Flux2BackboneWrapper
from refcompose.prompts.template import PromptTemplateBuilder, ScenePromptSlots
from refcompose.utils.image_ops import depth_to_rgb_tensor, draw_boxes_overlay, tensor_to_pil


@dataclass
class RefComposeOutput:
    """Complete output artifacts produced by a RefCompose generation call."""

    generated_image: Image.Image
    prompt: str
    layout_image: Optional[Image.Image]
    depth_map: Optional[np.ndarray]
    depth_vis_image: Optional[Image.Image]
    canvas_image: Image.Image
    boxes_overlay_image: Optional[Image.Image]
    localized_boxes: List[BoundingBox]
    depth_sorted_order: List[int]
    mean_box_depths: List[float]


class RefComposePipeline:
    """
    Production & research pipeline orchestrating Stage 1 (Prompt-Induced Layout),
    Stage 2 (Depth Scaffolding & Canvas Assembly), and Stage 3 (DSA LoRA Denoising).
    """

    def __init__(
        self,
        cfg: RefComposeConfig,
        lora_checkpoint_dir: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        use_mock_models: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_models = use_mock_models

        self.backbone = Flux2BackboneWrapper(
            cfg=cfg,
            device=self.device,
            use_mock_backbone=use_mock_models,
            inject_lora=True,
        )
        if lora_checkpoint_dir is not None and Path(lora_checkpoint_dir).exists():
            # Paper Section 4.3: loads Subject/Appearance LoRA before Depth/Structure LoRA
            load_dsa_lora_weights(
                self.backbone.transformer,
                lora_checkpoint_dir,
                load_appearance=True,
                load_structure=True,
                strict=False,
            )

        self.detector = GroundingDINOLocalizer(
            repo_id=cfg.model.grounding_dino_repo_id,
            cfg=cfg.detection,
            device=self.device,
            use_mock_detector=use_mock_models,
        )
        self.depth_estimator = DepthEstimator(
            da3_repo_id=cfg.model.depth_anything3_repo_id,
            fallback_repo_id=cfg.model.depth_fallback_repo_id,
            device=self.device,
            use_mock_depth=use_mock_models,
        )
        self.composer = ReferenceCanvasComposer(
            width=cfg.resolution.width,
            height=cfg.resolution.height,
            canvas_cfg=cfg.canvas,
            aug_cfg=cfg.training.augmentations,
        )

    def _set_merged_ablation_flag(self, enabled: bool) -> None:
        for _, proc in iter_dsa_processors(self.backbone.transformer):
            proc.cfg.merged_single_stream_ablation = enabled

    @torch.no_grad()
    def __call__(
        self,
        references: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        prompt: Optional[str] = None,
        prompt_slots: Optional[ScenePromptSlots] = None,
        override_layout_image: Optional[Image.Image] = None,
        override_boxes: Optional[Sequence[BoundingBox]] = None,
        override_depth_map: Optional[np.ndarray] = None,
        ablation_mode: str = "full",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> RefComposeOutput:
        """
        Execute the full 3-Stage RefCompose synthesis pipeline:
          Stage 1: L = F_theta(p),  B = G(L),  D_s = Psi(L)
          Stage 2: C = Paste(phi(I_{pi(1)}, b_{pi(1)}), ...),  z_C = E(C),  z_{D_s} = E(D_s)
          Stage 3: z_hat_0 = F_theta(p, z_T, z_C, z_{D_s}; Delta W_s + Delta W_a),  I_hat = E^{-1}(z_hat_0)
        """
        w, h = self.cfg.resolution.width, self.cfg.resolution.height

        # Resolve structured prompt p (Section 4.2)
        if prompt is None:
            if prompt_slots is not None:
                prompt = PromptTemplateBuilder.render(
                    prompt_slots, num_references=len(references)
                )
            else:
                prompt = PromptTemplateBuilder.from_object_descriptions(reference_descriptions)

        # ---------------- Stage 1: Prompt-Induced Layout Generation ----------------
        if override_layout_image is not None:
            layout_img = override_layout_image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        elif override_boxes is not None and override_depth_map is not None:
            layout_img = None
        else:
            layout_img = self.backbone.generate_layout_image(
                prompt=prompt,
                height=h,
                width=w,
                seed=seed,
            )

        if override_boxes is not None:
            boxes = list(override_boxes)
        else:
            assert layout_img is not None
            boxes = self.detector.localize_references(layout_img, reference_descriptions)

        if override_depth_map is not None:
            depth_map = override_depth_map
        elif ablation_mode == "wo_depth":
            depth_map = None
        else:
            assert layout_img is not None
            depth_map = self.depth_estimator.estimate_depth(layout_img)

        # ---------------- Stage 2: Reference Canvas & Depth Encoding ---------------
        black_canvas = ablation_mode == "wo_canvas"
        comp_out = self.composer.compose(
            references=references,
            boxes=boxes,
            depth_map=depth_map,
            higher_depth_is_farther=True,
            apply_stage3_perturbations=False,
            black_canvas_override=black_canvas,
        )

        # Constant-size latent encoding |z_C| = const regardless of N (Eq. 4 & Eq. 7)
        z_canvas: Optional[torch.Tensor] = None
        z_depth: Optional[torch.Tensor] = None
        app_scale = self.cfg.lora.appearance_scale
        struct_scale = self.cfg.lora.structure_scale

        if ablation_mode == "wo_canvas":
            app_scale = 0.0
        else:
            z_canvas = self.backbone.encode_image_to_packed_latents(comp_out.canvas_tensor)

        if ablation_mode == "wo_depth" or comp_out.depth_tensor is None:
            struct_scale = 0.0
        else:
            z_depth = self.backbone.encode_image_to_packed_latents(comp_out.depth_tensor)

        # ---------------- Stage 3: Decoupled Structure-Appearance Sampling ---------
        is_merged = ablation_mode == "merged_single_lora"
        self._set_merged_ablation_flag(is_merged)
        try:
            generated_list = self.backbone.sample_flow_matching(
                prompt=prompt,
                z_canvas=z_canvas,
                z_depth=z_depth,
                height=h,
                width=w,
                num_steps=num_inference_steps or self.cfg.sampling.num_inference_steps,
                guidance_scale=guidance_scale or self.cfg.sampling.guidance_scale,
                appearance_scale=app_scale,
                structure_scale=struct_scale,
                seed=seed,
            )
        finally:
            self._set_merged_ablation_flag(self.cfg.lora.merged_single_stream_ablation)

        depth_vis: Optional[Image.Image] = None
        if depth_map is not None:
            depth_vis = tensor_to_pil(depth_to_rgb_tensor(depth_map, normalize_minus1_to_1=True))

        overlay_vis: Optional[Image.Image] = None
        if layout_img is not None and boxes:
            overlay_vis = draw_boxes_overlay(
                layout_img,
                boxes_xywh=[b.to_xywh() for b in boxes],
                labels=[b.label for b in boxes],
                depths=comp_out.mean_depths if comp_out.mean_depths else None,
            )

        return RefComposeOutput(
            generated_image=generated_list[0],
            prompt=prompt,
            layout_image=layout_img,
            depth_map=depth_map,
            depth_vis_image=depth_vis,
            canvas_image=comp_out.canvas_pil,
            boxes_overlay_image=overlay_vis,
            localized_boxes=boxes,
            depth_sorted_order=comp_out.sorted_order,
            mean_box_depths=comp_out.mean_depths,
        )

    @torch.no_grad()
    def generate_multi_reference_swap(
        self,
        reference_sets: Sequence[Sequence[Image.Image]],
        reference_descriptions: Sequence[str],
        prompt: str,
        seed: int = 42,
    ) -> List[RefComposeOutput]:
        """
        Paper Figure 5 (Multi-reference flexibility):
        Induce layout L, boxes B, and depth D_s ONCE from `prompt`, then synthesize
        consistent scenes across multiple distinct reference sets {R_1, R_2, ...}.
        """
        if not reference_sets:
            return []

        first_out = self(
            references=reference_sets[0],
            reference_descriptions=reference_descriptions,
            prompt=prompt,
            seed=seed,
        )
        outputs = [first_out]
        for ref_set in reference_sets[1:]:
            swapped = self(
                references=ref_set,
                reference_descriptions=reference_descriptions,
                prompt=prompt,
                override_layout_image=first_out.layout_image,
                override_boxes=first_out.localized_boxes,
                override_depth_map=first_out.depth_map,
                seed=seed,
            )
            outputs.append(swapped)
        return outputs

    @torch.no_grad()
    def generate_depth_controlled(
        self,
        references: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        prompts: Sequence[str],
        depth_maps: Sequence[np.ndarray],
        shared_boxes: Optional[Sequence[BoundingBox]] = None,
        seed: int = 42,
    ) -> List[RefComposeOutput]:
        """
        Paper Figure 6 (Depth-controlled orientation):
        Hold `references` fixed while varying `depth_maps` (e.g., static sweet shop vs
        dynamic floating debris depth map).
        """
        outputs: List[RefComposeOutput] = []
        boxes_to_use = shared_boxes
        for idx, d_map in enumerate(depth_maps):
            p = prompts[min(idx, len(prompts) - 1)]
            if boxes_to_use is None:
                w, h = self.cfg.resolution.width, self.cfg.resolution.height
                boxes_to_use = self.detector._generate_fallback_grid_boxes(
                    len(references), w, h, reference_descriptions
                )
            out = self(
                references=references,
                reference_descriptions=reference_descriptions,
                prompt=p,
                override_boxes=boxes_to_use,
                override_depth_map=d_map,
                seed=seed,
            )
            outputs.append(out)
        return outputs

