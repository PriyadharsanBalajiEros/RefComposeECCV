"""
Qwen3-VL Multimodal Scene & Object Annotator (Paper Section 4.1, 4.2, 4.4):
  1. Stage 1 Object Enumeration: Enumerates all distinct actors, props, and landmarks.
  2. Stage 5 Scene & Crop Annotation: Populates the 10-slot compositional template (Section 4.2).
  3. LayoutSAM-Eval VQA Scoring: Evaluates region-wise Spatial, Color, Texture, and Shape fidelity.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image
import torch

from refcompose.prompts.template import PromptTemplateBuilder, ScenePromptSlots


class Qwen3VLAnnotator:
    """
    Wraps `Qwen/Qwen3-VL-8B-Instruct` via HuggingFace `transformers`.
    Supports `use_mock_vlm=True` for fast offline testing and dry runs.
    """

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_mock_vlm: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.use_mock_vlm = use_mock_vlm
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self.use_mock_vlm or self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def _generate_vlm_response(
        self,
        images: Sequence[Image.Image],
        instruction: str,
        max_new_tokens: int = 512,
    ) -> str:
        self._lazy_load()
        assert self._processor is not None and self._model is not None

        content = [{"type": "image", "image": img.convert("RGB")} for img in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._processor(
            text=[text_input],
            images=[img.convert("RGB") for img in images],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)
        return decoded[0].strip()

    def enumerate_scene_objects(self, shot_image: Image.Image) -> List[str]:
        """
        Dataset Stage 1 (Section 4.1): Enumerate all distinct objects present in a movie shot
        (actors, props, and landmarks) as concise open-vocabulary detection phrases.
        """
        if self.use_mock_vlm:
            return [
                "person in foreground",
                "wooden bed",
                "vintage table lamp",
                "potted plant",
                "backpack on floor",
                "framed wall poster",
            ]

        instruction = (
            "Enumerate all distinct visible objects in this scene, including actors/people, "
            "furniture, props, cultural artifacts, and architectural landmarks. "
            'Return ONLY a JSON list of concise noun phrases, e.g. ["wooden chair", "brass lamp", ...].'
        )
        raw = self._generate_vlm_response([shot_image], instruction, max_new_tokens=256)
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
                if isinstance(items, list):
                    return [str(x).strip() for x in items if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [line.strip("- *•\t ") for line in raw.splitlines() if line.strip()]

    def describe_crop(self, crop_image: Image.Image, coarse_label: str = "object") -> str:
        """Dataset Stage 5: Produce a concise semantic description for a canonicalized object crop."""
        if self.use_mock_vlm:
            return f"a detailed {coarse_label}"

        instruction = (
            f"Provide a concise one-phrase visual description of this {coarse_label}, "
            "mentioning its object category, material, and primary color in under 12 words."
        )
        return self._generate_vlm_response([crop_image], instruction, max_new_tokens=64)

    def annotate_scene_slots(
        self,
        scene_image: Image.Image,
        crop_descriptions: Sequence[str],
    ) -> Tuple[ScenePromptSlots, str]:
        """
        Dataset Stage 5 & Benchmark Evaluation (Section 4.2 & 4.4):
        Populate the 10-slot RefCompose prompt template from a scene image and crop descriptions.
        """
        if self.use_mock_vlm:
            prompt = PromptTemplateBuilder.from_object_descriptions(crop_descriptions)
            slots = ScenePromptSlots(
                primary_objects=", ".join(crop_descriptions[1:3]) if len(crop_descriptions) > 1 else "main subject",
                central_subject=crop_descriptions[0] if crop_descriptions else "central subject",
                secondary_objects=", ".join(crop_descriptions[3:]) if len(crop_descriptions) > 3 else "ambient props",
            )
            return slots, prompt

        instruction = PromptTemplateBuilder.build_qwen3_vl_slot_extraction_prompt(crop_descriptions)
        raw = self._generate_vlm_response([scene_image], instruction, max_new_tokens=384)
        slots = PromptTemplateBuilder.parse_qwen3_vl_json_response(raw)
        rendered = PromptTemplateBuilder.render(slots, num_references=len(crop_descriptions))
        return slots, rendered

    def score_region_attributes_vqa(
        self,
        generated_crop: Image.Image,
        reference_crop: Image.Image,
        object_label: str,
    ) -> Dict[str, float]:
        """
        LayoutSAM-Eval VQA region attribute scoring comparing a generated region crop
        against its target reference crop across Color, Texture, and Shape in [0.0, 1.0].
        """
        if self.use_mock_vlm:
            return {"color": 0.85, "texture": 0.85, "shape": 0.85}

        instruction = (
            f"Image 1 is the reference `{object_label}` and Image 2 is the generated `{object_label}` crop. "
            "Score how faithfully Image 2 preserves Image 1's Color, Texture, and Shape on a scale from 0.0 to 1.0. "
            'Respond ONLY with JSON: {"color": 0.0, "texture": 0.0, "shape": 0.0}.'
        )
        raw = self._generate_vlm_response([reference_crop, generated_crop], instruction, max_new_tokens=96)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return {
                    "color": float(max(0.0, min(1.0, float(parsed.get("color", 0.5))))),
                    "texture": float(max(0.0, min(1.0, float(parsed.get("texture", 0.5))))),
                    "shape": float(max(0.0, min(1.0, float(parsed.get("shape", 0.5))))),
                }
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        return {"color": 0.5, "texture": 0.5, "shape": 0.5}
