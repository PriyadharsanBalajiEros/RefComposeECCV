"""
Structured Inference Prompt Template & Scale Steering (Paper Section 4.2).

Exact paper template:
    "A [shot type] of a [scene/environment] featuring [primary objects] arranged
     around [central subject], alongside [secondary objects/details] placed naturally
     throughout the scene. Captured from a [camera perspective] using a [lens type],
     with [lighting condition], [mood/style descriptors], and [photorealistic/cinematic
     rendering style]."
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


PAPER_PROMPT_TEMPLATE = (
    "A {shot_type} of a {scene_environment} featuring {primary_objects} arranged "
    "around {central_subject}, alongside {secondary_objects} placed naturally "
    "throughout the scene. Captured from a {camera_perspective} using a {lens_type}, "
    "with {lighting_condition}, {mood_style}, and {rendering_style}."
)


@dataclass
class ScenePromptSlots:
    """Bracketed slots populated automatically from Qwen3-VL scene & crop descriptions."""

    shot_type: str = "wide-angle cinematic shot"
    scene_environment: str = "detailed realistic environment"
    primary_objects: str = "key foreground subjects"
    central_subject: str = "the main focal element in the center"
    secondary_objects: str = "complementary props and background elements"
    camera_perspective: str = "slightly elevated eye-level camera perspective"
    lens_type: str = "35mm cinematic prime lens"
    lighting_condition: str = "soft natural directional lighting"
    mood_style: str = "warm, balanced, culturally authentic atmosphere"
    rendering_style: str = "photorealistic cinematic rendering style"
    # Optional priority subject indices or names for relational scale boosting when N is large
    priority_subjects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenePromptSlots":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class PromptTemplateBuilder:
    """
    Builds object-centric compositional prompts `p` with positional and relational scale
    language while delegating fine-grained photometric identity to the reference canvas.
    """

    @staticmethod
    def render(
        slots: ScenePromptSlots,
        num_references: Optional[int] = None,
        priority_scale_descriptor: str = "prominent, large-scale foreground",
    ) -> str:
        """
        Render the exact Section 4.2 structured prompt.
        When `num_references` is large (e.g., N > 5) and `slots.priority_subjects` is given,
        applies relational scale descriptors so priority subjects receive proportionally
        larger boxes in L = F_theta(p) (Section 4.2, paragraph 2).
        """
        primary_str = slots.primary_objects.strip()
        central_str = slots.central_subject.strip()

        if slots.priority_subjects and (num_references is None or num_references > 5):
            highlighted = ", ".join(
                f"a {priority_scale_descriptor} {subj}" for subj in slots.priority_subjects
            )
            if highlighted.lower() not in primary_str.lower():
                primary_str = f"{highlighted}, {primary_str}"

        prompt = PAPER_PROMPT_TEMPLATE.format(
            shot_type=slots.shot_type.strip(),
            scene_environment=slots.scene_environment.strip(),
            primary_objects=primary_str,
            central_subject=central_str,
            secondary_objects=slots.secondary_objects.strip(),
            camera_perspective=slots.camera_perspective.strip(),
            lens_type=slots.lens_type.strip(),
            lighting_condition=slots.lighting_condition.strip(),
            mood_style=slots.mood_style.strip(),
            rendering_style=slots.rendering_style.strip(),
        )
        return re.sub(r"\s+", " ", prompt).strip()

    @staticmethod
    def from_object_descriptions(
        object_labels: Sequence[str],
        scene_environment: str = "cozy modern interior room",
        spatial_relations: Optional[Sequence[str]] = None,
        priority_indices: Optional[Sequence[int]] = None,
        shot_type: str = "cinematic medium-wide shot",
        camera_perspective: str = "eye-level camera perspective",
        lens_type: str = "35mm cinematic lens",
        lighting_condition: str = "soft natural studio and ambient lighting",
        mood_style: str = "highly detailed, balanced, inviting mood",
        rendering_style: str = "photorealistic cinematic rendering style",
    ) -> str:
        """
        Construct the Section 4.2 compositional prompt automatically from a list of N
        reference object labels and optional positional hints (e.g. 'on the left').
        """
        if not object_labels:
            return PAPER_PROMPT_TEMPLATE.format(**ScenePromptSlots().to_dict())

        annotated_objs: List[str] = []
        for idx, label in enumerate(object_labels):
            clean = label.strip()
            if spatial_relations and idx < len(spatial_relations) and spatial_relations[idx]:
                clean = f"{clean} ({spatial_relations[idx].strip()})"
            annotated_objs.append(clean)

        central_subject = annotated_objs[0]
        if len(annotated_objs) == 1:
            primary_objects = annotated_objs[0]
            secondary_objects = "subtle contextual details"
        elif len(annotated_objs) <= 4:
            primary_objects = ", ".join(annotated_objs[1:])
            secondary_objects = "natural environmental details"
        else:
            split_idx = max(2, (len(annotated_objs) + 1) // 2)
            primary_objects = ", ".join(annotated_objs[1:split_idx])
            secondary_objects = ", ".join(annotated_objs[split_idx:])

        priority_subjects: List[str] = []
        if priority_indices:
            for p_idx in priority_indices:
                if 0 <= p_idx < len(object_labels):
                    priority_subjects.append(object_labels[p_idx].strip())

        slots = ScenePromptSlots(
            shot_type=shot_type,
            scene_environment=scene_environment,
            primary_objects=primary_objects,
            central_subject=central_subject,
            secondary_objects=secondary_objects,
            camera_perspective=camera_perspective,
            lens_type=lens_type,
            lighting_condition=lighting_condition,
            mood_style=mood_style,
            rendering_style=rendering_style,
            priority_subjects=priority_subjects,
        )
        return PromptTemplateBuilder.render(slots, num_references=len(object_labels))

    @staticmethod
    def build_qwen3_vl_slot_extraction_prompt(crop_descriptions: Sequence[str]) -> str:
        """
        System/user instruction sent to Qwen3-VL (Section 4.1 & 4.2) to populate
        all 10 bracketed slots of the RefCompose compositional prompt template.
        """
        crops_text = "\n".join(f"- Object {i + 1}: {desc}" for i, desc in enumerate(crop_descriptions))
        return (
            "Analyze the scene image together with the following detected reference objects:\n"
            f"{crops_text}\n\n"
            "Return ONLY a valid JSON object populating these exact compositional slots "
            "(focus on spatial arrangement, framing, and lighting rather than fine-grained identity):\n"
            "{\n"
            '  "shot_type": "e.g., wide-angle interior shot / medium close-up shot",\n'
            '  "scene_environment": "e.g., cozy modern bedroom / regal palace courtyard",\n'
            '  "primary_objects": "positional list of main foreground objects",\n'
            '  "central_subject": "central anchor object/person and its placement",\n'
            '  "secondary_objects": "remaining props/landmarks and their natural positions",\n'
            '  "camera_perspective": "e.g., slightly elevated eye-level camera perspective",\n'
            '  "lens_type": "e.g., 24-28mm wide-angle lens / 50mm cinematic prime lens",\n'
            '  "lighting_condition": "e.g., soft golden-hour natural lighting",\n'
            '  "mood_style": "e.g., warm, royal, culturally authentic atmosphere",\n'
            '  "rendering_style": "photorealistic cinematic rendering style"\n'
            "}"
        )

    @staticmethod
    def parse_qwen3_vl_json_response(raw_text: str) -> ScenePromptSlots:
        """Robustly parse JSON slot output from Qwen3-VL into `ScenePromptSlots`."""
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return ScenePromptSlots()
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return ScenePromptSlots.from_dict(
                    {k: str(v) for k, v in parsed.items() if isinstance(v, (str, int, float))}
                )
        except json.JSONDecodeError:
            pass
        return ScenePromptSlots()

