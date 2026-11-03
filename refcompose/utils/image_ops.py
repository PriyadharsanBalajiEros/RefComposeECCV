"""Image, tensor, depth visualization, and grid rendering utilities."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def pil_to_tensor(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    normalize_minus1_to_1: bool = True,
) -> torch.Tensor:
    """
    Convert a PIL RGB image to a float32 tensor of shape (3, H, W).
    `target_size` uses (width, height) convention to match PIL.
    """
    img = image.convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize_minus1_to_1:
        tensor = tensor * 2.0 - 1.0
    return tensor


def tensor_to_pil(
    tensor: torch.Tensor,
    is_minus1_to_1: bool = True,
) -> Image.Image:
    """Convert a (3, H, W) or (1, 3, H, W) tensor back into a PIL RGB image."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    if is_minus1_to_1:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def depth_to_rgb_tensor(
    depth_map: np.ndarray | torch.Tensor,
    normalize_minus1_to_1: bool = True,
) -> torch.Tensor:
    """
    Convert a single-channel depth map D_s in R^{H x W} (Eq. 5) into a 3-channel
    tensor in R^{3 x H x W} suitable for VAE encoding z_{D_s} = E(D_s) (Eq. 7).
    """
    if isinstance(depth_map, np.ndarray):
        d = torch.from_numpy(depth_map).float()
    else:
        d = depth_map.detach().cpu().float()

    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    elif d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]

    d_min = float(d.min())
    d_max = float(d.max())
    if d_max - d_min > 1e-8:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = torch.zeros_like(d)

    rgb = d.unsqueeze(0).repeat(3, 1, 1).contiguous()
    if normalize_minus1_to_1:
        rgb = rgb * 2.0 - 1.0
    return rgb


def draw_boxes_overlay(
    image: Image.Image,
    boxes_xywh: Sequence[Tuple[float, float, float, float]],
    labels: Optional[Sequence[str]] = None,
    depths: Optional[Sequence[float]] = None,
) -> Image.Image:
    """Render annotated bounding boxes and depth order onto a copy of `image`."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    palette = [
        (239, 68, 68),
        (59, 130, 246),
        (16, 185, 129),
        (245, 158, 11),
        (168, 85, 247),
        (236, 72, 153),
        (20, 184, 166),
        (249, 115, 22),
    ]
    for idx, (x, y, w, h) in enumerate(boxes_xywh):
        color = palette[idx % len(palette)]
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=color, width=3)
        tag = labels[idx] if labels and idx < len(labels) else f"ref_{idx}"
        if depths is not None and idx < len(depths):
            tag = f"{tag} (d={depths[idx]:.2f})"
        draw.rectangle([x, max(0, y - 18), x + max(80, len(tag) * 7), y], fill=color)
        draw.text((x + 4, max(0, y - 16)), tag, fill=(255, 255, 255))
    return vis


def make_comparison_grid(
    rows: List[List[Image.Image]],
    column_headers: Optional[List[str]] = None,
    cell_size: Tuple[int, int] = (384, 216),
    padding: int = 8,
) -> Image.Image:
    """
    Assemble a multi-row, multi-column visual comparison grid (e.g., Paper Figure 4:
    Ground Truth | RefCompose | CreatiLayout | InstanceAssemble | UNO | XVerse).
    """
    if not rows or not rows[0]:
        raise ValueError("`rows` must contain at least one non-empty row of PIL Images.")

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows)
    cell_w, cell_h = cell_size
    header_h = 34 if column_headers else 0

    total_w = num_cols * cell_w + (num_cols + 1) * padding
    total_h = header_h + num_rows * cell_h + (num_rows + 1) * padding

    grid = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    if column_headers:
        for col_idx, title in enumerate(column_headers[:num_cols]):
            cx = padding + col_idx * (cell_w + padding) + cell_w // 2 - len(title) * 3
            draw.text((max(padding, cx), 10), title, fill=(20, 20, 20), font=font)

    for r_idx, row in enumerate(rows):
        y0 = header_h + padding + r_idx * (cell_h + padding)
        for c_idx, img in enumerate(row):
            x0 = padding + c_idx * (cell_w + padding)
            resized = img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.BICUBIC)
            grid.paste(resized, (x0, y0))

    return grid

