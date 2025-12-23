"""Common enums for GUI visualization"""
from enum import IntEnum
from dataclasses import dataclass, field

import torch


class VisMode(IntEnum):
    RENDER = 0
    TARGET = 2
    GROUND_TRUTH = 3
    WSUM = 5

@dataclass
class ShaderBinding:
    """Declarative shader configuration for renderer -> visualizer communication"""
    program: str                                              # program key: 'texture', 'upscale', 'colormap_lut', etc.
    textures: dict[str, str]                                  # uniform_name -> texture_slot_name
    uniforms_float: dict[str, float] = field(default_factory=dict)
    uniforms_vec2: dict[str, tuple[float, float]] = field(default_factory=dict)
    skip_texture_size: bool = False                           # skip texture_size uniform (for upscale etc.)
    geometry: str = 'fullscreen'                              # 'fullscreen' | 'sphere' | future types


def _make_checkerboard(h: int, w: int, cell_size: int = 16, device='cuda') -> torch.Tensor:
    """Пастельно-фиолетовый checkerboard [1, 3, H, W] torch формат"""
    color_light = torch.tensor([0.85, 0.75, 0.95], device=device)
    color_dark = torch.tensor([0.65, 0.55, 0.80], device=device)

    y_idx = torch.arange(h, device=device) // cell_size
    x_idx = torch.arange(w, device=device) // cell_size
    checker = (y_idx[:, None] + x_idx[None, :]) % 2  # [H, W]

    rgb = torch.where(checker[..., None] == 0,
                      color_light.view(1, 1, 3),
                      color_dark.view(1, 1, 3))  # [H, W, 3]

    # HWC -> CHW -> NCHW
    return rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def cv2_to_gl_rgba(img: "np.ndarray") -> torch.Tensor:
    """Convert cv2 BGR/BGRA uint8 image to HWC RGBA float32 tensor for GL upload"""
    import numpy as np
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w = img.shape[:2]
    if img.shape[2] == 4:
        # BGRA -> RGBA
        rgba = img[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    else:
        # BGR -> RGB + alpha
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[..., :3] = img[..., ::-1].astype(np.float32) / 255.0

    return torch.from_numpy(rgba.copy())


def chw_to_gl_rgba(tensor: torch.Tensor) -> torch.Tensor:
    """Convert NCHW/CHW tensor to HWC RGBA float for GL upload"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if tensor.dim() == 3:
        # CHW -> HWC
        out = tensor.detach().permute(1, 2, 0)
    else:
        # 2D -> stack to RGB
        out = tensor.detach()
        out = torch.stack([out] * 3, dim=-1)

    # Ensure RGBA
    if out.shape[2] == 3:
        alpha = torch.ones((*out.shape[:2], 1), dtype=out.dtype, device=out.device)
        out = torch.cat([out, alpha], dim=2)

    return torch.clamp(out, 0, 1)


def hwc_to_gl_rgba(tensor: torch.Tensor) -> torch.Tensor:
    """Convert HWC tensor to HWC RGBA float for GL upload"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    out = tensor.detach()
    if out.dim() == 2:
        out = torch.stack([out] * 3, dim=-1)

    if out.shape[2] == 3:
        alpha = torch.ones((*out.shape[:2], 1), dtype=out.dtype, device=out.device)
        out = torch.cat([out, alpha], dim=2)

    return torch.clamp(out, 0, 1)


def gradient_to_gl(gradient: torch.Tensor) -> torch.Tensor:
    """Prepare gradient tensor for GL upload (HWC RGBA)"""
    if gradient.dim() == 4:
        gradient = gradient.squeeze(0)
    # Input is already HWC
    if gradient.dim() == 2:
        gradient = gradient.unsqueeze(-1)

    # Replicate single channel to RGB
    if gradient.shape[2] == 1:
        gradient = gradient.repeat(1, 1, 3)

    # Add alpha
    if gradient.shape[2] == 3:
        alpha = torch.ones((*gradient.shape[:2], 1), dtype=gradient.dtype, device=gradient.device)
        gradient = torch.cat([gradient, alpha], dim=2)

    return gradient
