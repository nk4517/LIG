"""LIG-specific renderer for visualizer"""
import typing
from dataclasses import dataclass

import imgui
import torch

if typing.TYPE_CHECKING:
    from gaussianlig import LIG


@dataclass
class RenderRequest:
    """Parameters for model rendering from GUI"""
    level_mode: int = 0  # 0=base (L0), 1=current active level
    residual_mode: bool = False
    scale_idx: int = 0
    accumulated: torch.Tensor | None = None


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


def _render_lig_model(
    model: "LIG",
    request: RenderRequest,
    checkerboard_cache: dict | None = None
) -> dict[str, torch.Tensor | None] | None:
    """
    Render LIG model with given parameters.
    
    Returns dict with keys: rendered, dx, dy, dxy, wsum
    Returns None if model not ready.
    """
    if model is None:
        return None
    
    has_level_models = hasattr(model, 'level_models') and model.level_models
    
    if has_level_models:
        if request.level_mode == 0:
            level_idx = 0
        else:
            level_idx = request.scale_idx
        level_idx = min(level_idx, len(model.level_models) - 1)
        
        target_model = model.level_models[level_idx]
        first_param = next(target_model.parameters(), None)
        if first_param is None or not first_param.is_cuda:
            return None
        
        result = target_model()
        rendered = result["render"].float()
        
        # Denormalization for levels > 0
        if level_idx > 0 and model.store_max and model.store_min:
            store_max = model.store_max[level_idx - 1]
            store_min = model.store_min[level_idx - 1]
            
            if not request.residual_mode:
                if request.accumulated is not None:
                    acc_upscaled = torch.nn.functional.interpolate(
                        request.accumulated.to(rendered.device),
                        size=(rendered.shape[2], rendered.shape[3]),
                        mode='bilinear'
                    )
                    rendered = rendered * (store_max - store_min) + acc_upscaled - 0.5 + store_min
            else:
                # Residual mode - blend with checkerboard
                h, w = rendered.shape[2], rendered.shape[3]
                if checkerboard_cache is not None:
                    cb = checkerboard_cache.get('cb')
                    if cb is None or cb.shape[2:] != (h, w):
                        cb = _make_checkerboard(h, w, cell_size=16, device=rendered.device)
                        checkerboard_cache['cb'] = cb
                else:
                    cb = _make_checkerboard(h, w, cell_size=16, device=rendered.device)
                rendered = cb * 0.1 + rendered * 0.9
    else:
        first_param = next(model.parameters(), None)
        if first_param is None or not first_param.is_cuda:
            return None
        result = model()
        rendered = result["render"].float()
    
    return {
        "rendered": rendered,
        "dx": result.get("dx"),
        "dy": result.get("dy"),
        "dxy": result.get("dxy"),
        "wsum": result.get("wsum")
    }


class LIGRenderer:
    """LIG-specific renderer with GUI controls"""
    
    def __init__(self):
        self.model: "LIG | None" = None
        self.level_mode: int = 0  # 0=base (L0), 1=current active level
        self.residual_mode: bool = False
        self.scale_idx: int = 0
        self.accumulated: torch.Tensor | None = None
        self._checkerboard_cache: dict = {}
    
    def set_model(self, model: "LIG"):
        self.model = model
    
    def set_scale_idx(self, idx: int):
        self.scale_idx = idx
    
    def set_accumulated(self, accumulated: torch.Tensor | None):
        self.accumulated = accumulated
    
    def render(self) -> dict[str, torch.Tensor | None] | None:
        """Render current model state"""
        if self.model is None:
            return None
        
        request = RenderRequest(
            level_mode=self.level_mode,
            residual_mode=self.residual_mode,
            scale_idx=self.scale_idx,
            accumulated=self.accumulated
        )
        return _render_lig_model(self.model, request, self._checkerboard_cache)
    
    def handle_key(self, key: int) -> bool:
        """
        Handle LIG-specific hotkeys.
        Returns True if key was handled and update needed.
        
        Keys: B=66, C=67, D=68 (GLFW constants)
        """
        if key == 66:  # B - Base level
            self.level_mode = 0
            return True
        elif key == 67:  # C - Current level
            self.level_mode = 1
            return True
        elif key == 68:  # D - Toggle residual
            self.residual_mode = not self.residual_mode
            return True
        return False
    
    def draw_gui_controls(self) -> bool:
        """
        Draw LIG-specific ImGui controls.
        Returns True if settings changed and re-render needed.
        """
        changed = False
        
        # Level selection
        level_controls_disabled = self.scale_idx == 0
        if level_controls_disabled:
            imgui.text("Render Level: (N/A at L0)")
        else:
            imgui.text("Render Level:")
        
        if imgui.radio_button("Base (L0) [B]", self.level_mode == 0):
            self.level_mode = 0
            changed = True
        imgui.same_line()
        if imgui.radio_button("Current [C]", self.level_mode == 1):
            self.level_mode = 1
            changed = True
        
        # Residual mode checkbox
        clicked, self.residual_mode = imgui.checkbox("Residual Mode [D]", self.residual_mode)
        if clicked:
            changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show raw output (residual) instead of accumulated")
        
        imgui.text(f"Active scale: {self.scale_idx}")
        
        return changed
