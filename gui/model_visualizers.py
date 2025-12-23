"""Model-specific visualizer subclasses"""
import typing

import glfw
import imgui
import OpenGL.GL as gl
import torch

from .visualizer import VisualizerGUI
from .common import VisMode, GradientMode, ShaderBinding, _make_checkerboard, chw_to_gl_rgba, hwc_to_gl_rgba, gradient_to_gl
from upscaler_torch import torch_gradient_aware_upscale
from gsplat2d import gradient_aware_upscale

if typing.TYPE_CHECKING:
    from gaussianlig import LIG
    from progressive_module import ProgressiveGaussian2D
    from minimal_trainer_upscale import Gaussian2DMinimal


class Gaussian2DVisualizerGUI(VisualizerGUI):
    """Visualizer for ProgressiveGaussian2D models"""

    TEXTURE_SLOTS: tuple[tuple[str, int], ...] = (
        ('render', gl.GL_LINEAR),
        ('dx', gl.GL_NEAREST),
        ('dy', gl.GL_NEAREST),
        ('dxy', gl.GL_NEAREST),
        ('gt', gl.GL_LINEAR),
        ('wsum', gl.GL_NEAREST),
    )
    HOTKEYS: tuple[tuple[int, str], ...] = ()

    def __init__(self, width=800, height=600, use_cuda=True):
        self.model: "Gaussian2DMinimal|ProgressiveGaussian2D | None" = None
        self._gt_source_id: int | None = None
        self.gt_image_gl: torch.Tensor | None = None
        self.wsum_min: float = 0.0
        self.wsum_max: float = 1.0


        super().__init__(width, height, use_cuda)


    def set_model(self, model: "Gaussian2DMinimal|ProgressiveGaussian2D", gt_image: torch.Tensor | None = None):
        self.model = model
        if gt_image is not None:
            self.set_gt_image(gt_image)

    def set_gt_image(self, gt_image: torch.Tensor):
        new_id = id(gt_image)
        if self._gt_source_id != new_id:
            self._gt_source_id = new_id
            self.gt_image_gl = chw_to_gl_rgba(gt_image)

    def render_model(self) -> dict[str, torch.Tensor | None] | None:
        if self.model is None:
            return None
        first_param = next(self.model.parameters(), None)
        if first_param is None or not first_param.is_cuda:
            return None
        result = self.model()
        rendered = result["render"].float()
        return {
            "rendered": rendered,
            "render_hwc": result.get("render_hwc"),
            "dx": result.get("dx"),
            "dy": result.get("dy"),
            "dxy": result.get("dxy"),
            "wsum": result.get("wsum")
        }

    def handle_key(self, key: int) -> bool:
        return False

    def draw_gui_controls(self) -> bool:
        return False

    def draw_info_gui(self):
        pass

    def get_display_size(self) -> tuple[int, int] | None:
        if self.model is None:
            return None
        if hasattr(self.model, 'H') and hasattr(self.model, 'W'):
            return (self.model.H, self.model.W)
        return None

    def can_display(self, vis_mode: VisMode) -> bool:
        if vis_mode in (VisMode.RENDER, VisMode.UPSCALED, VisMode.UPSCALED_TORCH, VisMode.UPSCALED_CUDA, VisMode.GRADIENTS, VisMode.WSUM):
            return self.model is not None
        if vis_mode == VisMode.GROUND_TRUTH:
            return self.gt_image_gl is not None
        return False

    def get_shader_binding(self, vis_mode: VisMode, gradient_mode: GradientMode) -> ShaderBinding:
        if vis_mode == VisMode.UPSCALED:
            return ShaderBinding(
                program='upscale',
                textures={'texRender': 'nearest_src', 'texDx': 'dx', 'texDy': 'dy', 'texDxy': 'dxy'},
                skip_texture_size=True
            )
        if vis_mode == VisMode.WSUM:
            return ShaderBinding(
                program='colormap_lut',
                textures={'texData': 'wsum', 'texLUT': 'viridis'},
                uniforms_float={'data_min': self.wsum_min, 'data_max': self.wsum_max}
            )
        if vis_mode == VisMode.GRADIENTS and gradient_mode == GradientMode.MAGNITUDE:
            return ShaderBinding(
                program='magnitude_lut',
                textures={'texDx': 'dx', 'texDy': 'dy', 'texLUT': 'batlow'},
                uniforms_float={'scale': 2.0}
            )
        if vis_mode == VisMode.GRADIENTS:
            return ShaderBinding(
                program='gradient_lut',
                textures={'texSampler': 'nearest_src', 'texLUT': 'icefire'},
                uniforms_float={'scale': 5.0}
            )
        if vis_mode == VisMode.GROUND_TRUTH:
            return ShaderBinding(program='texture', textures={'texSampler': 'gt'})
        return ShaderBinding(program='texture', textures={'texSampler': 'render'})

    def try_capture(self, model: "ProgressiveGaussian2D",
                    gt: torch.Tensor | None = None, stats: dict | None = None,
                    changed: bool = True):
        need_render = changed or self.view_dirty.is_set()
        if not self.gui_ready.is_set() and not need_render:
            return

        self.model = model
        if gt is not None:
            self.set_gt_image(gt)

        if self.gui_ready.is_set() and need_render:
            with torch.no_grad():
                render_data = self.render_model()

            with self.lock:
                self.tensor_buffer = render_data if render_data else {}
                self.tensor_buffer["gt"] = self.gt_image_gl
                self.stats = stats or {}
                self.has_new_frame = True

            self.gui_ready.clear()
            self.view_dirty.clear()
        else:
            with self.lock:
                self.stats = stats or {}
                self.has_new_frame = need_render

    def prepare_for_display(
            self,
            render_data: dict,
            vis_mode: VisMode,
            gradient_mode: GradientMode = GradientMode.DX,
            target_size: tuple[int, int] | None = None,
            roi: tuple[int, int, int, int] | None = None) -> dict[str, torch.Tensor]:
        result = {}

        dx = render_data.get("dx")
        dy = render_data.get("dy")
        dxy = render_data.get("dxy")
        wsum = render_data.get("wsum")
        render_hwc = render_data.get("render_hwc")

        if vis_mode == VisMode.GROUND_TRUTH and render_data.get("gt") is not None:
            result["gt"] = render_data["gt"]
            return result

        if vis_mode == VisMode.WSUM and wsum is not None:
            if wsum.dim() == 4:
                wsum = wsum.squeeze(0)
            if wsum.dim() == 3:
                wsum = wsum.mean(dim=-1)
            self.wsum_min = float(torch.quantile(wsum.flatten(), 0.01))
            self.wsum_max = float(torch.quantile(wsum.flatten(), 0.99))
            wsum_rgba = wsum.unsqueeze(-1).repeat(1, 1, 4)
            wsum_rgba[..., 3] = 1.0
            result["wsum"] = wsum_rgba.float()
            return result

        if render_hwc is None:
            return result

        rendered_gl = hwc_to_gl_rgba(render_hwc)

        if vis_mode == VisMode.UPSCALED and dx is not None and dy is not None and dxy is not None:
            result["nearest_src"] = rendered_gl
            result["dx"] = gradient_to_gl(dx)
            result["dy"] = gradient_to_gl(dy)
            result["dxy"] = gradient_to_gl(dxy)
            return result

        if vis_mode == VisMode.GRADIENTS:
            if gradient_mode == GradientMode.MAGNITUDE and dx is not None and dy is not None:
                result["render"] = rendered_gl
                result["dx"] = gradient_to_gl(dx)
                result["dy"] = gradient_to_gl(dy)
                if dxy is not None:
                    result["dxy"] = gradient_to_gl(dxy)
                return result
            grad_map = {GradientMode.DX: dx, GradientMode.DY: dy, GradientMode.DXY: dxy}
            grad = grad_map.get(gradient_mode)
            if grad is not None:
                result["nearest_src"] = gradient_to_gl(grad.float())
                return result

        result["render"] = rendered_gl
        return result


class LIGVisualizerGUI(Gaussian2DVisualizerGUI):
    """Visualizer for LIG models with multi-level support"""

    TEXTURE_SLOTS: tuple[tuple[str, int], ...] = (
        ('render', gl.GL_LINEAR),
        ('dx', gl.GL_NEAREST),
        ('dy', gl.GL_NEAREST),
        ('dxy', gl.GL_NEAREST),
        ('target', gl.GL_LINEAR),
        ('gt', gl.GL_LINEAR),
        ('wsum', gl.GL_NEAREST),
    )

    HOTKEYS: tuple[tuple[int, str], ...] = (
        (glfw.KEY_B, "B - Base level (L0)"),
        (glfw.KEY_C, "C - Current level"),
        (glfw.KEY_D, "D - Toggle residual"),
    )

    def __init__(self, width=800, height=600, use_cuda=True):
        self.level_mode: int = 0
        self.residual_mode: bool = False
        self.scale_idx: int = 0
        self.accumulated: torch.Tensor | None = None
        self._acc_upscaled_cache: tuple[torch.Tensor, tuple[int, int]] | None = None
        self._target_source_id: int | None = None
        self.target_image_gl: torch.Tensor | None = None
        self._checkerboard_cache: dict = {}

        super().__init__(width, height, use_cuda)
        self.model: "LIG | None" = None

    def set_model(self, model: "LIG", gt_image: torch.Tensor | None = None):
        self.model = model
        if gt_image is not None:
            self.set_gt_image(gt_image)

    def set_scale_idx(self, idx: int):
        self.scale_idx = idx

    def set_accumulated(self, accumulated: torch.Tensor | None):
        if accumulated is not self.accumulated:
            self._acc_upscaled_cache = None
        self.accumulated = accumulated

    def set_target_image(self, target: torch.Tensor | None):
        if target is None:
            self._target_source_id = None
            self.target_image_gl = None
            return
        new_id = id(target)
        if self._target_source_id != new_id:
            self._target_source_id = new_id
            self.target_image_gl = chw_to_gl_rgba(target)

    def render_model(self) -> dict[str, torch.Tensor | None] | None:
        if self.model is None:
            return None

        has_level_models = hasattr(self.model, 'level_models') and self.model.level_models

        if has_level_models:
            if self.level_mode == 0:
                level_idx = 0
            else:
                level_idx = self.scale_idx
            level_idx = min(level_idx, len(self.model.level_models) - 1)

            target_model = self.model.level_models[level_idx]
            first_param = next(target_model.parameters(), None)
            if first_param is None or not first_param.is_cuda:
                return None

            result = target_model()
            rendered = result["render"]

            if level_idx > 0 and self.model.store_max and self.model.store_min:
                store_max = self.model.store_max[level_idx - 1]
                store_min = self.model.store_min[level_idx - 1]

                if not self.residual_mode:
                    if self.accumulated is not None:
                        target_size = (rendered.shape[2], rendered.shape[3])
                        if self._acc_upscaled_cache is not None and self._acc_upscaled_cache[1] == target_size:
                            acc_upscaled = self._acc_upscaled_cache[0]
                        else:
                            acc_upscaled = torch.nn.functional.interpolate(
                                self.accumulated.to(rendered.device),
                                size=target_size,
                                mode='bilinear'
                            )
                            self._acc_upscaled_cache = (acc_upscaled, target_size)
                        rendered = rendered * (store_max - store_min) + acc_upscaled - 0.5 + store_min
                else:
                    h, w = rendered.shape[2], rendered.shape[3]
                    cb = self._checkerboard_cache.get('cb')
                    if cb is None or cb.shape[2:] != (h, w):
                        cb = _make_checkerboard(h, w, cell_size=16, device=rendered.device)
                        self._checkerboard_cache['cb'] = cb
                    rendered = cb * 0.1 + rendered * 0.9
        else:
            first_param = next(self.model.parameters(), None)
            if first_param is None or not first_param.is_cuda:
                return None
            result = self.model()
            rendered = result["render"]

        return {
            "rendered": rendered,
            "render_hwc": result.get("render_hwc"),
            "dx": result.get("dx"),
            "dy": result.get("dy"),
            "dxy": result.get("dxy"),
            "wsum": result.get("wsum")
        }

    def handle_key(self, key: int) -> bool:
        if key == glfw.KEY_B:
            self.level_mode = 0
            return True
        elif key == glfw.KEY_C:
            self.level_mode = 1
            return True
        elif key == glfw.KEY_D:
            self.residual_mode = not self.residual_mode
            return True
        return False

    def draw_gui_controls(self) -> bool:
        changed = False

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

        clicked, self.residual_mode = imgui.checkbox("Residual Mode [D]", self.residual_mode)
        if clicked:
            changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show raw output (residual) instead of accumulated")

        imgui.text(f"Active scale: {self.scale_idx}")

        return changed

    def draw_info_gui(self):
        imgui.separator()
        for _, hint in self.HOTKEYS:
            imgui.text(hint)

    def can_display(self, vis_mode: VisMode) -> bool:
        if vis_mode == VisMode.TARGET:
            return self.target_image_gl is not None
        return super().can_display(vis_mode)

    def get_shader_binding(self, vis_mode: VisMode, gradient_mode: GradientMode) -> ShaderBinding:
        if vis_mode == VisMode.TARGET:
            return ShaderBinding(program='texture', textures={'texSampler': 'target'})
        return super().get_shader_binding(vis_mode, gradient_mode)

    def try_capture(self, model: "LIG", accumulated: torch.Tensor | None = None,
                    scale_idx: int = 0, target: torch.Tensor | None = None,
                    gt: torch.Tensor | None = None, stats: dict | None = None,
                    changed: bool = True):

        need_render = changed or self.view_dirty.is_set()
        if not self.gui_ready.is_set() and not need_render:
            return

        self.model = model
        self.scale_idx = scale_idx
        if accumulated is not None:
            self.set_accumulated(accumulated)
        if target is not None:
            self.set_target_image(target)
        if gt is not None:
            self.set_gt_image(gt)

        if self.gui_ready.is_set() and need_render:
            with torch.no_grad():
                render_data = self.render_model()

            with self.lock:
                self.tensor_buffer = render_data if render_data else {}
                self.tensor_buffer["target"] = self.target_image_gl
                self.tensor_buffer["gt"] = self.gt_image_gl
                self.stats = stats or {}
                self.has_new_frame = True

            self.gui_ready.clear()
            self.view_dirty.clear()

        else:
            with self.lock:
                self.stats = stats or {}
                self.has_new_frame = need_render

    def prepare_for_display(self, render_data: dict, vis_mode: VisMode,
                            gradient_mode: GradientMode = GradientMode.DX,
                            target_size: tuple[int, int] | None = None,
                            roi: tuple[int, int, int, int] | None = None) -> dict[str, torch.Tensor]:
        if vis_mode == VisMode.TARGET and render_data.get("target") is not None:
            return {"target": render_data["target"]}
        return super().prepare_for_display(render_data, vis_mode, gradient_mode, target_size, roi)
