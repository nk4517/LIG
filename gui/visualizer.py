from pathlib import Path
from dataclasses import dataclass
import typing

import glfw
import OpenGL.GL as gl
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
import torch

try:
    from cuda import cudart as cu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from .cmap_batlow import batlow as BATLOW_CMAP
from .common import VisMode, GradientMode, ShaderBinding


@dataclass
class TextureSlot:
    """GL texture with optional CUDA interop"""
    gl_id: int = 0
    cuda_resource: typing.Any = None
    width: int = 0
    height: int = 0
    last_tensor_obj_id: int = 0 # !!! НЕ ИСПОЛЬЗОВАТЬ data_ptr - они переиспользуются аллокатором !!!


def load_shader(shader_path):
    """Load shader source from file"""
    with open(shader_path, 'r') as f:
        return f.read()


def compile_shaders(vertex_source, fragment_source):
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, vertex_source)
    gl.glCompileShader(vertex_shader)

    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, fragment_source)
    gl.glCompileShader(fragment_shader)

    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    return program


def _upload_tensor_to_cuda_slot(tensor: torch.Tensor, slot: TextureSlot):
    """Upload tensor to texture slot via CUDA interop"""
    h, w = tensor.shape[:2]

    if slot.cuda_resource is None or slot.width != w or slot.height != h:
        if slot.cuda_resource is not None:
            cu.cudaGraphicsUnregisterResource(slot.cuda_resource)

        gl.glBindTexture(gl.GL_TEXTURE_2D, slot.gl_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        err, cuda_res = cu.cudaGraphicsGLRegisterImage(
            slot.gl_id, gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register texture")

        slot.cuda_resource = cuda_res
        slot.width = w
        slot.height = h

    (err,) = cu.cudaGraphicsMapResources(1, slot.cuda_resource, cu.cudaStreamLegacy)
    if err != cu.cudaError_t.cudaSuccess:
        raise RuntimeError("Unable to map graphics resource")

    err, array = cu.cudaGraphicsSubResourceGetMappedArray(slot.cuda_resource, 0, 0)
    if err != cu.cudaError_t.cudaSuccess:
        raise RuntimeError("Unable to get mapped array")

    tensor = tensor.contiguous()
    (err,) = cu.cudaMemcpy2DToArrayAsync(
        array, 0, 0, tensor.data_ptr(), 4 * 4 * w, 4 * 4 * w, h,
        cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice, cu.cudaStreamLegacy)

    (err,) = cu.cudaGraphicsUnmapResources(1, slot.cuda_resource, cu.cudaStreamLegacy)


def upload_to_slot(gl_tensor: torch.Tensor, slot: TextureSlot, use_cuda: bool):
    """Upload RGBA float tensor to slot via CUDA or CPU"""
    h, w = gl_tensor.shape[:2]
    current_obj_id = id(gl_tensor)
    if slot.last_tensor_obj_id == current_obj_id and slot.width == w and slot.height == h:
        return
    slot.last_tensor_obj_id = current_obj_id
    
    # print("upload_to_slot", slot.gl_id, w, h)
    if use_cuda and gl_tensor.is_cuda:
        _upload_tensor_to_cuda_slot(gl_tensor.float(), slot)
    else:
        gl.glBindTexture(gl.GL_TEXTURE_2D, slot.gl_id)
        np_data = gl_tensor.detach().cpu().numpy() if gl_tensor.is_cuda else gl_tensor.numpy()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, np_data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        slot.width, slot.height = w, h


def _create_texture(filter_mode: int = gl.GL_LINEAR):
    """Create and configure a texture with standard parameters"""
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    return texture


def _create_lut_texture(cmap_name: str, size: int = 256) -> int:
    """Create 1D LUT texture from matplotlib colormap"""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    lut = cmap(np.linspace(0, 1, size))[:, :3].astype(np.float32)  # [size, 3]

    texture = _create_texture()
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, size, 1, 0,
                    gl.GL_RGB, gl.GL_FLOAT, lut)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


def _create_icefire_lut(size: int = 256) -> int:
    """Create icefire (blue-black-red) diverging LUT texture"""
    t = np.linspace(0, 1, size).astype(np.float32)
    lut = np.zeros((size, 3), dtype=np.float32)
    # blue (0) -> black (0.5) -> red (1)
    lut[:, 0] = np.where(t < 0.5, 0, (t - 0.5) * 2)  # R
    lut[:, 2] = np.where(t < 0.5, 1 - t * 2, 0)  # B

    texture = _create_texture()
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, size, 1, 0,
                    gl.GL_RGB, gl.GL_FLOAT, lut)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


def _create_lut_from_array(lut_data: list, size: int = 256) -> int:
    """Create 1D LUT texture from RGB array data"""
    lut = np.array(lut_data, dtype=np.float32)
    if len(lut) != size:
        lut = np.array([lut_data[int(i * len(lut_data) / size)] for i in range(size)], dtype=np.float32)

    texture = _create_texture()
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, size, 1, 0,
                    gl.GL_RGB, gl.GL_FLOAT, lut)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


class VisualizerGUI:
    """Base visualizer GUI - subclass and implement abstract methods for specific models"""

    TEXTURE_SLOTS: tuple[tuple[str, int], ...] = ()
    HOTKEYS: tuple[tuple[int, str], ...] = ()

    def __init__(self, width=800, height=600, use_cuda=True):
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        if self.use_cuda:
            err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
            if err == cu.cudaError_t.cudaErrorUnknown:
                raise RuntimeError("cudaGLGetDevices failed")

        self.width = width
        self.height = height
        self.window = None
        self.impl = None

        self.pan_x = 0.0
        self.pan_y = 0.0
        self.display_width = float(width)  # Целевая ширина картинки в пикселях окна
        self.aspect = 1.0
        self.fit_to_window = True  # Автоподстройка зума под размер окна
        self.use_bilinear_filter = True  # GL_LINEAR vs GL_NEAREST для render/gt/target/wsum

        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.frame_skip_counter = 0

        # Sync primitives for trainer-GUI communication
        import threading
        self.gui_ready = threading.Event()
        self.view_dirty = threading.Event()
        self.lock = threading.Lock()
        self.tensor_buffer: dict = {}
        self.has_new_frame = False
        self.stats: dict = {}

        self.displayed_image_width = 0
        self.displayed_image_height = 0
        self.program = None
        self.vao = None

        self.upscale_program = None
        self.textures = {}

        self.colormap_lut_program = None
        self.magnitude_lut_program = None
        self.gradient_lut_program = None

        self.programs: dict[str, int] = {}
        self.show_info = True
        self.current_iter = 0
        self.current_psnr = 0.0
        self.current_loss = 0.0

        # Visualization mode
        self.vis_mode = VisMode.RENDER
        self.gradient_mode = GradientMode.MAGNITUDE
        self.active_vis_mode = VisMode.RENDER
        self.active_gradient_mode = GradientMode.MAGNITUDE

    @property
    def zoom(self):
        if self.displayed_image_width > 0:
            return self.display_width / self.displayed_image_width
        return 1.0

    def init(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(self.width, self.height, "LIG Visualizer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")

        glfw.make_context_current(self.window)
        imgui.create_context()

        # Initialize aspect ratio
        self.aspect = self.width / self.height

        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_window_close_callback(self.window, self.window_close_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        self.impl = GlfwRenderer(self.window, attach_callbacks=False)

        # Load shaders from files
        shader_dir = Path(__file__).parent / "shaders"
        vertex_source = load_shader(shader_dir / "fullscreen.vert")
        texture_fragment = load_shader(shader_dir / "texture.frag")
        upscale_fragment = load_shader(shader_dir / "upscale.frag")
        colormap_lut_fragment = load_shader(shader_dir / "colormap_lut.frag")
        magnitude_lut_fragment = load_shader(shader_dir / "magnitude_lut.frag")
        gradient_lut_fragment = load_shader(shader_dir / "gradient_lut.frag")

        self.program = compile_shaders(vertex_source, texture_fragment)
        self.upscale_program = compile_shaders(vertex_source, upscale_fragment)
        self.colormap_lut_program = compile_shaders(vertex_source, colormap_lut_fragment)
        self.magnitude_lut_program = compile_shaders(vertex_source, magnitude_lut_fragment)
        self.gradient_lut_program = compile_shaders(vertex_source, gradient_lut_fragment)

        self.programs = {
            'texture': self.program,
            'upscale': self.upscale_program,
            'colormap_lut': self.colormap_lut_program,
            'magnitude_lut': self.magnitude_lut_program,
            'gradient_lut': self.gradient_lut_program,
        }

        self.textures['viridis'] = TextureSlot(gl_id=_create_lut_texture('viridis'))
        self.textures['inferno'] = TextureSlot(gl_id=_create_lut_texture('inferno'))
        self.textures['batlow'] = TextureSlot(gl_id=_create_lut_from_array(BATLOW_CMAP))
        self.textures['icefire'] = TextureSlot(gl_id=_create_icefire_lut())
        self.vao = gl.glGenVertexArrays(1)

        # Создаём текстуры согласно TEXTURE_SLOTS
        for name, filt in self.TEXTURE_SLOTS:
            self.textures[name] = TextureSlot(gl_id=_create_texture(filt))
        
        # Separate always-nearest texture for upscale/colormap/gradient shaders
        self.textures['nearest_src'] = TextureSlot(gl_id=_create_texture(gl.GL_NEAREST))

        glfw.swap_interval(1)

    def _update_fit_zoom(self):
        """Update zoom to fit image in window"""
        if self.displayed_image_width > 0 and self.displayed_image_height > 0:
            scale_x = self.width / float(self.displayed_image_width)
            scale_y = self.height / float(self.displayed_image_height)
            fit_scale = min(scale_x, scale_y)
            self.display_width = self.displayed_image_width * fit_scale
            self.pan_x = 0.0
            self.pan_y = 0.0

    def _update_texture_filters(self):
        """Update GL filter mode for render/gt/target textures"""
        filt = gl.GL_LINEAR if self.use_bilinear_filter else gl.GL_NEAREST
        for name in ('render', 'gt', 'target'):
            if name in self.textures:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[name].gl_id)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filt)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filt)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def load_image_to_slot(self, image_path: str, slot_name: str):
        """Load image from file and upload to specified texture slot"""
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        tensor = cv2_to_gl_rgba(img)
        upload_to_slot(tensor, self.textures[slot_name], self.use_cuda)

    def window_close_callback(self, window):
        pass

    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.aspect = width / height if height > 0 else 1.0
        gl.glViewport(0, 0, width, height)

        # Пересчитываем zoom если включен fit_to_window
        if self.fit_to_window:
            self._update_fit_zoom()

    def cursor_pos_callback(self, window, xpos, ypos):
        if imgui.get_io().want_capture_mouse:
            self.is_panning = False
            return

        if self.is_panning:
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y
            self.fit_to_window = False
            # Pan хранится в пикселях окна
            self.pan_x -= dx
            self.pan_y -= dy

        self.last_mouse_x = xpos
        self.last_mouse_y = ypos

    def mouse_button_callback(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse:
            return

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.is_panning = (action == glfw.PRESS)

    def scroll_callback(self, window, xoffset, yoffset):
        if imgui.get_io().want_capture_mouse:
            return

        # При ручном изменении зума отключаем fit_to_window
        self.fit_to_window = False

        if self.displayed_image_width <= 0 or self.displayed_image_height <= 0:
            return

        old_zoom = self.zoom
        mouse_x, mouse_y = self.last_mouse_x, self.last_mouse_y

        # Позиция в текстурных координатах до зума
        center_offset_x = (self.width - self.displayed_image_width * old_zoom) * 0.5
        center_offset_y = (self.height - self.displayed_image_height * old_zoom) * 0.5
        tex_pos_x = mouse_x - center_offset_x + self.pan_x
        tex_pos_y = mouse_y - center_offset_y + self.pan_y

        zoom_speed = 1.1
        if yoffset > 0:
            self.display_width *= zoom_speed
        else:
            self.display_width /= zoom_speed
        self.display_width = max(100.0, min(50000.0, self.display_width))

        new_zoom = self.zoom
        # Корректировка pan чтобы точка под курсором осталась на месте
        center_offset_x_new = (self.width - self.displayed_image_width * new_zoom) * 0.5
        center_offset_y_new = (self.height - self.displayed_image_height * new_zoom) * 0.5
        tex_pos_x_new = tex_pos_x * (new_zoom / old_zoom)
        tex_pos_y_new = tex_pos_y * (new_zoom / old_zoom)
        self.pan_x = tex_pos_x_new - mouse_x + center_offset_x_new
        self.pan_y = tex_pos_y_new - mouse_y + center_offset_y_new

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:  # Reset view
                self.pan_x = 0.0
                self.pan_y = 0.0
                if self.fit_to_window:
                    self._update_fit_zoom()
                elif self.displayed_image_width > 0:
                    self.display_width = float(self.displayed_image_width)
            elif key == glfw.KEY_0:  # Zoom 1:1
                if self.displayed_image_width > 0:
                    self.display_width = float(self.displayed_image_width)
                    self.fit_to_window = False
                    self.pan_x = 0.0
                    self.pan_y = 0.0
            elif key == glfw.KEY_1:
                self.vis_mode = VisMode.RENDER
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_2:
                self.vis_mode = VisMode.UPSCALED
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_3:
                self.vis_mode = VisMode.TARGET
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_4:
                self.vis_mode = VisMode.GROUND_TRUTH
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_5:
                # Переключение градиентов по кругу
                if self.vis_mode != VisMode.GRADIENTS:
                    self.vis_mode = VisMode.GRADIENTS
                    self.gradient_mode = GradientMode.MAGNITUDE
                else:
                    # Циклическое переключение между dx, dy, dxy, magnitude
                    self.gradient_mode = GradientMode((self.gradient_mode + 1) % 4)
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_6:
                self.vis_mode = VisMode.WSUM
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_7:
                self.vis_mode = VisMode.UPSCALED_TORCH
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_8:
                self.vis_mode = VisMode.UPSCALED_CUDA
                self.view_dirty.set()
                self.gui_ready.set()
            elif key == glfw.KEY_F:
                self.use_bilinear_filter = not self.use_bilinear_filter
                self._update_texture_filters()
            elif any(key == k for k, _ in self.HOTKEYS):
                if self.handle_key(key):
                    self.view_dirty.set()
                    self.gui_ready.set()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)

    def _update_from_trainer(self):
        # Check if trainer provided new data
        with self.lock:
            if not self.has_new_frame:
                return

            render_data = self.tensor_buffer
            if render_data:
                # Inlined from _update_display_from_tensors
                roi = None
                target_size = None
                
                # For UPSCALED modes, use gt size for zoom calculation
                gt_tensor = render_data.get("gt")
                if gt_tensor is not None and self.vis_mode in (VisMode.GROUND_TRUTH, VisMode.UPSCALED_TORCH, VisMode.UPSCALED_CUDA):
                    src_h, src_w = gt_tensor.shape[:2]
                    self.displayed_image_width = src_w
                    self.displayed_image_height = src_h
                    if self.fit_to_window:
                        self._update_fit_zoom()
                else:
                    render_hwc = render_data.get("render_hwc")
                    if render_hwc is not None:
                        src_h, src_w = render_hwc.shape[:2]
                        self.displayed_image_width = src_w
                        self.displayed_image_height = src_h
                        if self.fit_to_window:
                            self._update_fit_zoom()

                if self.vis_mode in (VisMode.UPSCALED_TORCH, VisMode.UPSCALED_CUDA):
                    rendered = render_data.get("rendered")
                    if rendered is not None:
                        if rendered.dim() == 4:
                            _, _, src_h, src_w = rendered.shape
                        else:
                            _, src_h, src_w = rendered.shape
                        target_size = (int(src_h * self.zoom), int(src_w * self.zoom))

                        # Compute ROI for UPSCALED_CUDA when zoomed in
                        if self.vis_mode == VisMode.UPSCALED_CUDA and self.zoom >= 1.0:
                            center_offset_x = (self.width - src_w * self.zoom) * 0.5
                            center_offset_y = (self.height - src_h * self.zoom) * 0.5
                            x1_src = (self.pan_x - center_offset_x) / self.zoom
                            y1_src = (self.pan_y - center_offset_y) / self.zoom
                            x2_src = x1_src + self.width / self.zoom
                            y2_src = y1_src + self.height / self.zoom
                            x1, y1 = max(0.0, x1_src), max(0.0, y1_src)
                            x2, y2 = min(float(src_w), x2_src), min(float(src_h), y2_src)
                            roi = (x1, y1, x2, y2)
                            target_size = (round((y2 - y1) * self.zoom), round((x2 - x1) * self.zoom))

                prepared = self.prepare_for_display(
                    render_data, self.vis_mode, self.gradient_mode, target_size, roi)

                for slot_name, gl_tensor in prepared.items():
                    if slot_name not in self.textures:
                        continue
                    upload_to_slot(gl_tensor.float(), self.textures[slot_name], self.use_cuda)

                self.active_vis_mode = self.vis_mode
                self.active_gradient_mode = self.gradient_mode
                self.tensor_buffer = None

            stats = self.stats
            self.has_new_frame = False

        # Update stats from renderer
        if stats:
            self.current_iter = stats.get("iter", self.current_iter)
            self.current_psnr = stats.get("psnr", self.current_psnr)
            self.current_loss = stats.get("loss", self.current_loss)

    # Abstract methods - subclasses must implement
    def handle_key(self, key: int) -> bool:
        """Handle model-specific hotkeys. Returns True if key was handled."""
        return False

    def draw_gui_controls(self) -> bool:
        """Draw model-specific ImGui controls. Returns True if re-render needed."""
        return False

    def draw_info_gui(self):
        """Draw model-specific info in Info panel."""
        pass

    def can_display(self, vis_mode: VisMode) -> bool:
        """Check if visualizer can display given mode."""
        return False

    def get_shader_binding(self, vis_mode: VisMode, gradient_mode: GradientMode) -> ShaderBinding:
        """Return shader binding configuration for given visualization mode."""
        return ShaderBinding(program='texture', textures={'texSampler': 'render'})

    def prepare_for_display(
            self,
            render_data: dict,
            vis_mode: VisMode,
            gradient_mode: GradientMode = GradientMode.DX,
            target_size: tuple[int, int] | None = None,
            roi: tuple[int, int, int, int] | None = None) -> dict[str, torch.Tensor]:
        """Prepare tensors for GL upload. Returns dict[slot_name, HWC RGBA tensor]."""
        return {}

    def _bind_tex(self, slot: str, uniform: str, unit: int, prog):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[slot].gl_id)
        gl.glUniform1i(gl.glGetUniformLocation(prog, uniform), unit)

    def _set_common_uniforms(self, prog, skip_texture_size: bool = False):
        gl.glUniform2f(gl.glGetUniformLocation(prog, "pan"), float(self.pan_x), float(self.pan_y))
        gl.glUniform1f(gl.glGetUniformLocation(prog, "zoom"), self.zoom)
        gl.glUniform2f(gl.glGetUniformLocation(prog, "window_size"), float(self.width), float(self.height))
        if not skip_texture_size:
            gl.glUniform2f(gl.glGetUniformLocation(prog, "texture_size"),
                           float(self.displayed_image_width), float(self.displayed_image_height))

    def _setup_shader(self):
        """Setup and bind appropriate shader for rendering"""
        binding = self.get_shader_binding(self.active_vis_mode, self.active_gradient_mode)

        prog = self.programs[binding.program]
        gl.glUseProgram(prog)

        # Bind textures
        for unit, (uniform_name, slot_name) in enumerate(binding.textures.items()):
            self._bind_tex(slot_name, uniform_name, unit, prog)

        # Float uniforms
        for name, val in binding.uniforms_float.items():
            gl.glUniform1f(gl.glGetUniformLocation(prog, name), val)

        # Vec2 uniforms
        for name, (x, y) in binding.uniforms_vec2.items():
            gl.glUniform2f(gl.glGetUniformLocation(prog, name), x, y)

        # Upscale-specific: source_size and target_size
        if binding.program == 'upscale':
            gl.glUniform2f(gl.glGetUniformLocation(prog, "source_size"),
                           float(self.displayed_image_width), float(self.displayed_image_height))
            gl.glUniform2f(gl.glGetUniformLocation(prog, "target_size"),
                           float(self.displayed_image_width * self.zoom),
                           float(self.displayed_image_height * self.zoom))

        self._set_common_uniforms(prog, skip_texture_size=binding.skip_texture_size)

        # UPSCALED_TORCH/CUDA: texture already upscaled - use its actual size with zoom=1
        if self.active_vis_mode in (VisMode.UPSCALED_TORCH, VisMode.UPSCALED_CUDA):
            tex_slot = self.textures.get('render')
            if tex_slot and tex_slot.width > 0:
                gl.glUniform2f(gl.glGetUniformLocation(prog, "texture_size"),
                               float(tex_slot.width), float(tex_slot.height))
                gl.glUniform1f(gl.glGetUniformLocation(prog, "zoom"), 1.0)
                # UPSCALED_CUDA renders to window size with ROI - no pan needed
                if self.active_vis_mode == VisMode.UPSCALED_CUDA:
                    gl.glUniform2f(gl.glGetUniformLocation(prog, "pan"), 0.0, 0.0)

    def _draw_gui(self):
        """Draw ImGui interface"""
        if not self.show_info:
            return

        # Settings panel - верхний правый угол
        imgui.set_next_window_position(self.width - 320, 10)
        imgui.set_next_window_size(300, 400)
        imgui.begin("Settings")

        # Fit to window checkbox
        changed, self.fit_to_window = imgui.checkbox("Fit to Window", self.fit_to_window)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically adjust zoom to fit image in window")

        if changed and self.fit_to_window:
            self._update_fit_zoom()

        filter_changed, self.use_bilinear_filter = imgui.checkbox(
            "Bilinear GL Filter [F]", self.use_bilinear_filter)
        if imgui.is_item_hovered():
            imgui.set_tooltip("GL_LINEAR (checked) vs GL_NEAREST for render/gt/target/wsum")
        if filter_changed:
            self._update_texture_filters()

        imgui.separator()

        # Display mode radio buttons
        imgui.text("Display:")

        if imgui.radio_button("Render [1]", self.vis_mode == VisMode.RENDER):
            self.vis_mode = VisMode.RENDER
            self.view_dirty.set()
            self.gui_ready.set()
        imgui.same_line()
        if imgui.radio_button("Upscaled [2]", self.vis_mode == VisMode.UPSCALED):
            self.vis_mode = VisMode.UPSCALED
            self.view_dirty.set()
            self.gui_ready.set()

        if imgui.radio_button("Target [3]", self.vis_mode == VisMode.TARGET):
            self.vis_mode = VisMode.TARGET
            self.view_dirty.set()
            self.gui_ready.set()
        imgui.same_line()
        if imgui.radio_button("GT [4]", self.vis_mode == VisMode.GROUND_TRUTH):
            self.vis_mode = VisMode.GROUND_TRUTH
            self.view_dirty.set()
            self.gui_ready.set()

        # Градиенты - показываем текущий режим
        gradient_label = f"Gradients [{self.gradient_mode.name}] [5]"
        if imgui.radio_button(gradient_label, self.vis_mode == VisMode.GRADIENTS):
            self.vis_mode = VisMode.GRADIENTS
            self.view_dirty.set()
            self.gui_ready.set()

        if imgui.radio_button("Wsum [6]", self.vis_mode == VisMode.WSUM):
            self.vis_mode = VisMode.WSUM
            self.view_dirty.set()
            self.gui_ready.set()

        imgui.separator()

        # Model-specific controls
        if self.draw_gui_controls():
            self.view_dirty.set()
            self.gui_ready.set()

        imgui.separator()

        # Display width slider - управление видимым размером картинки
        label = "Display Width" if self.vis_mode != VisMode.UPSCALED else "Upscale Width"
        changed, self.display_width = imgui.slider_float(
            label,
            self.display_width,
            100.0,
            50000.0,
            f"{self.display_width:.0f}px"
        )
        if changed:
            self.fit_to_window = False

        if imgui.is_item_hovered():
            imgui.set_tooltip("Display width in pixels (also controlled by mouse wheel)")

        imgui.end()

        # Info panel - нижний правый угол
        imgui.set_next_window_position(self.width - 320, self.height - 420)
        imgui.set_next_window_size(300, 400)
        imgui.begin("Info")
        imgui.text(f"Mode: {'CUDA' if self.use_cuda else 'CPU'}")
        # Для градиентов показываем подрежим
        if self.vis_mode == VisMode.GRADIENTS:
            imgui.text(f"View: {self.vis_mode.name} ({self.gradient_mode.name})")
        else:
            imgui.text(f"View: {self.vis_mode.name}")
        imgui.separator()
        imgui.text(f"Iteration: {self.current_iter}")
        imgui.text(f"PSNR: {self.current_psnr:.2f} dB")
        imgui.text(f"Loss: {self.current_loss:.6f}")
        imgui.separator()
        imgui.text("Texture sizes:")
        lut_names = {'viridis', 'inferno', 'batlow', 'icefire'}
        for name, slot in self.textures.items():
            if name in lut_names:
                continue
            imgui.text(f"  {name}: {slot.width}x{slot.height}")
        imgui.separator()
        imgui.text(f"Zoom: {self.zoom:.2f}x")
        imgui.text(f"Pan: ({self.pan_x:.0f}, {self.pan_y:.0f}) px")
        # Model-specific info
        self.draw_info_gui()
        imgui.separator()
        imgui.text("Controls:")
        imgui.text("LMB - Pan")
        imgui.text("Scroll - Zoom")
        imgui.text("R - Reset view")
        imgui.text("0 - Set zoom to 1.0")
        imgui.text("1-6 - Switch view mode")
        imgui.text("5 - Cycle gradients")
        imgui.text("6 - Wsum view")
        imgui.text("7 - Upscale Torch")
        imgui.text("8 - Upscale CUDA")
        imgui.text("ESC - Exit")
        imgui.end()

    def main_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self._update_from_trainer()

            if self.can_display(self.active_vis_mode):
                self._setup_shader()
                gl.glBindVertexArray(self.vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # GUI
            self._draw_gui()

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            self.frame_skip_counter += 1
            if self.frame_skip_counter % 2 == 0:
                self.gui_ready.set()

    def cleanup(self):
        """Clean up resources"""
        if self.use_cuda:
            for slot in self.textures.values():
                if slot.cuda_resource is not None:
                    cu.cudaGraphicsUnregisterResource(slot.cuda_resource)
        if self.impl:
            self.impl.shutdown()
        if self.window:
            glfw.terminate()

    def run(self):
        """Run visualization"""
        self.init()
        self.main_loop()
        self.cleanup()
