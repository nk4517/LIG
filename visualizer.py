import threading
from enum import IntEnum
import typing

import glfw
import OpenGL.GL as gl
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
import torch
from pathlib import Path

from LIG.upscaler_torch import bicubic_spline_upscale_single_channel
from LIG.lig_renderer import LIGRenderer
from dataclasses import dataclass

try:
    from cuda import cudart as cu

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


@dataclass
class TextureSlot:
    """GL texture with optional CUDA interop"""
    gl_id: int = 0
    cuda_resource: typing.Any = None
    width: int = 0
    height: int = 0


class VisMode(IntEnum):
    RENDER = 0
    UPSCALED = 1
    TARGET = 2
    GROUND_TRUTH = 3
    GRADIENTS = 4
    WSUM = 5


class GradientMode(IntEnum):
    DX = 0
    DY = 1
    DXY = 2
    MAGNITUDE = 3


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
    if use_cuda and gl_tensor.is_cuda:
        _upload_tensor_to_cuda_slot(gl_tensor.float(), slot)
    else:
        gl.glBindTexture(gl.GL_TEXTURE_2D, slot.gl_id)
        np_data = gl_tensor.detach().cpu().numpy() if gl_tensor.is_cuda else gl_tensor.numpy()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, np_data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        slot.width, slot.height = w, h


def _create_texture():
    """Create and configure a texture with standard parameters"""
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    return texture

def _create_lut_texture(cmap_name: str, size: int = 256) -> int:
    """Create 1D LUT texture from matplotlib colormap"""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
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
    lut[:, 2] = np.where(t < 0.5, 1 - t * 2, 0)       # B
    
    texture = _create_texture()
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, size, 1, 0,
                    gl.GL_RGB, gl.GL_FLOAT, lut)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


def _gsplat_tensor_to_gl_tensor(rendered: torch.Tensor):
    """Prepare rendered tensor for texture update"""
    # Handle different tensor dimensions
    if rendered.dim() == 4:
        rendered = rendered.squeeze(0)

    if rendered.dim() == 3:
        # Convert CHW to HWC, keep on GPU if using CUDA
        image_tensor = rendered.detach().permute(1, 2, 0)
    else:
        image_tensor = rendered.detach()
        image_tensor = torch.stack([image_tensor] * 3, dim=-1)

    # Ensure RGBA
    if image_tensor.shape[2] == 3:
        alpha = torch.ones((*image_tensor.shape[:2], 1),
                           dtype=image_tensor.dtype,
                           device=image_tensor.device)
        image_tensor = torch.cat([image_tensor, alpha], dim=2)

    return torch.clamp(image_tensor, 0, 1)




class LIGVisualizer:
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

        # Events for synchronization (like in splatting_app)
        self.e_want_to_render = threading.Event()
        self.e_want_to_render.clear()
        self.e_finished_rendering = threading.Event()
        self.e_finished_rendering.set()

        # Lock for data synchronization
        self.lock = threading.Lock()

        self.pan_x = 0.0
        self.pan_y = 0.0
        self.display_width = 800.0  # Целевая ширина картинки в пикселях окна
        self.aspect = 1.0
        self.fit_to_window = True  # Автоподстройка зума под размер окна

        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Model and image data
        self.lig_renderer: LIGRenderer | None = None
        self.gt_image = None
        self.target_image = None  # Для хранения первой мелкой картинки
        self.displayed_image_width = 0
        self.displayed_image_height = 0
        self.displayed_image_aspect = 1.0
        self.program = None
        self.vao = None

        self.upscale_program = None
        self.textures = {}
        self.lut_textures = {}  # colormap LUTs
        
        self.colormap_lut_program = None
        self.magnitude_lut_program = None
        self.gradient_lut_program = None

        # Flag for visualization updates
        self.was_updated = False

        self.show_info = True
        self.current_iter = 0
        self.current_psnr = 0.0
        self.current_loss = 0.0

        # Visualization mode
        self.vis_mode = VisMode.RENDER
        self.gradient_mode = GradientMode.DX


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
        
        self.lut_textures['viridis'] = _create_lut_texture('viridis')
        self.lut_textures['inferno'] = _create_lut_texture('inferno')
        self.lut_textures['icefire'] = _create_icefire_lut()
        self.vao = gl.glGenVertexArrays(1)

        # Создаём все текстуры через unified storage
        for name in (
                'render', 'dx', 'dy', 'dxy', 'target', 'gt',
                'wsum',
        ):
            self.textures[name] = TextureSlot(gl_id=_create_texture())

        glfw.swap_interval(1)

    def _update_fit_zoom(self):
        """Update zoom to fit image in window"""
        if self.displayed_image_width > 0 and self.displayed_image_height > 0:
            scale_x = self.width / float(self.displayed_image_width)
            scale_y = self.height / float(self.displayed_image_height)
            fit_scale = min(scale_x, scale_y)
            self.display_width = self.displayed_image_width * fit_scale

    def window_close_callback(self, window):
        # Signal training to stop when window is closed
        self.e_want_to_render.clear()
        self.e_finished_rendering.set()

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

        zoom_speed = 1.1
        if yoffset > 0:
            self.display_width *= zoom_speed
        else:
            self.display_width /= zoom_speed
        self.display_width = max(100.0, min(16000.0, self.display_width))

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:  # Reset view
                self.pan_x = 0.0
                self.pan_y = 0.0
                if self.fit_to_window:
                    self._update_fit_zoom()
                elif self.displayed_image_width > 0:
                    self.display_width = float(self.displayed_image_width)
            elif key == glfw.KEY_1:
                self.vis_mode = VisMode.RENDER
                self.was_updated = True
            elif key == glfw.KEY_2:
                self.vis_mode = VisMode.UPSCALED
                self.was_updated = True
            elif key == glfw.KEY_3:
                self.vis_mode = VisMode.TARGET
                self.was_updated = True
            elif key == glfw.KEY_4:
                self.vis_mode = VisMode.GROUND_TRUTH
                self.was_updated = True
            elif key == glfw.KEY_5:
                # Переключение градиентов по кругу
                if self.vis_mode != VisMode.GRADIENTS:
                    self.vis_mode = VisMode.GRADIENTS
                    self.gradient_mode = GradientMode.DX
                else:
                    # Циклическое переключение между dx, dy, dxy, magnitude
                    self.gradient_mode = GradientMode((self.gradient_mode + 1) % 4)
                self.was_updated = True
            elif key == glfw.KEY_6:
                self.vis_mode = VisMode.WSUM
                self.was_updated = True
            elif key == glfw.KEY_0:
                # Установка display_width = image_width (zoom 1:1)
                self.display_width = float(self.displayed_image_width) if self.displayed_image_width > 0 else 800.0
                self.fit_to_window = False
            elif key in (glfw.KEY_B, glfw.KEY_C, glfw.KEY_D):
                if self.lig_renderer and self.lig_renderer.handle_key(key):
                    self.was_updated = True
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)

    def set_model(self, gaussian_model, gt_image):
        """Set the model and ground truth image for rendering"""
        if self.lig_renderer is None:
            self.lig_renderer = LIGRenderer()
        self.lig_renderer.set_model(gaussian_model)
        self.gt_image = gt_image

    def set_target_image(self, reference_image):
        """Set target image (e.g., first low-res image in multi-scale)"""
        self.target_image = reference_image
        if 'target' in self.textures:
            self.textures['target'].width = 0  # Reset to force texture reload

    def set_updated(self):
        """Signal that model was updated and needs re-rendering"""
        self.was_updated = True

    def update_stats(self, iteration, psnr, loss):
        """Update training statistics"""
        self.current_iter = iteration
        self.current_psnr = psnr
        self.current_loss = loss

    def set_current_scale(self, scale_idx):
        """Update current active scale index from training"""
        if self.lig_renderer:
            self.lig_renderer.set_scale_idx(scale_idx)

    def set_accumulated_image(self, accumulated):
        """Set accumulated image from previous levels for proper visualization"""
        if self.lig_renderer:
            self.lig_renderer.set_accumulated(accumulated)

    def _update_image_texture(self, gl_tensor: torch.Tensor, slot_name: str = 'render'):
        """Update OpenGL texture - chooses CPU or CUDA method based on configuration"""

        # Get dimensions from tensor
        h, w = gl_tensor.shape[:2]

        self.displayed_image_width = w
        self.displayed_image_height = h
        self.displayed_image_aspect = w / h if h > 0 else 1.0

        # Если включен fit_to_window - пересчитываем zoom
        if self.fit_to_window:
            self._update_fit_zoom()

        upload_to_slot(gl_tensor.float(), self.textures[slot_name], self.use_cuda)

    def _update_gradient_textures(self, dx: torch.Tensor, dy: torch.Tensor, dxy: torch.Tensor):
        """Обновление текстур градиентов для апскейлинга"""
        for slot_name, gradient in [('dx', dx), ('dy', dy), ('dxy', dxy)]:
            tensor = self._prepare_gradient_tensor(gradient)
            slot = self.textures[slot_name]
            upload_to_slot(tensor.float(), slot, self.use_cuda)

    def _update_wsum_texture(self, wsum: torch.Tensor):
        """Обновление текстуры wsum с нормализацией по 1 и 99 перцентилю"""
        if wsum.dim() == 4:
            wsum = wsum.squeeze(0)
        if wsum.dim() == 3:
            wsum = wsum.mean(dim=0)

        # Нормализация по перцентилям
        p1 = torch.quantile(wsum.flatten(), 0.01)
        p99 = torch.quantile(wsum.flatten(), 0.99)
        wsum_norm = (wsum - p1) / (p99 - p1 + 1e-8)
        wsum_norm = torch.clamp(wsum_norm, 0, 1)

        # Преобразование в RGBA
        wsum_rgba = wsum_norm.unsqueeze(-1).repeat(1, 1, 4)
        wsum_rgba[..., 3] = 1.0

        slot = self.textures['wsum']
        upload_to_slot(wsum_rgba.float(), slot, self.use_cuda)

    def _prepare_gradient_tensor(self, gradient: torch.Tensor):
        """Подготовка тензора градиента для текстуры"""
        if gradient.dim() == 4:
            gradient = gradient.squeeze(0)

        if gradient.dim() == 3:
            # CHW -> HWC
            gradient = gradient.permute(1, 2, 0)
        elif gradient.dim() == 2:
            # Single channel - add channel dimension
            gradient = gradient.unsqueeze(-1)

        # Ensure RGB (replicate single channel if needed)
        if gradient.shape[2] == 1:
            gradient = gradient.repeat(1, 1, 3)

        # Добавляем альфа-канал если нужно
        if gradient.shape[2] == 3:
            alpha = torch.ones((*gradient.shape[:2], 1),
                               dtype=gradient.dtype,
                               device=gradient.device)
            gradient = torch.cat([gradient, alpha], dim=2)

        return gradient

    def _render_model(self):
        """Render the gaussian model and update textures"""
        if self.lig_renderer is None or not self.was_updated:
            return

        self.e_finished_rendering.clear()
        self.e_want_to_render.set()

        with self.lock:
            try:
                with torch.no_grad():
                    if not self.lig_renderer:
                        return

                    render_data = self.lig_renderer.render()
                    
                    if render_data is None:
                        self.was_updated = False
                        return

                    self._update_display_from_tensors(render_data)
                    self.was_updated = False

            except Exception as e:
                print(f"Render error: {e}")
            finally:
                self.e_want_to_render.clear()
                self.e_finished_rendering.set()

    def _update_display_from_tensors(self, render_data):
        """Update display textures based on render tensors"""

        dx = render_data.get("dx")
        dy = render_data.get("dy")
        dxy = render_data.get("dxy")
        wsum = render_data.get("wsum")

        if self.vis_mode == VisMode.TARGET and self.target_image is not None:
            self._update_image_texture(_gsplat_tensor_to_gl_tensor(self.target_image), 'target')
            return
        if self.vis_mode == VisMode.GROUND_TRUTH and self.gt_image is not None:
            self._update_image_texture(_gsplat_tensor_to_gl_tensor(self.gt_image), 'gt')
            return
        if self.vis_mode == VisMode.WSUM and wsum is not None:
            self._update_wsum_texture(wsum)
            return

        rendered_gl = _gsplat_tensor_to_gl_tensor(render_data["rendered"])

        if self.vis_mode == VisMode.UPSCALED and dx is not None and dy is not None and dxy is not None:
            self._update_image_texture(rendered_gl)
            self._update_gradient_textures(dx, dy, dxy)
            return

        if self.vis_mode == VisMode.GRADIENTS:
            if self.gradient_mode == GradientMode.DX and dx is not None:
                self._update_image_texture(self._prepare_gradient_tensor(dx.float()))
                return
            if self.gradient_mode == GradientMode.DY and dy is not None:
                self._update_image_texture(self._prepare_gradient_tensor(dy.float()))
                return
            if self.gradient_mode == GradientMode.DXY and dxy is not None:
                self._update_image_texture(self._prepare_gradient_tensor(dxy.float()))
                return
            if self.gradient_mode == GradientMode.MAGNITUDE and dx is not None and dy is not None:
                self._update_gradient_textures(dx, dy, dxy if dxy is not None else dx)
                self._update_image_texture(rendered_gl)
                return

        self._update_image_texture(rendered_gl)

    def _setup_shader(self):
        """Setup and bind appropriate shader for rendering"""

        # Select shader and textures based on vis_mode
        gl.glActiveTexture(gl.GL_TEXTURE0)
        
        if self.vis_mode == VisMode.UPSCALED:
            prog = self.upscale_program
            gl.glUseProgram(prog)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['render'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texRender"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['dx'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texDx"), 1)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['dy'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texDy"), 2)
            gl.glActiveTexture(gl.GL_TEXTURE3)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['dxy'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texDxy"), 3)
            gl.glUniform2f(gl.glGetUniformLocation(prog, "source_size"),
                           float(self.displayed_image_width), float(self.displayed_image_height))
            gl.glUniform2f(gl.glGetUniformLocation(prog, "target_size"),
                           float(self.displayed_image_width * self.zoom), float(self.displayed_image_height * self.zoom))
        elif self.vis_mode == VisMode.WSUM:
            prog = self.colormap_lut_program
            gl.glUseProgram(prog)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['wsum'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texData"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.lut_textures['viridis'])
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texLUT"), 1)
        elif self.vis_mode == VisMode.GRADIENTS and self.gradient_mode == GradientMode.MAGNITUDE:
            prog = self.magnitude_lut_program
            gl.glUseProgram(prog)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['dx'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texDx"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['dy'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texDy"), 1)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.lut_textures['inferno'])
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texLUT"), 2)
            gl.glUniform1f(gl.glGetUniformLocation(prog, "scale"), 2.0)
        elif self.vis_mode == VisMode.GRADIENTS:
            prog = self.gradient_lut_program
            gl.glUseProgram(prog)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['render'].gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texSampler"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.lut_textures['icefire'])
            gl.glUniform1i(gl.glGetUniformLocation(prog, "texLUT"), 1)
            gl.glUniform1f(gl.glGetUniformLocation(prog, "scale"), 5.0)
        else:
            prog = self.program
            gl.glUseProgram(prog)
            if self.vis_mode == VisMode.TARGET:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['target'].gl_id)
            elif self.vis_mode == VisMode.GROUND_TRUTH:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['gt'].gl_id)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['render'].gl_id)

        # Common uniforms
        gl.glUniform2f(gl.glGetUniformLocation(prog, "pan"), float(self.pan_x), float(self.pan_y))
        gl.glUniform1f(gl.glGetUniformLocation(prog, "zoom"), self.zoom)
        gl.glUniform2f(gl.glGetUniformLocation(prog, "window_size"), float(self.width), float(self.height))
        if self.vis_mode != VisMode.UPSCALED:
            gl.glUniform2f(gl.glGetUniformLocation(prog, "texture_size"), float(self.displayed_image_width), float(self.displayed_image_height))

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

        imgui.separator()

        # Display mode radio buttons
        imgui.text("Display:")

        if imgui.radio_button("Render [1]", self.vis_mode == VisMode.RENDER):
            self.vis_mode = VisMode.RENDER
            self.was_updated = True
        imgui.same_line()
        if imgui.radio_button("Upscaled [2]", self.vis_mode == VisMode.UPSCALED):
            self.vis_mode = VisMode.UPSCALED
            self.was_updated = True

        if imgui.radio_button("Target [3]", self.vis_mode == VisMode.TARGET):
            self.vis_mode = VisMode.TARGET
            self.was_updated = True
        imgui.same_line()
        if imgui.radio_button("GT [4]", self.vis_mode == VisMode.GROUND_TRUTH):
            self.vis_mode = VisMode.GROUND_TRUTH
            self.was_updated = True

        # Градиенты - показываем текущий режим
        gradient_label = f"Gradients [{self.gradient_mode.name}] [5]"
        if imgui.radio_button(gradient_label, self.vis_mode == VisMode.GRADIENTS):
            self.vis_mode = VisMode.GRADIENTS
            self.was_updated = True

        if imgui.radio_button("Wsum [6]", self.vis_mode == VisMode.WSUM):
            self.vis_mode = VisMode.WSUM
            self.was_updated = True

        imgui.separator()

        # LIG-specific controls
        if self.lig_renderer:
            if self.lig_renderer.draw_gui_controls():
                self.was_updated = True

        imgui.separator()

        # Display width slider - управление видимым размером картинки
        label = "Display Width" if self.vis_mode != 1 else "Upscale Width"
        changed, self.display_width = imgui.slider_float(
            label,
            self.display_width,
            50.0,
            4000.0,
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
        r = self.textures.get('render', TextureSlot())
        imgui.text(f"  Render: {r.width}x{r.height}")
        dx = self.textures.get('dx', TextureSlot())
        imgui.text(f"  Grads: {dx.width}x{dx.height}")
        w = self.textures.get('wsum', TextureSlot())
        imgui.text(f"  Wsum: {w.width}x{w.height}")
        t = self.textures.get('target', TextureSlot())
        imgui.text(f"  Target: {t.width}x{t.height}")
        g = self.textures.get('gt', TextureSlot())
        imgui.text(f"  GT: {g.width}x{g.height}")
        imgui.separator()
        imgui.text(f"Zoom: {self.zoom:.2f}x")
        imgui.text(f"Pan: ({self.pan_x:.0f}, {self.pan_y:.0f}) px")
        imgui.separator()
        imgui.text("Controls:")
        imgui.text("LMB - Pan")
        imgui.text("Scroll - Zoom")
        imgui.text("R - Reset view")
        imgui.text("0 - Set zoom to 1.0")
        imgui.text("1-6 - Switch view mode")
        imgui.text("5 - Cycle gradients")
        imgui.text("6 - Wsum view")
        imgui.text("B - Base level (L0)")
        imgui.text("C - Current level")
        imgui.text("D - Toggle residual")
        imgui.text("ESC - Exit")
        imgui.end()

    def main_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            # Render model if needed
            self._render_model()

            # Render image
            # Рендерим в зависимости от режима
            if (self.vis_mode in (VisMode.RENDER, VisMode.UPSCALED, VisMode.GRADIENTS) or
                    (self.vis_mode == VisMode.TARGET and self.target_image is not None) or
                    (self.vis_mode == VisMode.GROUND_TRUTH and self.gt_image is not None) or
                    (self.vis_mode == VisMode.WSUM)):
                self._setup_shader()
                gl.glBindVertexArray(self.vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # GUI
            self._draw_gui()

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

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
