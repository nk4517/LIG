import threading
import typing

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import torch
from pathlib import Path

if typing.TYPE_CHECKING:
    from gaussianlig import LIG


from LIG.upscaler_torch import bicubic_spline_upscale_single_channel

try:
    from cuda import cudart as cu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

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


class LIGVisualizer:
    def __init__(self, width=800, height=600, use_cuda=True):
        self.width = width
        self.height = height
        self.use_cuda = use_cuda and CUDA_AVAILABLE
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
        self.zoom = 1.0
        self.aspect = 1.0
        self.fit_to_window = True  # Автоподстройка зума под размер окна
        
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Model and image data
        self.gaussian_model: "LIG|None" = None
        self.gt_image = None
        self.target_image = None  # Для хранения первой мелкой картинки
        self.current_image = None
        self.image_width = 1
        self.image_height = 1
        self.image_aspect = 1.0
        self.texture = None
        self.program = None
        self.vao = None
        
        # Дополнительные текстуры для target и ground truth
        self.texture_target = None
        self.texture_gt = None
        self.target_width = 1
        self.target_height = 1
        self.gt_width = 1
        self.gt_height = 1
        
        # Дополнительные текстуры для градиентов
        self.texture_dx = None
        self.texture_dy = None
        self.texture_dxy = None
        self.upscale_program = None
        
        # Flag for visualization updates
        self.was_updated = False
        
        self.show_info = True
        self.current_iter = 0
        self.current_psnr = 0.0
        self.current_loss = 0.0
        self.use_magnitude_shader = False
        
        # Visualization mode: 0=render, 1=upscaled, 2=target, 3=ground_truth, 4=gradients
        self.vis_mode = 0
        self.gradient_mode = 0  # 0=dx, 1=dy, 2=dxy, 3=magnitude (для vis_mode==4)
        self.vis_mode_names = ["Render", "Upscaled", "Target", "Ground Truth", "Gradients"]
        self.gradient_mode_names = ["dX", "dY", "dXY", "Magnitude"]
        
        # Флаги для апскейлинга
        self.use_upscale_shader = False
        self.use_gradient_shader = False
        self.use_magnitude_shader = False
        
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
        gradient_fragment = load_shader(shader_dir / "gradient.frag")
        upscale_fragment = load_shader(shader_dir / "upscale.frag")
        magnitude_fragment = load_shader(shader_dir / "magnitude.frag")
        
        self.program = compile_shaders(vertex_source, texture_fragment)
        self.upscale_program = compile_shaders(vertex_source, upscale_fragment)
        self.gradient_program = compile_shaders(vertex_source, gradient_fragment)
        self.magnitude_program = compile_shaders(vertex_source, magnitude_fragment)
        self.vao = gl.glGenVertexArrays(1)
        
        # Создаём все текстуры
        self.texture = self._create_texture()
        self.texture_dx = self._create_texture()
        self.texture_dy = self._create_texture()
        self.texture_dxy = self._create_texture()
        self.texture_target = self._create_texture()
        self.texture_gt = self._create_texture()
        
        self._init_cuda_interop()
        
        glfw.swap_interval(2)
        
    def _create_texture(self):
        """Create and configure a texture with standard parameters"""
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        return texture
        
    def _init_cuda_interop(self):
        """Initialize CUDA-OpenGL interop if available and enabled"""
        self.cuda_image = None
        if not self.use_cuda:
            return
            
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError("cudaGLGetDevices failed")
        
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
        if self.fit_to_window and self.image_width > 0 and self.image_height > 0:
            window_aspect = self.width / self.height if self.height > 0 else 1.0
            image_aspect = self.image_width / self.image_height if self.image_height > 0 else 1.0
            
            # Вычисляем базовый zoom чтобы изображение поместилось в окно
            zoom_x = self.width / float(self.image_width)
            zoom_y = self.height / float(self.image_height)
            # Берём минимальный zoom чтобы изображение гарантированно поместилось
            self.zoom = min(zoom_x, zoom_y)
        
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
            self.zoom *= zoom_speed
        else:
            self.zoom /= zoom_speed
        self.zoom = max(0.1, min(10.0, self.zoom))
        
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:  # Reset view
                self.pan_x = 0.0
                self.pan_y = 0.0
                # Если fit_to_window включен - пересчитываем zoom, иначе сбрасываем на 1.0
                if self.fit_to_window and self.image_width > 0 and self.image_height > 0:
                    window_aspect = self.width / self.height if self.height > 0 else 1.0
                    image_aspect = self.image_width / self.image_height if self.image_height > 0 else 1.0
                    
                    # Вычисляем базовый zoom чтобы изображение поместилось в окно
                    zoom_x = self.width / float(self.image_width)
                    zoom_y = self.height / float(self.image_height)
                    # Берём минимальный zoom чтобы изображение гарантированно поместилось
                    self.zoom = min(zoom_x, zoom_y)
                else:
                    self.zoom = 1.0
            elif key == glfw.KEY_1:
                self.vis_mode = 0  # Render
            elif key == glfw.KEY_2:
                self.vis_mode = 1  # Upscaled
            elif key == glfw.KEY_3:
                self.vis_mode = 2  # Target
            elif key == glfw.KEY_4:
                self.vis_mode = 3  # Ground Truth
            elif key == glfw.KEY_5:
                # Переключение градиентов по кругу
                if self.vis_mode != 4:
                    self.vis_mode = 4
                    self.gradient_mode = 0  # Начинаем с dx
                else:
                    # Циклическое переключение между dx, dy, dxy, magnitude
                    self.gradient_mode = (self.gradient_mode + 1) % 4
            elif key == glfw.KEY_0:
                # Установка zoom в 1.0
                self.zoom = 1.0
                self.fit_to_window = False
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
                
    def set_model(self, gaussian_model, gt_image):
        """Set the model and ground truth image for rendering"""
        self.gaussian_model = gaussian_model
        self.gt_image = gt_image
        
    def set_target_image(self, reference_image):
        """Set target image (e.g., first low-res image in multi-scale)"""
        self.target_image = reference_image
        
    def set_updated(self):
        """Signal that model was updated and needs re-rendering"""
        self.was_updated = True
        
    def update_stats(self, iteration, psnr, loss):
        """Update training statistics"""
        self.current_iter = iteration
        self.current_psnr = psnr
        self.current_loss = loss
        
    def _prepare_render_data(self, rendered):
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
        
    def _update_texture(self, image_np):
        """Update OpenGL texture - chooses CPU or CUDA method based on configuration"""
        self.current_image = image_np  # Keep for compatibility
        
        # Get dimensions from tensor
        if isinstance(image_np, torch.Tensor):
            h, w = image_np.shape[:2]
            img_tensor = image_np
        else:
            # Fallback for numpy arrays
            h, w = image_np.shape[:2]
            img_tensor = torch.from_numpy(image_np).cuda()
        
        self.image_width = w
        self.image_height = h
        self.image_aspect = w / h if h > 0 else 1.0
        
        # Если включен fit_to_window - пересчитываем zoom
        if self.fit_to_window:
            window_aspect = self.width / self.height if self.height > 0 else 1.0
            # Вычисляем базовый zoom чтобы изображение поместилось в окно
            zoom_x = self.width / float(w) if w > 0 else 1.0
            zoom_y = self.height / float(h) if h > 0 else 1.0
            # Берём минимальный zoom чтобы изображение гарантированно поместилось
            self.zoom = min(zoom_x, zoom_y)
        
        if self.use_cuda:
            self._update_texture_cuda(img_tensor, w, h)
        else:
            self._update_texture_cpu(img_tensor, w, h)
    
    def _update_texture_cpu(self, img_tensor, w, h):
        """Update texture using CPU path"""
        # Convert to numpy if needed
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.detach().cpu().numpy()
        else:
            img_np = img_tensor
        
        # Update texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                       gl.GL_RGBA, gl.GL_FLOAT, img_np)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def _update_texture_cuda(self, img_tensor, w, h):
        """Update texture using CUDA-OpenGL interop"""
        # Initialize or update texture if size changed
        if self.cuda_image is None or h != getattr(self, '_last_h', 0) or w != getattr(self, '_last_w', 0):
            # Unregister old CUDA image if exists
            if self.cuda_image is not None:
                cu.cudaGraphicsUnregisterResource(self.cuda_image)
            
            # Update texture size
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                           gl.GL_RGBA, gl.GL_FLOAT, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            
            # Register with CUDA
            err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
                self.texture,
                gl.GL_TEXTURE_2D,
                cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
            )
            if err != cu.cudaError_t.cudaSuccess:
                raise RuntimeError("Unable to register OpenGL texture")
            
            self._last_h = h
            self._last_w = w
        
        # Transfer data using CUDA
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        # Ensure tensor is contiguous and CUDA
        if not img_tensor.is_cuda:
            img_tensor = img_tensor.cuda()
        img_tensor = img_tensor.contiguous()
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img_tensor.data_ptr(),
            4 * 4 * w,  # 4 bytes per float * 4 channels * width
            4 * 4 * w,
            h,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")
    
    def _update_gradient_textures(self, dx, dy, dxy):
        """Обновление текстур градиентов для апскейлинга"""
        gradients = [
            (self.texture_dx, dx),
            (self.texture_dy, dy),
            (self.texture_dxy, dxy)
        ]
        
        for texture, gradient in gradients:
            tensor = self._prepare_gradient_tensor(gradient)
            h, w = tensor.shape[:2]
            
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            np_data = tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.numpy()
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                           gl.GL_RGBA, gl.GL_FLOAT, np_data)
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def _prepare_gradient_tensor(self, gradient):
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
        
    def _load_reference_texture(self, image, texture, mode):
        """Load reference texture (target or gt) if not already loaded"""
        # Check if already loaded
        width_attr = f'{mode}_width'
        height_attr = f'{mode}_height'
        
        if getattr(self, width_attr) == 1:  # Not loaded yet
            tensor = self._prepare_render_data(image)
            h, w = tensor.shape[:2]
            setattr(self, width_attr, w)
            setattr(self, height_attr, h)
            
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            np_data = tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.numpy()
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0,
                           gl.GL_RGBA, gl.GL_FLOAT, np_data)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
        # Update display dimensions
        self.image_width = getattr(self, width_attr)
        self.image_height = getattr(self, height_attr)
        self.image_aspect = self.image_width / self.image_height if self.image_height > 0 else 1.0

    def _render_model(self):
        """Render the gaussian model and update textures"""
        if self.gaussian_model is None or not self.was_updated:
            return

        self.e_finished_rendering.clear()
        self.e_want_to_render.set()

        with self.lock:
            try:
                with torch.no_grad():
                    # Check if model is on CUDA before rendering
                    if not self.gaussian_model and self.gaussian_model.level_models:
                        return

                    base_model = self.gaussian_model.level_models[0]
                    first_param = next(base_model.parameters(), None)
                    if first_param is not None and first_param.is_cuda:
                        # Render the current state without changing model mode
                        result = base_model()
                        rendered = result["render"].float()

                        # Store derivatives if available
                        dx = result.get("dx", None)
                        dy = result.get("dy", None)
                        dxy = result.get("dxy", None)

                        # Choose what to display based on vis_mode
                        if self.vis_mode == 0:
                            # Render mode
                            display_tensor = rendered
                            image_tensor = self._prepare_render_data(display_tensor)
                            self._update_texture(image_tensor)
                        elif self.vis_mode == 1:
                            # Upscaled mode - use GL shader upscaling
                            if dx is not None and dy is not None and dxy is not None:
                                self._update_gradient_textures(dx, dy, dxy)
                                display_tensor = rendered
                                image_tensor = self._prepare_render_data(display_tensor)
                                self._update_texture(image_tensor)
                                self.use_upscale_shader = True
                            else:
                                # Fallback to regular rendering if derivatives not available
                                display_tensor = rendered
                                image_tensor = self._prepare_render_data(display_tensor)
                                self._update_texture(image_tensor)
                                self.use_upscale_shader = False
                        elif self.vis_mode == 2:
                            # Target mode - показываем мелкую картинку
                            if self.target_image is not None:
                                self._load_reference_texture(self.target_image, self.texture_target, 'target')
                        elif self.vis_mode == 3:
                            # Ground Truth mode
                            if self.gt_image is not None:
                                self._load_reference_texture(self.gt_image, self.texture_gt, 'gt')
                        elif self.vis_mode == 4:
                            # Gradient mode - показываем градиенты в зависимости от gradient_mode
                            if self.gradient_mode == 0 and dx is not None:
                                # dX mode
                                display_tensor = dx.float()
                                image_tensor = self._prepare_gradient_tensor(display_tensor)
                                self._update_texture(image_tensor)
                                self.use_gradient_shader = True
                            elif self.gradient_mode == 1 and dy is not None:
                                # dY mode
                                display_tensor = dy.float()
                                image_tensor = self._prepare_gradient_tensor(display_tensor)
                                self._update_texture(image_tensor)
                                self.use_gradient_shader = True
                            elif self.gradient_mode == 2 and dxy is not None:
                                # dXY mode
                                display_tensor = dxy.float()
                                image_tensor = self._prepare_gradient_tensor(display_tensor)
                                self._update_texture(image_tensor)
                                self.use_gradient_shader = True
                            elif self.gradient_mode == 3:
                                # Magnitude mode
                                if dx is not None and dy is not None:
                                    # Update gradient textures for magnitude shader
                                    self._update_gradient_textures(dx, dy, dxy if dxy is not None else dx)
                                    # Use rendered image for texture (not used in magnitude shader but needed for consistency)
                                    display_tensor = rendered
                                    image_tensor = self._prepare_render_data(display_tensor)
                                    self._update_texture(image_tensor)
                                    self.use_magnitude_shader = True
                                else:
                                    # Fallback to regular rendering if derivatives not available
                                    display_tensor = rendered
                                    image_tensor = self._prepare_render_data(display_tensor)
                                    self._update_texture(image_tensor)
                                    self.use_magnitude_shader = False
                            else:
                                # Fallback to regular rendering
                                display_tensor = rendered
                                image_tensor = self._prepare_render_data(display_tensor)
                                self._update_texture(image_tensor)
                        else:
                            display_tensor = rendered
                            # Fallback to regular rendering
                            image_tensor = self._prepare_render_data(display_tensor)
                            self._update_texture(image_tensor)

                        # Clear the flag after rendering
                        self.was_updated = False
                    else:
                        # Model is on CPU, skip rendering but clear flag
                        self.was_updated = False

            except Exception as e:
                print(f"Render error: {e}")
            finally:
                self.e_want_to_render.clear()
                self.e_finished_rendering.set()
        
    def _setup_shader(self):
        """Setup and bind appropriate shader for rendering"""
        # Проверяем наличие данных в зависимости от режима
        if self.vis_mode == 2 and self.target_image is None:
            return
        elif self.vis_mode == 3 and self.gt_image is None:
            return
        elif self.vis_mode in [0, 1, 4] and self.current_image is None:
            return
            
        if self.use_upscale_shader:
            gl.glUseProgram(self.upscale_program)
            
            # Устанавливаем униформы для апскейлинга
            gl.glUniform2f(gl.glGetUniformLocation(self.upscale_program, "source_size"), 
                          float(self.image_width), float(self.image_height))
            gl.glUniform2f(gl.glGetUniformLocation(self.upscale_program, "target_size"), 
                          float(self.image_width * self.zoom), 
                          float(self.image_height * self.zoom))
            gl.glUniform2f(gl.glGetUniformLocation(self.upscale_program, "pan"), float(self.pan_x), float(self.pan_y))
            gl.glUniform1f(gl.glGetUniformLocation(self.upscale_program, "zoom"), self.zoom)
            gl.glUniform2f(gl.glGetUniformLocation(self.upscale_program, "window_size"), float(self.width), float(self.height))
            
            # Привязываем текстуры
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glUniform1i(gl.glGetUniformLocation(self.upscale_program, "texRender"), 0)
            
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dx)
            gl.glUniform1i(gl.glGetUniformLocation(self.upscale_program, "texDx"), 1)
            
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dy)
            gl.glUniform1i(gl.glGetUniformLocation(self.upscale_program, "texDy"), 2)
            
            gl.glActiveTexture(gl.GL_TEXTURE3)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dxy)
            gl.glUniform1i(gl.glGetUniformLocation(self.upscale_program, "texDxy"), 3)
            
            # Сбрасываем флаг после использования
            self.use_upscale_shader = False
        elif self.use_magnitude_shader:
            gl.glUseProgram(self.magnitude_program)
            gl.glUniform2f(gl.glGetUniformLocation(self.magnitude_program, "pan"), float(self.pan_x), float(self.pan_y))
            gl.glUniform1f(gl.glGetUniformLocation(self.magnitude_program, "zoom"), self.zoom)
            gl.glUniform2f(gl.glGetUniformLocation(self.magnitude_program, "window_size"), float(self.width), float(self.height))
            gl.glUniform2f(gl.glGetUniformLocation(self.magnitude_program, "texture_size"), float(self.image_width), float(self.image_height))
            
            # Bind gradient textures
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dx)
            gl.glUniform1i(gl.glGetUniformLocation(self.magnitude_program, "texDx"), 0)
            
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_dy)
            gl.glUniform1i(gl.glGetUniformLocation(self.magnitude_program, "texDy"), 1)
            
            # Сбрасываем флаг после использования
            self.use_magnitude_shader = False
        elif self.use_gradient_shader:
            gl.glUseProgram(self.gradient_program)
            gl.glUniform2f(gl.glGetUniformLocation(self.gradient_program, "pan"), float(self.pan_x), float(self.pan_y))
            gl.glUniform1f(gl.glGetUniformLocation(self.gradient_program, "zoom"), self.zoom)
            gl.glUniform2f(gl.glGetUniformLocation(self.gradient_program, "window_size"), float(self.width), float(self.height))
            gl.glUniform2f(gl.glGetUniformLocation(self.gradient_program, "texture_size"), float(self.image_width), float(self.image_height))
            
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            
            # Сбрасываем флаг после использования
            self.use_gradient_shader = False
        else:
            gl.glUseProgram(self.program)
            gl.glUniform2f(gl.glGetUniformLocation(self.program, "pan"), float(self.pan_x), float(self.pan_y))
            # Для GT режима корректируем зум на соотношение размеров
            if self.vis_mode == 3 and self.target_width > 0 and self.gt_width > 0:
                zoom_correction = self.target_width / self.gt_width
                effective_zoom = self.zoom * zoom_correction
            else:
                effective_zoom = self.zoom
            gl.glUniform1f(gl.glGetUniformLocation(self.program, "zoom"), effective_zoom)
            gl.glUniform2f(gl.glGetUniformLocation(self.program, "window_size"), float(self.width), float(self.height))
            gl.glUniform2f(gl.glGetUniformLocation(self.program, "texture_size"), float(self.image_width), float(self.image_height))
            
            gl.glActiveTexture(gl.GL_TEXTURE0)
            # Выбираем текстуру в зависимости от режима
            if self.vis_mode == 2:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_target)
            elif self.vis_mode == 3:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_gt)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        
    def _draw_gui(self):
        """Draw ImGui interface"""
        if not self.show_info:
            return
            
        # Settings panel - верхний правый угол
        imgui.set_next_window_position(self.width - 320, 10)
        imgui.set_next_window_size(300, 320)
        imgui.begin("Settings")
        
        # Fit to window checkbox
        changed, self.fit_to_window = imgui.checkbox("Fit to Window", self.fit_to_window)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically adjust zoom to fit image in window")
        
        # При включении fit_to_window пересчитываем zoom
        if changed and self.fit_to_window and self.image_width > 0 and self.image_height > 0:
            window_aspect = self.width / self.height if self.height > 0 else 1.0
            image_aspect = self.image_width / self.image_height if self.image_height > 0 else 1.0
            
            # Вычисляем базовый zoom чтобы изображение поместилось в окно
            zoom_x = self.width / float(self.image_width)
            zoom_y = self.height / float(self.image_height)
            # Берём минимальный zoom чтобы изображение гарантированно поместилось
            self.zoom = min(zoom_x, zoom_y)
        
        imgui.separator()
        
        # Display mode radio buttons
        imgui.text("Display:")
        
        if imgui.radio_button("Render [1]", self.vis_mode == 0):
            self.vis_mode = 0
        imgui.same_line()
        if imgui.radio_button("Upscaled [2]", self.vis_mode == 1):
            self.vis_mode = 1
        
        if imgui.radio_button("Target [3]", self.vis_mode == 2):
            self.vis_mode = 2
        imgui.same_line()
        if imgui.radio_button("GT [4]", self.vis_mode == 3):
            self.vis_mode = 3
        
        # Градиенты - показываем текущий режим
        gradient_label = f"Gradients [{self.gradient_mode_names[self.gradient_mode]}] [5]"
        if imgui.radio_button(gradient_label, self.vis_mode == 4):
            self.vis_mode = 4
        
        imgui.separator()
        
        # Zoom slider - общий для всех режимов
        label = "Upscaler zoom" if self.vis_mode == 1 else "Zoom"
        changed, self.zoom = imgui.slider_float(
            label, 
            self.zoom, 
            0.1, 
            10.0,
            f"{self.zoom:.2f}x"
        )
        if changed:
            self.fit_to_window = False
            
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom level (also controlled by mouse wheel)")
        
        imgui.end()
        
        # Info panel - нижний правый угол
        imgui.set_next_window_position(self.width - 320, self.height - 360)
        imgui.set_next_window_size(300, 350)
        imgui.begin("Info")
        imgui.text(f"Mode: {'CUDA' if self.use_cuda else 'CPU'}")
        # Для градиентов показываем подрежим
        if self.vis_mode == 4:
            imgui.text(f"View: {self.vis_mode_names[self.vis_mode]} ({self.gradient_mode_names[self.gradient_mode]})")
        else:
            imgui.text(f"View: {self.vis_mode_names[self.vis_mode]}")
        imgui.separator()
        imgui.text(f"Iteration: {self.current_iter}")
        imgui.text(f"PSNR: {self.current_psnr:.2f} dB")
        imgui.text(f"Loss: {self.current_loss:.6f}")
        imgui.separator()
        # Показываем реальный zoom
        if self.image_width > 0 and self.image_height > 0:
            window_scale_x = self.width / float(self.image_width)
            window_scale_y = self.height / float(self.image_height)
            window_scale = min(window_scale_x, window_scale_y)
            display_zoom = self.zoom * window_scale
        else:
            display_zoom = self.zoom
        imgui.text(f"Zoom: {display_zoom:.2f}x")
        imgui.text(f"Pan: ({self.pan_x:.0f}, {self.pan_y:.0f}) px")
        imgui.separator()
        imgui.text("Controls:")
        imgui.text("LMB - Pan")
        imgui.text("Scroll - Zoom")
        imgui.text("R - Reset view")
        imgui.text("0 - Set zoom to 1.0")
        imgui.text("1-4 - Switch view mode")
        imgui.text("5 - Cycle gradients")
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
            if ((self.vis_mode in [0, 1, 4] and self.current_image is not None) or
                (self.vis_mode == 2 and self.target_image is not None) or
                (self.vis_mode == 3 and self.gt_image is not None)):
                self._setup_shader()
                gl.glBindVertexArray(self.vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            
            # GUI
            # GUI
            self._draw_gui()
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
        
    def cleanup(self):
        """Clean up resources"""
        if self.use_cuda and self.cuda_image is not None:
            cu.cudaGraphicsUnregisterResource(self.cuda_image)
        if self.impl:
            self.impl.shutdown()
        if self.window:
            glfw.terminate()
            
    def run(self):
        """Run visualization"""
        self.init()
        self.main_loop()
        self.cleanup() 