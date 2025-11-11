import threading
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import torch

from LIG.upscaler_torch import bicubic_spline_upscale_single_channel

try:
    from cuda import cudart as cu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 pan;
uniform float zoom;
uniform float window_aspect;
uniform float image_aspect;
uniform int pixel_perfect;  // 0=normal, 1=pixel perfect
uniform vec2 window_size;
uniform vec2 texture_size;

void main()
{
    vec2 uv;
    
    if (pixel_perfect == 1) {
        // Pixel-perfect mode - map screen pixels directly to texture pixels
        vec2 screen_pos = texcoords * window_size;
        vec2 center_offset = (window_size - texture_size) * 0.5;
        vec2 tex_pos = screen_pos - center_offset;
        
        // Apply pan in pixel space
        tex_pos -= pan * texture_size;
        
        uv = tex_pos / texture_size;
    } else {
        // Normal mode with aspect correction
        vec2 centered = (texcoords - vec2(0.5));
    
    // Correct for window aspect ratio to maintain image aspect ratio
    float scale_x = 1.0;
    float scale_y = 1.0;
    
    if (window_aspect > image_aspect) {
        // Window is wider than image - scale x
        scale_x = window_aspect / image_aspect;
    } else {
        // Window is taller than image - scale y
        scale_y = image_aspect / window_aspect;
    }
    
    centered.x *= scale_x;
    centered.y *= scale_y;
    
        uv = centered / zoom + vec2(0.5) + pan;
    }
    
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        outputColour = texture(texSampler, uv);
    }
}
"""

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
        
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Model and image data
        self.gaussian_model = None
        self.gt_image = None
        self.current_image = None
        self.image_width = 1
        self.image_height = 1
        self.image_aspect = 1.0
        self.texture = None
        self.program = None
        self.vao = None
        
        # Flag for visualization updates
        self.was_updated = False
        
        self.show_info = True
        self.current_iter = 0
        self.current_psnr = 0.0
        self.current_loss = 0.0
        
        # Visualization mode: 0=render, 1=dx, 2=dy, 3=dxy
        self.vis_mode = 0
        self.vis_mode_names = ["Render", "dX", "dY", "dXY", "Upscaled"]
        self.pixel_perfect = False  # Flag for pixel-perfect rendering
        
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
        
        self.program = compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        self.vao = gl.glGenVertexArrays(1)
        
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        self._init_cuda_interop()
        
        glfw.swap_interval(2)
        
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
        
    def cursor_pos_callback(self, window, xpos, ypos):
        if imgui.get_io().want_capture_mouse:
            self.is_panning = False
            return
        
        if self.is_panning:
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y
            
            # Calculate aspect ratio correction (same as in shader)
            window_aspect = self.width / self.height if self.height > 0 else 1.0
            
            scale_x = 1.0
            scale_y = 1.0
            
            if window_aspect > self.image_aspect:
                # Window is wider than image - scale x
                scale_x = window_aspect / self.image_aspect
            else:
                # Window is taller than image - scale y
                scale_y = self.image_aspect / window_aspect
            
            # Convert pixel movement to UV space with aspect correction
            # Negative dx because moving mouse right should show content to the right (pan left)
            # Negative dy for correct vertical movement
            self.pan_x -= (dx / self.width) * scale_x / self.zoom
            self.pan_y -= (dy / self.height) * scale_y / self.zoom
            
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
                self.zoom = 1.0
            elif key >= glfw.KEY_1 and key <= glfw.KEY_5:
                # Switch visualization mode with keys 1-5
                self.vis_mode = key - glfw.KEY_1
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
                
    def setModel(self, gaussian_model, gt_image):
        """Set the model and ground truth image for rendering"""
        self.gaussian_model = gaussian_model
        self.gt_image = gt_image
        
    def set_updated(self):
        """Signal that model was updated and needs re-rendering"""
        self.was_updated = True
        
    def update_stats(self, iteration, psnr, loss):
        """Update training statistics"""
        self.current_iter = iteration
        self.current_psnr = psnr
        self.current_loss = loss
        
    def _prepare_gradient_data(self, gradient):
        """Prepare gradient tensor for visualization with red-blue colormap"""
        # Handle different tensor dimensions
        if gradient.dim() == 4:
            gradient = gradient.squeeze(0)
        
        if gradient.dim() == 3:
            # Average across channels (CHW -> HW)
            gradient = gradient.mean(dim=0)
        
        # Use quantiles for robust normalization, symmetric around zero
        q_low = torch.quantile(gradient.flatten(), 0.01)
        q_high = torch.quantile(gradient.flatten(), 0.99)
        
        # Make symmetric around zero
        max_abs = max(abs(q_low), abs(q_high))
        
        if max_abs > 1e-6:
            gradient_norm = torch.clamp(gradient / max_abs, -1.0, 1.0)
        else:
            gradient_norm = torch.zeros_like(gradient)
        
        # Create red-blue colormap
        # Negative values -> blue, positive values -> red
        red = torch.clamp(gradient_norm, 0, 1)
        blue = torch.clamp(-gradient_norm, 0, 1)
        green = torch.zeros_like(gradient_norm)
        alpha = torch.ones_like(gradient_norm)
        
        # Stack into RGBA tensor
        image_tensor = torch.stack([red, green, blue, alpha], dim=-1)
        
        return image_tensor
        
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
        
    def main_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()
            
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Check if we need to render
            if self.gaussian_model is not None:
                # Check if model was updated
                if self.was_updated:
                    self.e_finished_rendering.clear()
                    self.e_want_to_render.set()
                    
                    with self.lock:
                        try:
                            with torch.no_grad():
                                # Check if model is on CUDA before rendering
                                # Get first parameter to check device
                                first_param = next(self.gaussian_model.parameters(), None)
                                if first_param is not None and first_param.is_cuda:
                                    # Render the current state without changing model mode
                                    result = self.gaussian_model()
                                    rendered = result["render"].float()
                                    
                                    # Store derivatives if available
                                    dx = result.get("dx", None)
                                    dy = result.get("dy", None)
                                    dxy = result.get("dxy", None)
                                    
                                    # Choose what to display based on vis_mode
                                    if self.vis_mode == 0:
                                        display_tensor = rendered
                                        # Prepare tensor for OpenGL
                                        image_tensor = self._prepare_render_data(display_tensor)
                                    elif self.vis_mode == 1 and dx is not None:
                                        display_tensor = dx.float()
                                        # Use gradient visualization for derivatives
                                        image_tensor = self._prepare_gradient_data(display_tensor)
                                    elif self.vis_mode == 2 and dy is not None:
                                        display_tensor = dy.float()
                                        # Use gradient visualization for derivatives
                                        image_tensor = self._prepare_gradient_data(display_tensor)
                                    elif self.vis_mode == 3 and dxy is not None:
                                        display_tensor = dxy.float()
                                        # Use gradient visualization for derivatives
                                        image_tensor = self._prepare_gradient_data(display_tensor)
                                    elif self.vis_mode == 4:
                                        # Upscaled mode - use bicubic spline upscaling
                                        h, w = rendered.permute(0, 2, 3, 1).shape[1:3]
                                        scale_factor = 2
                                        new_h = h * scale_factor
                                        new_w = w * scale_factor
                                        
                                        # Ensure we have derivatives for upscaling
                                        if dx is not None and dy is not None and dxy is not None:
                                            # Process each channel separately to save memory
                                            channels = []
                                            rendered_hwc = rendered.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                                            dx_hwc = dx.squeeze(0).permute(1, 2, 0)
                                            dy_hwc = dy.squeeze(0).permute(1, 2, 0)
                                            dxy_hwc = dxy.squeeze(0).permute(1, 2, 0)
                                            
                                            for c in range(rendered_hwc.shape[2]):
                                                upscaled_channel = bicubic_spline_upscale_single_channel(
                                                    rendered_hwc[:, :, c], 
                                                    dx_hwc[:, :, c], 
                                                    dy_hwc[:, :, c], 
                                                    dxy_hwc[:, :, c], 
                                                    new_h, new_w
                                                )
                                                channels.append(upscaled_channel)
                                            
                                            # Stack channels back together
                                            display_tensor = torch.stack(channels, dim=-1)  # [new_h, new_w, C]
                                            image_tensor = torch.clamp(display_tensor, 0, 1)
                                            
                                            # Ensure RGBA
                                            if image_tensor.shape[2] == 3:
                                                alpha = torch.ones((*image_tensor.shape[:2], 1),
                                                                 dtype=image_tensor.dtype,
                                                                 device=image_tensor.device)
                                                image_tensor = torch.cat([image_tensor, alpha], dim=2)
                                        else:
                                            # Fallback to regular rendering if derivatives not available
                                            display_tensor = rendered
                                            image_tensor = self._prepare_render_data(display_tensor)
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
            
            # Render image
            if self.current_image is not None:
                gl.glUseProgram(self.program)
                gl.glUniform2f(gl.glGetUniformLocation(self.program, "pan"), self.pan_x, self.pan_y)
                gl.glUniform1f(gl.glGetUniformLocation(self.program, "zoom"), self.zoom)
                gl.glUniform1f(gl.glGetUniformLocation(self.program, "window_aspect"), self.aspect)
                gl.glUniform1f(gl.glGetUniformLocation(self.program, "image_aspect"), self.image_aspect)
                gl.glUniform1i(gl.glGetUniformLocation(self.program, "pixel_perfect"), 1 if self.pixel_perfect else 0)
                gl.glUniform2f(gl.glGetUniformLocation(self.program, "window_size"), float(self.width), float(self.height))
                gl.glUniform2f(gl.glGetUniformLocation(self.program, "texture_size"), float(self.image_width), float(self.image_height))
                
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
                gl.glBindVertexArray(self.vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            
            # GUI
            if self.show_info:
                imgui.set_next_window_position(10, 10)
                imgui.begin("Info", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
                imgui.text(f"Mode: {'CUDA' if self.use_cuda else 'CPU'}")
                imgui.text(f"View: {self.vis_mode_names[self.vis_mode]}")
                imgui.separator()
                imgui.text(f"Iteration: {self.current_iter}")
                imgui.text(f"PSNR: {self.current_psnr:.2f} dB")
                imgui.text(f"Loss: {self.current_loss:.6f}")
                imgui.separator()
                imgui.text(f"Zoom: {self.zoom:.2f}x")
                imgui.text(f"Pan: ({self.pan_x:.3f}, {self.pan_y:.3f})")
                imgui.separator()
                imgui.text("Controls:")
                imgui.text("LMB - Pan")
                imgui.text("Scroll - Zoom")
                imgui.text("R - Reset view")
                imgui.text("1-5 - Switch view mode")
                imgui.text("ESC - Exit")
                imgui.end()
            
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