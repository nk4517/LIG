from pathlib import Path

import glfw
import imgui
from OpenGL import GL as gl

from LIG.gui.common import VisMode, GradientMode, ShaderBinding
from LIG.gui.visualizer import VisualizerGUI, TextureSlot, _create_texture, load_shader, compile_shaders


class SphereVisualizerGUI(VisualizerGUI):
    """Visualizer with sphere view support (EAC/Equirect)"""

    TEXTURE_SLOTS: tuple[tuple[str, int], ...] = ()

    def __init__(self, width=800, height=600, use_cuda=True):
        super().__init__(width, height, use_cuda)
        self.sphere_yaw = 0.0    # radians, rotation around Y
        self.sphere_pitch = 0.0  # radians, rotation around X
        self.sphere_fov = 1.5708 # pi/2 = 90 degrees vertical FOV

    def init(self):
        super().init()
        self.textures['eac'] = TextureSlot(gl_id=_create_texture(gl.GL_NEAREST))
        self.textures['equirect'] = TextureSlot(gl_id=_create_texture(gl.GL_LINEAR))

        shader_dir = Path(__file__).parent / "shaders"
        vertex_source = load_shader(shader_dir / "fullscreen.vert")
        eac_sphere_fragment = load_shader(shader_dir / "eac_sphere.frag")
        equirect_sphere_fragment = load_shader(shader_dir / "equirect_sphere.frag")

        self.programs['eac_sphere'] = compile_shaders(vertex_source, eac_sphere_fragment)
        self.programs['equirect_sphere'] = compile_shaders(vertex_source, equirect_sphere_fragment)

    def cursor_pos_callback(self, window, xpos, ypos):
        if imgui.get_io().want_capture_mouse:
            self.is_panning = False
            return

        if self.is_panning:
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y

            if self.vis_mode in (VisMode.EAC_SPHERE, VisMode.EQUIRECT_SPHERE):
                sensitivity = 0.002
                self.sphere_yaw -= dx * sensitivity
                self.sphere_pitch -= dy * sensitivity
                self.sphere_pitch = max(-1.5, min(1.5, self.sphere_pitch))
            else:
                self.fit_to_window = False
                self.pan_x -= dx
                self.pan_y -= dy

        self.last_mouse_x = xpos
        self.last_mouse_y = ypos

    def scroll_callback(self, window, xoffset, yoffset):
        if imgui.get_io().want_capture_mouse:
            return

        if self.vis_mode in (VisMode.EAC_SPHERE, VisMode.EQUIRECT_SPHERE):
            if yoffset > 0:
                self.sphere_fov /= 1.1
            else:
                self.sphere_fov *= 1.1
            self.sphere_fov = max(0.3, min(1.5708, self.sphere_fov))
            return

        super().scroll_callback(window, xoffset, yoffset)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_9:
                self.vis_mode = VisMode.EAC_SPHERE
                self.view_dirty.set()
                self.gui_ready.set()
                return
            elif key == glfw.KEY_E:
                self.vis_mode = VisMode.EQUIRECT_SPHERE
                self.view_dirty.set()
                self.gui_ready.set()
                return
        super().key_callback(window, key, scancode, action, mods)

    def get_shader_binding(self, vis_mode: VisMode, gradient_mode: GradientMode) -> ShaderBinding:
        if vis_mode == VisMode.EAC_SPHERE:
            return ShaderBinding(program='eac_sphere', textures={'texSampler': 'eac'}, skip_texture_size=True)
        if vis_mode == VisMode.EQUIRECT_SPHERE:
            return ShaderBinding(program='equirect_sphere', textures={'texSampler': 'equirect'}, skip_texture_size=True)
        return super().get_shader_binding(vis_mode, gradient_mode)

    def _setup_shader(self):
        super()._setup_shader()
        if self.active_vis_mode in (VisMode.EAC_SPHERE, VisMode.EQUIRECT_SPHERE):
            binding = self.get_shader_binding(self.active_vis_mode, self.active_gradient_mode)
            prog = self.programs[binding.program]
            gl.glUniform1f(gl.glGetUniformLocation(prog, "yaw"), self.sphere_yaw)
            gl.glUniform1f(gl.glGetUniformLocation(prog, "pitch"), self.sphere_pitch)
            gl.glUniform1f(gl.glGetUniformLocation(prog, "fov"), self.sphere_fov)
            gl.glUniform2f(gl.glGetUniformLocation(prog, "window_size"),
                           float(self.width), float(self.height))

    def can_display(self, vis_mode: VisMode) -> bool:
        if vis_mode == VisMode.EAC_SPHERE:
            return self.textures.get('eac') is not None and self.textures['eac'].width > 0
        if vis_mode == VisMode.EQUIRECT_SPHERE:
            return self.textures.get('equirect') is not None and self.textures['equirect'].width > 0
        return False


def main():
    eac_path = Path(r"P:\3d_printing\_gsplat_sandbox\splatting_app\gsplat-2025\examples\data\1800_eac.png")
    # equirect_path = Path(r"p:\3d_printing\_gsplat_sandbox\splatting_app\gsplat-2025\examples\data\pano4s.png")
    viewer = SphereVisualizerGUI(width=1280, height=720, use_cuda=True)
    viewer.init()
    viewer.load_image_to_slot(str(eac_path), 'eac')
    # viewer.load_image_to_slot(str(equirect_path), 'equirect')
    viewer.vis_mode = VisMode.EAC_SPHERE
    viewer.active_vis_mode = VisMode.EAC_SPHERE
    # viewer.vis_mode = VisMode.EQUIRECT_SPHERE
    # viewer.active_vis_mode = VisMode.EQUIRECT_SPHERE
    viewer.main_loop()
    viewer.cleanup()


if __name__ == "__main__":
    main()
