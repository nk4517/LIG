#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform sampler2D texLUT;// 256x1 RGB, diverging colormap
uniform vec2 pan;
uniform float zoom;
uniform vec2 window_size;
uniform vec2 texture_size;
uniform float scale;// gradient amplification, default 5.0

void main()
{
    vec2 screen_pos = texcoords * window_size;
    vec2 center_offset = (window_size - texture_size * zoom) * 0.5;
    vec2 tex_pos = screen_pos - center_offset + pan;
    vec2 uv = tex_pos / (texture_size * zoom);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        vec3 gradient = clamp(texture(texSampler, uv).rgb, -1.0, 1.0);
        float avg = (gradient.r + gradient.g + gradient.b) / 3.0;
        float normalized = (avg * scale + 1.0) / 2.0;
        vec3 color = texture(texLUT, vec2(clamp(normalized, 0.0, 1.0), 0.5)).rgb;
        outputColour = vec4(color, 1.0);
    }
}
