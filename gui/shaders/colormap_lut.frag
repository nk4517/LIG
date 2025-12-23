#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texData;
uniform sampler2D texLUT;// 256x1 RGB
uniform vec2 pan;
uniform float data_min;  // default 0.0
uniform float data_max;  // default 1.0
uniform float zoom;
uniform vec2 window_size;
uniform vec2 texture_size;

void main()
{
    vec2 screen_pos = texcoords * window_size;
    vec2 center_offset = (window_size - texture_size * zoom) * 0.5;
    vec2 tex_pos = screen_pos - center_offset + pan;
    vec2 uv = tex_pos / (texture_size * zoom);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        float raw = texture(texData, uv).r;
        float value = clamp((raw - data_min) / (data_max - data_min + 1e-8), 0.0, 1.0);
        vec3 color = texture(texLUT, vec2(value, 0.5)).rgb;
        outputColour = vec4(color, 1.0);
    }
}
