#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 pan;
uniform float zoom;
uniform vec2 window_size;
uniform vec2 texture_size;

void main()
{
    vec2 uv;
    vec2 screen_pos = texcoords * window_size;
    vec2 center_offset = (window_size - texture_size * zoom) * 0.5;
    vec2 tex_pos = screen_pos - center_offset;

    // Pan в пикселях окна
    tex_pos += pan;

    uv = tex_pos / (texture_size * zoom);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        outputColour = texture(texSampler, uv);
    }
}