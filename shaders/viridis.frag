#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 pan;
uniform float zoom;
uniform vec2 window_size;
uniform vec2 texture_size;

// Viridis colormap approximation
vec3 colormap_viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    
    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    vec3 c1 = vec3(0.282623, 0.140926, 0.457517);
    vec3 c2 = vec3(0.253935, 0.265254, 0.529983);
    vec3 c3 = vec3(0.206756, 0.371758, 0.553117);
    vec3 c4 = vec3(0.163625, 0.471133, 0.558148);
    vec3 c5 = vec3(0.127568, 0.566949, 0.550556);
    vec3 c6 = vec3(0.134692, 0.658636, 0.517649);
    vec3 c7 = vec3(0.266941, 0.748751, 0.440573);
    vec3 c8 = vec3(0.477504, 0.821444, 0.318195);
    vec3 c9 = vec3(0.741388, 0.873449, 0.149561);
    vec3 c10 = vec3(0.993248, 0.906157, 0.143936);
    
    if (t < 0.1) return mix(c0, c1, t * 10.0);
    else if (t < 0.2) return mix(c1, c2, (t - 0.1) * 10.0);
    else if (t < 0.3) return mix(c2, c3, (t - 0.2) * 10.0);
    else if (t < 0.4) return mix(c3, c4, (t - 0.3) * 10.0);
    else if (t < 0.5) return mix(c4, c5, (t - 0.4) * 10.0);
    else if (t < 0.6) return mix(c5, c6, (t - 0.5) * 10.0);
    else if (t < 0.7) return mix(c6, c7, (t - 0.6) * 10.0);
    else if (t < 0.8) return mix(c7, c8, (t - 0.7) * 10.0);
    else if (t < 0.9) return mix(c8, c9, (t - 0.8) * 10.0);
    else return mix(c9, c10, (t - 0.9) * 10.0);
}

void main()
{
    vec2 uv;
    vec2 screen_pos = texcoords * window_size;
    vec2 center_offset = (window_size - texture_size * zoom) * 0.5;
    vec2 tex_pos = screen_pos - center_offset;
    
    tex_pos += pan;
    
    uv = tex_pos / (texture_size * zoom);
    
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        float wsum_value = texture(texSampler, uv).r;
        vec3 color = colormap_viridis(wsum_value);
        outputColour = vec4(color, 1.0);
    }
}