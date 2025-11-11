#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 pan;
uniform float zoom;
uniform float window_aspect;
uniform float image_aspect;
uniform int pixel_perfect;
uniform vec2 window_size;
uniform vec2 texture_size;

vec3 colormap_icefire(float v) {
    float vc = clamp(v, 0.0, 1.0);
    vec3 blue = vec3(0.0, 0.0, 1.0);
    vec3 black = vec3(0.0, 0.0, 0.0);
    vec3 red = vec3(1.0, 0.0, 0.0);
    vec3 a = mix(blue, black, vc * 2.0);
    vec3 b = mix(black, red, (vc - 0.5) * 2.0);
    return mix(a, b, step(0.5, vc));
}

void main()
{
    vec2 uv;
    
    if (pixel_perfect == 1) {
        vec2 screen_pos = texcoords * window_size;
        vec2 center_offset = (window_size - texture_size) * 0.5;
        vec2 tex_pos = screen_pos - center_offset;
        tex_pos -= pan * texture_size;
        uv = tex_pos / texture_size;
    } else {
        vec2 centered = (texcoords - vec2(0.5));
        
        float scale_x = 1.0;
        float scale_y = 1.0;
        
        if (window_aspect > image_aspect) {
            scale_x = window_aspect / image_aspect;
        } else {
            scale_y = image_aspect / window_aspect;
        }
        
        centered.x *= scale_x;
        centered.y *= scale_y;
        
        uv = centered / zoom + vec2(0.5) + pan;
    }
    
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        // Получаем значение градиента (усредненное по каналам)
        vec3 gradient = texture(texSampler, uv).rgb;
        float grad_value = (gradient.r + gradient.g + gradient.b) / 3.0;
        
        // Ограничиваем градиент диапазоном [-1, 1]
        grad_value = clamp(grad_value, -1.0, 1.0);
        
        // Нормализация и применение colormap
        float normalized = (grad_value * 5.0 + 1.0) / 2.0;
        vec3 color = colormap_icefire(normalized);
        outputColour = vec4(color, 1.0);
    }
}