#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texDx;      // градиенты по x
uniform sampler2D texDy;      // градиенты по y
uniform vec2 pan;
uniform float zoom;
uniform vec2 window_size;
uniform vec2 texture_size;

// Inferno colormap approximation
vec3 colormap_inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    
    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
    vec3 c1 = vec3(0.258234, 0.038571, 0.406485);
    vec3 c2 = vec3(0.572033, 0.139911, 0.403400);
    vec3 c3 = vec3(0.865006, 0.316822, 0.226055);
    vec3 c4 = vec3(0.987053, 0.641941, 0.139426);
    vec3 c5 = vec3(0.988362, 0.998364, 0.644924);
    
    if (t < 0.2) {
        return mix(c0, c1, t * 5.0);
    } else if (t < 0.4) {
        return mix(c1, c2, (t - 0.2) * 5.0);
    } else if (t < 0.6) {
        return mix(c2, c3, (t - 0.4) * 5.0);
    } else if (t < 0.8) {
        return mix(c3, c4, (t - 0.6) * 5.0);
    } else {
        return mix(c4, c5, (t - 0.8) * 5.0);
    }
}

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
        // Получаем градиенты
        vec3 dx = texture(texDx, uv).rgb;
        vec3 dy = texture(texDy, uv).rgb;
        
        // Вычисляем magnitude для каждого канала
        vec3 magnitude = sqrt(dx * dx + dy * dy);
        
        // Усредняем по каналам
        float avg_magnitude = (magnitude.r + magnitude.g + magnitude.b) / 3.0;
        
        // Нормализуем для визуализации (можно настроить масштаб)
        float normalized = clamp(avg_magnitude * 2.0, 0.0, 1.0);
        
        // Применяем colormap
        vec3 color = colormap_inferno(normalized);
        outputColour = vec4(color, 1.0);
    }
}