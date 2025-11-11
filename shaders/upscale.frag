#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texRender;  // основное изображение
uniform sampler2D texDx;      // градиенты по x
uniform sampler2D texDy;      // градиенты по y
uniform sampler2D texDxy;     // смешанные производные

uniform vec2 source_size;     // размер исходного изображения
uniform vec2 target_size;     // размер целевого изображения
uniform vec2 pan;
uniform float zoom;
uniform float window_aspect;
uniform float image_aspect;
uniform int pixel_perfect;
uniform vec2 window_size;

// Бикубическая сплайн-интерполяция
float spline_interp(mat4 f, vec2 p) {
    // Матрица преобразования для кубического сплайна
    mat4 m = mat4(
        1.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  0.0,
        -3.0, 3.0, -2.0, -1.0,
        2.0, -2.0,  1.0,  1.0
    );
    
    mat4 a = transpose(m) * f * m;
    
    vec4 tx = vec4(1.0, p.x, p.x * p.x, p.x * p.x * p.x);
    vec4 ty = vec4(1.0, p.y, p.y * p.y, p.y * p.y * p.y);
    
    return dot(tx, a * ty);
}

void main() {
    vec2 uv;
    
    if (pixel_perfect == 1) {
        // Pixel-perfect mode
        vec2 screen_pos = texcoords * window_size;
        vec2 center_offset = (window_size - target_size) * 0.5;
        vec2 tex_pos = screen_pos - center_offset;
        tex_pos -= pan * target_size;
        uv = tex_pos / target_size;
    } else {
        // Normal mode with aspect correction
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
        return;
    }
    
    // Преобразование UV в координаты исходного изображения
    // При апскейлинге нужно правильно центрировать координаты
    // Коэффициент масштабирования
    vec2 scale_factor = target_size / source_size;
    // Смещение должно быть в масштабе исходного изображения
    vec2 src_coord = uv * source_size - 0.5 / scale_factor;
    vec2 left_upper = floor(src_coord);
    vec2 frac_coord = fract(src_coord);
    
    // Обработка отрицательных координат
    if (src_coord.x < 0.0) {
        frac_coord.x = 1.0 - fract(-src_coord.x);
    }
    if (src_coord.y < 0.0) {
        frac_coord.y = 1.0 - fract(-src_coord.y);
    }
    
    vec3 color_result = vec3(0.0);
    
    // Интерполяция для каждого канала
    for (int c = 0; c < 3; c++) {
        mat4 f_values;
        
        // Собираем 4x4 матрицу значений для интерполяции
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                vec2 sample_pos = clamp(left_upper + vec2(i, j), vec2(0.0), source_size - 1.0) / source_size;
                
                vec4 z_v = texture(texRender, sample_pos);
                vec4 dx_v = texture(texDx, sample_pos);
                vec4 dy_v = texture(texDy, sample_pos);
                vec4 dxy_v = texture(texDxy, sample_pos);
                
                // Заполняем матрицу F согласно математическому описанию
                if (i == 0 && j == 0) {
                    f_values[0][0] = z_v[c];
                    f_values[2][0] = -dx_v[c];  // инвертируем знак как в WGSL
                    f_values[0][2] = -dy_v[c];  // инвертируем знак как в WGSL
                    f_values[2][2] = dxy_v[c];
                } else if (i == 1 && j == 0) {
                    f_values[1][0] = z_v[c];
                    f_values[3][0] = -dx_v[c];
                    f_values[1][2] = -dy_v[c];
                    f_values[3][2] = dxy_v[c];
                } else if (i == 0 && j == 1) {
                    f_values[0][1] = z_v[c];
                    f_values[2][1] = -dx_v[c];
                    f_values[0][3] = -dy_v[c];
                    f_values[2][3] = dxy_v[c];
                } else { // i == 1 && j == 1
                    f_values[1][1] = z_v[c];
                    f_values[3][1] = -dx_v[c];
                    f_values[1][3] = -dy_v[c];
                    f_values[3][3] = dxy_v[c];
                }
            }
        }
        
        // Интерполяция с перестановкой координат как в WGSL
        color_result[c] = spline_interp(f_values, vec2(frac_coord.y, frac_coord.x));
    }
    
    outputColour = vec4(clamp(color_result, 0.0, 1.0), 1.0);
}