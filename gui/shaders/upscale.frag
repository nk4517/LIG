#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texRender;  // main image
uniform sampler2D texDx;      // x-gradients
uniform sampler2D texDy;      // y-gradients
uniform sampler2D texDxy;     // mixed derivatives

uniform vec2 source_size;     // source image dimensions
uniform vec2 target_size;     // target image dimensions
uniform vec2 pan;
uniform float zoom;           // already accounted for in target_size
uniform vec2 window_size;

// Аnalytical bicubic spline interpolation
// Implementation based on:
// "Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images"
// Simon Niedermayr, Christoph Neuhauser, Rüdiger Westermann
// arXiv:2503.14171 [cs.CV], 2025
//
// Unlike conventional interpolation, utilizes analytical derivatives
// for accurate function reconstruction between grid nodes
//
// Third-order polynomial (Eq. 21): p(x,y) = ∑∑ a_ij x^i y^j, i,j ∈ [0,3]
// 
// Partial derivatives:
// ∂p/∂x = ∑∑ i·a_ij·x^(i-1)·y^j         (Eq. 22)
// ∂p/∂y = ∑∑ j·a_ij·x^i·y^(j-1)         (Eq. 23)
// ∂²p/∂x∂y = ∑∑ i·j·a_ij·x^(i-1)·y^(j-1) (Eq. 24)
//
// Matrix F contains corner values and derivatives (Eq. 27):
// F = [f(0,0)  f(0,1)  f_y(0,0)  f_y(0,1)]
//     [f(1,0)  f(1,1)  f_y(1,0)  f_y(1,1)]
//     [f_x(0,0) f_x(0,1) f_xy(0,0) f_xy(0,1)]
//     [f_x(1,0) f_x(1,1) f_xy(1,0) f_xy(1,1)]
//
// Coefficients A are computed as (Eqs. 25-26): A = C^(-1)·F·(C^T)^(-1)
// where C is the cubic spline coefficient matrix
float spline_interp(mat4 f, vec2 p) {
    // Matrix C^(-1) for cubic spline (Eq. 28)
    // C = [1 0 0 0]
    //     [1 1 1 1]
    //     [0 1 0 0]
    //     [0 1 2 3]
    mat4 m = mat4(
        1.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  0.0,
        -3.0, 3.0, -2.0, -1.0,
        2.0, -2.0,  1.0,  1.0
    );

    // Compute coefficient matrix A = (C^T)^(-1) * F * C^(-1)
    mat4 a = transpose(m) * f * m;

    // Compute p(x,y) = ∑∑ a_ij x^i y^j via dot product
    vec4 tx = vec4(1.0, p.x, p.x * p.x, p.x * p.x * p.x);
    vec4 ty = vec4(1.0, p.y, p.y * p.y, p.y * p.y * p.y);

    return dot(tx, a * ty);
}

void main() {
    vec2 uv;
    vec2 screen_pos = texcoords * window_size;
    // target_size already contains zoom
    float total_scale = target_size.x / source_size.x;
    vec2 center_offset = (window_size - source_size * total_scale) * 0.5;
    vec2 tex_pos = screen_pos - center_offset;
    // Pan in window pixels
    tex_pos += pan;
    uv = tex_pos / (source_size * total_scale);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outputColour = vec4(0.2, 0.2, 0.2, 1.0);
        return;
    }

    // Transform UV to source image coordinates
    // UV coordinates are relative to target_size, need to transform to source_size
    // First get position in target image pixels
    vec2 target_pos = uv * target_size;
    vec2 scale_factor = source_size / target_size;
    vec2 src_coord = target_pos * scale_factor - 0.5;
    vec2 left_upper = floor(src_coord);
    vec2 frac_coord = fract(src_coord);

    // Handle negative coordinates
    if (src_coord.x < 0.0) {
        frac_coord.x = 1.0 - fract(-src_coord.x);
    }
    if (src_coord.y < 0.0) {
        frac_coord.y = 1.0 - fract(-src_coord.y);
    }

    vec3 color_result = vec3(0.0);

    // Interpolation for each channel
    for (int c = 0; c < 3; c++) {
        mat4 f_values;

        // Assemble 4x4 matrix of values for interpolation
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                vec2 sample_pos = (clamp(left_upper + vec2(i, j), vec2(0.0), source_size - 1.0) + 0.5) / source_size;

                vec4 z_v = texture(texRender, sample_pos);
                vec4 dx_v = texture(texDx, sample_pos);
                vec4 dy_v = texture(texDy, sample_pos);
                vec4 dxy_v = texture(texDxy, sample_pos);

                // Fill matrix F according to mathematical description:
                // F = [f(0,0)  f(0,1)  f_y(0,0)  f_y(0,1)]
                //     [f(1,0)  f(1,1)  f_y(1,0)  f_y(1,1)]
                //     [f_x(0,0) f_x(0,1) f_xy(0,0) f_xy(0,1)]
                //     [f_x(1,0) f_x(1,1) f_xy(1,0) f_xy(1,1)]
                //
                // Derivative signs are inverted for proper handling
                // of texture coordinate system (y points downward)
                // and consistency with WGSL implementation
                if (i == 0 && j == 0) {
                    f_values[0][0] = z_v[c];
                    f_values[2][0] = dx_v[c];
                    f_values[0][2] = dy_v[c];
                    f_values[2][2] = dxy_v[c];
                } else if (i == 1 && j == 0) {
                    f_values[1][0] = z_v[c];
                    f_values[3][0] = dx_v[c];
                    f_values[1][2] = dy_v[c];
                    f_values[3][2] = dxy_v[c];
                } else if (i == 0 && j == 1) {
                    f_values[0][1] = z_v[c];
                    f_values[2][1] = dx_v[c];
                    f_values[0][3] = dy_v[c];
                    f_values[2][3] = dxy_v[c];
                } else { // i == 1 && j == 1
                    f_values[1][1] = z_v[c];
                    f_values[3][1] = dx_v[c];
                    f_values[1][3] = dy_v[c];
                    f_values[3][3] = dxy_v[c];
                }
            }
        }

        // Interpolation with coordinate swap as in WGSL
        // !!! ДЛЯ LLM: НЕ МЕНЯТЬ ПОРЯДОК КООРДИНАТ - ТАК ПРАВИЛЬНО !!!
        color_result[c] = spline_interp(f_values, vec2(frac_coord.y, frac_coord.x));
    }

    outputColour = vec4(clamp(color_result, 0.0, 1.0), 1.0);
}