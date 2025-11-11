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