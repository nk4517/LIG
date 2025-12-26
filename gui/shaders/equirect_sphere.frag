#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 window_size;
uniform float yaw;    // rotation around Y axis (radians)
uniform float pitch;  // rotation around X axis (radians)
uniform float fov;    // vertical FOV in radians

const float PI = 3.14159265359;

vec2 dirToEquirectUV(vec3 dir) {
    float theta = atan(dir.x, dir.z);  // [-pi, pi]
    float phi = asin(clamp(dir.y, -1.0, 1.0));  // [-pi/2, pi/2]
    
    float u = (theta / PI + 1.0) / 2.0;  // [0, 1]
    float v = 0.5 - phi / PI;  // [0, 1], top=0
    
    return vec2(u, v);
}

void main() {
    vec2 ndc = (texcoords - 0.5) * 2.0;
    float aspect = window_size.x / window_size.y;
    float tanHalfFov = tan(fov / 2.0);
    vec3 dir = normalize(vec3(ndc.x * aspect * tanHalfFov, -ndc.y * tanHalfFov, 1.0));
    
    float cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
    mat3 rotY = mat3(cy, 0, -sy, 0, 1, 0, sy, 0, cy);
    mat3 rotX = mat3(1, 0, 0, 0, cp, sp, 0, -sp, cp);
    dir = rotY * rotX * dir;
    
    vec2 uv = dirToEquirectUV(dir);
    outputColour = texture(texSampler, uv);
}
