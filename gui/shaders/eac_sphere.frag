#version 330

smooth in vec2 texcoords;
out vec4 outputColour;

uniform sampler2D texSampler;
uniform vec2 window_size;
uniform float yaw;    // rotation around Y axis (radians)
uniform float pitch;  // rotation around X axis (radians)
uniform float fov;    // vertical FOV in radians

const float PI = 3.14159265359;

// EAC face grid positions (row, col)
// left(0,0), front(0,1), right(0,2), bottom(1,0), back(1,1), top(1,2)

vec2 dirToEacUV(vec3 dir) {
    float ax = abs(dir.x);
    float ay = abs(dir.y);
    float az = abs(dir.z);
    
    int face;
    float u, v;
    
    if (az >= ax && az >= ay) {
        if (dir.z > 0.0) {
            // front +Z
            face = 1;
            u = dir.x / dir.z;
            v = -dir.y / dir.z;
        } else {
            // back -Z
            face = 4;
            u = -dir.x / (-dir.z);
            v = -dir.y / (-dir.z);
        }
    } else if (ax > ay && ax > az) {
        if (dir.x < 0.0) {
            // left -X
            face = 0;
            u = dir.z / (-dir.x);
            v = -dir.y / (-dir.x);
        } else {
            // right +X
            face = 2;
            u = -dir.z / dir.x;
            v = -dir.y / dir.x;
        }
    } else {
        if (dir.y > 0.0) {
            // top +Y
            face = 5;
            u = dir.x / dir.y;
            v = dir.z / dir.y;
        } else {
            // bottom -Y
            face = 3;
            u = dir.x / (-dir.y);
            v = -dir.z / (-dir.y);
        }
    }
    
    // EAC inverse transform: atan(t) / (pi/4)
    u = atan(u) / (PI / 4.0);
    v = atan(v) / (PI / 4.0);
    
    // Convert to [0,1] within face
    float px = (u + 1.0) / 2.0;
    float py = (v + 1.0) / 2.0;
    
    // Grid positions: face -> (row, col)
    int rows[6] = int[](0, 0, 0, 1, 1, 1);
    int cols[6] = int[](0, 1, 2, 0, 1, 2);
    
    // Apply face rotation for bottom row
    if (face == 3 || face == 5) { // bottom, top: CW 90
        float tmp = px;
        px = py;
        py = 1.0 - tmp;
    } else if (face == 4) { // back: 270
        float tmp = px;
        px = 1.0 - py;
        py = tmp;
    }
    
    return vec2((float(cols[face]) + px) / 3.0, (float(rows[face]) + py) / 2.0);
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
    vec2 uv = dirToEacUV(dir);
    outputColour = texture(texSampler, uv);
}
