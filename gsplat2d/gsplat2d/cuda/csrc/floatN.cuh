#pragma once
#include <cuda_runtime.h>


// (duplicated from helper_math.h)
// float2 operations
__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x; a.y += b.y;
}
__forceinline__ __host__ __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ float2 operator*(float b, float2 a) {
    return make_float2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// float3 operations
__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}
__forceinline__ __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

__forceinline__ __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// float4 operations
__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}
__forceinline__ __host__ __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

struct float5 {
    float data[5];
    __forceinline__ __host__ __device__ float& operator[](int i) { return data[i]; }
    __forceinline__ __host__ __device__ const float& operator[](int i) const { return data[i]; }
};

__forceinline__ __host__ __device__ float5 make_float5(float a, float b, float c, float d, float e) {
    float5 r;
    r.data[0] = a; r.data[1] = b; r.data[2] = c; r.data[3] = d; r.data[4] = e;
    return r;
}

__forceinline__ __host__ __device__ float5 operator+(float5 a, float5 b) {
    float5 r;
    #pragma unroll
    for (int i = 0; i < 5; ++i) r.data[i] = a.data[i] + b.data[i];
    return r;
}

__forceinline__ __host__ __device__ void operator+=(float5 &a, float5 b) {
    #pragma unroll
    for (int i = 0; i < 5; ++i) a.data[i] += b.data[i];
}

__forceinline__ __host__ __device__ float5 operator*(float5 a, float b) {
    float5 r;
    #pragma unroll
    for (int i = 0; i < 5; ++i) r.data[i] = a.data[i] * b;
    return r;
}

__forceinline__ __host__ __device__ float5 operator*(float b, float5 a) {
    float5 r;
    #pragma unroll
    for (int i = 0; i < 5; ++i) r.data[i] = b * a.data[i];
    return r;
}

__forceinline__ __host__ __device__ void operator*=(float5 &a, float b) {
    #pragma unroll
    for (int i = 0; i < 5; ++i) a.data[i] *= b;
}
