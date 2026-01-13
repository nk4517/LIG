#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "floatN.cuh"

template<typename T>
__global__ void gradient_aware_upscale_kernel(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const T* __restrict__ render,
    const T* __restrict__ dx,
    const T* __restrict__ dy,
    const T* __restrict__ dxy,
    T* __restrict__ output
);

template<typename T>
__global__ void gradient_aware_upscale_backward_kernel(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const T* __restrict__ grad_output,  // [dst_H, dst_W] of float3
    T* __restrict__ grad_render,        // [src_H, src_W] of float3
    T* __restrict__ grad_dx,
    T* __restrict__ grad_dy,
    T* __restrict__ grad_dxy
);
