#include "upscale.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
 * ============================================================================
 * Gradient-Aware Bicubic Spline Upscaling for 3D Gaussian Splatting
 * ============================================================================
 * 
 * Reference:
 *   Niedermayr, S., Neuhauser, C., & Westermann, R. (2025).
 *   "Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images"
 *   arXiv:2503.14171v2 [cs.CV]
 *   https://arxiv.org/abs/2503.14171
 * 
 * ============================================================================
 * OVERVIEW
 * ============================================================================
 * 
 * This implementation performs bicubic spline interpolation using analytical
 * image gradients from 3DGS rendering, rather than finite-difference 
 * approximations. The key insight is that 3DGS provides exact gradients 
 * ∂I/∂x, ∂I/∂y, ∂²I/∂x∂y at each pixel, enabling more accurate spline fitting.
 * 
 * ============================================================================
 * MATHEMATICAL FORMULATION
 * ============================================================================
 * 
 * BICUBIC SPLINE PARAMETERIZATION (Section G, Eq. 21):
 * 
 *   p(x,y) = Σᵢ₌₀³ Σⱼ₌₀³ aᵢⱼ xⁱ yʲ
 * 
 * The spline coefficients A ∈ ℝ⁴ˣ⁴ are computed by solving (Eq. 25-26):
 * 
 *   F = C · A · Cᵀ
 *   A = C⁻¹ · F · (Cᵀ)⁻¹
 * 
 * where F is the constraint matrix containing function values and derivatives
 * at the four corner points of the interpolation cell.
 * 
 * F MATRIX LAYOUT (Eq. 6, 27):
 * 
 *   F = [ f(0,0)    f(0,1)    fᵧ(0,0)   fᵧ(0,1)  ]
 *       [ f(1,0)    f(1,1)    fᵧ(1,0)   fᵧ(1,1)  ]
 *       [ fₓ(0,0)   fₓ(0,1)   fₓᵧ(0,0)  fₓᵧ(0,1) ]
 *       [ fₓ(1,0)   fₓ(1,1)   fₓᵧ(1,0)  fₓᵧ(1,1) ]
 * 
 * where:
 *   f(i,j)    = pixel value at corner (i,j)
 *   fₓ(i,j)   = ∂f/∂x at corner (i,j)  -- analytical gradient from 3DGS
 *   fᵧ(i,j)   = ∂f/∂y at corner (i,j)  -- analytical gradient from 3DGS
 *   fₓᵧ(i,j)  = ∂²f/∂x∂y at corner (i,j) -- mixed partial from 3DGS
 * 
 * C MATRIX (Eq. 28):
 * 
 * Derived from cubic polynomial constraints at x=0 and x=1:
 *   f(x)  = a₀ + a₁x + a₂x² + a₃x³
 *   f'(x) = a₁ + 2a₂x + 3a₃x²
 * 
 *   C = [ 1  0  0  0 ]    (f(0)  = a₀)
 *       [ 1  1  1  1 ]    (f(1)  = a₀ + a₁ + a₂ + a₃)
 *       [ 0  1  0  0 ]    (f'(0) = a₁)
 *       [ 0  1  2  3 ]    (f'(1) = a₁ + 2a₂ + 3a₃)
 * 
 *   C⁻¹ = [  1   0   0   0 ]
 *         [  0   0   1   0 ]
 *         [ -3   3  -2  -1 ]
 *         [  2  -2   1   1 ]
 * 
 * POLYNOMIAL EVALUATION (Eq. 7):
 * 
 *   p(x,y) = [1  x  x²  x³] · A · [1  y  y²  y³]ᵀ
 * 
 * ============================================================================
 * BACKWARD PASS (Section 8.B)
 * ============================================================================
 * 
 * For gradient backpropagation through the upscaling operation:
 * 
 *   p = pxᵀ · C⁻¹ · F · (C⁻¹)ᵀ · py
 * 
 * where px = [1, tx, tx², tx³]ᵀ and py = [1, ty, ty², ty³]ᵀ
 * 
 * The gradient w.r.t. each element of F is:
 * 
 *   ∂p/∂F[i,j] = (C⁻¹ᵀ · px)[i] · (C⁻¹ᵀ · py)[j]
 * 
 * This allows backpropagation of gradients to:
 *   - grad_render: gradients w.r.t. pixel values f(i,j)
 *   - grad_dx:     gradients w.r.t. x-derivatives fₓ(i,j)
 *   - grad_dy:     gradients w.r.t. y-derivatives fᵧ(i,j)
 *   - grad_dxy:    gradients w.r.t. mixed partials fₓᵧ(i,j)
 * 
 * ============================================================================
 * 3DGS ANALYTICAL GRADIENTS (Section 5)
 * ============================================================================
 * 
 * The image I(x,y) from 3DGS is computed via alpha blending (Eq. 3):
 * 
 *   I(x,y) = Σᵢ₌₁ᴺ Tᵢ(x,y) · αᵢ(x,y) · cᵢ
 * 
 * Analytical gradients are computed during rendering (Eq. 8-10):
 * 
 *   ∂I/∂x = Σᵢ cᵢ · (∂Tᵢ/∂x · αᵢ + Tᵢ · ∂αᵢ/∂x)
 * 
 *   ∂I/∂y = Σᵢ cᵢ · (∂Tᵢ/∂y · αᵢ + Tᵢ · ∂αᵢ/∂y)
 * 
 *   ∂²I/∂x∂y = Σᵢ cᵢ · (∂²Tᵢ/∂x∂y · αᵢ + ∂Tᵢ/∂x · ∂αᵢ/∂y 
 *                       + ∂Tᵢ/∂y · ∂αᵢ/∂x + Tᵢ · ∂²αᵢ/∂x∂y)
 * 
 * These gradients are computed iteratively during the blending loop with
 * minimal overhead, then passed to this upscaling kernel.
 * 
 * ============================================================================
 * IMPLEMENTATION NOTES
 * ============================================================================
 * 
 * - Forward kernel: computes A = C⁻¹ · F · (C⁻¹)ᵀ, then evaluates p(tx,ty)
 * - Backward kernel: computes ∂p/∂F and accumulates via atomicAdd
 * - Memory layout: CHW format (channels × height × width)
 * - ROI support: allows upscaling a subregion of the source image
 * 
 * ============================================================================
 */

// C^(-1) matrix for cubic spline (stored in row-major)
// C = [1 0 0 0;
//      1 1 1 1;
//      0 1 0 0;
//      0 1 2 3]
// C^(-1) = [1 0 0 0;
//           0 0 1 0;
//           -3 3 -2 -1;
//           2 -2 1 1]
__constant__ float C_inv[16] = {
    1.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f,
   -3.0f,  3.0f, -2.0f, -1.0f,
    2.0f, -2.0f,  1.0f,  1.0f
};

// Compute C_inv^T * p vector (used for both px and py)
__device__ __forceinline__ void compute_C_inv_T_p(float t, float* out) {
    // C_inv^T[i,k] = C_inv[k,i]
    // out[i] = sum_k C_inv[k*4+i] * p[k] where p = [1, t, t², t³]
    const float t2 = t * t;
    const float t3 = t2 * t;
    out[0] = C_inv[0] * 1.0f + C_inv[4] * t + C_inv[8] * t2 + C_inv[12] * t3;
    out[1] = C_inv[1] * 1.0f + C_inv[5] * t + C_inv[9] * t2 + C_inv[13] * t3;
    out[2] = C_inv[2] * 1.0f + C_inv[6] * t + C_inv[10] * t2 + C_inv[14] * t3;
    out[3] = C_inv[3] * 1.0f + C_inv[7] * t + C_inv[11] * t2 + C_inv[15] * t3;
}

// Interpolate single channel using precomputed coefficients
// F layout (column-major, matching original):
//   col0: [f00, f01, fx00, fx01]
//   col1: [f10, f11, fx10, fx11]  
//   col2: [fy00, fy01, fxy00, fxy01]
//   col3: [fy10, fy11, fxy10, fxy11]
// Result = (C_inv^T * px)^T * F * (C_inv^T * py)
template<typename T>
__device__ __forceinline__ T spline_interp(
    T f00, T f01, T f10, T f11,
    T fx00, T fx01, T fx10, T fx11,
    T fy00, T fy01, T fy10, T fy11,
    T fxy00, T fxy01, T fxy10, T fxy11,
    const float* cx, const float* cy
) {
    // F * cy (4 elements)
    T Fcy0 = f00 * cy[0] + f10 * cy[1] + fy00 * cy[2] + fy10 * cy[3];
    T Fcy1 = f01 * cy[0] + f11 * cy[1] + fy01 * cy[2] + fy11 * cy[3];
    T Fcy2 = fx00 * cy[0] + fx10 * cy[1] + fxy00 * cy[2] + fxy10 * cy[3];
    T Fcy3 = fx01 * cy[0] + fx11 * cy[1] + fxy01 * cy[2] + fxy11 * cy[3];
    
    // cx^T * (F * cy)
    return cx[0] * Fcy0 + cx[1] * Fcy1 + cx[2] * Fcy2 + cx[3] * Fcy3;
}

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
) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    const int roi_w = roi_x2 - roi_x1;
    const int roi_h = roi_y2 - roi_y1;

    // Map dst coords to src ROI coords
    const float src_x = roi_x1 + (dst_x + 0.5f) * roi_w / dst_w - 0.5f;
    const float src_y = roi_y1 + (dst_y + 0.5f) * roi_h / dst_h - 0.5f;

    // Get integer and fractional parts
    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    float tx = src_x - x0;
    float ty = src_y - y0;

    // Clamp indices
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));

    // Precompute C_inv^T * px and C_inv^T * py (same for all channels)
    float cx[4], cy[4];
    compute_C_inv_T_p(tx, cx);
    compute_C_inv_T_p(ty, cy);

    // Load float3 values at 4 corners (coalesced HWC access)
    const int idx00 = y0 * src_w + x0;
    const int idx01 = y0 * src_w + x1;
    const int idx10 = y1 * src_w + x0;
    const int idx11 = y1 * src_w + x1;

    const T f00 = render[idx00], f01 = render[idx01], f10 = render[idx10], f11 = render[idx11];
    const T fx00 = dx[idx00], fx01 = dx[idx01], fx10 = dx[idx10], fx11 = dx[idx11];
    const T fy00 = dy[idx00], fy01 = dy[idx01], fy10 = dy[idx10], fy11 = dy[idx11];
    const T fxy00 = dxy[idx00], fxy01 = dxy[idx01], fxy10 = dxy[idx10], fxy11 = dxy[idx11];

    // Interpolate each channel
    T result = spline_interp(
        f00, f01, f10, f11,
        fx00, fx01, fx10, fx11,
        fy00, fy01, fy10, fy11,
        fxy00, fxy01, fxy10, fxy11,
        cx, cy
    );

    output[dst_y * dst_w + dst_x] = result;
}

template __global__ void gradient_aware_upscale_kernel<float>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float* __restrict__ render,
    const float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ dxy,
    float* __restrict__ output);

template __global__ void gradient_aware_upscale_kernel<float2>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float2* __restrict__ render,
    const float2* __restrict__ dx,
    const float2* __restrict__ dy,
    const float2* __restrict__ dxy,
    float2* __restrict__ output);

template __global__ void gradient_aware_upscale_kernel<float3>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float3* __restrict__ render,
    const float3* __restrict__ dx,
    const float3* __restrict__ dy,
    const float3* __restrict__ dxy,
    float3* __restrict__ output);

template __global__ void gradient_aware_upscale_kernel<float4>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float4* __restrict__ render,
    const float4* __restrict__ dx,
    const float4* __restrict__ dy,
    const float4* __restrict__ dxy,
    float4* __restrict__ output);

template __global__ void gradient_aware_upscale_kernel<float5>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float5* __restrict__ render,
    const float5* __restrict__ dx,
    const float5* __restrict__ dy,
    const float5* __restrict__ dxy,
    float5* __restrict__ output);


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
) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    const int roi_w = roi_x2 - roi_x1;
    const int roi_h = roi_y2 - roi_y1;

    const float src_x = roi_x1 + (dst_x + 0.5f) * roi_w / dst_w - 0.5f;
    const float src_y = roi_y1 + (dst_y + 0.5f) * roi_h / dst_h - 0.5f;

    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    float tx = src_x - x0;
    float ty = src_y - y0;

    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));

    // Precompute C_inv^T * px and C_inv * py
    float cx[4], cy[4];
    compute_C_inv_T_p(tx, cx);
    compute_C_inv_T_p(ty, cy);

    // p = px^T * C_inv * F * C_inv^T * py
    // ∂p/∂F[i,j] = cx[i] * cy[j]
    //
    // F layout (row-major):
    // row 0: [f00,  f10,  fy00,  fy10 ]
    // row 1: [f01,  f11,  fy01,  fy11 ]
    // row 2: [fx00, fx10, fxy00, fxy10]
    // row 3: [fx01, fx11, fxy01, fxy11]

    const T g = grad_output[dst_y * dst_w + dst_x];

    // Corner indices in src image
    const int idx00 = y0 * src_w + x0;
    const int idx01 = y0 * src_w + x1;
    const int idx10 = y1 * src_w + x0;
    const int idx11 = y1 * src_w + x1;

    // grad_render: F[0,0]=f00, F[0,1]=f10, F[1,0]=f01, F[1,1]=f11
    atomic_add_scaled(grad_render, idx00, g, cx[0] * cy[0]);
    atomic_add_scaled(grad_render, idx01, g, cx[1] * cy[0]);
    atomic_add_scaled(grad_render, idx10, g, cx[0] * cy[1]);
    atomic_add_scaled(grad_render, idx11, g, cx[1] * cy[1]);

    // grad_dx: F[2,0]=fx00, F[2,1]=fx10, F[3,0]=fx01, F[3,1]=fx11
    atomic_add_scaled(grad_dx, idx00, g, cx[2] * cy[0]);
    atomic_add_scaled(grad_dx, idx01, g, cx[3] * cy[0]);
    atomic_add_scaled(grad_dx, idx10, g, cx[2] * cy[1]);
    atomic_add_scaled(grad_dx, idx11, g, cx[3] * cy[1]);

    // grad_dy: F[0,2]=fy00, F[0,3]=fy10, F[1,2]=fy01, F[1,3]=fy11
    atomic_add_scaled(grad_dy, idx00, g, cx[0] * cy[2]);
    atomic_add_scaled(grad_dy, idx01, g, cx[1] * cy[2]);
    atomic_add_scaled(grad_dy, idx10, g, cx[0] * cy[3]);
    atomic_add_scaled(grad_dy, idx11, g, cx[1] * cy[3]);

    // grad_dxy: F[2,2]=fxy00, F[2,3]=fxy10, F[3,2]=fxy01, F[3,3]=fxy11
    atomic_add_scaled(grad_dxy, idx00, g, cx[2] * cy[2]);
    atomic_add_scaled(grad_dxy, idx01, g, cx[3] * cy[2]);
    atomic_add_scaled(grad_dxy, idx10, g, cx[2] * cy[3]);
    atomic_add_scaled(grad_dxy, idx11, g, cx[3] * cy[3]);
}

template __global__ void gradient_aware_upscale_backward_kernel<float>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_render,
    float* __restrict__ grad_dx,
    float* __restrict__ grad_dy,
    float* __restrict__ grad_dxy);

template __global__ void gradient_aware_upscale_backward_kernel<float2>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float2* __restrict__ grad_output,
    float2* __restrict__ grad_render,
    float2* __restrict__ grad_dx,
    float2* __restrict__ grad_dy,
    float2* __restrict__ grad_dxy);

template __global__ void gradient_aware_upscale_backward_kernel<float3>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float3* __restrict__ grad_output,
    float3* __restrict__ grad_render,
    float3* __restrict__ grad_dx,
    float3* __restrict__ grad_dy,
    float3* __restrict__ grad_dxy);

template __global__ void gradient_aware_upscale_backward_kernel<float4>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float4* __restrict__ grad_output,
    float4* __restrict__ grad_render,
    float4* __restrict__ grad_dx,
    float4* __restrict__ grad_dy,
    float4* __restrict__ grad_dxy);

template __global__ void gradient_aware_upscale_backward_kernel<float5>(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float5* __restrict__ grad_output,
    float5* __restrict__ grad_render,
    float5* __restrict__ grad_dx,
    float5* __restrict__ grad_dy,
    float5* __restrict__ grad_dxy);


// Src-centric backward kernel: iterates over src pixels, no atomics needed
// Each src pixel (sx, sy) participates as corner in up to 4 cells:
//   role (0,0): cell [sx, sx+1) x [sy, sy+1)
//   role (0,1): cell [sx, sx+1) x [sy-1, sy)
//   role (1,0): cell [sx-1, sx) x [sy, sy+1)
//   role (1,1): cell [sx-1, sx) x [sy-1, sy)
template<typename T>
__global__ void gradient_aware_upscale_backward_src_centric_kernel(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_render,
    T* __restrict__ grad_dx,
    T* __restrict__ grad_dy,
    T* __restrict__ grad_dxy
) {
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= src_w || sy >= src_h) return;

    const float roi_w = roi_x2 - roi_x1;
    const float roi_h = roi_y2 - roi_y1;
    const float scale_x = (float)dst_w / roi_w;
    const float scale_y = (float)dst_h / roi_h;

    T acc_f{}, acc_fx{}, acc_fy{}, acc_fxy{};

    // Process all 4 roles this src pixel can have
    // role_x: 0 = src is left edge (x0), 1 = src is right edge (x1)
    // role_y: 0 = src is top edge (y0), 1 = src is bottom edge (y1)
    #pragma unroll
    for (int role_x = 0; role_x < 2; role_x++) {
        #pragma unroll
        for (int role_y = 0; role_y < 2; role_y++) {
            // Cell where this src pixel is corner (role_x, role_y)
            // cell_x0 = sx - role_x, cell_y0 = sy - role_y
            const int cell_x0 = sx - role_x;
            const int cell_y0 = sy - role_y;

            // Skip invalid cells (outside src bounds)
            if (cell_x0 < 0 || cell_x0 >= src_w - 1) continue;
            if (cell_y0 < 0 || cell_y0 >= src_h - 1) continue;

            // Find dst pixels that map to this cell
            // src_x = roi_x1 + (dst_x + 0.5) * roi_w / dst_w - 0.5
            // cell_x0 <= src_x < cell_x0 + 1
            // (cell_x0 + 0.5 - roi_x1) * scale_x - 0.5 <= dst_x < (cell_x0 + 1.5 - roi_x1) * scale_x - 0.5

            // For boundary handling: extend range to include clamped pixels
            float cell_x_lo = cell_x0 + 0.5f;
            float cell_x_hi = cell_x0 + 1.5f;
            float cell_y_lo = cell_y0 + 0.5f;
            float cell_y_hi = cell_y0 + 1.5f;
            
            // Extend to capture clamped dst pixels at boundaries
            if (cell_x0 == 0) cell_x_lo = -1e9f;
            if (cell_x0 == src_w - 2) cell_x_hi = 1e9f;
            if (cell_y0 == 0) cell_y_lo = -1e9f;
            if (cell_y0 == src_h - 2) cell_y_hi = 1e9f;
            
            const int dst_x_min = max(0, (int)ceilf((cell_x_lo - roi_x1) * scale_x - 0.5f));
            const int dst_x_max = min(dst_w - 1, (int)floorf((cell_x_hi - roi_x1) * scale_x - 0.5f - 1e-6f));
            const int dst_y_min = max(0, (int)ceilf((cell_y_lo - roi_y1) * scale_y - 0.5f));
            const int dst_y_max = min(dst_h - 1, (int)floorf((cell_y_hi - roi_y1) * scale_y - 0.5f - 1e-6f));

            // F matrix indices for this role:
            // role (0,0): f->F[0,0], fx->F[2,0], fy->F[0,2], fxy->F[2,2] => cx[0]*cy[0], cx[2]*cy[0], cx[0]*cy[2], cx[2]*cy[2]
            // role (0,1): f->F[1,0], fx->F[3,0], fy->F[1,2], fxy->F[3,2] => cx[1]*cy[0], cx[3]*cy[0], cx[1]*cy[2], cx[3]*cy[2]
            // role (1,0): f->F[0,1], fx->F[2,1], fy->F[0,3], fxy->F[2,3] => cx[0]*cy[1], cx[2]*cy[1], cx[0]*cy[3], cx[2]*cy[3]
            // role (1,1): f->F[1,1], fx->F[3,1], fy->F[1,3], fxy->F[3,3] => cx[1]*cy[1], cx[3]*cy[1], cx[1]*cy[3], cx[3]*cy[3]
            const int cx_f_idx = role_x;      // 0 or 1
            const int cx_d_idx = role_x + 2;  // 2 or 3
            const int cy_f_idx = role_y;      // 0 or 1
            const int cy_d_idx = role_y + 2;  // 2 or 3

            for (int dst_y = dst_y_min; dst_y <= dst_y_max; dst_y++) {
                const float src_y_f = roi_y1 + (dst_y + 0.5f) * roi_h / dst_h - 0.5f;
                const float ty = src_y_f - cell_y0;
                float cy[4];
                compute_C_inv_T_p(ty, cy);

                for (int dst_x = dst_x_min; dst_x <= dst_x_max; dst_x++) {
                    const float src_x_f = roi_x1 + (dst_x + 0.5f) * roi_w / dst_w - 0.5f;
                    const float tx = src_x_f - cell_x0;
                    float cx[4];
                    compute_C_inv_T_p(tx, cx);

                    const T g = grad_output[dst_y * dst_w + dst_x];

                    acc_f   += g * (cx[cx_f_idx] * cy[cy_f_idx]);
                    acc_fx  += g * (cx[cx_d_idx] * cy[cy_f_idx]);
                    acc_fy  += g * (cx[cx_f_idx] * cy[cy_d_idx]);
                    acc_fxy += g * (cx[cx_d_idx] * cy[cy_d_idx]);
                }
            }
        }
    }

    const int src_idx = sy * src_w + sx;
    grad_render[src_idx] = acc_f;
    grad_dx[src_idx] = acc_fx;
    grad_dy[src_idx] = acc_fy;
    grad_dxy[src_idx] = acc_fxy;
}

template __global__ void gradient_aware_upscale_backward_src_centric_kernel<float>(
    const int, const int, const int, const int,
    const float, const float, const float, const float,
    const float* __restrict__,
    float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__);

template __global__ void gradient_aware_upscale_backward_src_centric_kernel<float2>(
    const int, const int, const int, const int,
    const float, const float, const float, const float,
    const float2* __restrict__,
    float2* __restrict__, float2* __restrict__,
    float2* __restrict__, float2* __restrict__);

template __global__ void gradient_aware_upscale_backward_src_centric_kernel<float3>(
    const int, const int, const int, const int,
    const float, const float, const float, const float,
    const float3* __restrict__,
    float3* __restrict__, float3* __restrict__,
    float3* __restrict__, float3* __restrict__);

template __global__ void gradient_aware_upscale_backward_src_centric_kernel<float4>(
    const int, const int, const int, const int,
    const float, const float, const float, const float,
    const float4* __restrict__,
    float4* __restrict__, float4* __restrict__,
    float4* __restrict__, float4* __restrict__);

template __global__ void gradient_aware_upscale_backward_src_centric_kernel<float5>(
    const int, const int, const int, const int,
    const float, const float, const float, const float,
    const float5* __restrict__,
    float5* __restrict__, float5* __restrict__,
    float5* __restrict__, float5* __restrict__);
