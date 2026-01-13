#include "backward.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_render_wsum,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float2 xy_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float opacity_batch[MAX_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_render_w = v_render_wsum[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            opacity_batch[tr] = opacities ? opacities[g_id] : 1.0f;
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                const float opacity = opacity_batch[t];
                float2 xy = xy_batch[t];
                delta = {xy.x - px, xy.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.99f, opacity * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){

                const float fac = alpha;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];

                v_alpha += rgb.x * v_out.x;
                v_alpha += rgb.y * v_out.y;
                v_alpha += rgb.z * v_out.z; 
                
                // Gradient from weighted sum of alphas: ∂(Σ alpha_i)/∂alpha_i = 1
                v_alpha += v_render_w;

                // v_opacity = d(alpha)/d(opacity) * v_alpha = vis * v_alpha
                v_opacity_local = vis * v_alpha;

                const float v_sigma = -alpha * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                 v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y};

                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum2(v_xy_abs_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_xy_abs_ptr = (float*)(v_xy_abs);
                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
                
                if (v_opacity) {
                    atomicAdd(v_opacity + g, v_opacity_local);
                }
            }
        }
    }
}

__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float2* __restrict__ extents,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || (extents[idx].x <= 0.f || extents[idx].y <= 0.f)) {
        return;
    }

    v_mean2d[idx].x = v_xy[idx].x;
    v_mean2d[idx].y = v_xy[idx].y;

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
}

__global__ void project_gaussians_backward_kernel_cholesky(
    const int num_points,
    const float2* __restrict__ extents,
    const float3* __restrict__ cholesky,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cholesky,
    float2* __restrict__ v_mean2d
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points || (extents[idx].x <= 0.f || extents[idx].y <= 0.f)) {
        return;
    }

    v_mean2d[idx].x = v_xy[idx].x;
    v_mean2d[idx].y = v_xy[idx].y;

    // get v_cov2d first
    float3 v_cov2d;
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d);

    // chain rule: v_cholesky from v_cov2d
    // cov2d = [l11^2, l11*l21, l21^2 + l22^2]
    // v_l11 = 2*l11*v_a + l21*v_b
    // v_l21 = l11*v_b + 2*l21*v_c
    // v_l22 = 2*l22*v_c
    float3 chol = cholesky[idx];
    float l11 = chol.x;
    float l21 = chol.y;
    float l22 = chol.z;
    float v_a = v_cov2d.x;
    float v_b = v_cov2d.y;
    float v_c = v_cov2d.z;

    float v_l11 = 2.f * l11 * v_a + l21 * v_b;
    float v_l21 = l11 * v_b + 2.f * l21 * v_c;
    float v_l22 = 2.f * l22 * v_c;

    bool all_finite = isfinite(v_l11) & isfinite(v_l21) & isfinite(v_l22);
    v_cholesky[idx] = all_finite ? float3{v_l11, v_l21, v_l22} : float3{0.f, 0.f, 0.f};
}


/**
 * Backward pass for weighted sum rasterization with gradient-aware upscaling
 * 
 * ============================================================================
 * NOTE: UNSORTED WEIGHTED SUM vs SORTED ALPHA-COMPOSITING
 * ============================================================================
 * 
 * This implementation uses unsorted weighted sum (no depth sorting).
 * The original 3DGS uses sorted alpha-compositing with final_idx to track
 * early termination when transmittance T < threshold. final_idx optimization
 * is incompatible with unsorted rendering: without depth ordering, there is
 * no "tail" of low-visibility gaussians that can be safely skipped.
 * 
 * ============================================================================
 * MATHEMATICAL DERIVATION
 * ============================================================================
 * 
 * Forward outputs (all per-pixel):
 *   I(x,y)      = Σ c_i * α_i           -- color image
 *   ∂I/∂x       = Σ c_i * α_x,i         -- image x-gradient  
 *   ∂I/∂y       = Σ c_i * α_y,i         -- image y-gradient
 *   ∂²I/∂x∂y    = Σ c_i * α_xy,i        -- mixed partial
 *   W           = Σ α_i                 -- weight sum
 * 
 * where for each gaussian i:
 *   d = [px - μ_x, py - μ_y]            -- pixel-to-center offset
 *   σ = 0.5*(a*d_x² + c*d_y²) + b*d_x*d_y   -- quadratic form (conic = {a,b,c})
 *   α = exp(-σ)                         -- gaussian weight
 *   
 *   σ_x = ∂σ/∂x = a*d_x + b*d_y
 *   σ_y = ∂σ/∂y = b*d_x + c*d_y  
 *   σ_xy = ∂²σ/∂x∂y = b
 *   
 *   α_x  = ∂α/∂x = -α * σ_x
 *   α_y  = ∂α/∂y = -α * σ_y
 *   α_xy = ∂²α/∂x∂y = α * (σ_x * σ_y - σ_xy)
 * 
 * Incoming gradients from loss (via spline upscaler):
 *   v_I   = ∂L/∂I
 *   v_dx  = ∂L/∂(∂I/∂x)
 *   v_dy  = ∂L/∂(∂I/∂y)
 *   v_dxy = ∂L/∂(∂²I/∂x∂y)
 *   v_W   = ∂L/∂W
 * 
 * ============================================================================
 * GRADIENTS W.R.T. COLOR c_k
 * ============================================================================
 * 
 * ∂L/∂c_k = v_I * α_k + v_dx * α_x,k + v_dy * α_y,k + v_dxy * α_xy,k
 * 
 * ============================================================================
 * GRADIENTS W.R.T. GAUSSIAN PARAMETERS (via chain rule through α)
 * ============================================================================
 * 
 * Intermediate "virtual gradients" on α and its derivatives:
 *   v_α   = dot(v_I, c) + v_W
 *   v_αx  = dot(v_dx, c)
 *   v_αy  = dot(v_dy, c)
 *   v_αxy = dot(v_dxy, c)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. POSITION μ
 * ============================================================================
 * 
 * Note: d = pixel - μ, so ∂d/∂μ = -1
 * 
 * ∂α/∂μ_x = α * σ_x           (since ∂σ/∂μ_x = -σ_x)
 * ∂α/∂μ_y = α * σ_y
 * 
 * ∂α_x/∂μ_x = α * (a - σ_x²)
 * ∂α_x/∂μ_y = α * (b - σ_x * σ_y)
 * 
 * ∂α_y/∂μ_x = α * (b - σ_x * σ_y)
 * ∂α_y/∂μ_y = α * (c - σ_y²)
 * 
 * ∂α_xy/∂μ_x = α * (σ_x² * σ_y - 2*b*σ_x - a*σ_y)
 * ∂α_xy/∂μ_y = α * (σ_x * σ_y² - 2*b*σ_y - c*σ_x)
 * 
 * Total:
 * ∂L/∂μ_x = v_α * α * σ_x 
 *         + v_αx * α * (a - σ_x²)
 *         + v_αy * α * (b - σ_x * σ_y)
 *         + v_αxy * α * (σ_x² * σ_y - 2*b*σ_x - a*σ_y)
 * 
 * ∂L/∂μ_y = v_α * α * σ_y
 *         + v_αx * α * (b - σ_x * σ_y)
 *         + v_αy * α * (c - σ_y²)
 *         + v_αxy * α * (σ_x * σ_y² - 2*b*σ_y - c*σ_x)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. CONIC (inverse covariance) {a, b, c}
 * ============================================================================
 * 
 * ∂σ/∂a = 0.5*d_x²,  ∂σ/∂b = d_x*d_y,  ∂σ/∂c = 0.5*d_y²
 * ∂σ_x/∂a = d_x,     ∂σ_x/∂b = d_y,    ∂σ_x/∂c = 0
 * ∂σ_y/∂a = 0,       ∂σ_y/∂b = d_x,    ∂σ_y/∂c = d_y
 * ∂σ_xy/∂b = 1       (others = 0)
 * 
 * ∂α/∂a = -α * 0.5 * d_x²
 * ∂α/∂b = -α * d_x * d_y
 * ∂α/∂c = -α * 0.5 * d_y²
 * 
 * ∂α_x/∂a = α * (0.5 * d_x² * σ_x - d_x)
 * ∂α_x/∂b = α * (d_x * d_y * σ_x - d_y)
 * ∂α_x/∂c = α * (0.5 * d_y² * σ_x)
 * 
 * ∂α_y/∂a = α * (0.5 * d_x² * σ_y)
 * ∂α_y/∂b = α * (d_x * d_y * σ_y - d_x)
 * ∂α_y/∂c = α * (0.5 * d_y² * σ_y - d_y)
 * 
 * Let P = σ_x * σ_y - b (the term in α_xy = α * P)
 * ∂α_xy/∂a = α * (-0.5 * d_x² * P + d_x * σ_y)
 * ∂α_xy/∂b = α * (-d_x * d_y * P + d_y * σ_y + d_x * σ_x - 1)
 * ∂α_xy/∂c = α * (-0.5 * d_y² * P + d_y * σ_x)
 * 
 * Total for each conic component combines all four gradient paths.
 */


template<bool WITH_UPSCALE_GRADS>
__global__ void rasterize_backward_kernel_unified(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_render_wsum,
    const float3* __restrict__ v_output_dx,
    const float3* __restrict__ v_output_dy,
    const float3* __restrict__ v_output_dxy,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    const bool inside = (i < img_size.y && j < img_size.x);
    const int bin_final = inside ? final_index[pix_id] : 0;

    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float2 xy_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float opacity_batch[MAX_BLOCK_SIZE];

    const float3 v_out = v_output[pix_id];
    const float v_render_w = v_render_wsum[pix_id];

    float3 v_dx, v_dy, v_dxy;
    if constexpr (WITH_UPSCALE_GRADS) {
        v_dx = v_output_dx[pix_id];
        v_dy = v_output_dy[pix_id];
        v_dxy = v_output_dxy[pix_id];
    }

    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (int b = 0; b < num_batches; ++b) {
        block.sync();

        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            opacity_batch[tr] = opacities ? opacities[g_id] : 1.0f;
        }
        block.sync();

        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float2 delta;
            float3 conic;
            float vis;
            float sigma_x = 0.f, sigma_y = 0.f;
            if (valid) {
                conic = conic_batch[t];
                const float opacity = opacity_batch[t];
                float2 xy = xy_batch[t];
                delta = {px - xy.x, py - xy.y};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opacity * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                } else if constexpr (WITH_UPSCALE_GRADS) {
                    sigma_x = conic.x * delta.x + conic.y * delta.y;
                    sigma_y = conic.y * delta.x + conic.z * delta.y;
                }
            }
            if (!warp.any(valid)) {
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            if (valid) {
                const float3 c = rgbs_batch[t];
                float v_alpha = dot(c, v_out) + v_render_w;

                if constexpr (WITH_UPSCALE_GRADS) {
                    // ================================================================
                    // GRADIENT W.R.T. COLOR
                    // ∂L/∂c = v_I * α + v_dx * α_x + v_dy * α_y + v_dxy * α_xy
                    // ================================================================
                    const float c_a = conic.x, c_b = conic.y, c_c = conic.z;
                    const float sx = sigma_x, sy = sigma_y;
                    const float alpha_x = -alpha * sx;
                    const float alpha_y = -alpha * sy;
                    const float alpha_xy = alpha * (sx * sy - c_b);

                    v_rgb_local = v_out * alpha + v_dx * alpha_x + v_dy * alpha_y + v_dxy * alpha_xy;

                    // ================================================================
                    // VIRTUAL GRADIENTS (intermediate gradients on α and its derivatives)
                    // ================================================================
                    const float v_alpha_x = dot(v_dx, c);
                    const float v_alpha_y = dot(v_dy, c);
                    const float v_alpha_xy = dot(v_dxy, c);

                    // ================================================================
                    // GRADIENT W.R.T. POSITION μ
                    // Note: d = pixel - μ, so gradients w.r.t. μ have opposite sign
                    // ================================================================
                    const float sx2 = sx * sx, sy2 = sy * sy, sxsy = sx * sy;

                    // ================================================================
                    // GRADIENT W.R.T. CONIC {a, b, c}
                    // ================================================================
                    const float dx = delta.x, dy = delta.y;
                    const float dx2 = dx * dx, dy2 = dy * dy, dxdy = dx * dy;
                    const float P = sxsy - c_b;

                    v_xy_local.x = alpha * (v_alpha * sx + v_alpha_x * (c_a - sx2) + v_alpha_y * (c_b - sxsy) + v_alpha_xy * (sx2 * sy - 2.f * c_b * sx - c_a * sy));
                    v_xy_local.y = alpha * (v_alpha * sy + v_alpha_x * (c_b - sxsy) + v_alpha_y * (c_c - sy2) + v_alpha_xy * (sx * sy2 - 2.f * c_b * sy - c_c * sx));

                    v_conic_local.x = alpha * (v_alpha * (-0.5f * dx2) + v_alpha_x * (0.5f * dx2 * sx - dx) + v_alpha_y * (0.5f * dx2 * sy) + v_alpha_xy * (-0.5f * dx2 * P + dx * sy));
                    v_conic_local.y = alpha * (v_alpha * (-dxdy) + v_alpha_x * (dxdy * sx - dy) + v_alpha_y * (dxdy * sy - dx) + v_alpha_xy * (-dxdy * P + dy * sy + dx * sx - 1.f));
                    v_conic_local.z = alpha * (v_alpha * (-0.5f * dy2) + v_alpha_x * (0.5f * dy2 * sx) + v_alpha_y * (0.5f * dy2 * sy - dy) + v_alpha_xy * (-0.5f * dy2 * P + dy * sx));
                } else {
                    v_rgb_local = v_out * alpha;
                    // ∂α/∂σ = -α (since α = opacity * exp(-σ))
                    const float v_sigma = -alpha * v_alpha;
                    // ∂σ/∂conic: σ = 0.5*(a*dx² + c*dy²) + b*dx*dy
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                     v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    // ∂σ/∂xy: chain rule through delta
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                }
                // ∂α/∂opacity = vis (gaussian weight before opacity scaling)
                v_opacity_local = vis * v_alpha;
                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum2(v_xy_abs_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);

                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);

                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_xy_abs_ptr = (float*)(v_xy_abs);
                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);

                if (v_opacity) {
                    atomicAdd(v_opacity + g, v_opacity_local);
                }
            }
        }
    }
}

template __global__ void rasterize_backward_kernel_unified<false>(
    const dim3, const dim3, const int32_t* __restrict__, const int2* __restrict__, const float2* __restrict__,
    const float3* __restrict__, const float3* __restrict__, const float* __restrict__, const int* __restrict__, const float3* __restrict__,
    const float* __restrict__, const float3* __restrict__, const float3* __restrict__, const float3* __restrict__,
    float2* __restrict__, float2* __restrict__, float3* __restrict__, float3* __restrict__, float* __restrict__);

template __global__ void rasterize_backward_kernel_unified<true>(
    const dim3, const dim3, const int32_t* __restrict__, const int2* __restrict__, const float2* __restrict__,
    const float3* __restrict__, const float3* __restrict__, const float* __restrict__, const int* __restrict__, const float3* __restrict__,
    const float* __restrict__, const float3* __restrict__, const float3* __restrict__, const float3* __restrict__,
    float2* __restrict__, float2* __restrict__, float3* __restrict__, float3* __restrict__, float* __restrict__);
