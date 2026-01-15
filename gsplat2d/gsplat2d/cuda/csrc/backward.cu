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
 * NOTE: NO WEIGHT NORMALIZATION (I_norm = I/W)
 * ============================================================================
 * 
 * Weight normalization (dividing accumulated color by sum of alphas) provides
 * no noticeable quality improvement over penalty-based color clamping.
 * 
 * Additional complexity if normalization were used:
 * 
 * - Backward: I_norm = I/W requires storing both I_norm and W, gradients
 *   become (c - I_norm)/W instead of c, adding divisions per gaussian per pixel.
 * 
 * - Upscale derivatives: ∂I_norm/∂x involves quotient rule terms
 *   (I_x*W - I*W_x)/W², requiring W_x, W_y, W_xy storage.
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
 *   T           = Π (1 - α_i)           -- transmittance
 * 
 * where for each gaussian i:
 *   d = [μ_x - px, μ_y - py]            -- center-to-pixel offset (classical 3DGS)
 *   σ = 0.5*(a*d_x² + c*d_y²) + b*d_x*d_y   -- quadratic form (conic = {a,b,c})
 *   α = exp(-σ)                         -- gaussian weight
 *   
 *   g_x = ∂σ/∂d_x = a*d_x + b*d_y       -- gradient of σ w.r.t. d
 *   g_y = ∂σ/∂d_y = b*d_x + c*d_y
 *   
 *   Since d = μ - p, ∂d/∂p = -1, so:
 *   α_x  = ∂α/∂p_x = α * g_x            -- (chain: -α * g_x * (-1))
 *   α_y  = ∂α/∂p_y = α * g_y
 *   α_xy = ∂²α/∂p_x∂p_y = α * (g_x * g_y - b)
 * 
 * Incoming gradients from loss (via spline upscaler):
 *   v_I   = ∂L/∂I
 *   v_dx  = ∂L/∂(∂I/∂x)
 *   v_dy  = ∂L/∂(∂I/∂y)
 *   v_dxy = ∂L/∂(∂²I/∂x∂y)
 *   v_T   = ∂L/∂T
 * 
 * ============================================================================
 * ALPHA-BLENDING (SORTED) vs UNSORTED WEIGHTED SUM
 * ============================================================================
 * 
 * Alpha-blending (arXiv:2503.14171 Eq. 50-52) uses sorted forward pass with
 * recurrent accumulation through A_i:
 *   ∂A_i/∂x = ∂A_{i-1}/∂x·(1-α_i) + (1-A_{i-1})·∂α_i/∂x
 *   ∂²A_i/∂x∂y = ∂²A_{i-1}/∂x∂y·(1-α_i) + (1-A_{i-1})·∂²α_i/∂x∂y 
 *                - ∂A_{i-1}/∂x·∂α_i/∂y - ∂A_{i-1}/∂y·∂α_i/∂x
 * 
 * This unsorted implementation has no first accumulative pass. Instead it uses
 * S-formulation via logarithmic derivative of T = Π(1-α_i):
 *   T = Π(1-α_i)
 *   dT/dx = -T·S_x,     where S_x = Σ(∂α_i/∂x / (1-α_i))
 *   d²T/dxdy = T·(S_x·S_y - S_xy - S_xy_cross)
 * 
 * Equivalence: ln(T) = Σln(1-α_i), so d(ln T)/dx = -S_x, hence dT/dx = -T·S_x.
 * 
 * S_xy_cross = Σ(∂α/∂x·∂α/∂y)/(1-α)² corresponds to the cross-terms
 * ∂A_{i-1}/∂x·∂α_i/∂y + ∂A_{i-1}/∂y·∂α_i/∂x in the recurrent formulation,
 * where ∂A_{i-1}/∂x = Σ_{j<i}(∂α_j/∂x · Π_{k≠j}(1-α_k)).
 * 
 * Alpha-blending backward iterates in reverse sorted order, maintaining
 * recurrent state (∂A_{i-1}/∂x, etc.) to compute cross-terms on the fly.
 * 
 * Weighted sum optimization: no sorting, no first accumulative pass. Backward
 * iterates in arbitrary order without recurrent state. Cross-terms cannot be
 * reconstructed from T derivatives alone (Σ(a_i·b_i) ≠ f(Σa_i, Σb_i)), so
 * forward precomputes per-pixel constants (S_x, S_y, S_xy via T derivatives,
 * plus S_xy_cross explicitly) and passes them to backward.
 * 
 * Mathematically equivalent; trade-off is storage vs computation ordering.
 * 
 * ============================================================================
 * GRADIENTS W.R.T. TRANSMITTANCE T
 * ============================================================================
 * 
 * T = Π_i (1 - α_i)
 * ∂T/∂α_k = -T / (1 - α_k)
 * 
 * Contribution to v_α from T:
 *   v_α += v_T * ∂T/∂α = -v_T * T / (1 - α)
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
 *   v_α   = dot(v_I, c) - v_T * T / (1 - α)
 *   v_αx  = dot(v_dx, c) - v_T_dx * T / (1 - α)
 *   v_αy  = dot(v_dy, c) - v_T_dy * T / (1 - α)
 *   v_αxy = dot(v_dxy, c) - v_T_dxy * T / (1 - α)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. POSITION μ
 * ============================================================================
 * 
 * Note: d = μ - p, so ∂d/∂μ = +1, ∂σ/∂μ = g, ∂α/∂μ = -α*g
 * 
 * ∂α/∂μ_x = -α * g_x
 * ∂α/∂μ_y = -α * g_y
 * 
 * ∂α_x/∂μ_x = α * (a - g_x²)
 * ∂α_x/∂μ_y = α * (b - g_x * g_y)
 * 
 * ∂α_y/∂μ_x = α * (b - g_x * g_y)
 * ∂α_y/∂μ_y = α * (c - g_y²)
 * 
 * ∂α_xy/∂μ_x = α * (-g_x² * g_y + 2*b*g_x + a*g_y)
 * ∂α_xy/∂μ_y = α * (-g_x * g_y² + 2*b*g_y + c*g_x)
 * 
 * Total:
 * ∂L/∂μ_x = v_α * (-α * g_x) 
 *         + v_αx * α * (a - g_x²)
 *         + v_αy * α * (b - g_x * g_y)
 *         + v_αxy * α * (-g_x² * g_y + 2*b*g_x + a*g_y)
 * 
 * ∂L/∂μ_y = v_α * (-α * g_y)
 *         + v_αx * α * (b - g_x * g_y)
 *         + v_αy * α * (c - g_y²)
 *         + v_αxy * α * (-g_x * g_y² + 2*b*g_y + c*g_x)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. CONIC (inverse covariance) {a, b, c}
 * ============================================================================
 * 
 * ∂σ/∂a = 0.5*d_x²,  ∂σ/∂b = d_x*d_y,  ∂σ/∂c = 0.5*d_y²
 * ∂g_x/∂a = d_x,     ∂g_x/∂b = d_y,    ∂g_x/∂c = 0
 * ∂g_y/∂a = 0,       ∂g_y/∂b = d_x,    ∂g_y/∂c = d_y
 * 
 * ∂α/∂a = -α * 0.5 * d_x²
 * ∂α/∂b = -α * d_x * d_y
 * ∂α/∂c = -α * 0.5 * d_y²
 * 
 * ∂α_x/∂a = α * (d_x - 0.5 * d_x² * g_x)
 * ∂α_x/∂b = α * (d_y - d_x * d_y * g_x)
 * ∂α_x/∂c = -α * 0.5 * d_y² * g_x
 * 
 * ∂α_y/∂a = -α * 0.5 * d_x² * g_y
 * ∂α_y/∂b = α * (d_x - d_x * d_y * g_y)
 * ∂α_y/∂c = α * (d_y - 0.5 * d_y² * g_y)
 * 
 * Let P = g_x * g_y - b (the term in α_xy = α * P)
 * ∂α_xy/∂a = α * (-0.5 * d_x² * P + d_x * g_y)
 * ∂α_xy/∂b = α * (-d_x * d_y * P + d_y * g_y + d_x * g_x - 1)
 * ∂α_xy/∂c = α * (-0.5 * d_y² * P + d_y * g_x)
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
    const float* __restrict__ out_T,
    const float* __restrict__ out_T_dx,
    const float* __restrict__ out_T_dy,
    const float* __restrict__ out_T_dxy,
    const float* __restrict__ out_S_xy_cross,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_T,
    const float3* __restrict__ v_output_dx,
    const float3* __restrict__ v_output_dy,
    const float3* __restrict__ v_output_dxy,
    const float* __restrict__ v_T_dx,
    const float* __restrict__ v_T_dy,
    const float* __restrict__ v_T_dxy,
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
    const float v_T_val = v_T ? v_T[pix_id] : 0.f;
    const float pix_T = out_T ? out_T[pix_id] : 1.f;
    const float v_T_pix_T = v_T_val * pix_T;

    float v_T_dx_val = 0.f, v_T_dy_val = 0.f, v_T_dxy_val = 0.f;
    float S_x = 0.f, S_y = 0.f, S_xy = 0.f;
    if constexpr (WITH_UPSCALE_GRADS) {
        if (v_T_dx) {
            v_T_dx_val = v_T_dx[pix_id];
            v_T_dy_val = v_T_dy[pix_id];
            v_T_dxy_val = v_T_dxy[pix_id];
            // Recover S from T derivatives: T_dx = -T * S_x => S_x = -T_dx / T
            if (pix_T > 1e-8f) {
                float inv_T = 1.f / pix_T;
                S_x = -out_T_dx[pix_id] * inv_T;
                S_y = -out_T_dy[pix_id] * inv_T;
                // T_dxy = T * (S_x * S_y - S_xy - S_xy_cross)
                // => S_xy = S_x * S_y - S_xy_cross - T_dxy / T
                S_xy = S_x * S_y - out_S_xy_cross[pix_id] - out_T_dxy[pix_id] * inv_T;
            }
        }
    }

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
            float2 d;
            float3 conic;
            float vis;
            if (valid) {
                conic = conic_batch[t];
                const float opacity = opacity_batch[t];
                float2 xy = xy_batch[t];
                // d = μ - p (classical 3DGS convention: center - pixel)
                d = {xy.x - px, xy.y - py};
                float sigma = 0.5f * (conic.x * d.x * d.x +
                                      conic.z * d.y * d.y) +
                              conic.y * d.x * d.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opacity * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
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
                float v_alpha = dot(c, v_out) - v_T_pix_T / (1.f - alpha);

                // ∂L/∂c = v_I * α
                v_rgb_local = v_out * alpha;

                // g_x, g_y = gradient of sigma w.r.t. d
                const float dx = d.x, dy = d.y;
                const float dx2 = dx * dx, dy2 = dy * dy, dxdy = dx * dy;
                const float c_a = conic.x, c_b = conic.y, c_c = conic.z;
                const float g_x = c_a * dx + c_b * dy;
                const float g_y = c_b * dx + c_c * dy;

                v_opacity_local = vis * v_alpha;

                // ∂α/∂σ = -α, ∂σ/∂conic uses d directly
                const float v_sigma = -alpha * v_alpha;
                v_conic_local = {0.5f * v_sigma * dx2,
                                 v_sigma * dxdy,
                                 0.5f * v_sigma * dy2};

                // ∂α/∂μ = -α * g (since d = μ - p, ∂d/∂μ = +1, ∂σ/∂μ = g, ∂α/∂μ = -α*g)
                v_xy_local = {-alpha * v_alpha * g_x, -alpha * v_alpha * g_y};

                if constexpr (WITH_UPSCALE_GRADS) {
                    // α_x = α * g_x, α_y = α * g_y (since ∂d/∂p = -1)
                    const float alpha_x = alpha * g_x;
                    const float alpha_y = alpha * g_y;
                    // α_xy = α * (g_x * g_y - b)
                    const float alpha_xy = alpha * (g_x * g_y - c_b);

                    v_rgb_local += v_dx * alpha_x + v_dy * alpha_y + v_dxy * alpha_xy;

                    const float one_minus_alpha = 1.f - alpha;
                    const float T_over_oma = pix_T / one_minus_alpha;

                    const float v_alpha_x = dot(v_dx, c) - v_T_dx_val * T_over_oma;
                    const float v_alpha_y = dot(v_dy, c) - v_T_dy_val * T_over_oma;
                    const float v_alpha_xy = dot(v_dxy, c) - v_T_dxy_val * T_over_oma;

                    // ∂T_dx/∂α = T/(1-α) * (S_x - α_x/(1-α)), same for y, xy
                    const float oma_inv = 1.f / one_minus_alpha;
                    v_alpha += v_T_dx_val * T_over_oma * (S_x - g_x * oma_inv);
                    v_alpha += v_T_dy_val * T_over_oma * (S_y - g_y * oma_inv);
                    v_alpha += v_T_dxy_val * T_over_oma * (S_xy - (g_x * g_y - c_b) * oma_inv);

                    // Position gradients from α derivatives (∂α_x/∂μ, ∂α_y/∂μ, ∂α_xy/∂μ)
                    // Note: no minus sign here unlike ∂α/∂μ = -α*g
                    // For ∂α_x/∂μ_x = ∂(α*g_x)/∂μ_x = (∂α/∂μ_x)*g_x + α*(∂g_x/∂μ_x) = (-α*g_x)*g_x + α*a = α*(a - g_x²)
                    // The minus from ∂α/∂μ is absorbed into the final formulas.
                    const float gx2 = g_x * g_x, gy2 = g_y * g_y, gxgy = g_x * g_y;
                    v_xy_local.x += alpha * (v_alpha_x * (c_a - gx2) + v_alpha_y * (c_b - gxgy) + v_alpha_xy * (-gx2 * g_y + 2.f * c_b * g_x + c_a * g_y));
                    v_xy_local.y += alpha * (v_alpha_x * (c_b - gxgy) + v_alpha_y * (c_c - gy2) + v_alpha_xy * (-g_x * gy2 + 2.f * c_b * g_y + c_c * g_x));

                    // Conic gradients from α derivatives
                    const float P = gxgy - c_b;
                    v_conic_local.x += alpha * (v_alpha_x * (dx - 0.5f * dx2 * g_x) + v_alpha_y * (-0.5f * dx2 * g_y) + v_alpha_xy * (-0.5f * dx2 * P + dx * g_y));
                    v_conic_local.y += alpha * (v_alpha_x * (dy - dxdy * g_x) + v_alpha_y * (dx - dxdy * g_y) + v_alpha_xy * (-dxdy * P + dy * g_y + dx * g_x - 1.f));
                    v_conic_local.z += alpha * (v_alpha_x * (-0.5f * dy2 * g_x) + v_alpha_y * (dy - 0.5f * dy2 * g_y) + v_alpha_xy * (-0.5f * dy2 * P + dy * g_x));
                }
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
    const float3* __restrict__, const float3* __restrict__, const float* __restrict__, const int* __restrict__,
    const float* __restrict__, const float* __restrict__, const float* __restrict__, const float* __restrict__, const float* __restrict__,
    const float3* __restrict__, const float* __restrict__,
    const float3* __restrict__, const float3* __restrict__, const float3* __restrict__,
    const float* __restrict__, const float* __restrict__, const float* __restrict__,
    float2* __restrict__, float2* __restrict__, float3* __restrict__, float3* __restrict__, float* __restrict__);

template __global__ void rasterize_backward_kernel_unified<true>(
    const dim3, const dim3, const int32_t* __restrict__, const int2* __restrict__, const float2* __restrict__,
    const float3* __restrict__, const float3* __restrict__, const float* __restrict__, const int* __restrict__,
    const float* __restrict__, const float* __restrict__, const float* __restrict__, const float* __restrict__, const float* __restrict__,
    const float3* __restrict__, const float* __restrict__,
    const float3* __restrict__, const float3* __restrict__, const float3* __restrict__,
    const float* __restrict__, const float* __restrict__, const float* __restrict__,
    float2* __restrict__, float2* __restrict__, float3* __restrict__, float3* __restrict__, float* __restrict__);
