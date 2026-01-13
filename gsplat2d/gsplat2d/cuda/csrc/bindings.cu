#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "upscale.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d,
    const float* __restrict__ opacities, float* __restrict__ conics, float2* __restrict__ extents
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float2 extent;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    float opacity = opacities ? opacities[row] : 1.0f;
    compute_cov2d_bounds(cov2d, opacity, conic, extent);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    extents[row] = extent;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output extents
compute_cov2d_bounds_tensor(
    const int num_pts,
    torch::Tensor &covs2d,
    const c10::optional<torch::Tensor> &opacities) {
    DEVICE_GUARD(covs2d);
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor extents =
        torch::zeros({num_pts, 2}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
        conics.contiguous().data_ptr<float>(),
        (float2*)extents.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, extents);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &cov2d,
    torch::Tensor &means2d,
    const c10::optional<torch::Tensor> &opacities,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width
) {
    DEVICE_GUARD(cov2d);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, cov2d.options().dtype(torch::kFloat32));

    torch::Tensor extents_d =
        torch::zeros({num_points, 2}, cov2d.options().dtype(torch::kFloat32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, cov2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, cov2d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)cov2d.contiguous().data_ptr<float>(),
        (float2 *)means2d.contiguous().data_ptr<float>(),
        opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
        tile_bounds_dim3,
        block_width,
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        (float2 *)extents_d.contiguous().data_ptr<float>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
            xys_d, extents_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &extent,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
){
    DEVICE_GUARD(conics);
    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, conics.options().dtype(torch::kFloat32));

    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, conics.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)extent.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_mean2d);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_cholesky_tensor(
    const int num_points,
    torch::Tensor &cholesky,
    torch::Tensor &means2d,
    const c10::optional<torch::Tensor> &opacities,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width
) {
    DEVICE_GUARD(cholesky);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, cholesky.options().dtype(torch::kFloat32));

    torch::Tensor extent_d =
        torch::zeros({num_points, 2}, cholesky.options().dtype(torch::kFloat32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, cholesky.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, cholesky.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel_cholesky<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)cholesky.contiguous().data_ptr<float>(),
        (float2 *)means2d.contiguous().data_ptr<float>(),
        opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
        tile_bounds_dim3,
        block_width,
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        (float2 *)extent_d.contiguous().data_ptr<float>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, extent_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_cholesky_tensor(
    const int num_points,
    torch::Tensor &extent,
    torch::Tensor &cholesky,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
){
    DEVICE_GUARD(conics);

    torch::Tensor v_cholesky =
        torch::zeros({num_points, 3}, conics.options().dtype(torch::kFloat32));

    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, conics.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel_cholesky<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)extent.contiguous().data_ptr<float>(),
        (float3 *)cholesky.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_cholesky.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cholesky, v_mean2d);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &extent,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(extent);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        (float2 *)extent.contiguous().data_ptr<float>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        block_width,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted, 
    const std::tuple<int, int, int> tile_bounds
) {
    DEVICE_GUARD(isect_ids_sorted);
    CHECK_INPUT(isect_ids_sorted);
    int num_tiles = std::get<0>(tile_bounds) * std::get<1>(tile_bounds);
    torch::Tensor tile_bins = torch::zeros(
        {num_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const c10::optional<torch::Tensor> &opacities,
    unsigned extras
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    if (opacities.has_value()) {
        CHECK_INPUT(opacities.value());
    }

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );

    bool compute_T = extras & RASTERIZE_EXTRAS_T;
    bool compute_upscale_grads = extras & RASTERIZE_EXTRAS_UPSCALE_GRADS;
    bool compute_T_upscale = compute_T && compute_upscale_grads;

    torch::Tensor out_T;
    if (compute_T) {
        out_T = torch::zeros(
            {img_height, img_width, 1}, xys.options().dtype(torch::kFloat32)
        );
    }
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    torch::Tensor out_img_dx, out_img_dy, out_img_dxy;
    torch::Tensor out_T_dx, out_T_dy, out_T_dxy, out_S_xy_cross;

    if (compute_upscale_grads) {
        out_img_dx = torch::zeros(
            {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
        );
        out_img_dy = torch::zeros(
            {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
        );
        out_img_dxy = torch::zeros(
            {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
        );
        if (compute_T_upscale) {
            out_T_dx = torch::zeros(
                {img_height, img_width, 1}, xys.options().dtype(torch::kFloat32)
            );
            out_T_dy = torch::zeros(
                {img_height, img_width, 1}, xys.options().dtype(torch::kFloat32)
            );
            out_T_dxy = torch::zeros(
                {img_height, img_width, 1}, xys.options().dtype(torch::kFloat32)
            );
            out_S_xy_cross = torch::zeros(
                {img_height, img_width, 1}, xys.options().dtype(torch::kFloat32)
            );
        }

        rasterize_forward_unified<true><<<tile_bounds_dim3, block_dim3>>>(
	        tile_bounds_dim3,
	        img_size_dim3,
	        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
	        (int2 *)tile_bins.contiguous().data_ptr<int>(),
	        (float2 *)xys.contiguous().data_ptr<float>(),
	        (float3 *)conics.contiguous().data_ptr<float>(),
	        (float3 *)colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
	        final_idx.contiguous().data_ptr<int>(),
	        (float3 *)out_img.contiguous().data_ptr<float>(),
            compute_T ? out_T.contiguous().data_ptr<float>() : nullptr,
	        (float3 *)out_img_dx.contiguous().data_ptr<float>(),
	        (float3 *)out_img_dy.contiguous().data_ptr<float>(),
	        (float3 *)out_img_dxy.contiguous().data_ptr<float>(),
	        compute_T_upscale ? out_T_dx.contiguous().data_ptr<float>() : nullptr,
	        compute_T_upscale ? out_T_dy.contiguous().data_ptr<float>() : nullptr,
	        compute_T_upscale ? out_T_dxy.contiguous().data_ptr<float>() : nullptr,
	        compute_T_upscale ? out_S_xy_cross.contiguous().data_ptr<float>() : nullptr
        );
    } else {
        rasterize_forward_unified<false><<<tile_bounds_dim3, block_dim3>>>(
            tile_bounds_dim3,
            img_size_dim3,
            gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
            (int2 *)tile_bins.contiguous().data_ptr<int>(),
            (float2 *)xys.contiguous().data_ptr<float>(),
            (float3 *)conics.contiguous().data_ptr<float>(),
            (float3 *)colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
            final_idx.contiguous().data_ptr<int>(),
            (float3 *)out_img.contiguous().data_ptr<float>(),
            compute_T ? out_T.contiguous().data_ptr<float>() : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        );
    }

    return std::make_tuple(out_img, out_T, out_img_dx, out_img_dy, out_img_dxy, out_T_dx, out_T_dy, out_T_dxy, out_S_xy_cross, final_idx);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dxy_abs
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacities
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const c10::optional<torch::Tensor> &opacities,
        const torch::Tensor &final_idx,
        const c10::optional<torch::Tensor> &out_T,
        const c10::optional<torch::Tensor> &out_T_dx,
        const c10::optional<torch::Tensor> &out_T_dy,
        const c10::optional<torch::Tensor> &out_T_dxy,
        const c10::optional<torch::Tensor> &out_S_xy_cross,
        const torch::Tensor &v_output,
        const c10::optional<torch::Tensor> &v_T,
        const c10::optional<torch::Tensor> &v_output_dx,
        const c10::optional<torch::Tensor> &v_output_dy,
        const c10::optional<torch::Tensor> &v_output_dxy,
        const c10::optional<torch::Tensor> &v_T_dx,
        const c10::optional<torch::Tensor> &v_T_dy,
        const c10::optional<torch::Tensor> &v_T_dxy
    ) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);
    if (opacities.has_value()) {
        CHECK_INPUT(opacities.value());
    }

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_xy_abs = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = opacities.has_value() 
        ? torch::zeros({num_points}, xys.options())
        : torch::Tensor();

    if (v_output_dx.has_value()) {
        rasterize_backward_kernel_unified<true><<<tile_bounds, block>>>(
            tile_bounds,
            img_size,
            gaussians_ids_sorted.contiguous().data_ptr<int>(),
            (int2 *)tile_bins.contiguous().data_ptr<int>(),
            (float2 *)xys.contiguous().data_ptr<float>(),
            (float3 *)conics.contiguous().data_ptr<float>(),
            (float3 *)colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
            final_idx.contiguous().data_ptr<int>(),
            out_T.has_value() ? out_T.value().contiguous().data_ptr<float>() : nullptr,
            out_T_dx.has_value() ? out_T_dx.value().contiguous().data_ptr<float>() : nullptr,
            out_T_dy.has_value() ? out_T_dy.value().contiguous().data_ptr<float>() : nullptr,
            out_T_dxy.has_value() ? out_T_dxy.value().contiguous().data_ptr<float>() : nullptr,
            out_S_xy_cross.has_value() ? out_S_xy_cross.value().contiguous().data_ptr<float>() : nullptr,
            (float3 *)v_output.contiguous().data_ptr<float>(),
            v_T.has_value() ? v_T.value().contiguous().data_ptr<float>() : nullptr,
            (float3 *)v_output_dx.value().contiguous().data_ptr<float>(),
            (float3 *)v_output_dy.value().contiguous().data_ptr<float>(),
            (float3 *)v_output_dxy.value().contiguous().data_ptr<float>(),
            v_T_dx.has_value() ? v_T_dx.value().contiguous().data_ptr<float>() : nullptr,
            v_T_dy.has_value() ? v_T_dy.value().contiguous().data_ptr<float>() : nullptr,
            v_T_dxy.has_value() ? v_T_dxy.value().contiguous().data_ptr<float>() : nullptr,
            (float2 *)v_xy.contiguous().data_ptr<float>(),
            (float2 *)v_xy_abs.contiguous().data_ptr<float>(),
            (float3 *)v_conic.contiguous().data_ptr<float>(),
            (float3 *)v_colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? v_opacity.contiguous().data_ptr<float>() : nullptr
        );
    } else {
        rasterize_backward_kernel_unified<false><<<tile_bounds, block>>>(
            tile_bounds,
            img_size,
            gaussians_ids_sorted.contiguous().data_ptr<int>(),
            (int2 *)tile_bins.contiguous().data_ptr<int>(),
            (float2 *)xys.contiguous().data_ptr<float>(),
            (float3 *)conics.contiguous().data_ptr<float>(),
            (float3 *)colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? opacities.value().contiguous().data_ptr<float>() : nullptr,
            final_idx.contiguous().data_ptr<int>(),
            out_T.has_value() ? out_T.value().contiguous().data_ptr<float>() : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            (float3 *)v_output.contiguous().data_ptr<float>(),
            v_T.has_value() ? v_T.value().contiguous().data_ptr<float>() : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            (float2 *)v_xy.contiguous().data_ptr<float>(),
            (float2 *)v_xy_abs.contiguous().data_ptr<float>(),
            (float3 *)v_conic.contiguous().data_ptr<float>(),
            (float3 *)v_colors.contiguous().data_ptr<float>(),
            opacities.has_value() ? v_opacity.contiguous().data_ptr<float>() : nullptr
        );
    }

    return std::make_tuple(v_xy, v_xy_abs, v_conic, v_colors, v_opacity);
}

torch::Tensor gradient_aware_upscale_forward_tensor(
    const torch::Tensor &render,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
) {
    DEVICE_GUARD(render);
    CHECK_INPUT(render);
    CHECK_INPUT(dx);
    CHECK_INPUT(dy);
    CHECK_INPUT(dxy);

    const int src_h = render.size(0);
    const int src_w = render.size(1);
    const int channels = render.size(2);

    const float roi_x1 = std::get<0>(roi);
    const float roi_y1 = std::get<1>(roi);
    const float roi_x2 = std::get<2>(roi);
    const float roi_y2 = std::get<3>(roi);

    torch::Tensor output = torch::zeros(
        {dst_h, dst_w, channels},
        render.options().dtype(torch::kFloat32)
    );

    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

#define UPSCALE_FORWARD_DISPATCH(T) \
    gradient_aware_upscale_kernel<T><<<grid, block>>>( \
        dst_h, dst_w, src_h, src_w, \
        roi_x1, roi_y1, roi_x2, roi_y2, \
        (T*)render.contiguous().data_ptr<float>(), \
        (T*)dx.contiguous().data_ptr<float>(), \
        (T*)dy.contiguous().data_ptr<float>(), \
        (T*)dxy.contiguous().data_ptr<float>(), \
        (T*)output.data_ptr<float>());

    switch (channels) {
    case 1: UPSCALE_FORWARD_DISPATCH(float); break;
    case 2: UPSCALE_FORWARD_DISPATCH(float2); break;
    case 3: UPSCALE_FORWARD_DISPATCH(float3); break;
    case 4: UPSCALE_FORWARD_DISPATCH(float4); break;
    case 5: UPSCALE_FORWARD_DISPATCH(float5); break;
    default: AT_ERROR("gradient_aware_upscale: unsupported channels ", channels);
    }
#undef UPSCALE_FORWARD_DISPATCH

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward_tensor(
    const torch::Tensor &grad_output,
    const torch::Tensor &render,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
) {
    DEVICE_GUARD(grad_output);
    CHECK_INPUT(grad_output);

    const int src_h = render.size(0);
    const int src_w = render.size(1);
    const int channels = render.size(2);

    const float roi_x1 = std::get<0>(roi);
    const float roi_y1 = std::get<1>(roi);
    const float roi_x2 = std::get<2>(roi);
    const float roi_y2 = std::get<3>(roi);

    auto grad_render = torch::zeros_like(render);
    auto grad_dx = torch::zeros_like(dx);
    auto grad_dy = torch::zeros_like(dy);
    auto grad_dxy = torch::zeros_like(dxy);

    dim3 block(16, 16);
    dim3 grid_src(
        (src_w + block.x - 1) / block.x,
        (src_h + block.y - 1) / block.y
    );
    dim3 grid_dst(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

#define UPSCALE_BACKWARD_DISPATCH(T) \
    if constexpr (USE_SRC_CENTRIC_UPSCALE_BACKWARD) { \
        gradient_aware_upscale_backward_src_centric_kernel<T><<<grid_src, block>>>( \
            dst_h, dst_w, src_h, src_w, \
            roi_x1, roi_y1, roi_x2, roi_y2, \
            (T*)grad_output.contiguous().data_ptr<float>(), \
            (T*)grad_render.data_ptr<float>(), \
            (T*)grad_dx.data_ptr<float>(), \
            (T*)grad_dy.data_ptr<float>(), \
            (T*)grad_dxy.data_ptr<float>()); \
    } else { \
        gradient_aware_upscale_backward_kernel<T><<<grid_dst, block>>>( \
            dst_h, dst_w, src_h, src_w, \
            roi_x1, roi_y1, roi_x2, roi_y2, \
            (T*)grad_output.contiguous().data_ptr<float>(), \
            (T*)grad_render.data_ptr<float>(), \
            (T*)grad_dx.data_ptr<float>(), \
            (T*)grad_dy.data_ptr<float>(), \
            (T*)grad_dxy.data_ptr<float>()); \
    }

    switch (channels) {
    case 1: UPSCALE_BACKWARD_DISPATCH(float); break;
    case 2: UPSCALE_BACKWARD_DISPATCH(float2); break;
    case 3: UPSCALE_BACKWARD_DISPATCH(float3); break;
    case 4: UPSCALE_BACKWARD_DISPATCH(float4); break;
    case 5: UPSCALE_BACKWARD_DISPATCH(float5); break;
    default: AT_ERROR("gradient_aware_upscale_backward: unsupported channels ", channels);
    }
#undef UPSCALE_BACKWARD_DISPATCH

    return std::make_tuple(grad_render, grad_dx, grad_dy, grad_dxy);
}