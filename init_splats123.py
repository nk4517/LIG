import math

import torch
from torch.nn import functional as F

from simple_knn._C import distCUDA2

_MULTINOMIAL_LIMIT = 2 ** 24 - 1

def init_splats(
    num_points: int, H: int, W: int, device,
    mode: str = 'random',
    weights: torch.Tensor | None = None,
    image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize splat parameters.

    Returns:
        positions: [N, 2] (x, y)
        rgbs: [N, 3]
        scales: [N] (single value per splat for circular gaussians)
    """
    # === Positions ===
    if mode == 'grid':
        aspect = W / H
        ny = max(1, int(math.sqrt(num_points / aspect)))
        nx = max(1, int(num_points / ny))
        ys = torch.linspace(0.5, H - 0.5, ny, device=device)
        xs = torch.linspace(0.5, W - 0.5, nx, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        if positions.shape[0] < num_points:
            extra = num_points - positions.shape[0]
            extra_pos = torch.rand(extra, 2, device=device) * torch.tensor([W, H], device=device)
            positions = torch.cat([positions, extra_pos], dim=0)
        positions = positions[:num_points]
    elif weights is not None:
        weights = weights.squeeze()
        weights = weights - weights.min()
        weights = weights / (weights.sum() + 1e-8)
        h_orig, w_orig = weights.shape
        total_pixels = h_orig * w_orig

        if total_pixels > _MULTINOMIAL_LIMIT:
            scale = (_MULTINOMIAL_LIMIT / total_pixels) ** 0.5
            h_small, w_small = max(1, int(h_orig * scale)), max(1, int(w_orig * scale))
            weights_small = F.interpolate(
                weights.unsqueeze(0).unsqueeze(0), size=(h_small, w_small), mode='area'
            ).squeeze()
            weights_small = (weights_small - weights_small.min()) / (weights_small.sum() + 1e-8)
            flat_weights = weights_small.flatten().to(device)
            indices = torch.multinomial(flat_weights, num_points, replacement=True)
            h_idx, w_idx = indices // w_small, indices % w_small
            scale_h, scale_w = H / h_small, W / w_small
            h_init = (h_idx.float() + torch.rand_like(h_idx.float())) * scale_h
            w_init = (w_idx.float() + torch.rand_like(w_idx.float())) * scale_w
        else:
            if (h_orig, w_orig) != (H, W):
                weights = F.interpolate(
                    weights.unsqueeze(0).unsqueeze(0), size=(H, W), mode='area'
                ).squeeze()
                weights = (weights - weights.min()) / (weights.sum() + 1e-8)
            flat_weights = weights.flatten().to(device)
            indices = torch.multinomial(flat_weights, num_points, replacement=True)
            h_idx, w_idx = indices // W, indices % W
            w_init = w_idx.float() + torch.rand_like(w_idx.float())
            h_init = h_idx.float() + torch.rand_like(h_idx.float())
        positions = torch.stack([w_init, h_init], dim=1)
    else:
        w_init = torch.rand(num_points, device=device) * W
        h_init = torch.rand(num_points, device=device) * H
        positions = torch.stack([w_init, h_init], dim=1)

    # === Scales via KNN ===
    positions_3d = F.pad(positions, (0, 1), value=0.0)  # [N, 2] -> [N, 3] with z=0
    dist2 = distCUDA2(positions_3d.float().contiguous())

    # Resample splats too close to neighbors (< 0.2 px), max 4 iterations
    min_dist_sq = 0.25 ** 2
    for _ in range(4):
        too_close = dist2 < min_dist_sq
        n_bad = too_close.sum().item()
        if n_bad == 0:
            break
        if weights is not None:
            flat_w = weights.flatten() if weights.dim() == 2 else weights
            flat_w = flat_w / (flat_w.sum() + 1e-8)
            indices = torch.multinomial(flat_w, n_bad, replacement=True)
            h_idx, w_idx = indices // W, indices % W
            positions[too_close, 0] = w_idx.float() + torch.rand(n_bad, device=device)
            positions[too_close, 1] = h_idx.float() + torch.rand(n_bad, device=device)
        else:
            positions[too_close, 0] = torch.rand(n_bad, device=device) * W
            positions[too_close, 1] = torch.rand(n_bad, device=device) * H
        positions_3d = F.pad(positions, (0, 1), value=0.0)
        dist2 = distCUDA2(positions_3d.float().contiguous())

    # Remove remaining too-close splats
    too_close = dist2 < min_dist_sq
    if too_close.any():
        keep = ~too_close
        positions = positions[keep]
        dist2 = dist2[keep]

    scales = (torch.sqrt(dist2)/2).clamp(min=0.05)

    # === RGBs from image ===
    if image is not None:
        img = image.float().squeeze(0) if image.dim() == 4 else image  # [C, H, W]
        img_h, img_w = img.shape[1], img.shape[2]
        if (img_h, img_w) != (H, W):
            img = F.interpolate(img.unsqueeze(0), size=(H, W), mode='area').squeeze(0)
        x_coords = positions[:, 0].long().clamp(0, W - 1)
        y_coords = positions[:, 1].long().clamp(0, H - 1)
        rgbs = img[:, y_coords, x_coords].T  # [N, 3]
    else:
        rgbs = torch.full((positions.shape[0], 3), 0.5, dtype=torch.float32, device=device)

    return positions.contiguous(), rgbs.contiguous(), scales.contiguous()
