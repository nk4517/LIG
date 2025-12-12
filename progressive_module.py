import math

from adan import Adan
from utils import *
import torch
import torch.nn as nn

from LIG.utils import loss_fn, _elongation_penalty
from gsplat2d import project_gaussians_cholesky, rasterize_gaussians


class ProgressiveGaussian2D(nn.Module):
    """Gaussian2D with ability to add points progressively"""

    def __init__(self, num_points: int, H: int, W: int, device, lr: float,
                 init_weights: torch.Tensor | None = None, opt_type: str = "adan"):
        super().__init__()
        self.device = device
        self.H = H
        self.W = W
        self.B_SIZE = 16
        self.lr = lr
        self.dog_weights: torch.Tensor | None = None
        self.opt_type = opt_type

        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5], device=device).view(1, 3))

        # Initialize parameters
        self.means = nn.Parameter(self._sample_positions(num_points, H, W, init_weights))
        self._cholesky = nn.Parameter(torch.rand(num_points, 3, device=device))
        self._rgb_logits = nn.Parameter(torch.zeros(num_points, 3, device=device))

        self._init_optimizer()

    _MULTINOMIAL_LIMIT = 2**24 - 1

    def _sample_positions(self, num_points: int, H: int, W: int,
                          weights: torch.Tensor | None) -> torch.Tensor:
        if weights is not None:
            weights = weights.squeeze()
            weights = weights - weights.min()
            weights = weights / (weights.sum() + 1e-8)

            h_orig, w_orig = weights.shape
            total_pixels = h_orig * w_orig

            if total_pixels > self._MULTINOMIAL_LIMIT:
                scale = (self._MULTINOMIAL_LIMIT / total_pixels) ** 0.5
                h_small = max(1, int(h_orig * scale))
                w_small = max(1, int(w_orig * scale))
                weights_small = F.interpolate(
                    weights.unsqueeze(0).unsqueeze(0),
                    size=(h_small, w_small), mode='area'
                ).squeeze()
                weights_small = weights_small - weights_small.min()
                weights_small = weights_small / (weights_small.sum() + 1e-8)
                flat_weights = weights_small.flatten().to(self.device)
                indices = torch.multinomial(flat_weights, num_points, replacement=True)
                h_idx = indices // w_small
                w_idx = indices % w_small
                scale_h = H / h_small
                scale_w = W / w_small
                h_init = ((h_idx.float() + torch.rand_like(h_idx.float())) * scale_h).unsqueeze(1)
                w_init = ((w_idx.float() + torch.rand_like(w_idx.float())) * scale_w).unsqueeze(1)
            else:
                # Scale weights to target resolution
                if (h_orig, w_orig) != (H, W):
                    weights = F.interpolate(
                        weights.unsqueeze(0).unsqueeze(0),
                        size=(H, W), mode='area'
                    ).squeeze()
                    weights = weights - weights.min()
                    weights = weights / (weights.sum() + 1e-8)

                flat_weights = weights.flatten().to(self.device)
                indices = torch.multinomial(flat_weights, num_points, replacement=True)
                h_idx = indices // W
                w_idx = indices % W
                w_init = (w_idx.float() + torch.rand_like(w_idx.float())).unsqueeze(1)
                h_init = (h_idx.float() + torch.rand_like(h_idx.float())).unsqueeze(1)
        else:
            w_init = torch.rand(num_points, 1, device=self.device) * W
            h_init = torch.rand(num_points, 1, device=self.device) * H

        return torch.cat((w_init, h_init), dim=1).to(self.device)

    def set_dog_weights(self, dog: torch.Tensor):
        """Установка DoG весов для взвешивания лосса. dog: [1, 1, H, W] или [H, W]"""
        dog = dog.squeeze()
        dog = dog - dog.min()
        if dog.max() > 0:
            dog = dog / dog.max()
        self.dog_weights = dog.to(self.device)

    def _init_optimizer(self):
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam([
                {'params': self._rgb_logits, 'lr': self.lr},
                {'params': self.means, 'lr': self.lr * 2},
                {'params': self._cholesky, 'lr': self.lr * 5}
            ])
        else:
            s = 3
            self.optimizer = Adan([
                {'params': self._rgb_logits, 'lr': self.lr * s},
                {'params': self.means, 'lr': self.lr * 2 * s},
                {'params': self._cholesky, 'lr': self.lr * 1 * s}
            ], betas=(0.98, 0.92, 0.99), fused=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7000, gamma=0.7)

    def add_points(self, num_new: int, weights: torch.Tensor):
        """Add new points with positions sampled from weights distribution"""
        with torch.no_grad():
            new_means = self._sample_positions(num_new, self.H, self.W, weights)
            new_cholesky = torch.rand(num_new, 3, device=self.device)
            new_rgb_logits = torch.zeros(num_new, 3, device=self.device)

            # Concatenate with existing
            self.means = nn.Parameter(torch.cat([self.means.data, new_means], dim=0))
            self._cholesky = nn.Parameter(torch.cat([self._cholesky.data, new_cholesky], dim=0))
            self._rgb_logits = nn.Parameter(torch.cat([self._rgb_logits.data, new_rgb_logits], dim=0))

        # Reinitialize optimizer with new parameters
        self._init_optimizer()

    def update_resolution(self, H: int, W: int):
        """Update target resolution and scale positions accordingly"""
        with torch.no_grad():
            scale_x = W / self.W
            scale_y = H / self.H
            self.means.data[:, 0] *= scale_x
            self.means.data[:, 1] *= scale_y
        self.H = H
        self.W = W

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound
    
    @property
    def get_rgbs(self):
        return torch.sigmoid(self._rgb_logits)

    @property
    def num_points(self):
        return self.means.shape[0]

    def forward(self):
        xys, radii, conics, num_tiles_hit = project_gaussians_cholesky(
            self.get_cholesky_elements,
            self.means,
            self.H, self.W,
            self.B_SIZE,
        )
        out_img, out_wsum, dx, dy, dxy = rasterize_gaussians(
            xys, radii, conics, num_tiles_hit,
            self.get_rgbs,
            self.H, self.W,
            self.B_SIZE,
        )

        out_img = out_img[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        dx = dx[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        dy = dy[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        dxy = dxy[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img, "wsum": out_wsum, "dx": dx, "dy": dy, "dxy": dxy}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        if self.dog_weights is not None:
            weight_map = self.dog_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            per_pixel_loss = ((image - gt_image) ** 2).mean(dim=1, keepdim=True)  # [1, 1, H, W]
            weighted_loss = (per_pixel_loss * weight_map).sum() / weight_map.sum()
            loss = weighted_loss
        else:
            loss = loss_fn(image, gt_image, "L2", lambda_value=0.7)
        # Штраф за пиксели вне [0, 1] (weighted sum может выходить за диапазон)
        # clamp_penalty = (F.relu(-image) + F.relu(image - 1)).mean()
        # loss = loss + 0.1 * clamp_penalty
        elongation_penalty = _elongation_penalty(self.get_cholesky_elements)
        loss = loss + 0.01 * elongation_penalty
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr
