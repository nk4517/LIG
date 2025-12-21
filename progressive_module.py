import math

from adan import Adan
from utils import *
import torch
import torch.nn as nn

from LIG.utils import loss_fn, _covariance_penalty
from gsplat2d import project_gaussians_cholesky, rasterize_gaussians


class ProgressiveGaussian2D(nn.Module):
    """Gaussian2D with ability to add points progressively"""

    def __init__(self, num_points: int, H: int, W: int, device, lr: float,
                 init_weights: torch.Tensor | None = None, opt_type: str = "adan",
                 iterations: int = 3000):
        super().__init__()
        self.device = device
        self.H = H
        self.W = W
        self.B_SIZE = 16
        self.lr = lr
        self.iterations = iterations
        self.loss_weights: torch.Tensor | None = None
        self.opt_type = opt_type

        # Initialize parameters
        self.means = nn.Parameter(self._sample_positions(num_points, H, W, init_weights))
        self._cholesky = nn.Parameter(torch.rand(num_points, 3, device=device))
        self._rgb_logits = nn.Parameter(torch.zeros(num_points, 3, device=device))
        self._opacity_logits = nn.Parameter(torch.zeros(num_points, device=device))

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

    def set_loss_weights(self, weights: torch.Tensor):
        """Установка DoG весов для взвешивания лосса. dog: [1, 1, H, W] или [H, W]"""
        weights = weights.squeeze()
        weights -= weights.min()
        if weights.max() > 0:
            weights /= weights.max()
        self.loss_weights = weights.to(self.device)

    def _init_optimizer(self):
        # lr для позиций масштабируется пропорционально разрешению (сплаты двигаются в пикселях)
        pos_lr_scale = max(self.H, self.W) / 512.0
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam([
                {'params': self._rgb_logits, 'lr': self.lr},
                {'params': self.means, 'lr': self.lr * 2 * pos_lr_scale},
                {'params': self._cholesky, 'lr': self.lr * 5 * pos_lr_scale},
                {'params': self._opacity_logits, 'lr': self.lr}
            ])
        else:
            s = 5
            self.optimizer = Adan([
                {'params': self._rgb_logits, 'lr': self.lr * 5 * s},
                {'params': self.means, 'lr': self.lr * 2 * pos_lr_scale * s},
                {'params': self._cholesky, 'lr': self.lr * 1 * pos_lr_scale * s},
                {'params': self._opacity_logits, 'lr': self.lr * s}
            ], betas=(0.98, 0.92, 0.99), fused=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.iterations, T_mult=1, eta_min=self.lr * 0.01)

    def add_points(self, num_new: int, weights: torch.Tensor | None = None,
                   target_image: torch.Tensor | None = None):
        """Add new points with positions sampled from weights distribution"""
        with torch.no_grad():
            new_means = self._sample_positions(num_new, self.H, self.W, weights)
            new_cholesky = torch.rand(num_new, 3, device=self.device)
            new_opacity_logits = torch.zeros(num_new, device=self.device)
            
            if target_image is not None:
                # Сэмплировать цвет из target_image [1, 3, H, W] по позициям new_means
                x_coords = new_means[:, 0].long().clamp(0, self.W - 1)
                y_coords = new_means[:, 1].long().clamp(0, self.H - 1)
                colors = target_image[0, :, y_coords, x_coords].T  # [num_new, 3]
                new_rgb_logits = torch.logit(colors.clamp(0.001, 0.999))
            else:
                new_rgb_logits = torch.zeros(num_new, 3, device=self.device)

            # Concatenate with existing
            self.means = nn.Parameter(torch.cat([self.means.data, new_means], dim=0))
            self._cholesky = nn.Parameter(torch.cat([self._cholesky.data, new_cholesky], dim=0))
            self._rgb_logits = nn.Parameter(torch.cat([self._rgb_logits.data, new_rgb_logits], dim=0))
            self._opacity_logits = nn.Parameter(torch.cat([self._opacity_logits.data, new_opacity_logits], dim=0))

        # Reinitialize optimizer with new parameters
        self._init_optimizer()

    def scale_to(self, new_H: int, new_W: int):
        """Scale parameters to new resolution"""
        scale_x = new_W / self.W
        scale_y = new_H / self.H

        with torch.no_grad():
            self.means.data[:, 0] *= scale_x
            self.means.data[:, 1] *= scale_y

            # L11, L22: softplus -> scale -> inverse softplus
            L11 = F.softplus(self._cholesky.data[:, 0])
            L22 = F.softplus(self._cholesky.data[:, 2])
            L11_new = L11 * scale_x
            L22_new = L22 * scale_y
            self._cholesky.data[:, 0] = torch.log(torch.clamp(torch.exp(L11_new) - 1, min=1e-8))
            self._cholesky.data[:, 2] = torch.log(torch.clamp(torch.exp(L22_new) - 1, min=1e-8))
            # L21: no softplus, just scale
            self._cholesky.data[:, 1] *= scale_y

        self.H = new_H
        self.W = new_W

    def start_stage(self, H: int, W: int, iterations: int):
        """Начало новой стадии: обновить разрешение и пересоздать optimizer"""
        if H != self.H or W != self.W:
            self.scale_to(H, W)
        self.iterations = iterations
        self._init_optimizer()

    def update_resolution(self, H: int, W: int):
        """Обновить целевое разрешение (для финального теста)."""
        if H != self.H or W != self.W:
            self.scale_to(H, W)

    @property
    def cholesky(self):
        L11 = F.softplus(self._cholesky[:, 0])
        L21 = self._cholesky[:, 1]  # no softplus
        L22 = F.softplus(self._cholesky[:, 2])
        return torch.stack([L11, L21, L22], dim=1)
    
    @property
    def rgbs(self):
        return torch.sigmoid(self._rgb_logits)

    @property
    def opacities(self):
        return torch.sigmoid(self._opacity_logits)

    @property
    def num_points(self):
        return self.means.shape[0]

    def forward(self):
        xys, extents, conics, num_tiles_hit = project_gaussians_cholesky(
            self.cholesky,
            self.means,
            self.H, self.W,
            self.B_SIZE,
            self.opacities
        )
        out_img, out_wsum, dx, dy, dxy = rasterize_gaussians(
            xys, extents, conics, num_tiles_hit,
            self.rgbs,
            self.opacities,
            self.H, self.W,
            self.B_SIZE,
        )

        out_img_chw = out_img[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img_chw, "render_hwc": out_img, "wsum": out_wsum, "dx": dx, "dy": dy, "dxy": dxy}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        if self.loss_weights is not None:
            weight_map = self.loss_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            per_pixel_loss = ((image - gt_image) ** 2).mean(dim=1, keepdim=True)  # [1, 1, H, W]
            weighted_loss = (per_pixel_loss * weight_map).sum() / weight_map.sum()
            loss = weighted_loss
        else:
            loss = loss_fn(image, gt_image, "L2", lambda_value=0.7)
        # Штраф за пиксели вне [0, 1] (weighted sum может выходить за диапазон)
        # clamp_penalty = (F.relu(-image) + F.relu(image - 1)).mean()
        # loss = loss + 0.1 * clamp_penalty
        covariance_penalty = _covariance_penalty(self.cholesky)
        loss = loss + 0.01 * covariance_penalty
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr