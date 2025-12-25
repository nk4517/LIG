import sys

from LIG.utils import _covariance_penalty

sys.path.insert(0, '../splatting_app/gsplat-2025/examples')
from lib_dog import fast_dog
from gsplat2d.project_gaussians_cholesky import project_gaussians_cholesky
from gsplat2d.rasterize import rasterize_gaussians
from utils import *
import torch
import torch.nn as nn
import math
# from optimizer import Adan
from adan import Adan

class LIG(nn.Module):
    def __init__(self, loss_type="L2", gt_image=None, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]

        self.n_scales = kwargs["n_scales"]
        self.allo_ratio = kwargs["allo_ratio"]
        self.level_models = []

        self.store_min = []
        self.store_max = []

        for s in range(self.n_scales):

            H = int(self.H * pow(2.0, -self.n_scales + s + 1))
            W = int(self.W * pow(2.0, -self.n_scales + s + 1))

            if s != self.n_scales - 1:
                num_points = int(kwargs["num_points"] * pow(2.0, (-self.n_scales + s + 1)*2) * self.allo_ratio)
            else:
                num_points = kwargs["num_points"]
                for i in range(s):
                    num_points -= int(kwargs["num_points"] * pow(2.0, (-self.n_scales + i + 1)*2) * self.allo_ratio)

            # Для первого уровня - взвешенная инициализация по DoG
            dog_weights = None
            if s == 0 and gt_image is not None:
                img_small = torch.nn.functional.interpolate(
                    gt_image, size=(H, W), mode='area'
                )
                dog_weights = fast_dog(img_small, sigma=1.25, k=2.6)

            iterations = kwargs.get("iterations", 3000)
            self.level_models.append(Gaussian2D(loss_type=self.loss_type, opt_type=kwargs['opt_type'],
                                                   num_points=num_points,
                                                   H=H, W=W, BLOCK_H=kwargs['BLOCK_H'], BLOCK_W=kwargs['BLOCK_W'],
                                                   device=kwargs['device'], lr=kwargs['lr'],
                                                   iterations=iterations,
                                                   init_weights=dog_weights,
                                                   init_mode=kwargs.get('init_mode', 'dog')))

class Gaussian2D(nn.Module):
    def __init__(self, loss_type="L2", init_weights=None, init_mode='dog', **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]

        self.B_SIZE = 16

        self.device = kwargs["device"]

        self.last_size = (self.H, self.W)
        self.loss_weights: torch.Tensor | None = None
        self.init_mode = init_mode

        self.means = nn.Parameter(self._sample_positions(init_weights if init_mode == 'dog' else None))

        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        d = 3
        self._rgb_logits = nn.Parameter(torch.zeros(self.init_num_points, d, device=self.device))
        self._opacity_logits = nn.Parameter(torch.zeros(self.init_num_points, device=self.device))

        self.means.requires_grad = True
        self._cholesky.requires_grad = True
        self._rgb_logits.requires_grad = True

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam([
                {'params': self._rgb_logits, 'lr': kwargs["lr"]},
                {'params': self.means, 'lr': kwargs["lr"] * 2},
                {'params': self._cholesky, 'lr': kwargs["lr"] * 5},
                {'params': self._opacity_logits, 'lr': kwargs["lr"]}
            ])
        else:
            s = 1
            self.optimizer = Adan([
                {'params': self._rgb_logits, 'lr': kwargs["lr"] * 5 * s},
                {'params': self.means, 'lr': kwargs["lr"] * 2 * s},
                {'params': self._cholesky, 'lr': kwargs["lr"] * 5 * s},
                {'params': self._opacity_logits, 'lr': kwargs["lr"] * s}
            ],
                betas=(0.98, 0.92, 0.99),
                fused=True)
        
        iterations = kwargs.get("iterations", 3000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=iterations, T_mult=1, eta_min=kwargs["lr"] * 0.01)

    _MULTINOMIAL_LIMIT = 2**24 - 1 # 4096x4096

    def _sample_positions(self, weights: torch.Tensor | None) -> torch.Tensor:
        """Выборка позиций: взвешенная если weights не None, иначе равномерная"""
        if self.init_mode == 'grid':
            aspect = self.W / self.H
            ny = max(1, int(math.sqrt(self.init_num_points / aspect)))
            nx = max(1, int(self.init_num_points / ny))
            ys = torch.linspace(0.5, self.H - 0.5, ny, device=self.device)
            xs = torch.linspace(0.5, self.W - 0.5, nx, device=self.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            if positions.shape[0] < self.init_num_points:
                extra = self.init_num_points - positions.shape[0]
                extra_pos = torch.rand(extra, 2, device=self.device) * torch.tensor([self.W, self.H], device=self.device)
                positions = torch.cat([positions, extra_pos], dim=0)
            return positions[:self.init_num_points]
        
        if weights is not None:
            weights = weights.squeeze()
            weights = (weights - weights.min())
            weights = weights / weights.sum()
            
            h_orig, w_orig = weights.shape
            total_pixels = h_orig * w_orig
            
            if total_pixels > self._MULTINOMIAL_LIMIT:
                scale = (self._MULTINOMIAL_LIMIT / total_pixels) ** 0.5
                h_dim, w_dim = max(1, int(h_orig * scale)), max(1, int(w_orig * scale))
                weights = torch.nn.functional.interpolate(
                    weights.unsqueeze(0).unsqueeze(0), size=(h_dim, w_dim), mode='area'
                ).squeeze()
                weights = (weights - weights.min())
                weights = weights / weights.sum()
                scale_h, scale_w = h_orig / h_dim, w_orig / w_dim
            else:
                h_dim, w_dim = h_orig, w_orig
                scale_h, scale_w = 1.0, 1.0
            
            indices = torch.multinomial(weights.flatten().to(self.device), self.init_num_points, replacement=True)
            w_init = ((indices % w_dim).float() + torch.rand(self.init_num_points, device=self.device)) * scale_w
            h_init = ((indices // w_dim).float() + torch.rand(self.init_num_points, device=self.device)) * scale_h
        else:
            w_init = torch.rand(self.init_num_points, device=self.device) * self.W
            h_init = torch.rand(self.init_num_points, device=self.device) * self.H
        
        return torch.stack((w_init, h_init), dim=1).to(self.device)

    def reinit_positions(self, weights: torch.Tensor):
        with torch.no_grad():
            self.means.data = self._sample_positions(weights)

    def set_loss_weights(self, weights: torch.Tensor | None):
        """Установка DoG весов для взвешивания лосса. dog: [1, 1, H, W] или [H, W]"""
        if weights is None:
            self.loss_weights = weights
            return
        weights = weights.clone().squeeze()
        weights -= weights.min()
        if weights.max() > 0:
            weights /= weights.max()
        self.loss_weights = weights.to(self.device)

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

    def forward(self):
        (
            xys,
            extents,
            conics,
            num_tiles_hit,
        ) = project_gaussians_cholesky(
            self.cholesky,
            self.means,
            self.H,
            self.W,
            self.B_SIZE,
            self.opacities
        )
        out_img, out_wsum, dx, dy, dxy = rasterize_gaussians(
                xys,
                extents,
                conics,
                num_tiles_hit,
                self.rgbs,
                self.opacities,
                self.H,
                self.W,
                self.B_SIZE,
                compute_upscale_gradients=True
            )

        out_img_r = out_img[..., :3].view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img_r, "render_hwc": out_img, "wsum": out_wsum, "dx": dx, "dy": dy, "dxy": dxy}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        if self.loss_weights is not None:
            weight_map = self.loss_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            per_pixel_loss = ((image - gt_image) ** 2).mean(dim=1, keepdim=True)  # [1, 1, H, W]
            weighted_loss = (per_pixel_loss * weight_map).sum() / weight_map.sum()
            loss = weighted_loss
        else:
            loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        
        covariance_penalty = _covariance_penalty(self.cholesky)
        loss += 0.01 * covariance_penalty
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr
