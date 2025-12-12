import sys
sys.path.insert(0, '../splatting_app/gsplat-2025/examples')
from lib_dog import fast_dog
from gsplat2d.project_gaussians import project_gaussians
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
                dog_weights = fast_dog(img_small, sigma=2.5, k=1.6)

            self.level_models.append(Gaussian2D(loss_type=self.loss_type, opt_type=kwargs['opt_type'],
                                                   num_points=num_points,
                                                   H=H, W=W, BLOCK_H=kwargs['BLOCK_H'], BLOCK_W=kwargs['BLOCK_W'],
                                                   device=kwargs['device'], lr=kwargs['lr'],
                                                   init_weights=dog_weights))

class Gaussian2D(nn.Module):
    def __init__(self, loss_type="L2", init_weights=None, **kwargs):
        super().__init__()
        self.use_cuda_cholesky = kwargs.get("use_cuda_cholesky", True)
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]

        self.B_SIZE = 16

        self.device = kwargs["device"]

        self.last_size = (self.H, self.W)

        self.means = nn.Parameter(self._sample_positions(init_weights))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5], device=self.device).view(1, 3))

        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        d = 3
        self._rgb_logits = nn.Parameter(torch.zeros(self.init_num_points, d, device=self.device))

        self.means.requires_grad = True
        self._cholesky.requires_grad = True
        self._rgb_logits.requires_grad = True

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam([
                {'params': self._rgb_logits, 'lr': kwargs["lr"]},
                {'params': self.means, 'lr': kwargs["lr"] * 2},
                {'params': self._cholesky, 'lr': kwargs["lr"] * 5}
            ])
        else:
            s = 8
            self.optimizer = Adan([
                {'params': self._rgb_logits, 'lr': kwargs["lr"] / 2},
                {'params': self.means, 'lr': kwargs["lr"] * 2 * s},
                {'params': self._cholesky, 'lr': kwargs["lr"] * 5 * s}
            ],
                betas=(0.98, 0.92, 0.99),
                fused=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=70000, gamma=0.7)

    _MULTINOMIAL_LIMIT = 2**24 - 1 # 4096x4096

    def _sample_positions(self, weights: torch.Tensor | None) -> torch.Tensor:
        """Выборка позиций: взвешенная если weights не None, иначе равномерная"""
        if weights is not None:
            weights = weights.squeeze()
            weights = weights - weights.min()
            weights = weights / weights.sum()

            h_orig, w_orig = weights.shape
            total_pixels = h_orig * w_orig

            if total_pixels > self._MULTINOMIAL_LIMIT:
                scale = (self._MULTINOMIAL_LIMIT / total_pixels) ** 0.5
                h_small = max(1, int(h_orig * scale))
                w_small = max(1, int(w_orig * scale))
                weights_small = torch.nn.functional.interpolate(
                    weights.unsqueeze(0).unsqueeze(0),
                    size=(h_small, w_small),
                    mode='area'
                ).squeeze()
                weights_small = weights_small - weights_small.min()
                weights_small = weights_small / weights_small.sum()
                flat_weights = weights_small.flatten().to(self.device)
                indices = torch.multinomial(flat_weights, self.init_num_points, replacement=True)
                h_idx_small = indices // w_small
                w_idx_small = indices % w_small
                scale_h = h_orig / h_small
                scale_w = w_orig / w_small
                h_init = ((h_idx_small.float() + torch.rand_like(h_idx_small.float())) * scale_h).unsqueeze(1)
                w_init = ((w_idx_small.float() + torch.rand_like(w_idx_small.float())) * scale_w).unsqueeze(1)
            else:
                flat_weights = weights.flatten().to(self.device)
                indices = torch.multinomial(flat_weights, self.init_num_points, replacement=True)
                h_idx = indices // w_orig
                w_idx = indices % w_orig
                w_init = (w_idx.float() + torch.rand_like(w_idx.float())).unsqueeze(1)
                h_init = (h_idx.float() + torch.rand_like(h_idx.float())).unsqueeze(1)
        else:
            w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W
            h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H
        return torch.cat((w_init, h_init), dim=1).to(self.device)

    def reinit_positions(self, weights: torch.Tensor):
        with torch.no_grad():
            self.means.data = self._sample_positions(weights)

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    @property
    def get_rgbs(self):
        return torch.sigmoid(self._rgb_logits)

    def _cholesky_to_cov2d(self, cholesky):
        l11 = cholesky[:, 0]
        l21 = cholesky[:, 1]
        l22 = cholesky[:, 2]
        return torch.stack([l11 * l11, l11 * l21, l21 * l21 + l22 * l22], dim=1)

    def forward(self):
        if self.use_cuda_cholesky:
            (
                xys,
                radii,
                conics,
                num_tiles_hit,
            ) = project_gaussians_cholesky(
                self.get_cholesky_elements,
                self.means,
                self.H,
                self.W,
                self.B_SIZE,
            )
        else:
            cov2d = self._cholesky_to_cov2d(self.get_cholesky_elements)
            (
                xys,
                radii,
                conics,
                num_tiles_hit,
            ) = project_gaussians(
                cov2d,
                self.means,
                self.H,
                self.W,
                self.B_SIZE,
            )
        out_img, out_wsum, dx, dy, dxy = rasterize_gaussians(
                xys,
                radii,
                conics,
                num_tiles_hit,
                self.get_rgbs,
                self.H,
                self.W,
                self.B_SIZE,
            )

        out_img = out_img[..., :3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        # Gradients can be negative, don't clamp them
        dx = dx[..., :3]
        dx = dx.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        dy = dy[..., :3]
        dy = dy.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        dxy = dxy[..., :3]
        dxy = dxy.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img, "wsum": out_wsum, "dx": dx, "dy": dy, "dxy": dxy}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr
