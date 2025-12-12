import math
import time
import typing
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import sys
from PIL import Image

from progressive_module import ProgressiveGaussian2D
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import threading

sys.path.insert(0, '../splatting_app/gsplat-2025/examples')
from lib_dog import fast_dog

if typing.TYPE_CHECKING:
    from visualizer import LIGVisualizer


@dataclass
class StageConfig:
    scale: float  # 1/scale по каждой стороне
    iterations: int
    points_to_add: int  # сколько точек добавить ПОСЛЕ этой стадии (0 для последней)


class ProgressiveTrainer:
    def __init__(
        self,
        image_path: Path,
        log_path: str,
        total_points: int,
        stages: list[StageConfig],
        lr: float = 0.018,
        save_imgs: bool = True,
        gui: "LIGVisualizer|None" = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        
        image_path = Path(image_path)
        self.image_name = image_path.stem
        self.log_dir = Path(log_path) / self.image_name
        self.save_imgs = save_imgs
        self.gui = gui
        self.stages = stages
        self.lr = lr
        
        # Calculate initial points for first stage
        total_added = sum(s.points_to_add for s in stages)
        initial_points = total_points - total_added
        
        # First stage resolution
        first_scale = stages[0].scale
        H0 = int(self.H / first_scale)
        W0 = int(self.W / first_scale)
        
        # Initial weights from DoG on smallest image
        img_small = F.interpolate(self.gt_image, size=(H0, W0), mode='area')
        dog_weights = fast_dog(img_small, sigma=2.5, k=1.6)
        
        self.model = ProgressiveGaussian2D(
            num_points=initial_points,
            H=H0, W=W0,
            device=self.device,
            lr=lr,
            init_weights=dog_weights
        ).to(self.device)
        
        if self.gui:
            self.gui.set_model(self.model, self.gt_image)
        
        self.logwriter = LogWriter(self.log_dir)
    
    def _get_target_at_scale(self, scale: float) -> torch.Tensor:
        H = int(self.H / scale)
        W = int(self.W / scale)
        return F.interpolate(self.gt_image, size=(H, W), mode='area')
    
    def _compute_error_weights(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 error map for weighted point placement"""
        error = (rendered - target) ** 2
        return error.mean(dim=1, keepdim=True).squeeze()
    
    def train(self):
        psnr_list, iter_list = [], []
        global_iter = 0
        start_time = time.time()
        
        for stage_idx, stage in enumerate(self.stages):
            # Update resolution
            H_stage = int(self.H / stage.scale)
            W_stage = int(self.W / stage.scale)
            
            if stage_idx > 0:
                self.model.update_resolution(H_stage, W_stage)
            
            target = self._get_target_at_scale(stage.scale)
            
            if self.gui:
                self.gui.set_target_image(target)
            
            progress_bar = tqdm(
                range(1, stage.iterations + 1),
                desc=f"Stage {stage_idx} (1/{stage.scale:.0f}x, {self.model.num_points} pts)"
            )
            self.model.train()
            
            for iter in range(1, stage.iterations + 1):
                if self.gui:
                    if self.gui.e_want_to_render.is_set():
                        self.gui.e_finished_rendering.wait()
                    self.gui.lock.acquire(blocking=True)
                
                loss, psnr = self.model.train_iter(target)
                psnr_list.append(psnr)
                iter_list.append(global_iter)
                global_iter += 1
                
                with torch.no_grad():
                    if iter % 10 == 0:
                        progress_bar.set_postfix({
                            "Loss": f"{loss.item():.7f}",
                            "PSNR": f"{psnr:.4f}"
                        })
                        progress_bar.update(10)
                
                if self.gui:
                    self.gui.set_updated()
                    self.gui.update_stats(global_iter, psnr, loss.item())
                    self.gui.lock.release()
            
            progress_bar.close()
            
            # Add points after stage (except last)
            if stage.points_to_add > 0:
                with torch.no_grad():
                    rendered = self.model()["render"]
                    weights = self._compute_error_weights(rendered, target)
                
                self.model.add_points(stage.points_to_add, weights)
                self.logwriter.write(
                    f"Stage {stage_idx}: added {stage.points_to_add} points, "
                    f"total now {self.model.num_points}"
                )
                
                if self.save_imgs:
                    transform = transforms.ToPILImage()
                    img = transform(torch.clamp(rendered, 0, 1).squeeze(0))
                    img.save(str(self.log_dir / f"{self.image_name}_stage{stage_idx}.png"))
        
        end_time = time.time() - start_time
        
        # Final test at full resolution
        self.model.update_resolution(self.H, self.W)
        psnr_value, ms_ssim_value = self.test()
        
        # Benchmark
        with torch.no_grad():
            self.model.eval()
            test_start = time.time()
            for _ in range(100):
                _ = self.model()
            test_end = (time.time() - test_start) / 100
        
        self.logwriter.write(
            f"Training Complete in {end_time:.4f}s, "
            f"Eval time:{test_end:.8f}s, FPS:{1/test_end:.4f}"
        )
        
        torch.save(self.model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {
            "iterations": iter_list,
            "training_psnr": psnr_list,
            "training_time": end_time,
            "psnr": psnr_value,
            "ms-ssim": ms_ssim_value,
            "rendering_time": test_end,
            "rendering_fps": 1/test_end
        })
        
        return psnr_value, ms_ssim_value, end_time, test_end, 1/test_end
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model()["render"].float()
        out = torch.clamp(out, 0, 1)
        mse_loss = F.mse_loss(out, self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out, self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write(f"Test PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")
        
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out.squeeze(0))
            img.save(str(self.log_dir / f"{self.image_name}_fitting.png"))
        
        return psnr, ms_ssim_value


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Progressive Gaussian training")
    parser.add_argument("-d", "--dataset", type=str, default='./dataset/DIV2K_valid_HR')
    parser.add_argument("--data_name", type=str, default='single_image')
    parser.add_argument("--num_points", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--save_imgs", action="store_true", default=True)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    
    # Stage configuration
    parser.add_argument("--scales", type=str, default="12,4,1",
                        help="Comma-separated scale divisors (e.g., '12,4,1' means 1/12, 1/4, 1/1)")
    parser.add_argument("--iters_per_stage", type=str, default="1000,2000,3000",
                        help="Comma-separated iterations per stage")
    parser.add_argument("--points_ratio", type=str, default="0.1,0.3",
                        help="Comma-separated ratios of total points to add after each stage (except last)")
    
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    
    # Parse stage configuration
    scales = [float(x) for x in args.scales.split(',')]
    iters = [int(x) for x in args.iters_per_stage.split(',')]
    ratios = [float(x) for x in args.points_ratio.split(',')]
    
    stages = []
    for i, (scale, it) in enumerate(zip(scales, iters)):
        pts = int(args.num_points * ratios[i]) if i < len(ratios) else 0
        stages.append(StageConfig(scale=scale, iterations=it, points_to_add=pts))
    
    log_path = f"./checkpoints/{args.data_name}/Progressive_{sum(iters)}_{args.num_points}"
    logwriter = LogWriter(log_path)
    
    gui = None
    gui_thread = None
    if args.visualize:
        try:
            from visualizer import LIGVisualizer
            gui = LIGVisualizer(width=800, height=600)
            gui_thread = threading.Thread(target=gui.run)
            gui_thread.start()
            time.sleep(1)
        except ImportError:
            print("Visualizer not available")
            gui = None
    
    image_path = Path(args.dataset)
    trainer = ProgressiveTrainer(
        image_path=image_path,
        log_path=log_path,
        total_points=args.num_points,
        stages=stages,
        lr=args.lr,
        save_imgs=args.save_imgs,
        gui=gui,
    )
    
    psnr, ms_ssim_val, train_time, eval_time, fps = trainer.train()
    logwriter.write(
        f"{trainer.image_name}: {trainer.H}x{trainer.W}, "
        f"PSNR:{psnr:.4f}, MS-SSIM:{ms_ssim_val:.4f}, "
        f"Training:{train_time:.4f}s, Eval:{eval_time:.8f}s, FPS:{fps:.4f}"
    )
    
    if gui:
        gui.e_want_to_render.clear()
        gui.e_finished_rendering.set()
        if gui_thread:
            gui_thread.join(timeout=2)


if __name__ == "__main__":
    main(sys.argv[1:])
