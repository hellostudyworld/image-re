from typing import Dict, Any, Optional
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from .mixins import ImageLoggerMixin


__all__ = [
    "ModelCheckpoint",
    "ImageLogger"
]


def _to_grayscale_nchw(image: torch.Tensor) -> torch.Tensor:
    """Convert replicated RGB grayscale (R=G=B) to single-channel for display."""
    if image.ndim != 4:
        return image
    if image.shape[1] == 1:
        return image
    if image.shape[1] == 3:
        if torch.allclose(image[:, 0], image[:, 1]) and torch.allclose(image[:, 1], image[:, 2]):
            return image[:, :1]
        return image.mean(dim=1, keepdim=True)
    return image


def _grid_to_chw_tensor(grid: torch.Tensor) -> torch.Tensor:
    """CHW tensor in [0, 1] for TensorBoard add_image."""
    grid = grid.detach().cpu().float()
    if grid.max() > 1.0:
        grid = grid / 255.0
    return grid.clamp(0, 1)


def _get_tensorboard_writer(logger) -> Optional[Any]:
    if logger is None:
        return None
    if hasattr(logger, "experiment"):
        return logger.experiment
    if hasattr(logger, "loggers"):
        for child_logger in logger.loggers:
            writer = _get_tensorboard_writer(child_logger)
            if writer is not None:
                return writer
    return None


class ImageLogger(Callback):
    """
    Log images during training or validating.
    
    TODO: Support validating.
    """
    
    def __init__(
        self,
        log_every_n_steps: int=2000,
        max_images_each_step: int=4,
        log_images_kwargs: Dict[str, Any]=None
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if pl_module.global_step % self.log_every_n_steps == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.freeze()
            
            with torch.no_grad():
                # returned images should be: nchw, rgb, [0, 1]
                images: Dict[str, torch.Tensor] = pl_module.log_images(batch, **self.log_images_kwargs)

            tb_writer = _get_tensorboard_writer(pl_module.logger)
            global_step = pl_module.global_step

            save_root = getattr(pl_module.logger, "save_dir", trainer.default_root_dir)
            save_dir = os.path.join(save_root, "image_log", "train")
            os.makedirs(save_dir, exist_ok=True)

            comparison_panels = []
            compare_order = ("lq", "control", "samples", "hq")

            for image_key in images:
                image = images[image_key].detach().cpu()
                image = _to_grayscale_nchw(image)
                N = min(self.max_images_each_step, len(image))
                grid = torchvision.utils.make_grid(image[:N], nrow=4, normalize=False)
                grid_np = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)

                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key, global_step, pl_module.current_epoch, batch_idx
                )
                Image.fromarray(grid_np).save(os.path.join(save_dir, filename))

                if tb_writer is not None:
                    tb_writer.add_image(
                        f"images/{image_key}",
                        _grid_to_chw_tensor(grid),
                        global_step,
                    )

            if tb_writer is not None:
                for key in compare_order:
                    if key in images and len(images[key]) > 0:
                        panel = _to_grayscale_nchw(images[key].detach().cpu())[0:1]
                        comparison_panels.append(panel)
                if comparison_panels:
                    comparison = torch.cat(comparison_panels, dim=3)
                    tb_writer.add_image(
                        "images/comparison_lq_control_samples_hq",
                        comparison[0].clamp(0, 1),
                        global_step,
                    )

            if is_train:
                pl_module.unfreeze()
