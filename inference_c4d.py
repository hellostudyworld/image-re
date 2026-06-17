"""
Standalone inference script for FlowIE on C4D dataset.
Handles custom AutoencoderKL (non-diffusers) properly.
"""
import os, sys, math, argparse
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.xformers_state import disable_xformers
from utils.image import auto_resize, pad, wavelet_reconstruction, adaptive_instance_normalization
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def forward_flowie_one_step(model, latents, prompt_embeds, timestep=400, c=None):
    ts = torch.ones(latents.shape[0], device=latents.device) * timestep
    control = model.control_model(latents, hint=c, timesteps=ts, context=prompt_embeds)
    noise_pred = model.model.diffusion_model(
        latents, timesteps=ts, context=prompt_embeds,
        control=control, only_mid_control=model.only_mid_control
    )
    return noise_pred


@torch.no_grad()
def process(model, control_imgs, color_fix_type, disable_preprocess_model,
            tiled, tile_size, tile_stride, preprocess_model=None, vae=None):
    n_samples = len(control_imgs)
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    y_null_all = torch.load("./weights/null_token_1024.pth", map_location="cpu")
    y_null_ori = y_null_all['null_prompt_embeds'].to(control.device)
    y_null = y_null_ori.repeat((control.shape[0], 1, 1))

    if not disable_preprocess_model:
        control = preprocess_model(control)

    img_buffer = torch.zeros_like(control).to(control.device)
    height, width = control.size(-2), control.size(-1)
    h, w = height // 8, width // 8

    control_norm = control * 2 - 1
    posterior = vae.encode(control_norm)
    # Handle both diffusers VAE and custom VAE
    if hasattr(posterior, 'latent_dist'):
        c_latent = posterior.latent_dist.mode().to(torch.float32)
    else:
        c_latent = posterior.mode().to(torch.float32)

    init_noise = torch.randn(c_latent.shape, device=c_latent.device)

    if not tiled:
        latents = forward_flowie_one_step(model, init_noise, y_null, c=torch.cat([c_latent], 1))
        # Handle scaling - model scale_factor from config (0.18215)
        scale_factor = getattr(model, 'scale_factor', 0.18215)
        if torch.is_tensor(scale_factor):
            scale_factor = scale_factor.item()
        latents = latents.detach() / scale_factor
        decoded = vae.decode(latents)
        # Handle both diffusers (returns .sample) and custom VAE (returns tensor directly)
        if hasattr(decoded, 'sample'):
            img_buffer = decoded.sample / 2 + 0.5
        else:
            img_buffer = decoded / 2 + 0.5
    else:
        from utils.image import _sliding_windows
        tiles_iterator = tqdm(_sliding_windows(h, w, tile_size // 8, tile_stride // 8))
        shape = (n_samples, 4, height // 8, width // 8)
        count = torch.zeros(shape, dtype=torch.long, device=init_noise.device)
        noise_buffer = torch.zeros_like(init_noise).to(init_noise.device)

        for hi, hi_end, wi, wi_end in tiles_iterator:
            tiles_iterator.set_description(f"Tile ({hi}-{hi_end}, {wi}-{wi_end})")
            tile_noise = init_noise[:, :, hi:hi_end, wi:wi_end]
            tile_cond = c_latent[:, :, hi:hi_end, wi:wi_end]
            tile_latents = forward_flowie_one_step(model, tile_noise, y_null, c=torch.cat([tile_cond], 1))
            noise_buffer[:, :, hi:hi_end, wi:wi_end] += tile_latents
            count[:, :, hi:hi_end, wi:wi_end] += 1

        noise_buffer.div_(count)
        count.zero_()

        for hi, hi_end, wi, wi_end in _sliding_windows(h, w, tile_size // 8, tile_stride // 8):
            tile_latents = noise_buffer[:, :, hi:hi_end, wi:wi_end]
            scale_factor = getattr(model, 'scale_factor', 0.18215)
            if torch.is_tensor(scale_factor):
                scale_factor = scale_factor.item()
            tile_latents = tile_latents.detach() / scale_factor
            tile_img_pixel = vae.decode(tile_latents)
            if hasattr(tile_img_pixel, 'sample'):
                tile_img_pixel = tile_img_pixel.sample / 2 + 0.5
            else:
                tile_img_pixel = tile_img_pixel / 2 + 0.5

            tile_cond_img = control[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8]
            if color_fix_type == "adain":
                tile_img_pixel = adaptive_instance_normalization(tile_img_pixel, tile_cond_img)
            elif color_fix_type == "wavelet":
                tile_img_pixel = wavelet_reconstruction(tile_img_pixel, tile_cond_img)

            img_buffer[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += tile_img_pixel
            count[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += 1
        img_buffer.div_(count)

    samples = img_buffer.clamp(0, 1)
    x_samples = (einops.rearrange(samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control_out = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control_out[i] for i in range(n_samples)]
    return preds, stage1_preds


def check_device(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"
    else:
        disable_xformers()
    print(f'Using device: {device}')
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--config", default='configs/model/cldm_bsr_eval.yaml', type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=448)
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    args.device = check_device(args.device)

    # Load model from config
    config = OmegaConf.load(args.config)
    # Modify config to avoid HF download (use custom VAE)
    if 'first_stage_config' in config:
        config.first_stage_config.use_fp16 = False
    model = instantiate_from_config(config)
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=False)

    # Extract VAE and preprocess model
    vae = model.first_stage_model
    preprocess_model = model.preprocess_model
    preprocess_model.to(args.device)
    model.freeze()
    model.to(args.device)

    assert os.path.isdir(args.input), f"Input directory not found: {args.input}"
    os.makedirs(args.output, exist_ok=True)

    files = list_image_files(args.input, follow_links=True)
    print(f"Processing {len(files)} images...")

    for file_path in tqdm(files):
        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        if not args.tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
        x = pad(np.array(lq_resized), scale=64)

        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        save_path = os.path.join(parent_path, f"{stem}.png")
        os.makedirs(parent_path, exist_ok=True)

        preds, stage1_preds = process(
            model, [x],
            color_fix_type=args.color_fix_type,
            disable_preprocess_model=args.disable_preprocess_model,
            tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride,
            vae=vae, preprocess_model=preprocess_model
        )
        pred = preds[0]
        pred = pred[:lq_resized.height, :lq_resized.width, :]
        # Convert to true grayscale: average channels, replicate back to 3-ch
        pred_gray = np.mean(pred.astype(np.float32), axis=2, keepdims=True)
        pred = np.repeat(pred_gray, 3, axis=2).astype(np.uint8)
        Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
