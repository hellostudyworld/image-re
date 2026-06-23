"""
Quick smoke test: short training + inference to verify grayscale + TensorBoard changes.

Usage:
    cd FlowIE_c4d
    python quick_smoke_test.py
"""
import glob
import os
import shutil
import subprocess
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config, load_state_dict

SMOKE_DIR = "./work_dirs/c4d_smoke_test"
CKPT_GLOB = os.path.join(SMOKE_DIR, "**", "checkpoints", "step-*.ckpt")
INFER_INPUT = "C:/code/diffusion-re/C4DataSet_flowie/test/noisy"
INFER_OUTPUT = os.path.join(SMOKE_DIR, "inference_results")
NUM_INFER = 3


def run_training():
    config = OmegaConf.load("./configs/train_c4d_smoke.yaml")
    pl.seed_everything(config.lightning.seed, workers=True)

    data_module = instantiate_from_config(config.data)
    model_config = OmegaConf.load(config.model.config)
    model_config.params.first_stage_config.use_fp16 = False
    model = instantiate_from_config(model_config)

    if config.model.get("resume"):
        weights = torch.load(config.model.resume, map_location="cpu")
        load_state_dict(model, weights, strict=False)
        print(f"Loaded base weights: {config.model.resume}")

    trainer_config = dict(config.lightning.trainer)
    callbacks = [instantiate_from_config(cb) for cb in config.lightning.callbacks]
    logger = instantiate_from_config(config.lightning.logger)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_config)
    print("=" * 60)
    print("Smoke test training (12 steps)")
    print(f"TensorBoard: tensorboard --logdir {logger.log_dir}")
    print("=" * 60)
    trainer.fit(model, datamodule=data_module)
    return logger.log_dir


def find_latest_checkpoint():
    ckpts = glob.glob(CKPT_GLOB, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {SMOKE_DIR}")
    return max(ckpts, key=os.path.getmtime)


def run_inference(ckpt_path: str):
    infer_input = os.path.join(SMOKE_DIR, "infer_input")
    if os.path.isdir(infer_input):
        shutil.rmtree(infer_input)
    os.makedirs(infer_input, exist_ok=True)

    import glob as g
    from PIL import Image

    noisy_files = sorted(g.glob(os.path.join(INFER_INPUT, "*")))[:NUM_INFER]
    for src in noisy_files:
        shutil.copy2(src, infer_input)

    if os.path.isdir(INFER_OUTPUT):
        shutil.rmtree(INFER_OUTPUT)

    cmd = [
        sys.executable, "inference_c4d.py",
        "--ckpt", ckpt_path,
        "--config", "configs/model/cldm_bsr_eval.yaml",
        "--input", infer_input,
        "--output", INFER_OUTPUT,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    print("=" * 60)
    print("Running inference:", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, check=True)


def verify_outputs(tb_log_dir: str, ckpt_path: str):
    events = glob.glob(os.path.join(tb_log_dir, "**", "events.out.tfevents.*"), recursive=True)
    image_logs = glob.glob(os.path.join(SMOKE_DIR, "image_log", "train", "*.png"))
    infer_outputs = glob.glob(os.path.join(INFER_OUTPUT, "**", "*.png"), recursive=True)

    print("\n" + "=" * 60)
    print("Smoke test summary")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"TensorBoard events: {len(events)} file(s)")
    print(f"Training image logs: {len(image_logs)} file(s)")
    print(f"Inference outputs: {len(infer_outputs)} file(s)")
    if image_logs:
        print("Sample train logs:", image_logs[:4])
    if infer_outputs:
        print("Inference results:", infer_outputs)
    print("=" * 60)

    ok = bool(ckpt_path) and len(events) > 0 and len(image_logs) > 0 and len(infer_outputs) >= NUM_INFER
    if not ok:
        raise RuntimeError("Smoke test verification failed — see summary above.")
    print("Smoke test PASSED.")


def main():
    tb_log_dir = run_training()
    ckpt = find_latest_checkpoint()
    run_inference(ckpt)
    verify_outputs(tb_log_dir, ckpt)


if __name__ == "__main__":
    main()
