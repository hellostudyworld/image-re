"""
FlowIE C4D Grayscale Training - Resume Script
Run this to continue or restart training.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python resume_training.py
"""
import os, sys, torch, pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.common import instantiate_from_config, load_state_dict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load training config
config = OmegaConf.load('./configs/train_c4d.yaml')
pl.seed_everything(config.lightning.seed, workers=True)

# Data module
data_module = instantiate_from_config(config.data)

# Model
model_config = OmegaConf.load(config.model.config)
model_config.params.first_stage_config.use_fp16 = False
model = instantiate_from_config(model_config)

# Load pretrained weights
if config.model.get('resume'):
    weights = torch.load(config.model.resume, map_location='cpu')
    load_state_dict(model, weights, strict=False)
    print(f"Loaded base weights from: {config.model.resume}")

# Trainer
trainer_config = dict(config.lightning.trainer)
trainer_config['devices'] = [0]
trainer_config['default_root_dir'] = './work_dirs/c4d_train_grayscale'
trainer_config['num_sanity_val_steps'] = 0

callbacks = []
for cb_config in config.lightning.callbacks:
    callbacks.append(instantiate_from_config(cb_config))

trainer = pl.Trainer(callbacks=callbacks, **trainer_config)

print("=" * 60)
print("Starting/Resuming GRAYSCALE training on C4D dataset")
print(f"Steps: {trainer_config.get('max_steps', 'N/A')}, "
      f"Batch: {config.data.params.train_config.data_loader.batch_size}, "
      f"Accumulate: {trainer_config.get('accumulate_grad_batches', 1)}")
print(f"Output: {trainer_config['default_root_dir']}")
print("=" * 60)

trainer.fit(model, datamodule=data_module)
