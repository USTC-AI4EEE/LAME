# Location-Aware Memory Enhancement (LAME) for Wind Speed Super-Resolution

This repository implements the LAME method described in `lame.md`: a location-aware memory enhancement module built on top of DRCT for 4x wind speed super-resolution from ERA5 to CERRA. LAME decouples geographic priors from model parameters, learns location-specific wind features via self-supervision, and fuses them with windowed attention to reduce over-smoothing in complex terrain.

## Highlights
- Location-aware memory: learnable condition tensors store site-specific wind statistics independent of model weights.
- Cross-window attention and optional gating fuse memory with DRCT feature extraction.
- Works with ERA5 to CERRA LMDB pairs (2009-2020) for 4x spatial upscaling; supports single-channel mode and temporal windows.

## Repository Layout
- `train.py`: Lightning CLI entry (fit/validate) binding `WindSRDRCT` and `WeatherDataModule`.
- `DRCT.py`: LightningModule wrapping the DRCT backbone, memory parameters, metrics, and schedulers.
- `arch/drct.py`: DRCT network with cross-window attention blocks and memory dropout options.
- `arch/callbacks/ema.py`: Exponential Moving Average callback.
- `configs/32.yaml`: Paper-style training/validation config (4x SR, 256 memory channels, gating disabled).
- `validate.sh`: Convenience wrapper for `python train.py validate --config ...`.
- `CONFIG.md`: Notes on editing YAML configs.
- `lame.md`: Full paper text; use it as reference for method and experiments.

## Environment
Create an environment with PyTorch and Lightning (CUDA build recommended):
```bash
conda create -n WindSR python=3.10
conda activate WindSR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA version
pip install lightning torchmetrics numpy wandb
# Optional: bitsandbytes for 8-bit optimizers
pip install bitsandbytes
```
Required external modules:
- `weatherdata` package providing `WeatherDataModule` (LMDB loader for ERA5/CERRA pairs).
- `strategy.drct.MyStrategy` if you keep the custom strategy in the YAML; otherwise replace with a built-in Lightning strategy (for example `ddp`).

## Data Preparation (ERA5 to CERRA)
- Training expects paired LMDB datasets, for example:
  - `./data/WindSRdata/CERRA/CERRA_2009_2020_paperRange.lmdb`
  - `./data/WindSRdata/ERA5/ERA5_2009_2020_paperRange.lmdb`
- Pairs are 4x upscaling (128x128 to 512x512) with temporal window length 2 by default. Resize, crop, and time-window options live in the YAML under `data.train_opt` and `data.val_opts`.
- Update the LMDB paths in `configs/32.yaml` to your local data location. Make sure the `offset` and `len` fields match your splits.

## Training
Run with the provided config (edit paths first):
```bash
python train.py fit --config configs/32.yaml
```
Key knobs (see `model.args` in the YAML):
- `ref_chans`, `condition_size`, `cross_mode`, `use_gating`: control the location memory.
- `single_channel`, `time_window`: input/output format (single wind speed channel and temporal context).
- `depths`, `num_heads`, `embed_dim`: DRCT backbone depth and width.
- `learning_rate`, `weight_decay`, `lr_scheduler`: optimization setup.
Checkpoints and logs default to `./logs` (W&B logger runs in offline mode by default).

## Validation and Testing
```bash
python train.py validate --config configs/32.yaml
# or
./validate.sh configs/32.yaml
```
Validation averages PSNR/SSIM and, in single-channel mode, also reports PSNR/SSIM/MSE/MAE on the wind-speed magnitude scale.

## Reproducing Paper Settings
- Use the provided YAML (4x SR, 256 memory channels, gating off).
- Batch size 4, window size 16, embed dimension 180, depths `[6,6,6,6,6,6]`, heads `[6,6,6,6,6,6]`.
- Adam-based optimizer with optional EMA (enabled via callback). Set `val_inference_times > 1` to enable Monte Carlo averaging during validation.

## Citation
If you reference this work, please cite the LAME paper in `lame.md`:
```
@article{lame2025wind,
  title={Location-Aware Memory Enhancement for Wind Speed Super-Resolution},
  author={He, Zhiyang and Liu, Lei and Huang, Jie and Dong, Xue and Zhao, Hongwei and Li, Bin},
  year={2025},
  note={Manuscript}
}
```

For questions, reach out to liulei13@ustc.edu.cn.
