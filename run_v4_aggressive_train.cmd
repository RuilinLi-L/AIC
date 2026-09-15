@echo off
setlocal

set "DATA_ROOT="
for /d %%D in (*) do if exist "%%D\train" if exist "%%D\test" set "DATA_ROOT=%%D"
if not defined DATA_ROOT (
  echo Could not find a dataset folder containing train and test.
  exit /b 1
)

python.exe train_lora.py ^
  --train-dir "%DATA_ROOT%\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --output-dir ".\outputs\robust_visual_v4_aggressive" ^
  --feature-cache ".\outputs\frozen_clip_train.npy" ^
  --head-init-checkpoint ".\outputs\robust_visual_v3_val05_tta_calibrated\model.pt" ^
  --device cuda ^
  --batch-size 8 ^
  --workers 0 ^
  --epochs 8 ^
  --gradient-accumulation 4 ^
  --val-ratio 0.05 ^
  --augmentation light ^
  --lora-rank 8 ^
  --lora-alpha 16 ^
  --lora-layers 6 ^
  --lora-targets q_proj,k_proj,v_proj,out_proj ^
  --tune-layernorm ^
  --tune-visual-projection ^
  --lora-lr 0.00001 ^
  --head-lr 0.00005 ^
  --clean-fraction 0.75 ^
  --weight-floor 0.15 ^
  --label-smoothing 0.05 ^
  --anchor-weight 0.15 ^
  --pseudo-start-epoch 2 ^
  --pseudo-threshold 0.62 ^
  --pseudo-weight 0.72 ^
  --temporal-decay 0.85 ^
  --ema-decay 0.997

if errorlevel 1 exit /b %errorlevel%
echo Aggressive V4 training completed.
