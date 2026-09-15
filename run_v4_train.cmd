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
  --output-dir ".\outputs\robust_visual_v4_lora" ^
  --feature-cache ".\outputs\frozen_clip_train.npy" ^
  --head-init-checkpoint ".\outputs\robust_visual_v3_val05_tta_calibrated\model.pt" ^
  --device cuda ^
  --batch-size 16 ^
  --workers 0 ^
  --epochs 6 ^
  --gradient-accumulation 2 ^
  --val-ratio 0.05 ^
  --augmentation light ^
  --lora-rank 8 ^
  --lora-alpha 16 ^
  --lora-layers 4 ^
  --lora-targets q_proj,v_proj ^
  --lora-lr 0.00002 ^
  --head-lr 0.00008 ^
  --clean-fraction 0.80 ^
  --weight-floor 0.20 ^
  --label-smoothing 0.03 ^
  --anchor-weight 0.10 ^
  --pseudo-start-epoch 2 ^
  --pseudo-threshold 0.70 ^
  --pseudo-weight 0.60 ^
  --temporal-decay 0.80 ^
  --ema-decay 0.995

if errorlevel 1 exit /b %errorlevel%
echo V4 training completed.
