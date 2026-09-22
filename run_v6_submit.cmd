@echo off
setlocal

set "DATA_ROOT="
for /d %%D in (*) do if exist "%%D\train" if exist "%%D\test" set "DATA_ROOT=%%D"
if not defined DATA_ROOT (
  echo Could not find a dataset folder containing train and test.
  exit /b 1
)

set "RUN_NAME=%~1"
if "%RUN_NAME%"=="" set "RUN_NAME=robust_visual_v6"
set "DEVICE=%~2"
if "%DEVICE%"=="" set "DEVICE=cuda"
set "RUN_DIR=.\outputs\%RUN_NAME%"

if not exist "%RUN_DIR%\best_model.pt" (
  echo Missing V6 validation checkpoint: %RUN_DIR%\best_model.pt
  exit /b 1
)
if not exist "%RUN_DIR%\model.pt" (
  echo Missing V6 full-data checkpoint: %RUN_DIR%\model.pt
  exit /b 1
)

python.exe evaluate_tta_v6.py ^
  --validation-checkpoint "%RUN_DIR%\best_model.pt" ^
  --final-checkpoint "%RUN_DIR%\model.pt" ^
  --train-dir "%DATA_ROOT%\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --original-val-logits "%RUN_DIR%\val_original_logits.npy" ^
  --hflip-val-logits "%RUN_DIR%\val_hflip_logits.npy" ^
  --output-metrics "%RUN_DIR%\calibration.json" ^
  --output-checkpoint "%RUN_DIR%\model.pt" ^
  --device "%DEVICE%" ^
  --batch-size 64 ^
  --workers 4 ^
  --force
if errorlevel 1 exit /b %errorlevel%

python.exe predict.py ^
  --checkpoint "%RUN_DIR%\model.pt" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir "%DATA_ROOT%\test" ^
  --output "%RUN_DIR%\pred_results.csv" ^
  --logits-output "%RUN_DIR%\test_logits.npy" ^
  --tta hflip ^
  --device "%DEVICE%" ^
  --batch-size 64 ^
  --workers 4
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -Command "Compress-Archive -LiteralPath '%RUN_DIR%\pred_results.csv' -DestinationPath '%RUN_DIR%\pred_results.zip' -Force"
if errorlevel 1 exit /b %errorlevel%

echo V6 submission ready: %RUN_DIR%\pred_results.zip
