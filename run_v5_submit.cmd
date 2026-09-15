@echo off
setlocal

set "DATA_ROOT="
for /d %%D in (*) do if exist "%%D\train" if exist "%%D\test" set "DATA_ROOT=%%D"
if not defined DATA_ROOT (
  echo Could not find a dataset folder containing train and test.
  exit /b 1
)

set "RUN_NAME=%~1"
if "%RUN_NAME%"=="" set "RUN_NAME=robust_visual_v5"
set "RUN_DIR=.\outputs\%RUN_NAME%"
set "FINAL_DIR=.\outputs\%RUN_NAME%_tta_calibrated"
set "CHECKPOINT=%RUN_DIR%\model.pt"
if not exist "%CHECKPOINT%" set "CHECKPOINT=%RUN_DIR%\best_model.pt"
if not exist "%CHECKPOINT%" (
  echo Could not find model.pt or best_model.pt under %RUN_DIR%.
  exit /b 1
)

python.exe predict.py ^
  --checkpoint "%CHECKPOINT%" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir "%DATA_ROOT%\test" ^
  --output "%RUN_DIR%\pred_results.csv" ^
  --logits-output "%RUN_DIR%\test_logits.npy" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 0
if errorlevel 1 exit /b %errorlevel%

python.exe evaluate_tta_v5.py ^
  --checkpoint "%CHECKPOINT%" ^
  --train-dir "%DATA_ROOT%\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --original-val-logits "%RUN_DIR%\val_logits.npy" ^
  --hflip-val-logits "%RUN_DIR%\val_hflip_logits.npy" ^
  --output-metrics "%RUN_DIR%\tta_metrics.json" ^
  --output-checkpoint "%FINAL_DIR%\model.pt" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 0 ^
  --force
if errorlevel 1 exit /b %errorlevel%

python.exe predict.py ^
  --checkpoint "%FINAL_DIR%\model.pt" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir "%DATA_ROOT%\test" ^
  --output "%FINAL_DIR%\pred_results.csv" ^
  --logits-output "%FINAL_DIR%\test_logits.npy" ^
  --tta hflip ^
  --base-logits-input "%RUN_DIR%\test_logits.npy" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 0
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -Command "Compress-Archive -LiteralPath '%FINAL_DIR%\pred_results.csv' -DestinationPath '%FINAL_DIR%\pred_results.zip' -Force"
if errorlevel 1 exit /b %errorlevel%

echo Submission ready: %FINAL_DIR%\pred_results.zip
