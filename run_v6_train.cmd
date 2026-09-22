@echo off
setlocal

set "DATA_ROOT="
for /d %%D in (*) do if exist "%%D\train" set "DATA_ROOT=%%D"
if not defined DATA_ROOT (
  echo Could not find a dataset folder containing train.
  exit /b 1
)

set "RUN_NAME=%~1"
if "%RUN_NAME%"=="" set "RUN_NAME=robust_visual_v6"
set "DEVICE=%~2"
if "%DEVICE%"=="" set "DEVICE=cuda"
set "RESUME_ARG="
if not "%~3"=="" set "RESUME_ARG=--resume "%~3""

echo V6 run name: %RUN_NAME%
echo Device: %DEVICE%

python.exe train_v6.py ^
  --train-dir "%DATA_ROOT%\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --output-dir ".\outputs\%RUN_NAME%" ^
  --feature-cache ".\outputs\%RUN_NAME%\frozen_clip_multiview_v6.npy" ^
  --device "%DEVICE%" ^
  --batch-size 64 ^
  --workers 4 ^
  %RESUME_ARG%

if errorlevel 1 exit /b %errorlevel%
echo V6 training completed: .\outputs\%RUN_NAME%\model.pt
