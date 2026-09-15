@echo off
call "%~dp0run_v5_train.cmd" robust_visual_v5_refit3 3 "%~1"
exit /b %errorlevel%
