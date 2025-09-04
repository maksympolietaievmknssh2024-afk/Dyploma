@echo off
echo ============================================================
echo GPU Setup for Diffusion Model Project
echo ============================================================
echo.

echo Checking current system configuration...
echo.

:: Run the Python setup check
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" setup_gpu.py

echo.
echo ============================================================
echo GPU Setup Options
echo ============================================================
echo.
echo 1. Install CUDA-enabled PyTorch (recommended)
echo 2. Check system status only
echo 3. Exit
echo.
set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" goto install_gpu
if "%choice%"=="2" goto check_only
if "%choice%"=="3" goto exit

echo Invalid choice. Exiting...
goto exit

:install_gpu
echo.
echo Installing CUDA-enabled PyTorch...
echo This may take a few minutes...
echo.

:: Uninstall CPU version
echo Removing CPU-only PyTorch...
pip uninstall torch torchvision torchaudio -y

:: Install GPU version
echo Installing GPU-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Verifying installation...
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU devices:', torch.cuda.device_count())"

echo.
echo Installation complete! You can now use GPU acceleration.
goto exit

:check_only
echo System check complete. See output above for recommendations.
goto exit

:exit
echo.
echo Press any key to exit...
pause >nul