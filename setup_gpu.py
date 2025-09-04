#!/usr/bin/env python3
"""
GPU Setup Script for Diffusion Model Project

This script helps configure the project to use GPU acceleration.
It checks system compatibility and provides installation instructions.
"""

import subprocess
import sys
import platform

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available on the system."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_cuda_version():
    """Check CUDA version if available."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    return line.strip()
        return None
    except FileNotFoundError:
        return None

def check_current_torch():
    """Check current PyTorch installation."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        return version, cuda_available, device_count
    except ImportError:
        return None, False, 0

def get_recommended_torch_command():
    """Get the recommended PyTorch installation command."""
    cuda_version = check_cuda_version()
    
    if cuda_version and '12.' in cuda_version:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_version and '11.' in cuda_version:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

def main():
    print("=" * 60)
    print("GPU Setup Check for Diffusion Model Project")
    print("=" * 60)
    
    # Check system information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print()
    
    # Check NVIDIA GPU
    has_nvidia = check_nvidia_gpu()
    print(f"NVIDIA GPU Available: {'Yes' if has_nvidia else 'No'}")
    
    if has_nvidia:
        # Check CUDA
        cuda_info = check_cuda_version()
        print(f"CUDA Installation: {cuda_info if cuda_info else 'Not found'}")
        print()
        
        # Check current PyTorch
        torch_version, cuda_available, device_count = check_current_torch()
        if torch_version:
            print(f"Current PyTorch Version: {torch_version}")
            print(f"CUDA Support in PyTorch: {'Yes' if cuda_available else 'No'}")
            print(f"Available GPU Devices: {device_count}")
        else:
            print("PyTorch not installed")
        print()
        
        # Provide recommendations
        if not cuda_available:
            print("🔧 RECOMMENDATION: Install CUDA-enabled PyTorch")
            print("Run the following command to install GPU-enabled PyTorch:")
            print()
            print(get_recommended_torch_command())
            print()
            print("Alternative: Install from requirements-gpu.txt:")
            print("pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118")
        else:
            print("✅ GPU support is already configured!")
            print(f"Your system can use {device_count} GPU device(s)")
    else:
        print()
        print("❌ No NVIDIA GPU detected.")
        print("GPU acceleration requires an NVIDIA GPU with CUDA support.")
        print("The project will run on CPU only.")
    
    print()
    print("=" * 60)
    print("Setup check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()