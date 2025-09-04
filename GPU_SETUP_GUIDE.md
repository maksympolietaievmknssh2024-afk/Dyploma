# GPU Setup Guide for Diffusion Model Project

This guide will help you configure your system to use GPU acceleration for faster image generation.

## Current Status

✅ **NVIDIA GPU Detected**: Your system has an NVIDIA GPU  
❌ **CUDA Not Installed**: CUDA drivers/toolkit not found  
❌ **CPU-only PyTorch**: Current PyTorch version (2.8.0+cpu) doesn't support GPU  

## Step-by-Step GPU Setup

### Step 1: Install CUDA Toolkit

1. **Download CUDA Toolkit**:
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 10/11 → exe (local)
   - Download and install CUDA Toolkit 11.8 or 12.1

2. **Verify Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Step 2: Install GPU-enabled PyTorch

**Option A: Quick Install (Recommended)**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B: Using requirements-gpu.txt**
```bash
pip uninstall torch torchvision torchaudio
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify GPU Setup

Run the setup check script:
```bash
python setup_gpu.py
```

Or test manually:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Performance Comparison

| Configuration | Typical Generation Time |
|---------------|------------------------|
| CPU Only      | 2-5 minutes per image  |
| GPU (CUDA)    | 10-30 seconds per image|

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or image resolution in the generation script.

### Issue: "CUDA driver version is insufficient"
**Solution**: Update your NVIDIA GPU drivers from https://www.nvidia.com/drivers

### Issue: PyTorch still uses CPU after installation
**Solution**: 
1. Restart your terminal/IDE
2. Verify CUDA installation with `nvidia-smi`
3. Reinstall PyTorch with the correct CUDA version

## Code Changes (Already Configured)

The project is already configured to automatically use GPU when available:

```python
# In generate_custom.py (line 26)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# In models/diffusion_model.py (line 15)
def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
    self.device = device
    # Models are automatically moved to GPU
    self.text_encoder.to(device)
    self.unet.to(device)
```

## Running with GPU

Once GPU is configured, simply run your generation script as usual:

```bash
python generate_custom.py --prompt "a beautiful landscape" --steps 50
```

The script will automatically detect and use your GPU for faster generation!

## Additional GPU Optimization Tips

1. **Use Mixed Precision**: Add `torch.cuda.amp` for even faster training
2. **Increase Batch Size**: Process multiple images simultaneously
3. **Monitor GPU Usage**: Use `nvidia-smi` to monitor GPU utilization
4. **Close Other GPU Applications**: Free up GPU memory for better performance

---

**Need Help?** Run `python setup_gpu.py` to check your current configuration and get specific recommendations.