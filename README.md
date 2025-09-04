# Diffusion Model Image Generation

## Setup

This project has been set up with Python 3.10.0 and all required dependencies have been installed.

### GPU Acceleration (Recommended)

🚀 **For significantly faster image generation**, configure GPU support:

1. **Quick Check**: Run `python setup_gpu.py` to check your system
2. **Full Guide**: See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for detailed instructions
3. **Quick Install**: If you have an NVIDIA GPU:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

**Performance**: GPU acceleration reduces generation time from 2-5 minutes to 10-30 seconds per image!

## Running the Model

You can use the provided batch file to run Python commands without typing the full path:

```
run_python.bat examples\simple_generation.py
```

Or you can use the full Python path:

```
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" examples\simple_generation.py
```

## Project Structure

- `app.py` - Main application entry point
- `data/` - Dataset information and preparation
- `examples/` - Example scripts to demonstrate functionality
  - `simple_generation.py` - Simple image generation example
- `generate.py` - Script for generating images
- `models/` - Model definitions
  - `diffusion_model.py` - Diffusion model implementation
- `train.py` - Script for training the model
- `utils/` - Utility functions
  - `text_processing.py` - Text processing utilities

## Generated Images

Generated images are saved in the `output/` directory.

### Custom Generation

Use the enhanced generation script with custom prompts:

```bash
run_python.bat generate_custom.py --prompt "your custom prompt" --steps 30 --guidance 7.5
```

The script automatically uses GPU if available and includes enhanced text processing for better results.

## Training

To train the model, use:

```
run_python.bat train.py
```

## Evaluation

To evaluate the model, use:

```
run_python.bat evaluate.py
```