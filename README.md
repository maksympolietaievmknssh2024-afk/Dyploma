# Diffusion Model

Text-to-image generation with PyTorch.

## Setup

Requires Python 3.10.0.

```bash
setup_gpu.bat  # For NVIDIA GPU
pip install -r requirements.txt
```

## Usage

```bash
run_python.bat generate.py     # Basic generation
run_python.bat generate_hq.py  # High quality
```

## Structure

- `generate.py` - Basic image generation
- `generate_hq.py` - High quality generation
- `train.py` - Model training
- `models/diffusion_model.py` - Model implementation
- `output/` - Generated images

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