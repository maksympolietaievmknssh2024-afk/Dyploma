# Image Generation System

Text-to-image generation with PyTorch.

## Overview (Updated)

This project provides image generation with diffusion models, with an emphasis on accurate reproduction of highlighted objects, batch runs, and reproducible results (fixed `seed`, metadata logging to `outputs/web/*.meta.json`). Below are the current commands, structure, and tools for comparison/evaluation.

## Requirements and GPU

- Python `3.10`
- NVIDIA CUDA (if GPU is available), up-to-date drivers
- Install dependencies:

```bash
setup_gpu.bat   # (optional) GPU check/setup
pip install -r requirements.txt
# or for a GPU-focused environment
pip install -r requirements-gpu.txt
```

## Generation (Current)

Basic CLI generation:

```bash
python generate.py --prompt "your prompt" \
  --negative_prompt "optional negative" \
  --steps 30 --guidance_scale 7.5 \
  --height 512 --width 512 \
  --seed 42 \
  --output_dir outputs/web \
  --output_name demo
```

Windows helper script:

```bash
run_python.bat generate.py --prompt "your prompt" --output_dir outputs/web --output_name demo
```

Outputs are saved to `outputs/web/` (PNG/JSON). Metadata is logged to `*.meta.json` for traceability.

## Batch Runs and Offline Mode

- Batch generation: `scripts/run_batch_generation.py`
- Offline generator calls: `scripts/offline_call_generate.py`
- Payload examples: `scripts/payloads/*.json` (including `one_prompt_camera*.json`, `batch_*`)

Example: run CLIPScore evaluation on web metadata:

```bash
python scripts/evaluate_quality.py --meta_dir outputs/web \
  --out outputs/evaluation/clipscore_report.json --print_top 5
```

## Model Comparison and Summaries

- Comparison under identical prompts/settings: `scripts/compare_models.py`
- Build visual grids from a report: `scripts/build_grids_from_report.py`
- Summaries/aggregated findings: `scripts/summarize_compare.py`
- Reports/diagrams: `outputs/compare/*`, `outputs/diagrams/*`, `outputs/evaluation/*`

Recommendation: fix `seed`, keep identical `steps`, `guidance_scale`, resolution, and sampler (e.g., DPM-Solver) to ensure fair comparisons.

## Project Structure (Updated)

- `src/` — core modules
  - `generation/image_generator.py` — image generation
  - `training/model_trainer.py`, `training/auto_retraining_system.py` — training/retraining
  - `datasets/dataset_processor.py`, `datasets/enhanced_dataset_generator.py` — dataset processing
  - `evaluation/clip_score.py` — CLIPScore computation
  - `main.py` — CLI entry point
- `scripts/` — utilities for comparison/evaluation/batch runs
- `outputs/` — generation results, comparisons, metrics
- `user_data/` — local cache of models/user data

## Training

Use the updated training scripts:

- `train_enhanced.py` — enhanced training pipeline
- `train_optimized_gpu.py` — GPU-optimized training pipeline

Run examples:

```bash
python train_enhanced.py --help
python train_optimized_gpu.py --help
```

## Recommended Defaults for Generation

For balanced quality without extra tuning, recommended defaults:

- `--negative_preset`: `standard`
- `--steps`: `80` (sweet spot 60–90)
- `--guidance_scale`: `9.5`
- `--guidance_rescale`: `0.3`

Example:

```bash
python generate.py --prompt "portrait, soft lighting, 85mm" --output_name demo
```

Tip: enable prompt diagnostics via `--debug_prompt` for structured parsing and weights.

## Objective Quality Metric (CLIPScore)

Compute CLIPScore to evaluate semantic consistency between prompt and image and obtain aggregate statistics.

```bash
python scripts/evaluate_quality.py --meta_dir outputs/web --out outputs/evaluation/clipscore_report.json --print_top 5
```

Report includes:
- `count`: number of examples
- `mean`, `median`, `std`, `min`, `max`: CLIPScore statistics
- `above_thresholds`: percentage of images above thresholds (8/12/16/20)
- `scores`: per-sample values for traceability

Use identical prompts and generation settings to compare under the same CLIP variant (`openai/clip-vit-base-patch32`).

## Project Layout

```
src/
├── generation/           # Image generation module
│   ├── __init__.py
│   └── image_generator.py
├── training/             # Model training module
│   ├── __init__.py
│   └── model_trainer.py
├── datasets/             # Dataset processing module
│   ├── __init__.py
│   └── dataset_processor.py
├── utils/                # Project utilities
│   ├── __init__.py
│   └── text_processing.py
├── __init__.py
└── main.py               # CLI entry point
```

## Usage (CLI entry)

```bash
python src/main.py generate --prompt "your text prompt"
python src/main.py train --dataset "path/to/dataset"
```

## Setup

Requires Python 3.10.

```bash
setup_gpu.bat  # For NVIDIA GPU
pip install -r requirements.txt
```

## Notes

- Generated assets are saved under `outputs/` (including `outputs/web/`).
- Metadata files `*.meta.json` support reproducibility and evaluation pipelines.
- Prefer `generate.py`, `train_enhanced.py`, and `train_optimized_gpu.py` over older scripts.

- `--negative_preset`: `standard` (common artifact suppression)
- `--steps`: `80` (quality sweet spot 60–90)
- `--guidance_scale`: `9.5` (strong, but stable)
- `--guidance_rescale`: `0.3` (stabilizes high guidance)

Example:

```
python generate.py --prompt "portrait, soft lighting, 85mm" --output_name demo
```

Tip: enable prompt diagnostics with `--debug_prompt` to see structured parsing and phrase weights.

## Objective Quality Metric (CLIPScore)

You can compute an objective, numeric quality metric based on CLIPScore for generated images. This evaluates semantic alignment between prompts and images and produces aggregated statistics (mean, median, std, min/max, and percentage above thresholds).

Run the evaluator on previously saved web outputs:

```
python scripts/evaluate_quality.py --meta_dir outputs/web --out outputs/evaluation/clipscore_report.json --print_top 5
```

The report includes:
- `count`: number of evaluated samples
- `mean`, `median`, `std`, `min`, `max`: CLIPScore statistics
- `above_thresholds`: percentage of images above fixed thresholds (8/12/16/20)
- `scores`: per-image scores for traceability

Use the resulting numbers to compare different models under the same prompts and CLIP variant (`openai/clip-vit-base-patch32`). Ensure identical generation presets and seeds for fair comparison.