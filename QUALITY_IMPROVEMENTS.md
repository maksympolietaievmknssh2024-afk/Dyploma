# Image Quality Improvements Guide

This document outlines the comprehensive quality improvements implemented in the diffusion model for enhanced image generation.

## 🚀 Key Quality Enhancements

### 1. Enhanced Generation Parameters

#### **Increased Default Steps**
- **Before**: 50 steps
- **After**: 75 steps (50% increase)
- **Impact**: More detailed denoising process, smoother gradients

#### **Improved Guidance Scale**
- **Before**: 7.5
- **After**: 10.0 (recommended range: 7.5-15.0)
- **Impact**: Better adherence to text prompts, more accurate generation

#### **Higher Resolution Support**
- **Before**: 512x512 pixels
- **After**: 768x768 pixels (default), supports up to 1024x1024
- **Impact**: More detailed images, better fine-grained features

### 2. Negative Prompting System

#### **Implementation**
```python
# Enhanced generation with negative prompting
image = model.generate_image(
    prompt="a beautiful landscape",
    negative_prompt="blurry, low quality, distorted, bad anatomy",
    num_inference_steps=75,
    guidance_scale=10.0
)
```

#### **Benefits**
- **Quality Control**: Actively avoid unwanted artifacts
- **Style Refinement**: Remove specific undesired elements
- **Consistency**: More predictable output quality

### 3. Advanced Sampling Improvements

#### **Proper Noise Initialization**
- Added `init_noise_sigma` scaling for better starting conditions
- Explicit `dtype=torch.float32` for numerical stability
- Device-aware scheduler timestep setting

#### **Enhanced Guidance Implementation**
- Clearer variable naming (`negative_embeddings`, `positive_embeddings`)
- Improved classifier-free guidance calculation
- Better separation of conditional and unconditional paths

### 4. Image Decoding Pipeline

#### **VAE-Style Decoding**
- Proper latent scaling (`1 / 0.18215`)
- Multi-channel handling for different latent dimensions
- Robust normalization to 0-255 range
- Direct PIL Image output

## 📊 Quality Presets

The `generate_hq.py` script includes predefined quality presets:

### Draft Quality
- **Steps**: 25
- **Guidance**: 7.5
- **Resolution**: 512x512
- **Use Case**: Quick previews, concept testing

### Standard Quality
- **Steps**: 50
- **Guidance**: 10.0
- **Resolution**: 768x768
- **Use Case**: General purpose generation

### High Quality (Default)
- **Steps**: 75
- **Guidance**: 12.0
- **Resolution**: 768x768
- **Use Case**: Production-ready images

### Ultra Quality
- **Steps**: 100
- **Guidance**: 15.0
- **Resolution**: 1024x1024
- **Use Case**: Maximum quality, detailed artwork

## 🛠️ Usage Examples

### Basic Enhanced Generation
```bash
python generate.py --prompt "a majestic mountain landscape" \
                   --negative_prompt "blurry, low quality" \
                   --steps 75 \
                   --guidance_scale 10.0 \
                   --height 768 \
                   --width 768
```

### High-Quality Preset Generation
```bash
python generate_hq.py --prompt "a detailed fantasy castle" \
                       --quality high \
                       --negative_prompt "blurry, distorted, bad anatomy"
```

### Ultra Quality with Custom Settings
```bash
python generate_hq.py --prompt "intricate steampunk machinery" \
                       --quality ultra \
                       --steps 120 \
                       --guidance_scale 15.0
```

## 🎯 Quality Optimization Tips

### For Better Detail
1. **Increase Steps**: 75-150 for high detail
2. **Higher Resolution**: 768x768 or 1024x1024
3. **Detailed Prompts**: Include specific descriptive terms

### For Style Control
1. **Negative Prompting**: Specify what to avoid
2. **Guidance Scale**: 10-15 for strong prompt adherence
3. **Style Keywords**: Add "detailed", "high quality", "masterpiece"

### For Consistency
1. **Fixed Seeds**: Use same random seed for reproducible results
2. **Deterministic Sampling**: Keep eta=0.0
3. **Consistent Parameters**: Use same settings for series

## 📈 Performance Considerations

### Memory Usage
- **768x768**: ~4GB VRAM recommended
- **1024x1024**: ~6GB VRAM recommended
- **CPU Fallback**: Available for systems without sufficient GPU memory

### Generation Time
- **50 steps**: ~30-60 seconds (GPU) / 5-10 minutes (CPU)
- **75 steps**: ~45-90 seconds (GPU) / 8-15 minutes (CPU)
- **100+ steps**: ~60-120 seconds (GPU) / 10-20 minutes (CPU)

## 🔧 Technical Improvements

### Code Quality
- Fixed deprecation warnings (`unet.config.in_channels`)
- Enhanced error handling and validation
- Improved documentation and type hints
- Better separation of concerns

### Architecture
- Modular quality preset system
- Flexible parameter override options
- Extensible negative prompting framework
- Robust image decoding pipeline

## 🎨 Recommended Workflows

### Concept Development
1. Start with **draft** quality for rapid iteration
2. Refine prompts and negative prompts
3. Move to **standard** quality for evaluation
4. Use **high** quality for final generation

### Production Pipeline
1. Use **high** or **ultra** quality presets
2. Implement consistent negative prompting
3. Maintain quality logs for reproducibility
4. Consider batch processing for efficiency

---

*These improvements provide a significant enhancement to image generation quality while maintaining flexibility and ease of use.*