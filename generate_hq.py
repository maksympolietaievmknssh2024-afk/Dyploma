#!/usr/bin/env python3
"""
High-Quality Image Generation Script
Demonstrates enhanced image generation with quality improvements.
"""

import argparse
import os
import sys
import torch
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

def generate_hq_image(args):
    """
    Generate high-quality images with optimized settings.
    """
    # Set device
    device = "cpu"  # Using CPU to avoid CUDA memory issues
    print(f"Using device: {device}")
    
    # Initialize model
    model = ImageGenerationModel(device=device)
    
    # Load trained model if available
    if args.model_path and os.path.exists(args.model_path):
        model.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Using base model (no custom training loaded)")
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Enhance the prompt
    enhanced_prompt = text_processor.enhance_prompt(args.prompt)
    print(f"Original prompt: {args.prompt}")
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    # Quality presets
    quality_presets = {
        'draft': {'steps': 25, 'guidance': 7.5, 'resolution': 512},
        'standard': {'steps': 50, 'guidance': 10.0, 'resolution': 768},
        'high': {'steps': 75, 'guidance': 12.0, 'resolution': 768},
        'ultra': {'steps': 100, 'guidance': 15.0, 'resolution': 1024}
    }
    
    preset = quality_presets.get(args.quality, quality_presets['high'])
    
    # Override with custom settings if provided
    steps = args.steps if args.steps else preset['steps']
    guidance_scale = args.guidance_scale if args.guidance_scale else preset['guidance']
    resolution = args.resolution if args.resolution else preset['resolution']
    
    print(f"Quality preset: {args.quality}")
    print(f"Generation settings: {steps} steps, guidance {guidance_scale}, {resolution}x{resolution}")
    
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    
    # Generate the image
    print("Generating high-quality image...")
    image = model.generate_image(
        enhanced_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=resolution,
        width=resolution,
        negative_prompt=args.negative_prompt,
        eta=args.eta
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename with quality info
    filename = f"{args.output_name}_{args.quality}_{steps}steps_{resolution}px.png"
    output_path = os.path.join(args.output_dir, filename)
    
    # Save the image
    image.save(output_path)
    print(f"High-quality image saved to {output_path}")
    
    # Display if requested
    if args.display:
        image.show()
    
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate high-quality images with preset configurations")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, ugly, bad anatomy", 
                       help="Negative prompt for quality control")
    parser.add_argument("--quality", type=str, choices=['draft', 'standard', 'high', 'ultra'], default='high',
                       help="Quality preset (draft/standard/high/ultra)")
    parser.add_argument("--model_path", type=str, default="./output/final_model.pt", help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated images")
    parser.add_argument("--output_name", type=str, default="hq_generated", help="Base name for output file")
    
    # Override options
    parser.add_argument("--steps", type=int, help="Override steps from preset")
    parser.add_argument("--guidance_scale", type=float, help="Override guidance scale from preset")
    parser.add_argument("--resolution", type=int, help="Override resolution from preset")
    parser.add_argument("--eta", type=float, default=0.0, help="Sampling noise (0.0=deterministic)")
    parser.add_argument("--display", action="store_true", help="Display the generated image")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_hq_image(args)