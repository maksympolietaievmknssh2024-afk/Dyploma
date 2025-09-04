#!/usr/bin/env python3
"""
Simple script to generate a yellow car on a gray road.
"""

import os
import sys
import torch
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

def generate_yellow_car():
    """
    Generate a yellow car on a gray road.
    """
    # Set device to CPU to avoid CUDA issues
    device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = ImageGenerationModel(device=device)
    
    # Load trained model if available
    model_path = "./output/final_model.pt"
    if os.path.exists(model_path):
        model.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Using base model (no custom training loaded)")
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Define the prompt
    prompt = "a bright yellow car driving on a gray asphalt road, realistic, detailed"
    negative_prompt = "blurry, low quality, distorted, bad anatomy"
    
    # Enhance the prompt
    enhanced_prompt = text_processor.enhance_prompt(prompt)
    print(f"Original prompt: {prompt}")
    print(f"Enhanced prompt: {enhanced_prompt}")
    print(f"Negative prompt: {negative_prompt}")
    
    # Generate the image with quality settings
    print("Generating yellow car image...")
    print("Settings: 50 steps, guidance scale 10.0, resolution 512x512")
    
    image = model.generate_image(
        enhanced_prompt,
        num_inference_steps=50,
        guidance_scale=10.0,
        height=512,
        width=512,
        negative_prompt=negative_prompt
    )
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Save the image
    output_path = "./output/yellow_car_gray_road.png"
    image.save(output_path)
    print(f"Yellow car image saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    try:
        output_path = generate_yellow_car()
        print(f"\nSuccess! Generated image saved at: {output_path}")
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()