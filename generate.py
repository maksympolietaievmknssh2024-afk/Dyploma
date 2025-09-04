import os
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

def generate_image(args):
    # Set device (force CPU to avoid CUDA memory issues)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ImageGenerationModel(device=device)
    
    # Load the trained model if specified
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model.load_model(args.model_path)
    
    # Initialize the text processor
    text_processor = TextProcessor()
    
    # Process the prompt to enhance context understanding
    enhanced_prompt = text_processor.enhance_prompt(args.prompt)
    print(f"Original prompt: {args.prompt}")
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    # Extract objects from the prompt
    objects = text_processor.extract_objects_from_prompt(args.prompt)
    print(f"Detected objects: {', '.join(objects)}")
    
    # Generate the image with enhanced quality settings
    print("Generating high-quality image...")
    print(f"Settings: {args.steps} steps, guidance scale {args.guidance_scale}, resolution {args.width}x{args.height}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    
    image = model.generate_image(
        enhanced_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
        eta=args.eta
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the image (now a PIL Image)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    image.save(output_path)
    print(f"High-quality image saved to {output_path}")
    
    # Display the image if requested
    if args.display:
        image.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate high-quality images from text prompts")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt - what to avoid in generation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated images")
    parser.add_argument("--output_name", type=str, default="generated", help="Name for the output image file")
    parser.add_argument("--steps", type=int, default=75, help="Number of inference steps (higher = better quality, 50-150 recommended)")
    parser.add_argument("--guidance_scale", type=float, default=10.0, help="Scale for classifier-free guidance (7.5-15 recommended)")
    parser.add_argument("--height", type=int, default=768, help="Height of the generated image (multiple of 64)")
    parser.add_argument("--width", type=int, default=768, help="Width of the generated image (multiple of 64)")
    parser.add_argument("--eta", type=float, default=0.0, help="Amount of noise during sampling (0.0=deterministic, 0.1-1.0=creative)")
    parser.add_argument("--display", action="store_true", help="Display the generated image")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_image(args)