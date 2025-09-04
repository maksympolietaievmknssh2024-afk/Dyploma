import os
import sys
import torch
from PIL import Image
import argparse

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate images from text prompts')
    parser.add_argument('--prompt', type=str, default="a flying saucer in the night sky",
                        help='Text prompt for image generation')
    parser.add_argument('--output', type=str, default="output/generated_image.png",
                        help='Output file path')
    parser.add_argument('--steps', type=int, default=30,
                        help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='Guidance scale')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ImageGenerationModel(device=device)
    
    # Initialize the text processor
    text_processor = TextProcessor()
    
    # Process the prompt to enhance context understanding
    enhanced_prompt = text_processor.enhance_prompt(args.prompt)
    
    # Print the original and enhanced prompts
    print(f"Original: {args.prompt}")
    print(f"Enhanced: {enhanced_prompt}")
    print()
    
    # Generate image
    print(f"Generating image for: {enhanced_prompt}")
    latents = model.generate_image(
        enhanced_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    )
    
    # Normalize latents for visualization
    latents_normalized = (latents - latents.min()) / (latents.max() - latents.min())
    
    # Convert to PIL Image
    image = Image.fromarray(
        (latents_normalized[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    )
    
    # Save the image
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()