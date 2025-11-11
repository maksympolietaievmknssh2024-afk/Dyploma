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
    # Set device (use GPU if available for faster generation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU cache if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
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
    
    # Optional debugging: show structured parse and phrase weights
    if getattr(args, 'debug_prompt', False):
        try:
            structure = text_processor.parse_prompt_structure(args.prompt)
            enhanced_with_weights, weight_map = text_processor.enhance_prompt_with_weights(args.prompt)
            print("\n[debug_prompt] Structured parse:")
            for k, v in structure.items():
                print(f"  - {k}: {v}")
            print("[debug_prompt] Phrase weights:")
            if weight_map:
                for phrase, w in weight_map.items():
                    print(f"  - '{phrase}': {w}")
            else:
                print("  (no weights detected)")
            if enhanced_with_weights != enhanced_prompt:
                print(f"[debug_prompt] Enhanced with weights: {enhanced_with_weights}")
        except Exception as e:
            print(f"[debug_prompt] Failed to parse/weight prompt: {e}")
    
    # Extract objects from the prompt
    objects = text_processor.extract_objects_from_prompt(args.prompt)
    print(f"Detected objects: {', '.join(objects)}")
    
    # Generate the image with enhanced quality settings
    # Negative presets temporarily disabled; use only user-supplied negative prompt
    combined_negative = args.negative_prompt
    print("Generating high-quality image...")
    print(f"Settings: {args.steps} steps, guidance scale {args.guidance_scale}, guidance rescale {args.guidance_rescale}, resolution {args.width}x{args.height}")
    if combined_negative:
        print(f"Negative prompt (final): {combined_negative}")
    else:
        print("Negative prompt: (none)")

    # Optional: override embedding weights from CLI
    if args.embedding_weights:
        try:
            weights = [float(x) for x in args.embedding_weights.split(',')]
            if len(weights) != 3:
                print("⚠️ --embedding_weights must have exactly 3 comma-separated values (CLIP, semantic, positional)")
            else:
                print(f"Embedding weights override: CLIP={weights[0]}, semantic={weights[1]}, positional={weights[2]}")
                if hasattr(model, 'set_embedding_weights_override'):
                    model.set_embedding_weights_override(weights)
        except Exception as e:
            print(f"⚠️ Failed to parse --embedding_weights: {e}")
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
    else:
        print("Using random seed")
    
    image = model.generate_image(
        enhanced_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        negative_prompt=combined_negative,
        eta=args.eta,
        seed=args.seed,
        guidance_rescale=args.guidance_rescale
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
    parser.add_argument("--negative_preset", type=str, default="none", choices=["none", "draft", "standard", "high", "ultra"],
                        help="(Disabled) Negative presets are temporarily turned off; provide --negative_prompt manually")
    parser.add_argument("--embedding_weights", type=str, default=None,
                        help="Override embedding weights as 'clip,semantic,positional' (e.g., 1.0,0.8,0.6)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated images")
    parser.add_argument("--output_name", type=str, default="generated", help="Name for the output image file")
    parser.add_argument("--steps", type=int, default=80, help="Number of inference steps (higher = better quality, 60-90 recommended)")
    parser.add_argument("--guidance_scale", type=float, default=9.5, help="Scale for classifier-free guidance (8.5-10.5 recommended)")
    parser.add_argument("--height", type=int, default=768, help="Height of the generated image (multiple of 64)")
    parser.add_argument("--width", type=int, default=768, help="Width of the generated image (multiple of 64)")
    parser.add_argument("--eta", type=float, default=0.0, help="Amount of noise during sampling (0.0=deterministic, 0.1-1.0=creative)")
    parser.add_argument("--guidance_rescale", type=float, default=0.3, help="Rescale CFG output to stabilize high guidance (0.3-0.7 recommended)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation (optional)")
    parser.add_argument("--display", action="store_true", help="Display the generated image")
    parser.add_argument("--debug_prompt", action="store_true", help="Log structured parsing and phrase weights for the prompt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_image(args)