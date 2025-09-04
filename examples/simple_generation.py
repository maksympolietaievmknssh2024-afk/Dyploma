import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor
from utils.visualization import visualize_generated_images

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ImageGenerationModel(device=device)
    
    # Initialize the text processor
    text_processor = TextProcessor()
    
    # Define some example prompts
    prompts = [
        "a flying saucer in the night sky",
        "a bat flying in a cave",
        "a baseball player swinging a bat",
        "a mouse on a computer desk",
        "a small mouse eating cheese"
    ]
    
    # Process the prompts to enhance context understanding
    enhanced_prompts = [text_processor.enhance_prompt(prompt) for prompt in prompts]
    
    # Print the original and enhanced prompts
    print("Original vs. Enhanced Prompts:")
    for original, enhanced in zip(prompts, enhanced_prompts):
        print(f"Original: {original}")
        print(f"Enhanced: {enhanced}")
        print()
    
    # Generate images for the first prompt as an example
    print(f"Generating image for: {enhanced_prompts[0]}")
    latents = model.generate_image(
        enhanced_prompts[0],
        num_inference_steps=30,  # Fewer steps for faster generation
        guidance_scale=7.5
    )
    
    # Normalize latents for visualization
    latents_normalized = (latents - latents.min()) / (latents.max() - latents.min())
    
    # Convert to PIL Image
    # This is a placeholder - in a real implementation, we would use a proper decoder
    image = Image.fromarray(
        (latents_normalized[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    )
    
    # Save the image
    os.makedirs("output", exist_ok=True)
    image.save("output/example_generation.png")
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(enhanced_prompts[0])
    plt.axis('off')
    plt.show()
    
    print(f"Image saved to output/example_generation.png")

if __name__ == "__main__":
    main()