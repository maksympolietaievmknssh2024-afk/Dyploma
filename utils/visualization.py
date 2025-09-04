import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid

def visualize_generated_images(images, prompts, output_path=None, figsize=(15, 10)):
    """
    Visualize a batch of generated images with their corresponding prompts.
    
    Args:
        images: List of PIL images or tensor of shape (B, C, H, W)
        prompts: List of text prompts used to generate the images
        output_path: Path to save the visualization (optional)
        figsize: Figure size (width, height) in inches
    """
    # Convert tensor to list of PIL images if needed
    if isinstance(images, torch.Tensor):
        images = tensor_to_pil(images)
    
    # Create the figure
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 1, figsize=figsize)
    
    # Handle the case of a single image
    if n_images == 1:
        axes = [axes]
    
    # Plot each image with its prompt
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        axes[i].imshow(np.array(img))
        axes[i].set_title(prompt, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()

def tensor_to_pil(tensor):
    """
    Convert a tensor of shape (B, C, H, W) to a list of PIL images.
    
    Args:
        tensor: Tensor of shape (B, C, H, W)
        
    Returns:
        List of PIL images
    """
    # Ensure the tensor is on CPU and detached from the computation graph
    tensor = tensor.cpu().detach()
    
    # Normalize to [0, 1] if needed
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Convert to numpy arrays and then to PIL images
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    
    return images

def visualize_attention(image, attention_map, prompt, output_path=None, figsize=(12, 5)):
    """
    Visualize the attention map overlaid on the generated image.
    
    Args:
        image: PIL image or tensor of shape (C, H, W)
        attention_map: Tensor of shape (H, W) representing the attention map
        prompt: Text prompt used to generate the image
        output_path: Path to save the visualization (optional)
        figsize: Figure size (width, height) in inches
    """
    # Convert tensor to PIL image if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image.unsqueeze(0))[0]
    
    # Convert attention map to numpy array
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().detach().numpy()
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot the original image
    axes[0].imshow(np.array(image))
    axes[0].set_title("Generated Image", fontsize=12)
    axes[0].axis('off')
    
    # Plot the attention map
    im = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title("Attention Map", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot the attention map overlaid on the image
    img_np = np.array(image)
    attention_map_resized = np.resize(attention_map, (img_np.shape[0], img_np.shape[1]))
    attention_map_rgb = plt.cm.jet(attention_map_resized)[:, :, :3]
    overlay = img_np * 0.7 + attention_map_rgb * 255 * 0.3
    overlay = overlay.astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis('off')
    
    # Add the prompt as a subtitle
    plt.suptitle(f"Prompt: {prompt}", fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()

def visualize_evaluation_results(results, output_dir):
    """
    Visualize the evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save the visualizations
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by correctness
    correct_results = [r for r in results if r["correct"]]
    incorrect_results = [r for r in results if not r["correct"]]
    
    # Create a summary figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Correct", "Incorrect"], [len(correct_results), len(incorrect_results)])
    ax.set_ylabel("Number of Test Cases")
    ax.set_title("Evaluation Results")
    
    # Add accuracy as text
    accuracy = len(correct_results) / len(results) if results else 0
    ax.text(0.5, 0.9, f"Accuracy: {accuracy:.2f}", transform=ax.transAxes, ha="center")
    
    # Save the summary figure
    plt.savefig(os.path.join(output_dir, "evaluation_summary.png"), bbox_inches='tight')
    
    # Visualize individual results
    for i, result in enumerate(results):
        # Load the generated image
        image = Image.open(result["image_path"])
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.array(image))
        ax.axis('off')
        
        # Add information as text
        status = "✓ Correct" if result["correct"] else "✗ Incorrect"
        info_text = f"Prompt: {result['prompt']}\n" \
                   f"Enhanced: {result['enhanced_prompt']}\n" \
                   f"Detected: {', '.join(result['detected_objects'])}\n" \
                   f"Expected: {', '.join(result['expected_objects'])}\n" \
                   f"Status: {status}"
        
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"result_{i}.png"), bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")