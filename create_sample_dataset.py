#!/usr/bin/env python3
"""
Create Sample Dataset for Training and Testing

This script generates a small synthetic dataset with simple colored images
and corresponding text descriptions for training and testing the diffusion model.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_simple_image(color, shape, size=(512, 512)):
    """
    Create a simple colored image with a basic shape.
    
    Args:
        color: RGB color tuple
        shape: Shape type ('circle', 'square', 'triangle')
        size: Image size tuple
    
    Returns:
        PIL Image
    """
    img = Image.new('RGB', size, color=(240, 240, 240))  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Calculate shape dimensions
    center_x, center_y = size[0] // 2, size[1] // 2
    shape_size = min(size) // 3
    
    if shape == 'circle':
        left = center_x - shape_size
        top = center_y - shape_size
        right = center_x + shape_size
        bottom = center_y + shape_size
        draw.ellipse([left, top, right, bottom], fill=color)
    
    elif shape == 'square':
        left = center_x - shape_size
        top = center_y - shape_size
        right = center_x + shape_size
        bottom = center_y + shape_size
        draw.rectangle([left, top, right, bottom], fill=color)
    
    elif shape == 'triangle':
        points = [
            (center_x, center_y - shape_size),  # Top
            (center_x - shape_size, center_y + shape_size),  # Bottom left
            (center_x + shape_size, center_y + shape_size)   # Bottom right
        ]
        draw.polygon(points, fill=color)
    
    return img

def generate_sample_dataset(output_dir, num_samples=100):
    """
    Generate a sample dataset with simple images and captions.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'processed', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Define colors and shapes
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42)
    }
    
    shapes = ['circle', 'square', 'triangle']
    
    # Generate captions data
    captions_data = {}
    
    print(f"Generating {num_samples} sample images...")
    
    for i in range(num_samples):
        # Randomly select color and shape
        color_name = random.choice(list(colors.keys()))
        color_rgb = colors[color_name]
        shape = random.choice(shapes)
        
        # Create image
        img = create_simple_image(color_rgb, shape)
        
        # Save image
        image_id = f"{i:06d}"
        image_path = os.path.join(images_dir, f"{image_id}.png")
        img.save(image_path)
        
        # Generate captions
        base_captions = [
            f"a {color_name} {shape}",
            f"{color_name} {shape} on gray background",
            f"simple {color_name} {shape} drawing",
            f"geometric {shape} in {color_name} color"
        ]
        
        # Add some variation to captions
        captions = []
        for caption in base_captions:
            captions.append(caption)
            if random.random() > 0.5:  # 50% chance to add variation
                if 'simple' not in caption:
                    captions.append(f"simple {caption}")
                if 'bright' not in caption and color_name in ['red', 'blue', 'green', 'yellow']:
                    captions.append(caption.replace(color_name, f"bright {color_name}"))
        
        # Store caption data
        captions_data[image_id] = {
            "captions": captions[:5],  # Limit to 5 captions per image
            "metadata": {
                "color": color_name,
                "shape": shape,
                "width": 512,
                "height": 512,
                "file_name": f"{image_id}.png"
            }
        }
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_samples} images")
    
    # Save captions file
    captions_path = os.path.join(output_dir, 'processed', 'captions.json')
    with open(captions_path, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    # Create metadata file
    metadata = {
        "dataset_name": "synthetic_shapes",
        "num_samples": num_samples,
        "image_size": [512, 512],
        "colors": list(colors.keys()),
        "shapes": shapes,
        "description": "Synthetic dataset with simple colored shapes for training and testing"
    }
    
    metadata_path = os.path.join(output_dir, 'processed', 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Generated {num_samples} images in {images_dir}")
    print(f"Captions saved to {captions_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return captions_data

def create_train_test_split(captions_data, output_dir, test_ratio=0.2):
    """
    Create train/test split for the dataset.
    
    Args:
        captions_data: Dictionary with caption data
        output_dir: Output directory
        test_ratio: Ratio of data to use for testing
    """
    image_ids = list(captions_data.keys())
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * (1 - test_ratio))
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:]
    
    # Create train data
    train_data = {img_id: captions_data[img_id] for img_id in train_ids}
    train_path = os.path.join(output_dir, 'processed', 'train_captions.json')
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Create test data
    test_data = {img_id: captions_data[img_id] for img_id in test_ids}
    test_path = os.path.join(output_dir, 'processed', 'test_captions.json')
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nTrain/Test split created:")
    print(f"Training samples: {len(train_ids)} ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"Testing samples: {len(test_ids)} ({len(test_ids)/len(image_ids)*100:.1f}%)")
    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")

def main():
    output_dir = './data'
    num_samples = 200  # Small dataset for quick training
    
    print("Creating sample dataset for diffusion model training...")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print()
    
    # Generate the dataset
    captions_data = generate_sample_dataset(output_dir, num_samples)
    
    # Create train/test split
    create_train_test_split(captions_data, output_dir)
    
    print("\n" + "="*60)
    print("Sample dataset creation complete!")
    print("You can now run training with:")
    print("python train.py --data_dir ./data --batch_size 2 --num_epochs 10")
    print("="*60)

if __name__ == "__main__":
    main()