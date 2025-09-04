#!/usr/bin/env python3
"""
Simplified Training Script for Image Generation Model

This script provides a simplified training approach using a basic CNN model
to demonstrate the training pipeline before moving to complex diffusion models.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import time

# Import our custom modules
from utils.text_processing import TextProcessor

# Define a simple CNN model for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        
        # Text encoder (simplified)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Image decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 1, 0),  # 1x1 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),     # 64x64 -> 128x128
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),      # 128x128 -> 256x256
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, text_features):
        # Encode text
        text_encoded = self.text_encoder(text_features)
        
        # Reshape for decoder
        text_encoded = text_encoded.view(-1, 128, 1, 1)
        
        # Generate image
        generated_image = self.decoder(text_encoded)
        
        return generated_image

# Dataset class (reuse from main training script)
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load captions based on split
        if split == 'train':
            caption_file = 'train_captions.json'
        elif split == 'test':
            caption_file = 'test_captions.json'
        else:
            caption_file = 'captions.json'
            
        caption_path = os.path.join(data_dir, 'processed', caption_file)
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                self.captions = json.load(f)
        else:
            # Fallback to main captions file
            with open(os.path.join(data_dir, 'processed/captions.json'), 'r') as f:
                self.captions = json.load(f)
        
        # Get list of image IDs
        self.image_ids = list(self.captions.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, f'processed/images/{image_id}.png')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get a random caption for this image
        captions = self.captions[image_id]['captions']
        caption = str(np.random.choice(captions))
        
        return image, caption

def simple_text_to_features(text, feature_dim=512):
    """
    Convert text to a simple feature vector.
    This is a placeholder - in practice you'd use a proper text encoder.
    """
    # Simple hash-based features (for demonstration)
    features = torch.zeros(feature_dim)
    for i, char in enumerate(text[:feature_dim]):
        features[i % feature_dim] += ord(char) / 255.0
    
    # Normalize
    features = features / (features.norm() + 1e-8)
    
    return features

def train_simple(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the simple model
    model = SimpleCNN().to(device)
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Smaller size for faster training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Set up data loading
    dataset = ImageTextDataset(args.data_dir, transform=transform, split='train')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print(f"Training dataset size: {len(dataset)} samples")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Training metrics
    training_losses = []
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for images, captions in progress_bar:
            # Convert captions to features
            text_features = torch.stack([simple_text_to_features(caption) for caption in captions])
            
            # Move data to device
            images = images.to(device)
            text_features = text_features.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            generated_images = model(text_features)
            
            # Resize target images to match generated images
            target_images = torch.nn.functional.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
            
            # Compute loss
            loss = criterion(generated_images, target_images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            loss_value = loss.item()
            epoch_loss += loss_value
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss_value:.4f}',
                'Avg Loss': f'{epoch_loss/batch_count:.4f}'
            })
        
        # Calculate epoch statistics
        avg_epoch_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_epoch_loss)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"  Estimated time remaining: {(elapsed_time/(epoch+1))*(args.num_epochs-epoch-1)/60:.1f} minutes")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"simple_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_epoch_loss
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "simple_final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'text_dim': 512},
        'training_losses': training_losses
    }, final_model_path)
    
    # Save training metrics
    final_metrics_path = os.path.join(args.output_dir, "simple_training_metrics.json")
    with open(final_metrics_path, 'w') as f:
        json.dump({
            'losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0,
            'total_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'total_training_time_minutes': (time.time() - start_time) / 60,
            'dataset_size': len(dataset)
        }, f, indent=2)
    
    print(f"\nSimple training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training metrics saved to: {final_metrics_path}")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")

def parse_args():
    parser = argparse.ArgumentParser(description="Train simple image generation model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_simple(args)