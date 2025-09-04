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
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

# Define a custom dataset class
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
        caption = str(np.random.choice(captions))  # Convert to regular string
        
        return image, caption

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ImageGenerationModel(device=device)
    
    # Initialize the text processor
    text_processor = TextProcessor()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
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
    
    # Set up optimizer
    optimizer = optim.AdamW(model.unet.parameters(), lr=args.learning_rate)
    
    # Training metrics
    training_losses = []
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        model.unet.train()
        epoch_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for images, captions in progress_bar:
            # Process captions to enhance context understanding
            enhanced_captions = [text_processor.enhance_prompt(caption) for caption in captions]
            
            # Move data to device
            images = images.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass - compute diffusion loss
            try:
                loss = model.compute_loss(images, enhanced_captions)
            except Exception as e:
                print(f"Error computing loss: {e}")
                continue
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
            
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
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            model.save_model(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # Save training metrics
            metrics_path = os.path.join(args.output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump({
                    'losses': training_losses,
                    'epochs_completed': epoch + 1,
                    'total_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size
                }, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    model.save_model(final_model_path)
    
    # Save final training metrics
    final_metrics_path = os.path.join(args.output_dir, "final_training_metrics.json")
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
    
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training metrics saved to: {final_metrics_path}")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")

def parse_args():
    parser = argparse.ArgumentParser(description="Train image generation model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)