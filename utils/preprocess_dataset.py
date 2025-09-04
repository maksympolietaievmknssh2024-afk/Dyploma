import os
import argparse
import json
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def preprocess_coco(data_dir, output_dir):
    """
    Preprocess the MS-COCO dataset.
    
    Args:
        data_dir: Directory containing the raw COCO dataset
        output_dir: Directory to save the processed dataset
    """
    print("Preprocessing MS-COCO dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load annotations
    annotations_file = os.path.join(data_dir, "annotations/captions_train2017.json")
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    # Process images and captions
    captions = {}
    for annotation in tqdm(annotations["annotations"], desc="Processing annotations"):
        image_id = str(annotation["image_id"]).zfill(12)
        caption = annotation["caption"]
        
        if image_id not in captions:
            captions[image_id] = {"captions": [], "metadata": {}}
        
        captions[image_id]["captions"].append(caption)
    
    # Copy and resize images
    for image_info in tqdm(annotations["images"], desc="Processing images"):
        image_id = str(image_info["id"]).zfill(12)
        
        # Skip if we don't have captions for this image
        if image_id not in captions:
            continue
        
        # Add metadata
        captions[image_id]["metadata"] = {
            "width": image_info["width"],
            "height": image_info["height"],
            "file_name": image_info["file_name"]
        }
        
        # Copy and resize the image
        src_path = os.path.join(data_dir, f"train2017/{image_id}.jpg")
        dst_path = os.path.join(output_dir, f"images/{image_id}.jpg")
        
        if os.path.exists(src_path):
            try:
                img = Image.open(src_path)
                
                # Resize to a standard size
                img = img.resize((512, 512), Image.LANCZOS)
                
                # Save the processed image
                img.save(dst_path)
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
    
    # Save the captions
    with open(os.path.join(output_dir, "captions.json"), "w") as f:
        json.dump(captions, f)
    
    print(f"Preprocessing complete. Processed {len(captions)} images.")

def preprocess_laion(data_dir, output_dir, sample_size=10000):
    """
    Preprocess a sample of the LAION dataset.
    
    Args:
        data_dir: Directory containing the raw LAION dataset
        output_dir: Directory to save the processed dataset
        sample_size: Number of samples to process
    """
    print("Preprocessing LAION dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load metadata
    metadata_file = os.path.join(data_dir, "metadata.jsonl")
    
    # Process images and captions
    captions = {}
    processed_count = 0
    
    with open(metadata_file, "r") as f:
        for line in tqdm(f, desc="Processing samples"):
            if processed_count >= sample_size:
                break
            
            try:
                sample = json.loads(line)
                image_id = str(processed_count).zfill(8)
                caption = sample["caption"]
                image_url = sample["url"]
                
                # Download the image
                try:
                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    
                    # Resize to a standard size
                    img = img.resize((512, 512), Image.LANCZOS)
                    
                    # Save the processed image
                    img.save(os.path.join(output_dir, f"images/{image_id}.jpg"))
                    
                    # Store the caption
                    captions[image_id] = {
                        "captions": [caption],
                        "metadata": {
                            "width": img.width,
                            "height": img.height,
                            "url": image_url
                        }
                    }
                    
                    processed_count += 1
                except Exception as e:
                    print(f"Error downloading image {image_url}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Save the captions
    with open(os.path.join(output_dir, "captions.json"), "w") as f:
        json.dump(captions, f)
    
    print(f"Preprocessing complete. Processed {len(captions)} images.")

def preprocess_conceptual_captions(data_dir, output_dir):
    """
    Preprocess the Conceptual Captions dataset.
    
    Args:
        data_dir: Directory containing the raw Conceptual Captions dataset
        output_dir: Directory to save the processed dataset
    """
    print("Preprocessing Conceptual Captions dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load TSV files
    train_file = os.path.join(data_dir, "Train_GCC-training.tsv")
    
    # Process images and captions
    captions = {}
    processed_count = 0
    
    with open(train_file, "r") as f:
        for line in tqdm(f, desc="Processing samples"):
            try:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                
                image_url, caption = parts
                image_id = str(processed_count).zfill(8)
                
                # Download the image
                try:
                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    
                    # Resize to a standard size
                    img = img.resize((512, 512), Image.LANCZOS)
                    
                    # Save the processed image
                    img.save(os.path.join(output_dir, f"images/{image_id}.jpg"))
                    
                    # Store the caption
                    captions[image_id] = {
                        "captions": [caption],
                        "metadata": {
                            "width": img.width,
                            "height": img.height,
                            "url": image_url
                        }
                    }
                    
                    processed_count += 1
                except Exception as e:
                    print(f"Error downloading image {image_url}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Save the captions
    with open(os.path.join(output_dir, "captions.json"), "w") as f:
        json.dump(captions, f)
    
    print(f"Preprocessing complete. Processed {len(captions)} images.")

def create_directory_structure(output_dir):
    """
    Create the directory structure for the processed dataset.
    
    Args:
        output_dir: Base directory for the processed dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed/images"), exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess datasets for image generation")
    parser.add_argument("--dataset", type=str, required=True, choices=["coco", "laion", "conceptual_captions"], 
                        help="Dataset to preprocess")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw dataset")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Directory to save the processed dataset")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of samples to process (for LAION)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create the directory structure
    create_directory_structure(args.output_dir)
    
    # Preprocess the dataset
    if args.dataset == "coco":
        preprocess_coco(args.data_dir, args.output_dir)
    elif args.dataset == "laion":
        preprocess_laion(args.data_dir, args.output_dir, args.sample_size)
    elif args.dataset == "conceptual_captions":
        preprocess_conceptual_captions(args.data_dir, args.output_dir)