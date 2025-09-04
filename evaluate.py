import os
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

def evaluate_object_understanding(model, text_processor, test_cases, output_dir):
    """
    Evaluate the model's ability to understand and generate images based on object descriptions.
    
    Args:
        model: The image generation model
        text_processor: The text processor for enhancing prompts
        test_cases: List of test cases with prompts and expected objects
        output_dir: Directory to save generated images and results
    """
    results = []
    correct_count = 0
    
    for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating object understanding")):
        prompt = test_case["prompt"]
        expected_objects = test_case["expected_objects"]
        
        # Process the prompt
        enhanced_prompt = text_processor.enhance_prompt(prompt)
        
        # Extract objects from the prompt
        detected_objects = text_processor.extract_objects_from_prompt(prompt)
        
        # Generate the image
        latents = model.generate_image(
            enhanced_prompt,
            num_inference_steps=30,  # Fewer steps for faster evaluation
            guidance_scale=7.5
        )
        
        # Normalize latents for visualization
        latents_normalized = (latents - latents.min()) / (latents.max() - latents.min())
        
        # Save the generated image
        image_path = os.path.join(output_dir, f"test_case_{i}.png")
        latents_np = latents_normalized[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((latents_np * 255).astype(np.uint8)).save(image_path)
        
        # Check if the detected objects match the expected objects
        # This is a simplified evaluation - in a real system, you would use
        # more sophisticated image understanding techniques
        correct = all(obj in detected_objects for obj in expected_objects)
        if correct:
            correct_count += 1
        
        # Store the results
        results.append({
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "detected_objects": detected_objects,
            "expected_objects": expected_objects,
            "correct": correct,
            "image_path": image_path
        })
    
    # Calculate accuracy
    accuracy = correct_count / len(test_cases) if test_cases else 0
    
    # Save the results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    
    print(f"Evaluation complete. Accuracy: {accuracy:.2f}")
    print(f"Results saved to {results_path}")
    
    return accuracy, results

def load_test_cases(test_file):
    """
    Load test cases from a JSON file.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        List of test cases
    """
    if not os.path.exists(test_file):
        # Create some default test cases if the file doesn't exist
        test_cases = [
            {
                "prompt": "a flying saucer in the night sky",
                "expected_objects": ["flying saucer", "sky"]
            },
            {
                "prompt": "a bat flying in a cave",
                "expected_objects": ["bat", "cave"]
            },
            {
                "prompt": "a baseball player swinging a bat",
                "expected_objects": ["baseball player", "bat"]
            },
            {
                "prompt": "a mouse on a computer desk",
                "expected_objects": ["mouse", "desk"]
            },
            {
                "prompt": "a small mouse eating cheese",
                "expected_objects": ["mouse", "cheese"]
            }
        ]
        
        # Save the default test cases
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "w") as f:
            json.dump(test_cases, f, indent=2)
    else:
        # Load the test cases from the file
        with open(test_file, "r") as f:
            test_cases = json.load(f)
    
    return test_cases

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image generation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_file", type=str, default="./data/test_cases.json", help="Path to test cases file")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory to save evaluation results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ImageGenerationModel(device=device)
    
    # Load the trained model
    print(f"Loading model from {args.model_path}")
    model.load_model(args.model_path)
    
    # Initialize the text processor
    text_processor = TextProcessor()
    
    # Load test cases
    test_cases = load_test_cases(args.test_file)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Evaluate the model
    evaluate_object_understanding(model, text_processor, test_cases, args.output_dir)