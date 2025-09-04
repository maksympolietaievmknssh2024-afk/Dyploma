import os
import argparse
import torch
from PIL import Image
import io
import base64
from flask import Flask, render_template, request, jsonify

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import TextProcessor

app = Flask(__name__)

# Initialize the model and text processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
text_processor = TextProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt from the request
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Process the prompt to enhance context understanding
    enhanced_prompt = text_processor.enhance_prompt(prompt)
    
    # Extract objects from the prompt
    objects = text_processor.extract_objects_from_prompt(prompt)
    
    # Generate the image
    try:
        latents = model.generate_image(
            enhanced_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        # In a complete implementation, we would convert latents to image using a VAE decoder
        # For now, we'll just normalize and save the latents as an image for visualization
        latents = (latents - latents.min()) / (latents.max() - latents.min())
        
        # Convert to PIL Image
        # This is a placeholder - in a real implementation, we would use a proper decoder
        image = Image.fromarray((latents[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        
        # Convert to base64 for sending to frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'image': img_str,
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'detected_objects': objects
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_model(model_path):
    global model
    model = ImageGenerationModel(device=device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_model(model_path)
    else:
        print("Using default model weights")

def parse_args():
    parser = argparse.ArgumentParser(description="Web interface for image generation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Create a simple HTML template
    with open("templates/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Image Generation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .input-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .prompt-input {
            padding: 10px;
            font-size: 16px;
            width: 100%;
        }
        .generate-button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .generate-button:hover {
            background-color: #45a049;
        }
        .result-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .image-container {
            width: 100%;
            max-width: 512px;
            margin: 0 auto;
        }
        .image-container img {
            width: 100%;
            height: auto;
        }
        .info-section {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            font-style: italic;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Image Generation from Text</h1>
    <div class="container">
        <div class="input-section">
            <label for="prompt">Enter your prompt:</label>
            <textarea id="prompt" class="prompt-input" rows="4" placeholder="Describe the image you want to generate..."></textarea>
            <button id="generate" class="generate-button">Generate Image</button>
            <div id="loading" class="loading">Generating image... This may take a minute.</div>
        </div>
        <div class="result-section">
            <div id="image-container" class="image-container">
                <!-- Generated image will be displayed here -->
            </div>
            <div id="info-section" class="info-section" style="display: none;">
                <h3>Details:</h3>
                <p><strong>Original Prompt:</strong> <span id="original-prompt"></span></p>
                <p><strong>Enhanced Prompt:</strong> <span id="enhanced-prompt"></span></p>
                <p><strong>Detected Objects:</strong> <span id="detected-objects"></span></p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('generate').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('info-section').style.display = 'none';
            document.getElementById('image-container').innerHTML = '';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt }),
                });

                const data = await response.json();

                if (data.error) {
                    alert(`Error: ${data.error}`);
                    return;
                }

                // Display the image
                document.getElementById('image-container').innerHTML = `
                    <img src="data:image/png;base64,${data.image}" alt="Generated image">
                `;

                // Display information
                document.getElementById('original-prompt').textContent = data.original_prompt;
                document.getElementById('enhanced-prompt').textContent = data.enhanced_prompt;
                document.getElementById('detected-objects').textContent = data.detected_objects.join(', ');
                document.getElementById('info-section').style.display = 'block';
            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
        """)
    
    # Load the model
    load_model(args.model_path)
    
    # Start the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)