import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler

class ImageGenerationModel(nn.Module):
    """
    Neural network model for generating images based on textual descriptions.
    This model combines a text encoder for understanding object descriptions
    with a diffusion model for high-quality image generation.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        
        # Text understanding components
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # Image generation components
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        
        # Move models to device
        self.text_encoder.to(device)
        self.unet.to(device)
        
        # Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def encode_text(self, prompt):
        """
        Encode the input text prompt into a latent representation.
        
        Args:
            prompt (str): The text description of the image to generate
            
        Returns:
            torch.Tensor: The encoded text embedding
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings
    
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512, negative_prompt="", eta=0.0):
        """
        Generate an image based on the input text prompt with enhanced quality settings.
        
        Args:
            prompt (str): The text description of the image to generate
            num_inference_steps (int): Number of denoising steps (higher = better quality)
            guidance_scale (float): Scale for classifier-free guidance (7.5-15 recommended)
            height (int): Height of the generated image (multiple of 64)
            width (int): Width of the generated image (multiple of 64)
            negative_prompt (str): What to avoid in the image generation
            eta (float): Amount of noise to add during sampling (0.0 = deterministic)
            
        Returns:
            PIL.Image: The generated image
        """
        # Encode the positive prompt
        text_embeddings = self.encode_text(prompt)
        
        # Prepare negative prompt embeddings for enhanced control
        if negative_prompt:
            negative_embeddings = self.encode_text(negative_prompt)
        else:
            # Use empty prompt for unconditional generation
            uncond_input = self.tokenizer(
                "", padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            negative_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate negative and positive embeddings for classifier-free guidance
        text_embeddings = torch.cat([negative_embeddings, text_embeddings])
        
        # Initialize random noise with proper scaling
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=torch.float32
        )
        
        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set up scheduler with enhanced settings
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform enhanced guidance
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Convert latents to image using VAE decoder
        return self.decode_latents(latents)
    
    def decode_latents(self, latents):
        """
        Decode latents to PIL Image using VAE decoder.
        
        Args:
            latents (torch.Tensor): Latent representations to decode
            
        Returns:
            PIL.Image: Decoded image
        """
        from PIL import Image
        import numpy as np
        
        # Scale latents back to original range
        latents = 1 / 0.18215 * latents
        
        # For now, convert latents to image using simple normalization
        # In a full implementation, this would use a proper VAE decoder
        with torch.no_grad():
            # Convert to numpy and normalize
            image = latents.squeeze(0).cpu().numpy()
            
            # Take mean across channels if multiple channels
            if image.shape[0] > 3:
                image = np.mean(image, axis=0, keepdims=True)
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            
            # Normalize to 0-255 range
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image)
    
    def save_model(self, path):
        """
        Save the model weights to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'scheduler_config': self.scheduler.config,
        }, path)
    
    def load_model(self, path):
        """
        Load the model weights from disk.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.scheduler = DDPMScheduler.from_config(checkpoint['scheduler_config'])
    
    def compute_loss(self, images, prompts):
        """
        Compute the diffusion training loss.
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            prompts (list): List of text prompts
            
        Returns:
            torch.Tensor: The computed loss
        """
        batch_size = images.shape[0]
        
        # Encode text prompts
        text_embeddings = []
        for prompt in prompts:
            text_emb = self.encode_text(prompt)
            text_embeddings.append(text_emb)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # Convert images to latent space (simplified - normally would use VAE)
        # For this implementation, we'll downsample the images and add a channel to match UNet input
        latents = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
        
        # UNet expects 4 channels, so we need to convert RGB (3 channels) to 4 channels
        # Simple approach: duplicate one channel or add a zero channel
        if latents.shape[1] == 3:
            # Add a zero channel to make it 4 channels
            zero_channel = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)
            latents = torch.cat([latents, zero_channel], dim=1)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
        
        # Add noise to latents according to timesteps
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict the noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Compute MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return loss