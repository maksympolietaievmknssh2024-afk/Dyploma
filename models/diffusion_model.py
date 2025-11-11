import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, PNDMScheduler, AutoencoderKL, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import sys
import os
import shutil
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.positional_encoding import WordOrderAwareEncoder, create_attention_mask
from utils.text_processing import EnhancedTextProcessor

# Глобальна змінна для кешування SentenceTransformer
_GLOBAL_SEMANTIC_ENCODER = None

def get_global_semantic_encoder():
    """Отримання глобального семантичного енкодера (singleton pattern)"""
    global _GLOBAL_SEMANTIC_ENCODER
    if _GLOBAL_SEMANTIC_ENCODER is None:
        # Тримаймо SentenceTransformer на CPU, щоб не займати VRAM
        _GLOBAL_SEMANTIC_ENCODER = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        # Freeze parameters
        for param in _GLOBAL_SEMANTIC_ENCODER.parameters():
            param.requires_grad = False
    return _GLOBAL_SEMANTIC_ENCODER

# Допоміжна утиліта: визначити пристрій модуля
def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')

class SemanticContextualEncoder(nn.Module):
    """
    Семантичний контекстуальний енкодер для кращого розуміння значень слів у контексті.
    """
    
    def __init__(self, clip_dim=512, semantic_dim=384, hidden_dim=768):
        super().__init__()
        self.clip_dim = clip_dim
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        
        # Використовуємо глобальний семантичний енкодер
        self.semantic_encoder = get_global_semantic_encoder()
        
        # Проекційні шари для різних типів ембедингів
        self.clip_projection = nn.Linear(768, hidden_dim)  # CLIP: 768 -> hidden_dim
        self.semantic_projection = nn.Linear(semantic_dim, hidden_dim)  # Semantic: 384 -> hidden_dim
        self.positional_projection = nn.Linear(clip_dim, hidden_dim)  # Positional: 512 -> hidden_dim
        
        # Attention механізм для комбінування різних типів ембедингів
        self.multi_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Контекстуальний трансформер
        self.contextual_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Фінальна проекція - повертаємо 768 для сумісності з UNet cross-attention
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 768),  # Змінено з clip_dim на 768 для UNet
            nn.LayerNorm(768)
        )
        
        # Адаптивні ваги для комбінування
        self.embedding_weights = nn.Parameter(torch.ones(3))  # [CLIP, Semantic, Positional]
        
        # Freeze semantic encoder
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False
    
    def get_semantic_embeddings(self, texts, to_device: bool = True):
        """Отримання семантичних ембедингів. Якщо to_device=False, залишаємо на CPU"""
        target_device = self.clip_projection.weight.device
        # Обчислення семантики завжди на CPU — це суттєво зменшує VRAM
        with torch.no_grad():
            semantic_embeddings = self.semantic_encoder.encode(
                texts, convert_to_tensor=True, device='cpu'
            )
        if to_device:
            semantic_embeddings = semantic_embeddings.to(target_device)
        return semantic_embeddings.clone().detach()
    
    def encode_text(self, text):
        """
        Кодування одного тексту для семантичної втрати.
        Повертає семантичний ембединг для одного тексту (на CPU для мінімального VRAM).
        """
        if isinstance(text, str):
            text = [text]
        # Для семантичної втрати повертаємо CPU‑тензор
        semantic_embeddings = self.get_semantic_embeddings(text, to_device=False)
        # Повертаємо перший ембединг якщо це був один текст
        if len(text) == 1:
            return semantic_embeddings[0]
        return semantic_embeddings

    def forward(self, clip_embeddings, texts, positional_embeddings=None, weights_override=None):
        batch_size = clip_embeddings.size(0)
        seq_len = clip_embeddings.size(1)

        # Project CLIP embeddings per token
        clip_projected = self.clip_projection(clip_embeddings)  # (B, seq_len, hidden_dim)

        # Semantic embeddings → expand across tokens
        semantic_embeddings = self.get_semantic_embeddings(texts)  # (B, semantic_dim)
        semantic_projected = self.semantic_projection(semantic_embeddings).unsqueeze(1)  # (B, 1, hidden_dim)
        semantic_expanded = semantic_projected.expand(-1, seq_len, -1)  # (B, seq_len, hidden_dim)

        # Positional embeddings → expand across tokens (if provided)
        if positional_embeddings is not None:
            positional_projected = self.positional_projection(positional_embeddings).unsqueeze(1)  # (B, 1, hidden_dim)
            positional_expanded = positional_projected.expand(-1, seq_len, -1)  # (B, seq_len, hidden_dim)
        else:
            positional_expanded = torch.zeros_like(clip_projected)

        # Adaptive weights and per-token combination
        weights = F.softmax(self.embedding_weights if weights_override is None else weights_override, dim=0)
        combined_embeddings = (
            weights[0] * clip_projected +
            weights[1] * semantic_expanded +
            weights[2] * positional_expanded
        )  # (B, seq_len, hidden_dim)

        # Multi-modal attention across tokens
        attended_embeddings, _ = self.multi_modal_attention(
            combined_embeddings, combined_embeddings, combined_embeddings
        )  # (B, seq_len, hidden_dim)

        # Contextual transformer
        contextual_embeddings = self.contextual_transformer(attended_embeddings)  # (B, seq_len, hidden_dim)

        # Final projection to 768 per token
        final_embeddings = self.final_projection(contextual_embeddings)  # (B, seq_len, 768)
        final_embeddings = F.normalize(final_embeddings, p=2, dim=-1)    # (B, seq_len, 768)
        return final_embeddings

class ImageGenerationModel(nn.Module):
    """
    Покращена модель генерації зображень з семантичним розумінням контексту.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", use_positional_encoding=True, use_semantic_context=True):
        super().__init__()
        self.device = device
        self.use_positional_encoding = use_positional_encoding
        self.use_semantic_context = use_semantic_context
        
        # Визначаємо джерело моделі та налаштування кешу/офлайн режиму
        base_repo = os.environ.get("HF_LOCAL_MODEL_DIR") or "runwayml/stable-diffusion-v1-5"
        cache_dir = os.environ.get("HF_CACHE_DIR")
        local_only = os.environ.get("HF_OFFLINE", "0") == "1"
        
        # Завантаження базових моделей
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_repo, subfolder="tokenizer", cache_dir=cache_dir, local_files_only=local_only
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_repo, subfolder="text_encoder", cache_dir=cache_dir, local_files_only=local_only
        )
        
        # UNet для дифузії
        self.unet = UNet2DConditionModel.from_pretrained(
            base_repo, 
            subfolder="unet",
            cache_dir=cache_dir,
            local_files_only=local_only
        )
        
        # Scheduler для дифузійного процесу
        self.scheduler = PNDMScheduler.from_pretrained(
            base_repo, 
            subfolder="scheduler",
            cache_dir=cache_dir,
            local_files_only=local_only
        )
        # Назва активного семплера (для метаданих)
        self.sampler_name = "pndm"
        
        # VAE для кодування/декодування зображень
        self.vae = AutoencoderKL.from_pretrained(
            base_repo,
            subfolder="vae",
            cache_dir=cache_dir,
            local_files_only=local_only
        )
        
        # Покращений текстовий процесор
        self.text_processor = EnhancedTextProcessor()
        
        # LRU-кеш для фразових ембедингів (на CPU) щоб зменшити повторні виклики CLIP
        self.phrase_cache = OrderedDict()
        self.phrase_cache_max = 256
        
        # Позиційний енкодер (якщо використовується)
        if self.use_positional_encoding:
            vocab_size = self.tokenizer.vocab_size
            self.word_order_encoder = WordOrderAwareEncoder(
                vocab_size=vocab_size,
                d_model=512,
                clip_dim=512,
                num_layers=4,
                num_heads=8,
                max_len=self.tokenizer.model_max_length
            )
        
        # Семантичний контекстуальний енкодер
        if self.use_semantic_context:
            self.semantic_contextual_encoder = SemanticContextualEncoder(
                clip_dim=512,
                semantic_dim=384,
                hidden_dim=768
            )
        
        # Шар для об'єднання ембеддінгів (якщо потрібно)
        if self.use_positional_encoding and not self.use_semantic_context:
            self.embedding_fusion = nn.Sequential(
                nn.Linear(768 + 512, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 512),
                nn.LayerNorm(512)
            )
        
        # Move models to device
        self.text_encoder.to('cpu')
        self.unet.to(device)
        self.vae.to('cpu')
        # Move additional components to devices: offload context to CPU to save VRAM
        if self.use_positional_encoding:
            self.word_order_encoder.to(device)
        
        if self.use_semantic_context:
            # Semantic contextual encoder fully on GPU
            self.semantic_contextual_encoder.to(device)
            # Тримаймо SentenceTransformer строго на CPU для економії VRAM
            try:
                self.semantic_contextual_encoder.semantic_encoder = self.semantic_contextual_encoder.semantic_encoder.to('cpu')
            except Exception:
                pass
        
        if self.use_positional_encoding and not self.use_semantic_context:
            self.embedding_fusion.to(device)
        
        # Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing on UNet to save VRAM
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception:
            pass
        # Увімкнення пам’ять‑ефективної уваги
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                self.unet.enable_attention_slicing("auto")
            except Exception:
                pass
        
        # Freeze VAE parameters and set eval mode for non-trainable modules
        for param in self.vae.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.vae.eval()
        
        # Оверрайд ваг ембеддингів (за замовчуванням вимкнено)
        self.embedding_weights_override = None

        # Проєкція латентів у простір для контрастної втрати (текст↔зображення)
        # Використовуємо усереднення по просторових осях і лінійний шар: [B, 4, H, W] → [B, 768]
        self.image_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.unet.config.in_channels, 768)
        )
        # Температура для InfoNCE/CLIP‑подібної втрати
        self.contrastive_tau = 0.07
    
    def encode_text(self, text, apply_enhancement: bool = True):
        """
        Покращене кодування тексту з семантичним контекстуальним розумінням
        """
        # Попередня обробка тексту для кращого розуміння
        if isinstance(text, str):
            text = [text]

        # Режим дослівного промпту: без покращень і вагових карт
        if not apply_enhancement:
            enhanced_texts = list(text)
            weight_maps = [{} for _ in enhanced_texts]
        else:
            enhanced_texts = []
            weight_maps = []
            for t in text:
                # Отримуємо покращений промпт та карту ваг фраз
                try:
                    enhanced_text, weight_map = self.text_processor.enhance_prompt_with_weights(t)
                except Exception:
                    enhanced_text = self.text_processor.enhance_prompt_advanced(t)
                    weight_map = {}
                enhanced_texts.append(enhanced_text)
                weight_maps.append(weight_map)
        
        # Токенізація тексту
        inputs = self.tokenizer(
            enhanced_texts, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        text_dev = _module_device(self.text_encoder)
        inputs = {k: v.to(text_dev) for k, v in inputs.items()}
        
        # Отримання стандартних CLIP ембедингів
        with torch.no_grad():
            clip_outputs = self.text_encoder(**inputs)
            clip_embeddings = clip_outputs.last_hidden_state.to(self.device)  # [batch_size, seq_len, 768]
            
            # Пер-фразове підсилення ембедингів (без складного мапінгу токенів)
            # Для кожної фрази з вагою >1 додамо невеликий резидуальний вектор;
            # при вагах <1 — віднімемо.
            if apply_enhancement:
                try:
                    residual_scale = 0.15  # обмежуємо вплив, щоб уникнути перенасичення
                    for i, wm in enumerate(weight_maps):
                        if not wm:
                            continue
                        # агрегуємо фразові вектори
                        accum = None
                        for phrase, w in wm.items():
                            if abs(w - 1.0) < 1e-3:
                                continue
                            # Кешування фразового ембединга на CPU
                            if phrase in self.phrase_cache:
                                phrase_emb_cpu = self.phrase_cache[phrase]
                                # Переміщується на поточний пристрій лише для розрахунку
                                phrase_emb = phrase_emb_cpu.to(self.device)
                            else:
                                # Ембединг фрази: усереднене last_hidden_state
                                phrase_inputs = self.tokenizer(
                                    [phrase], return_tensors="pt", padding='max_length', truncation=True,
                                    max_length=self.tokenizer.model_max_length
                                )
                                text_dev = _module_device(self.text_encoder)
                                phrase_inputs = {k: v.to(text_dev) for k, v in phrase_inputs.items()}
                                phrase_out = self.text_encoder(**phrase_inputs)
                                phrase_emb = phrase_out.last_hidden_state.mean(dim=1).to(self.device)  # [1, 768]
                                # Зберігаємо на CPU з LRU
                                phrase_emb_cpu = phrase_emb.detach().cpu()
                                self.phrase_cache[phrase] = phrase_emb_cpu
                                # Обмежуємо розмір кешу
                                if len(self.phrase_cache) > self.phrase_cache_max:
                                    # видаляємо найстаріший
                                    self.phrase_cache.popitem(last=False)
                            # Резидуальний внесок
                            delta = (w - 1.0) * residual_scale * phrase_emb  # [1, 768]
                            accum = delta if accum is None else accum + delta
                        if accum is not None:
                            # Додаємо/віднімаємо однаково до всіх токенів послідовності
                            seq_len = clip_embeddings.shape[1]
                            clip_embeddings[i, :, :] = clip_embeddings[i, :, :] + accum.repeat(1, seq_len, 1).squeeze(0)
                except Exception as e:
                    print(f"WARNING: token weighting residual failed: {e}")
        
        # Використання семантичного контекстуального енкодера
        if self.use_semantic_context:
            positional_embeddings = None

            # Отримання позиційних ембедингів (на GPU) за потреби
            if self.use_positional_encoding:
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = create_attention_mask(input_ids)
                positional_embeddings = self.word_order_encoder(
                    input_ids,
                    attention_mask
                )  # [batch_size, 512] on GPU

            # Семантичне контекстуальне кодування (всі компоненти на GPU)
            final_embeddings = self.semantic_contextual_encoder(
                clip_embeddings=clip_embeddings,
                texts=enhanced_texts,
                positional_embeddings=positional_embeddings,
                weights_override=self.embedding_weights_override
            )

            # Повертаємо на основний пристрій для Unet
            return final_embeddings
        
        # Fallback до старого методу (для зворотної сумісності)
        elif self.use_positional_encoding:
            # Отримання позиційно-усвідомлених ембедингів (енкодер на GPU)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = create_attention_mask(input_ids)
            positional_embeddings = self.word_order_encoder(
                input_ids,
                attention_mask
            )  # [batch_size, 512] на GPU
            
            # Pooling CLIP ембедингів до того ж розміру
            clip_pooled = clip_outputs.pooler_output.to(self.device)  # [batch_size, 768]
            
            # Комбінування CLIP та позиційних ембедингів
            positional_embeddings = positional_embeddings.to(self.device)
            combined_embeddings = torch.cat([clip_pooled, positional_embeddings], dim=-1)  # [batch_size, 768+512]
            fused_embeddings = self.embedding_fusion(combined_embeddings)
            
            # Нормалізація для стабільності
            pooled_output = F.normalize(fused_embeddings, p=2, dim=1)
            
            return pooled_output
        else:
            # Стандартний CLIP pooling з нормалізацією
            pooled_output = clip_outputs.pooler_output.to(self.device)
            pooled_output = F.normalize(pooled_output, p=2, dim=1)
            return pooled_output

    def set_embedding_weights_override(self, weights):
        """
        Встановлює оверрайд ваг для комбінування ембеддингів [CLIP, Semantic, Positional].
        Приймає список/кортеж/рядок з трьома значеннями або тензор.
        """
        if weights is None:
            self.embedding_weights_override = None
            return
        if isinstance(weights, str):
            parts = [p.strip() for p in weights.replace(';', ',').split(',') if p.strip() != ""]
            if len(parts) != 3:
                print("WARNING: Expected 3 comma-separated values for embedding weights override; ignoring.")
                return
            try:
                values = [float(x) for x in parts]
            except Exception:
                print("WARNING: Could not parse embedding weights override; ignoring.")
                return
            self.embedding_weights_override = torch.tensor(values, dtype=torch.float32, device=self.device)
            return
        if isinstance(weights, (list, tuple)):
            if len(weights) != 3:
                print("WARNING: Expected 3 values in list/tuple for embedding weights override; ignoring.")
                return
            self.embedding_weights_override = torch.tensor(list(weights), dtype=torch.float32, device=self.device)
            return
        if isinstance(weights, torch.Tensor):
            if weights.numel() != 3:
                print("WARNING: Expected tensor with 3 elements for embedding weights override; ignoring.")
                return
            self.embedding_weights_override = weights.to(self.device).float()
            return
        print("WARNING: Invalid type for embedding weights override; ignoring.")

    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512, negative_prompt="", eta=0.0, seed=None, guidance_rescale=0.0, apply_enhancement: bool = True):
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
            seed (int, optional): Random seed for reproducible generation. If None, uses random seed.
            guidance_rescale (float): Optional rescale factor (0.0-1.0) to stabilize CFG.
            
        Returns:
            PIL.Image: The generated image
        """
        # Set random seed for reproducible generation
        gen: torch.Generator = None
        if seed is not None:
            # Use a dedicated generator bound to the model device for determinism
            try:
                gen = torch.Generator(device=self.device)
            except Exception:
                gen = torch.Generator()
            gen.manual_seed(int(seed))
            # Keep global seeds in sync as a fallback
            try:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(int(seed))
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass
        
        # Encode the positive prompt (можна вимкнути покращення)
        text_embeddings = self.encode_text(prompt, apply_enhancement=apply_enhancement)
        
        # Prepare negative prompt embeddings for enhanced control
        if negative_prompt:
            # Негативний промпт завжди дослівно
            negative_embeddings = self.encode_text(negative_prompt, apply_enhancement=False)
        else:
            # Use empty prompt for unconditional generation - use encode_text for consistency
            negative_embeddings = self.encode_text("", apply_enhancement=False)
        
        # Concatenate negative and positive embeddings for classifier-free guidance
        # text_embeddings та negative_embeddings вже мають правильний формат [batch, seq_len, 768]
        text_embeddings = torch.cat([negative_embeddings, text_embeddings])
        
        # Initialize latents with proper scaling for VAE
        if gen is not None:
            latents = torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=torch.float32,
                generator=gen,
            )
        else:
            latents = torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=torch.float32,
            )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set up scheduler with enhanced settings
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Clamp timesteps to valid range to avoid out-of-bounds indexing
        timesteps = self.scheduler.timesteps
        try:
            max_t = getattr(self.scheduler.config, "num_train_timesteps", 1000) - 1
            if torch.is_tensor(timesteps):
                timesteps = timesteps.clamp(0, max_t)
            else:
                timesteps = torch.tensor(timesteps, device=self.device).clamp(0, max_t)
        except Exception:
            # Fallback if scheduler doesn't expose config
            timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for t in timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform enhanced guidance
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)

            # Guidance rescale helps when using high guidance_scale
            if guidance_rescale and guidance_rescale > 0.0:
                noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_positive, guidance_rescale)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Convert latents to image using VAE decoder
        return self.decode_latents(latents)
    
    def decode_latents(self, latents):
        """Decode latents to image using VAE decoder"""
        # Ensure latents are on the VAE device (CPU) for decoding
        latents = latents.to(_module_device(self.vae))
        
        # Scale latents back from scheduler scaling
        latents = 1 / 0.18215 * latents
        
        # Decode using VAE
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        # Convert to PIL Image format
        image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
        image = image.cpu().permute(0, 2, 3, 1).numpy()  # Convert to HWC format
        image = (image * 255).astype("uint8")
        
        from PIL import Image
        return Image.fromarray(image[0])

    # --- NEW: sampler control ---
    def set_sampler(self, sampler: str):
        """Перемикає семплер/шедулер за назвою.
        Доступні: 'auto'/'pndm', 'dpm', 'euler_a'"""
        name = (sampler or "").strip().lower()
        try:
            if name in ("pndm", "auto"):
                self.scheduler = PNDMScheduler.from_config(self.scheduler.config)
                self.sampler_name = "pndm"
            elif name in ("dpm", "dpmsolver", "dpmsolver_multistep"):
                self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler.config)
                self.sampler_name = "dpm"
            elif name in ("euler_a", "euler-ancestral"):
                self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler.config)
                self.sampler_name = "euler_a"
            else:
                # Fallback to PNDM
                self.scheduler = PNDMScheduler.from_config(self.scheduler.config)
                self.sampler_name = "pndm"
        except Exception:
            # If conversion fails, keep existing scheduler but record requested name
            self.sampler_name = name or self.sampler_name

    def get_sampler_name(self) -> str:
        return getattr(self, "sampler_name", "pndm")

    def _rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale the combined CFG prediction to match the standard deviation of the
        text-conditioned prediction and blend with the original based on guidance_rescale.

        Args:
            noise_cfg (torch.Tensor): CFG-combined noise prediction.
            noise_pred_text (torch.Tensor): Text-conditioned prediction.
            guidance_rescale (float): Blend factor in [0, 1].

        Returns:
            torch.Tensor: Rescaled and blended noise prediction.
        """
        if guidance_rescale <= 0.0:
            return noise_cfg

        dims_text = list(range(1, noise_pred_text.ndim))
        std_text = noise_pred_text.std(dim=dims_text, keepdim=True)
        dims_cfg = list(range(1, noise_cfg.ndim))
        std_cfg = noise_cfg.std(dim=dims_cfg, keepdim=True)

        eps = 1e-8
        noise_rescaled = noise_cfg * (std_text / (std_cfg + eps))
        return guidance_rescale * noise_rescaled + (1.0 - guidance_rescale) * noise_cfg
    
    def save_model(self, path):
        """
        Save the model weights to disk.
        
        Args:
            path (str): Path to save the model
        """
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"[save_model] Починаємо збереження у: {path}")
        except Exception:
            pass
        # Переміщуємо всі тензори стану на CPU перед збереженням,
        # щоб уникнути проблем із CUDA IPC та zip‑серіалізацією на Windows
        def _cpu_state_dict(module):
            sd = module.state_dict()
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    sd[k] = v.detach().cpu()
            return sd

        save_dict = {
            'unet_state_dict': _cpu_state_dict(self.unet),
            'scheduler_config': self.scheduler.config,
            'vae_state_dict': _cpu_state_dict(self.vae),
        }
        
        # Зберігаємо додаткові модулі лише якщо вони існують у конфігурації
        if self.use_positional_encoding and hasattr(self, 'word_order_encoder'):
            save_dict['word_order_encoder_state_dict'] = _cpu_state_dict(self.word_order_encoder)
        if hasattr(self, 'embedding_fusion'):
            save_dict['embedding_fusion_state_dict'] = _cpu_state_dict(self.embedding_fusion)
        # Атомарний запис: спочатку у тимчасовий файл, потім replace
        tmp_path = f"{path}.tmp"
        # Використовуємо legacy серіалізацію без zip, щоб уникнути помилки
        # "unexpected pos ..." у torch.serialization на Windows
        try:
            # На нових версіях PyTorch надійніше використовувати weights_only=True
            try:
                torch.save(save_dict, tmp_path, weights_only=True)
            except TypeError:
                # Якщо параметр недоступний — використовуємо legacy серіалізацію без zip
                torch.save(save_dict, tmp_path, _use_new_zipfile_serialization=False)
            try:
                os.replace(tmp_path, path)
            except Exception:
                try:
                    shutil.copyfile(tmp_path, path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            try:
                size_mb = os.path.getsize(path) / (1024*1024)
                logger.info(f"[save_model] Збережено чекпоінт: {path} ({size_mb:.2f} MB)")
                # Створюємо простий файл‑маячок поруч з чекпоінтом для відлагодження
                try:
                    with open(path + ".ok", "w", encoding="utf-8") as f:
                        f.write("saved")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            try:
                logger.warning(f"[save_model] Помилка при збереженні у {path}: {e}")
            except Exception:
                pass
            # Спробуємо очистити CUDA кеш, якщо доступний, і повторити ще один раз
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                try:
                    torch.save(save_dict, tmp_path, weights_only=True)
                except TypeError:
                    torch.save(save_dict, tmp_path, _use_new_zipfile_serialization=False)
                os.replace(tmp_path, path)
                try:
                    size_mb = os.path.getsize(path) / (1024*1024)
                    logger.info(f"[save_model] Повторне збереження вдалося: {path} ({size_mb:.2f} MB)")
                    try:
                        with open(path + ".ok", "w", encoding="utf-8") as f:
                            f.write("saved")
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception as e2:
                try:
                    logger.error(f"[save_model] Повторне збереження не вдалося для {path}: {e2}")
                except Exception:
                    pass
    
    def load_model(self, path):
        """
        Load the model weights from disk.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.scheduler = PNDMScheduler.from_config(checkpoint['scheduler_config'])
        
        if 'vae_state_dict' in checkpoint:
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        # Завантажуємо додаткові модулі, якщо вони є в чекпоінті та в поточній конфігурації
        if 'word_order_encoder_state_dict' in checkpoint and hasattr(self, 'word_order_encoder'):
            self.word_order_encoder.load_state_dict(checkpoint['word_order_encoder_state_dict'])
        if 'embedding_fusion_state_dict' in checkpoint and hasattr(self, 'embedding_fusion'):
            self.embedding_fusion.load_state_dict(checkpoint['embedding_fusion_state_dict'])
    
    def forward(self, images, prompts):
        """
        Forward pass for the model - computes diffusion loss
        
        Args:
            images: Batch of images
            prompts: List of text prompts
            
        Returns:
            loss: Computed MSE loss
        """
        return self.compute_loss(images, prompts)
    
    def compute_loss(self, images, prompts):
        """
        Compute the diffusion loss for training
        
        Args:
            images: Batch of images
            prompts: List of text prompts
            
        Returns:
            loss: Computed MSE loss
        """
        # Encode text prompts
        # Compute text embeddings without autograd to reduce VRAM usage
        with torch.no_grad():
            text_embeddings = self.encode_text(prompts)
        
        # Convert images to latent space using VAE encoder
        with torch.no_grad():
            # Кодування через VAE на його пристрої (CPU), потім перенос латентів на основний
            vae_dev = _module_device(self.vae)
            latents = self.vae.encode(images.to(vae_dev)).latent_dist.sample()
            latents = latents * 0.18215  # Scale latents
            latents = latents.to(self.device)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                 (latents.shape[0],), device=latents.device)
        
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

        # Контрастне вирівнювання семантики тексту та зображення
        # text_embeddings має форму [B, 1, 768] → агрегуємо по seq_len
        text_vec = text_embeddings.mean(dim=1)  # [B, 768]
        # latents: [B, 4, H/8, W/8] → [B, 768]
        img_vec = self.image_proj(latents)
        # Нормалізація для косинусної подібності
        text_vec = F.normalize(text_vec, dim=-1)
        img_vec = F.normalize(img_vec, dim=-1)
        # Двосторонні логіти (текст→зображення та зображення→текст)
        logits_ti = torch.matmul(text_vec, img_vec.t()) / self.contrastive_tau
        logits_it = torch.matmul(img_vec, text_vec.t()) / self.contrastive_tau
        targets = torch.arange(logits_ti.size(0), device=logits_ti.device)
        contrastive_loss = (
            F.cross_entropy(logits_ti, targets) + F.cross_entropy(logits_it, targets)
        ) * 0.1

        # Основна MSE‑втрата дифузії + семантичне вирівнювання
        loss = F.mse_loss(noise_pred, noise) + contrastive_loss
        return loss