import os
import torch
from typing import Optional
from diffusers import DiffusionPipeline, PNDMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler


class PretrainedPipelineGenerator:
    """
    Обгортка над Diffusers-пайплайном для генерації зображень без донавчання.
    Має метод generate_image з інтерфейсом, сумісним із поточним сервером.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        hf_cache_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Опціонально налаштовуємо кеш Hugging Face поза проєктом
        if hf_cache_dir:
            try:
                os.makedirs(hf_cache_dir, exist_ok=True)
                os.environ["HF_HOME"] = hf_cache_dir
                os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            except Exception:
                pass

        # Визначаємо dtype: fp16 на CUDA, інакше fp32
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Відкриваємо модель (локальна папка або Hugging Face id)
        model_id_or_path = model_path or "runwayml/stable-diffusion-v1-5"
        self.pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch_dtype)

        # Оптимізації пам'яті
        try:
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)
        except Exception:
            # Fallback на .to(device)
            self.pipeline.to(self.device)
        # Назва активного семплера (для метаданих)
        self.sampler_name = "pndm"

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        negative_prompt: str = "",
        eta: float = 0.0,
        seed: Optional[int] = None,
        guidance_rescale: float = 0.0,
        apply_enhancement: bool = False,  # параметр приймається для сумісності, не використовується
    ):
        """Генерація одного зображення через Diffusers-пайплайн."""
        # Налаштовуємо генератор випадковості
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Викликаємо пайплайн; повертаємо PIL.Image
        out = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        return out.images[0]

    # --- NEW: sampler control ---
    def set_sampler(self, sampler: str):
        name = (sampler or "").strip().lower()
        try:
            if name in ("pndm", "auto"):
                self.pipeline.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
                self.sampler_name = "pndm"
            elif name in ("dpm", "dpmsolver", "dpmsolver_multistep"):
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
                self.sampler_name = "dpm"
            elif name in ("euler_a", "euler-ancestral"):
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
                self.sampler_name = "euler_a"
            else:
                self.pipeline.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
                self.sampler_name = "pndm"
        except Exception:
            self.sampler_name = name or self.sampler_name

    def get_sampler_name(self) -> str:
        return getattr(self, "sampler_name", "pndm")