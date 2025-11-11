"""
Об'єднаний модуль для генерації зображень
Включає функціональність з generate.py, generate_custom.py, generate_hq.py, generate_sd.py, generate_yellow_car.py
"""

import os
import argparse
import torch
import json
from PIL import Image
import numpy as np
from datetime import datetime

# Імпортуємо необхідні модулі
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import EnhancedTextProcessor

class ImageGenerator:
    def __init__(self, model_path=None, device=None):
        """Ініціалізація генератора зображень"""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.text_processor = EnhancedTextProcessor()
        
    def _load_model(self, model_path):
        """Завантаження моделі"""
        print(f"Завантаження моделі з {model_path if model_path else 'стандартного шляху'}")
        model = ImageGenerationModel()
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def generate(self, prompt, output_path=None, num_inference_steps=50, 
                 guidance_scale=7.5, height=512, width=512, seed=None, hq_mode=False):
        """Генерація зображення за текстовим промптом"""
        # Встановлюємо seed для відтворюваності
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Обробка промпту
        processed_prompt = self.text_processor.process(prompt)
        print(f"Оброблений промпт: {processed_prompt}")
        
        # Генерація зображення
        print(f"Генерація зображення з {num_inference_steps} кроками та guidance_scale={guidance_scale}")
        with torch.no_grad():
            image = self.model.generate(
                prompt=processed_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                hq_mode=hq_mode
            )
        
        # Збереження зображення
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            image.save(output_path)
            print(f"Зображення збережено в {output_path}")
        
        return image
    
    def generate_yellow_car(self, output_path=None):
        """Спеціальна функція для генерації жовтої машини (демо)"""
        return self.generate(
            prompt="a yellow car on the road, high quality, detailed",
            output_path=output_path,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42
        )
    
    def generate_batch(self, prompts, output_dir="generated_images", 
                       num_inference_steps=50, guidance_scale=7.5, 
                       height=512, width=512, hq_mode=False):
        """Генерація партії зображень за списком промптів"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, prompt in enumerate(prompts):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"image_{i}_{timestamp}.png")
            
            try:
                image = self.generate(
                    prompt=prompt,
                    output_path=output_path,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    hq_mode=hq_mode
                )
                
                results.append({
                    "prompt": prompt,
                    "output_path": output_path,
                    "success": True
                })
            except Exception as e:
                print(f"Помилка при генерації зображення для промпту '{prompt}': {e}")
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                })
        
        # Зберігаємо результати
        results_path = os.path.join(output_dir, "generation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Основна функція для запуску з командного рядка"""
    parser = argparse.ArgumentParser(description="Генерація зображень за текстовим промптом")
    parser.add_argument("--prompt", type=str, help="Текстовий промпт для генерації")
    parser.add_argument("--prompts_file", type=str, help="Файл з промптами для пакетної генерації")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Директорія для збереження зображень")
    parser.add_argument("--model_path", type=str, help="Шлях до моделі")
    parser.add_argument("--steps", type=int, default=50, help="Кількість кроків інференсу")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Коефіцієнт guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Висота зображення")
    parser.add_argument("--width", type=int, default=512, help="Ширина зображення")
    parser.add_argument("--seed", type=int, help="Seed для відтворюваності")
    parser.add_argument("--hq", action="store_true", help="Використовувати режим високої якості")
    parser.add_argument("--yellow_car", action="store_true", help="Генерувати демо з жовтою машиною")
    
    args = parser.parse_args()
    
    # Ініціалізуємо генератор
    generator = ImageGenerator(model_path=args.model_path)
    
    # Визначаємо режим роботи
    if args.yellow_car:
        # Демо з жовтою машиною
        output_path = os.path.join(args.output_dir, "yellow_car.png")
        generator.generate_yellow_car(output_path=output_path)
    
    elif args.prompts_file:
        # Пакетна генерація з файлу
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
        
        generator.generate_batch(
            prompts=prompts,
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            hq_mode=args.hq
        )
    
    elif args.prompt:
        # Генерація одного зображення
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"image_{timestamp}.png")
        
        generator.generate(
            prompt=args.prompt,
            output_path=output_path,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            hq_mode=args.hq
        )
    
    else:
        print("Потрібно вказати --prompt, --prompts_file або --yellow_car")


if __name__ == "__main__":
    main()