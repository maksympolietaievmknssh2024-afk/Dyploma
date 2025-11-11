"""
Головний файл для запуску проекту
"""

import os
import argparse
import sys

from generation.image_generator import ImageGenerator
from training.model_trainer import ModelTrainer
from utils.text_processing import EnhancedTextProcessor

def main():
    parser = argparse.ArgumentParser(description="Система генерації зображень")
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")
    
    # Парсер для генерації зображень
    generate_parser = subparsers.add_parser("generate", help="Генерація зображень")
    generate_parser.add_argument("--prompt", type=str, help="Текстовий промпт для генерації")
    generate_parser.add_argument("--prompts_file", type=str, help="Файл з промптами для пакетної генерації")
    generate_parser.add_argument("--output_dir", type=str, default="generated_images", help="Директорія для збереження зображень")
    generate_parser.add_argument("--model_path", type=str, help="Шлях до моделі")
    generate_parser.add_argument("--steps", type=int, default=50, help="Кількість кроків інференсу")
    generate_parser.add_argument("--guidance_scale", type=float, default=7.5, help="Коефіцієнт guidance scale")
    generate_parser.add_argument("--height", type=int, default=512, help="Висота зображення")
    generate_parser.add_argument("--width", type=int, default=512, help="Ширина зображення")
    generate_parser.add_argument("--seed", type=int, help="Seed для відтворюваності")
    generate_parser.add_argument("--hq", action="store_true", help="Використовувати режим високої якості")
    
    # Парсер для навчання моделі
    train_parser = subparsers.add_parser("train", help="Навчання моделі")
    train_parser.add_argument("--dataset", type=str, required=True, help="Шлях до тренувального датасету")
    train_parser.add_argument("--val_dataset", type=str, help="Шлях до валідаційного датасету")
    train_parser.add_argument("--output_dir", type=str, default="trained_model", help="Директорія для збереження моделі")
    train_parser.add_argument("--pretrained_model", type=str, help="Шлях до попередньо навченої моделі")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Розмір батчу")
    train_parser.add_argument("--epochs", type=int, default=10, help="Кількість епох")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Швидкість навчання")
    train_parser.add_argument("--save_interval", type=int, default=2, help="Інтервал збереження моделі (епохи)")
    train_parser.add_argument("--no_mixed_precision", action="store_true", help="Вимкнути mixed precision")
    train_parser.add_argument("--no_memory_optimization", action="store_true", help="Вимкнути оптимізацію пам'яті")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generator = ImageGenerator(model_path=args.model_path)
        
        if args.prompts_file:
            import json
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
            from datetime import datetime
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
            print("Потрібно вказати --prompt або --prompts_file")
    
    elif args.command == "train":
        trainer = ModelTrainer(output_dir=args.output_dir)
        
        trainer.train(
            dataset_path=args.dataset,
            validation_dataset_path=args.val_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_interval=args.save_interval,
            pretrained_model_path=args.pretrained_model,
            use_mixed_precision=not args.no_mixed_precision,
            optimize_memory=not args.no_memory_optimization
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()