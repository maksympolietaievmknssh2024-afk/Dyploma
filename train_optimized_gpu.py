#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимізований скрипт для навчання моделі з використанням GPU
Використовує комбінований датасет для кращого розуміння англійських промптів
"""

import os
# Зменшуємо фрагментацію CUDA-пам'яті
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import torch
# Дозволяємо TF32 для пришвидшення
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import time
import logging
from datetime import datetime
import gc
from PIL import Image
import numpy as np

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import EnhancedTextProcessor

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedDataset(torch.utils.data.Dataset):
    """Оптимізований датасет для ефективного навчання на GPU"""
    
    def __init__(self, dataset_file, transform=None, image_size=512, max_samples=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.image_size = image_size
        self.text_processor = EnhancedTextProcessor()
        self.image_cache = {}  # Кеш для зображень
        self.max_samples = max_samples
        
        # Завантажуємо комбінований датасет
        logger.info(f"Завантаження датасету з {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Обмежуємо кількість зразків, якщо потрібно
        if max_samples and max_samples < len(data):
            logger.info(f"Обмеження датасету до {max_samples} зразків (з {len(data)})")
            data = data[:max_samples]
        
        self.data = data
        logger.info(f"Завантажено {len(self.data)} зразків")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Отримуємо текстовий промпт
        prompt = item.get('prompt', '')
        
        # Покращуємо промпт для кращого розуміння
        enhanced_prompt = self.text_processor.enhance_prompt(prompt)
        
        # Завантажуємо зображення
        image_path = item.get('image_path', '')
        
        # Перевіряємо, чи є зображення в кеші
        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                # Кешуємо зображення, якщо кеш не занадто великий
                if len(self.image_cache) < 1000:  # Обмежуємо розмір кешу
                    self.image_cache[image_path] = image
            except Exception as e:
                logger.warning(f"Помилка завантаження зображення {image_path}: {e}")
                # Створюємо пусте зображення у випадку помилки
                image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        # Застосовуємо трансформації
        if self.transform:
            image = self.transform(image)
        
        return image, enhanced_prompt

def train_model(args):
    """Функція для навчання моделі з оптимізацією для GPU"""
    
    # Перевіряємо доступність GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Використовуємо пристрій: {device}")
    
    if device.type == "cuda":
        # Очищаємо кеш GPU
        torch.cuda.empty_cache()
        # Виводимо інформацію про GPU
        logger.info(f"Використовуємо GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Доступна пам'ять GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
    
    # Створюємо директорію для збереження моделі
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ініціалізуємо модель
    logger.info("Ініціалізація моделі...")
    model = ImageGenerationModel(
        device=device,
        use_positional_encoding=True,
        use_semantic_context=True
    )
    
    # Переносимо модель на GPU
    model.to(device)
    
    # Налаштовуємо оптимізатор з градієнтним накопиченням
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Налаштовуємо планувальник швидкості навчання
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # Завантажуємо датасет
    logger.info("Завантаження датасету...")
    train_dataset = OptimizedDataset(
        args.train_dataset,
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    # Створюємо DataLoader з оптимізацією для GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    
    # Налаштовуємо змішану точність для економії пам'яті GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Початок навчання
    logger.info(f"Початок навчання на {len(train_dataset)} зразках, {args.epochs} епох")
    start_time = time.time()
    
    # Метрики для відстеження
    metrics = {
        "epoch": [],
        "loss": [],
        "learning_rate": [],
        "time_per_epoch": []
    }
    
    # Цикл навчання
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        # Прогрес-бар для відстеження
        progress_bar = tqdm(train_loader, desc=f"Епоха {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, prompts) in enumerate(progress_bar):
            # Переносимо дані на GPU
            images = images.to(device, non_blocking=True)
            
            # zero_grad лише на початку вікна акумуляції
            if batch_idx % getattr(args, 'grad_accum_steps', 1) == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # Використовуємо змішану точність для економії пам'яті
            if scaler:
                with torch.cuda.amp.autocast():
                    # Обчислюємо втрату
                    loss = model.compute_loss(images, prompts)
                
                # Масштабуємо втрату і обчислюємо градієнти (з акумуляцією)
                scaler.scale(loss / max(1, getattr(args, 'grad_accum_steps', 1))).backward()
                
                if (batch_idx + 1) % max(1, getattr(args, 'grad_accum_steps', 1)) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Оновлюємо параметри з масштабуванням
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Звичайне навчання без змішаної точності
                loss = model.compute_loss(images, prompts)
                (loss / max(1, getattr(args, 'grad_accum_steps', 1))).backward()
                if (batch_idx + 1) % max(1, getattr(args, 'grad_accum_steps', 1)) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # Оновлюємо загальну втрату
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Розширене логування VRAM
            try:
                if torch.cuda.is_available() and ((batch_idx + 1) % getattr(args, 'log_mem_every', 50) == 0 or batch_idx < 20):
                    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                    logger.info(f"GPU mem batch {batch_idx+1}: alloc={mem_alloc:.1f}MB, reserved={mem_reserved:.1f}MB")
                    if mem_reserved - mem_alloc > 512 and (batch_idx + 1) % getattr(args, 'log_mem_every', 50) == 0:
                        torch.cuda.empty_cache()
            except Exception:
                pass
            
            # Очищаємо кеш GPU кожні N батчів
            if device.type == "cuda" and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Обчислюємо середню втрату за епоху
        avg_loss = total_loss / len(train_loader)
        
        # Оновлюємо планувальник швидкості навчання
        scheduler.step()
        
        # Зберігаємо метрики
        metrics["epoch"].append(epoch + 1)
        metrics["loss"].append(avg_loss)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
        metrics["time_per_epoch"].append(time.time() - epoch_start_time)
        
        # Виводимо інформацію про епоху
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Епоха {epoch+1}/{args.epochs} завершена за {epoch_time:.2f} с, "
                   f"втрата: {avg_loss:.4f}, "
                   f"швидкість навчання: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Зберігаємо модель кожні N епох
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            try:
                # Зберігаємо у форматі, сумісному з веб-сервером
                model.save_model(model_path)
                logger.info(f"Модель (сумісний чекпоїнт) збережено в {model_path}")
            except Exception as e:
                # Резервне збереження стандартним способом
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "loss": avg_loss,
                }, model_path)
                logger.info(f"Модель збережено (резервний формат) в {model_path}: {e}")
            
            # Зберігаємо метрики
            metrics_path = os.path.join(args.output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
    
    # Завершення навчання
    total_time = time.time() - start_time
    logger.info(f"Навчання завершено за {total_time:.2f} с")
    logger.info(f"Середня втрата: {sum(metrics['loss']) / len(metrics['loss']):.4f}")
    
    # Зберігаємо фінальну модель
    final_model_path = os.path.join(args.output_dir, "model_final.pt")
    try:
        model.save_model(final_model_path)
        logger.info(f"Фінальна модель (сумісний чекпоїнт) збережена в {final_model_path}")
    except Exception as e:
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": avg_loss,
        }, final_model_path)
        logger.info(f"Фінальна модель збережена (резервний формат) в {final_model_path}: {e}")
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Оптимізоване навчання моделі з використанням GPU")
    
    # Параметри датасету
    parser.add_argument("--train_dataset", type=str, default="combined_dataset_train.json",
                        help="Шлях до тренувального датасету")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Розмір зображень для навчання")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Максимальна кількість зразків для навчання (None = всі)")
    
    # Параметри навчання
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Розмір батчу для навчання")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Кількість епох навчання")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Швидкість навчання")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Регуляризація вагів")
    
    # Параметри оптимізації
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Кількість робочих потоків для завантаження даних")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Інтервал збереження моделі (епохи)")
    parser.add_argument("--output_dir", type=str, default="optimized_model",
                        help="Директорія для збереження моделі")
    
    args = parser.parse_args()
    
    # Запускаємо навчання
    train_model(args)

if __name__ == "__main__":
    main()