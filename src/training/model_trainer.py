"""
Об'єднаний модуль для навчання моделей
Включає функціональність з train.py, train_optimized_gpu.py, train_enhanced.py, train_simple.py
"""

import os
import argparse
import torch
import json
import time
from datetime import datetime

# Імпортуємо необхідні модулі
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import EnhancedTextProcessor

class ModelTrainer:
    def __init__(self, output_dir="trained_model", device=None):
        """Ініціалізація тренера моделей"""
        self.output_dir = output_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.text_processor = EnhancedTextProcessor()
        
        # Створюємо директорію для збереження моделі
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_dataset(self, dataset_path):
        """Завантаження датасету"""
        print(f"Завантаження датасету з {dataset_path}")
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        return dataset
    
    def _initialize_model(self, pretrained_model_path=None):
        """Ініціалізація моделі"""
        print(f"Ініціалізація моделі на пристрої {self.device}")
        model = ImageGenerationModel()
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Завантаження попередньо навченої моделі з {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            try:
                # Підтримка формату навчального скрипта з ключем 'model_state_dict'
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                # Підтримка власного формату збереження через ImageGenerationModel.save_model
                elif isinstance(checkpoint, dict) and ('unet_state_dict' in checkpoint or 'vae_state_dict' in checkpoint):
                    model.load_model(pretrained_model_path)
                else:
                    # Звичайний state_dict
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Не вдалося завантажити ваги з {pretrained_model_path}: {e}")
        
        model.to(self.device)
        self.model = model
        return model
    
    def train(self, dataset_path, validation_dataset_path=None, batch_size=4, 
              epochs=10, learning_rate=1e-4, save_interval=2, pretrained_model_path=None,
              use_mixed_precision=True, optimize_memory=True):
        """Навчання моделі"""
        # Завантаження датасету
        dataset = self._load_dataset(dataset_path)
        validation_dataset = None
        if validation_dataset_path:
            validation_dataset = self._load_dataset(validation_dataset_path)
        
        # Ініціалізація моделі
        model = self._initialize_model(pretrained_model_path)
        model.train()
        
        # Налаштування оптимізатора
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Налаштування mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == "cuda" else None
        
        # Метрики навчання
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset_size": len(dataset),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device)
        }
        
        print(f"Початок навчання на {len(dataset)} зразках, {epochs} епох, розмір батчу {batch_size}")
        
        # Цикл навчання
        for epoch in range(epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            
            # Перемішуємо датасет
            import random
            random.shuffle(dataset)
            
            # Розбиваємо на батчі
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                
                # Підготовка даних
                prompts = [item["prompt"] for item in batch]
                processed_prompts = [self.text_processor.process(prompt) for prompt in prompts]
                
                # Оптимізація пам'яті
                if optimize_memory and i > 0 and i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Навчання з mixed precision
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        loss = model.training_step(processed_prompts)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model.training_step(processed_prompts)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                # Виводимо прогрес
                if i % (10 * batch_size) == 0:
                    print(f"Епоха {epoch+1}/{epochs}, Батч {i//batch_size}/{len(dataset)//batch_size}, Втрата: {loss.item():.4f}")
            
            # Обчислюємо середню втрату за епоху
            epoch_loss = running_loss / (len(dataset) // batch_size)
            metrics["train_loss"].append(epoch_loss)
            metrics["epochs"].append(epoch + 1)
            
            # Валідація
            val_loss = None
            if validation_dataset:
                model.eval()
                val_running_loss = 0.0
                
                with torch.no_grad():
                    for i in range(0, min(len(validation_dataset), 100), batch_size):
                        val_batch = validation_dataset[i:i+batch_size]
                        val_prompts = [item["prompt"] for item in val_batch]
                        processed_val_prompts = [self.text_processor.process(prompt) for prompt in val_prompts]
                        
                        val_loss = model.training_step(processed_val_prompts)
                        val_running_loss += val_loss.item()
                
                val_loss = val_running_loss / (min(len(validation_dataset), 100) // batch_size)
                metrics["val_loss"].append(val_loss)
                model.train()
            
            # Зберігаємо модель
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                model_path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pt")
                try:
                    # Зберігаємо у форматі, сумісному з веб-сервером
                    self.model.save_model(model_path)
                except Exception:
                    # Якщо власний формат недоступний, зберігаємо стандартний state_dict
                    torch.save(self.model.state_dict(), model_path)
                print(f"Модель збережено в {model_path}")
            
            # Зберігаємо метрики
            metrics_path = os.path.join(self.output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Виводимо статистику епохи
            epoch_time = time.time() - epoch_start_time
            print(f"Епоха {epoch+1}/{epochs} завершена за {epoch_time:.2f} сек, Втрата: {epoch_loss:.4f}" + 
                  (f", Валідаційна втрата: {val_loss:.4f}" if val_loss else ""))
        
        print(f"Навчання завершено. Модель та метрики збережено в {self.output_dir}")
        return metrics


def main():
    """Основна функція для запуску з командного рядка"""
    parser = argparse.ArgumentParser(description="Навчання моделі генерації зображень")
    parser.add_argument("--dataset", type=str, required=True, help="Шлях до тренувального датасету")
    parser.add_argument("--val_dataset", type=str, help="Шлях до валідаційного датасету")
    parser.add_argument("--output_dir", type=str, default="trained_model", help="Директорія для збереження моделі")
    parser.add_argument("--pretrained_model", type=str, help="Шлях до попередньо навченої моделі")
    parser.add_argument("--batch_size", type=int, default=4, help="Розмір батчу")
    parser.add_argument("--epochs", type=int, default=10, help="Кількість епох")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Швидкість навчання")
    parser.add_argument("--save_interval", type=int, default=2, help="Інтервал збереження моделі (епохи)")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Вимкнути mixed precision")
    parser.add_argument("--no_memory_optimization", action="store_true", help="Вимкнути оптимізацію пам'яті")
    
    args = parser.parse_args()
    
    # Ініціалізуємо тренер
    trainer = ModelTrainer(output_dir=args.output_dir)
    
    # Запускаємо навчання
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


if __name__ == "__main__":
    main()