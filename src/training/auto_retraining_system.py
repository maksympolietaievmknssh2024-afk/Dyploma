#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система автоматичного донавчання моделі на основі введених користувачем промптів
Збирає дані про використання та періодично перенавчає модель для покращення якості
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import torch
from pathlib import Path

from .model_trainer import ModelTrainer
from ..utils.text_processing import EnhancedTextProcessor

class AutoRetrainingSystem:
    def __init__(self, 
                 data_collection_dir: str = "user_data",
                 min_samples_for_retraining: int = 100,
                 retraining_interval_hours: int = 24,
                 max_dataset_size: int = 10000,
                 backup_models: bool = True):
        """
        Ініціалізація системи автоматичного донавчання
        
        Args:
            data_collection_dir: Директорія для збереження зібраних даних
            min_samples_for_retraining: Мінімальна кількість нових зразків для запуску донавчання
            retraining_interval_hours: Інтервал перевірки необхідності донавчання (години)
            max_dataset_size: Максимальний розмір датасету для донавчання
            backup_models: Чи створювати резервні копії моделей
        """
        self.data_collection_dir = Path(data_collection_dir)
        self.min_samples_for_retraining = min_samples_for_retraining
        self.retraining_interval_hours = retraining_interval_hours
        self.max_dataset_size = max_dataset_size
        self.backup_models = backup_models
        
        # Створюємо необхідні директорії
        self.data_collection_dir.mkdir(exist_ok=True)
        (self.data_collection_dir / "prompts").mkdir(exist_ok=True)
        (self.data_collection_dir / "feedback").mkdir(exist_ok=True)
        (self.data_collection_dir / "datasets").mkdir(exist_ok=True)
        (self.data_collection_dir / "models").mkdir(exist_ok=True)
        
        # Файли для збереження даних
        self.prompts_file = self.data_collection_dir / "prompts" / "user_prompts.jsonl"
        self.feedback_file = self.data_collection_dir / "feedback" / "user_feedback.jsonl"
        self.metadata_file = self.data_collection_dir / "metadata.json"
        
        # Ініціалізуємо компоненти
        self.text_processor = EnhancedTextProcessor()
        self.trainer = None
        
        # Налаштування логування
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Завантажуємо метадані
        self.metadata = self._load_metadata()
        
        # Запускаємо фоновий процес моніторингу
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def _load_metadata(self) -> Dict:
        """Завантаження метаданих системи"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "total_prompts_collected": 0,
                "last_retraining_time": None,
                "retraining_count": 0,
                "current_model_version": "base",
                "collection_start_time": datetime.now().isoformat()
            }
    
    def _save_metadata(self):
        """Збереження метаданих системи"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def collect_prompt_data(self, 
                           original_prompt: str,
                           enhanced_prompt: str,
                           negative_prompt: str = "",
                           detected_objects: List[str] = None,
                           generation_success: bool = True,
                           user_feedback: Optional[str] = None,
                           generation_time: float = 0.0):
        """
        Збір даних про використання промптів
        
        Args:
            original_prompt: Оригінальний промпт користувача
            enhanced_prompt: Покращений промпт після обробки
            negative_prompt: Негативний промпт
            detected_objects: Виявлені об'єкти в промпті
            generation_success: Чи успішно згенеровано зображення
            user_feedback: Відгук користувача (опціонально)
            generation_time: Час генерації в секундах
        """
        if detected_objects is None:
            detected_objects = []
            
        # Створюємо запис про використання
        prompt_data = {
            "timestamp": datetime.now().isoformat(),
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "detected_objects": detected_objects,
            "generation_success": generation_success,
            "generation_time": generation_time,
            "prompt_length": len(original_prompt),
            "language": self._detect_language(original_prompt),
            "complexity_score": self._calculate_complexity_score(original_prompt)
        }
        
        # Зберігаємо в файл
        with open(self.prompts_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
        
        # Оновлюємо метадані
        self.metadata["total_prompts_collected"] += 1
        self._save_metadata()
        
        # Зберігаємо відгук користувача, якщо є
        if user_feedback:
            self.collect_user_feedback(original_prompt, user_feedback)
        
        self.logger.info(f"Зібрано дані про промпт: {original_prompt[:50]}...")
    
    def collect_user_feedback(self, prompt: str, feedback: str, rating: Optional[int] = None):
        """
        Збір відгуків користувачів
        
        Args:
            prompt: Промпт, до якого відноситься відгук
            feedback: Текстовий відгук
            rating: Оцінка від 1 до 5 (опціонально)
        """
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "feedback": feedback,
            "rating": rating
        }
        
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Зібрано відгук користувача для промпту: {prompt[:30]}...")
    
    def _detect_language(self, text: str) -> str:
        """Простий детектор мови"""
        # Підрахунок кириличних символів
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        total_letters = sum(1 for char in text if char.isalpha())
        
        if total_letters == 0:
            return "unknown"
        
        cyrillic_ratio = cyrillic_count / total_letters
        
        if cyrillic_ratio > 0.5:
            return "ukrainian"
        else:
            return "english"
    
    def _calculate_complexity_score(self, prompt: str) -> float:
        """Розрахунок складності промпту"""
        words = prompt.split()
        
        # Базові метрики
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        unique_words = len(set(words))
        
        # Складність на основі різних факторів
        complexity = (
            word_count * 0.1 +  # Довжина
            avg_word_length * 0.2 +  # Складність слів
            (unique_words / max(word_count, 1)) * 0.3  # Різноманітність
        )
        
        return min(complexity, 10.0)  # Обмежуємо максимальним значенням
    
    def check_retraining_needed(self) -> bool:
        """Перевірка необхідності донавчання"""
        # Перевіряємо кількість нових зразків
        if not self.prompts_file.exists():
            return False
        
        # Підраховуємо кількість зразків з останнього донавчання
        last_retraining = self.metadata.get("last_retraining_time")
        
        new_samples_count = 0
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if last_retraining is None or data["timestamp"] > last_retraining:
                    if data["generation_success"]:  # Враховуємо тільки успішні генерації
                        new_samples_count += 1
        
        # Перевіряємо умови для донавчання
        enough_samples = new_samples_count >= self.min_samples_for_retraining
        
        # Перевіряємо часовий інтервал
        time_passed = True
        if last_retraining:
            last_time = datetime.fromisoformat(last_retraining)
            time_passed = datetime.now() - last_time >= timedelta(hours=self.retraining_interval_hours)
        
        self.logger.info(f"Перевірка донавчання: {new_samples_count} нових зразків, "
                        f"потрібно: {self.min_samples_for_retraining}, "
                        f"час пройшов: {time_passed}")
        
        return enough_samples and time_passed
    
    def prepare_retraining_dataset(self) -> str:
        """Підготовка датасету для донавчання"""
        dataset_path = self.data_collection_dir / "datasets" / f"retraining_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Збираємо всі успішні промпти
        training_samples = []
        
        if self.prompts_file.exists():
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data["generation_success"]:
                        # Створюємо зразок для навчання
                        sample = {
                            "prompt": data["enhanced_prompt"],
                            "original_prompt": data["original_prompt"],
                            "negative_prompt": data["negative_prompt"],
                            "objects": data["detected_objects"],
                            "timestamp": data["timestamp"],
                            "language": data["language"],
                            "complexity": data["complexity_score"]
                        }
                        training_samples.append(sample)
        
        # Обмежуємо розмір датасету
        if len(training_samples) > self.max_dataset_size:
            # Сортуємо за часом та беремо найновіші
            training_samples.sort(key=lambda x: x["timestamp"], reverse=True)
            training_samples = training_samples[:self.max_dataset_size]
        
        # Зберігаємо датасет
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Підготовлено датасет для донавчання: {len(training_samples)} зразків")
        return str(dataset_path)
    
    def start_retraining(self, base_model_path: str = None) -> bool:
        """
        Запуск процесу донавчання
        
        Args:
            base_model_path: Шлях до базової моделі для донавчання
            
        Returns:
            True якщо донавчання успішно запущено
        """
        try:
            # Підготовка датасету
            dataset_path = self.prepare_retraining_dataset()
            
            # Створення директорії для нової моделі
            model_version = f"retrained_v{self.metadata['retraining_count'] + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = self.data_collection_dir / "models" / model_version
            
            # Резервне копіювання поточної моделі
            if self.backup_models and base_model_path and os.path.exists(base_model_path):
                backup_dir = self.data_collection_dir / "models" / f"backup_{self.metadata['current_model_version']}"
                backup_dir.mkdir(exist_ok=True)
                # Тут можна додати логіку копіювання моделі
            
            # Ініціалізація тренера
            self.trainer = ModelTrainer(output_dir=str(output_dir))
            
            # Запуск донавчання в окремому потоці
            retraining_thread = threading.Thread(
                target=self._run_retraining,
                args=(dataset_path, str(output_dir), base_model_path)
            )
            retraining_thread.daemon = True
            retraining_thread.start()
            
            self.logger.info(f"Запущено донавчання моделі: {model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Помилка при запуску донавчання: {e}")
            return False
    
    def _run_retraining(self, dataset_path: str, output_dir: str, base_model_path: str = None):
        """Виконання донавчання в окремому потоці"""
        try:
            # Параметри донавчання
            training_params = {
                "batch_size": 2,  # Малий батч для економії пам'яті
                "epochs": 5,  # Невелика кількість епох для донавчання
                "learning_rate": 1e-6,  # Низька швидкість навчання для fine-tuning
                "save_interval": 2,
                "use_mixed_precision": True,
                "optimize_memory": True
            }
            
            # Запуск навчання
            self.trainer.train(
                dataset_path=dataset_path,
                pretrained_model_path=base_model_path,
                **training_params
            )
            
            # Оновлення метаданих після успішного донавчання
            self.metadata["last_retraining_time"] = datetime.now().isoformat()
            self.metadata["retraining_count"] += 1
            self.metadata["current_model_version"] = os.path.basename(output_dir)
            self._save_metadata()
            
            self.logger.info("Донавчання успішно завершено")
            
        except Exception as e:
            self.logger.error(f"Помилка під час донавчання: {e}")
    
    def start_monitoring(self):
        """Запуск фонового моніторингу для автоматичного донавчання"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Моніторинг вже запущено")
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Запущено фоновий моніторинг для автоматичного донавчання")
    
    def stop_monitoring_process(self):
        """Зупинка фонового моніторингу"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Зупинено фоновий моніторинг")
    
    def _monitoring_loop(self):
        """Основний цикл моніторингу"""
        while not self.stop_monitoring:
            try:
                if self.check_retraining_needed():
                    self.logger.info("Виявлено необхідність донавчання, запускаємо процес...")
                    self.start_retraining()
                
                # Чекаємо перед наступною перевіркою (1 година)
                for _ in range(3600):  # 3600 секунд = 1 година
                    if self.stop_monitoring:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Помилка в циклі моніторингу: {e}")
                time.sleep(60)  # Чекаємо хвилину перед повторною спробою
    
    def get_statistics(self) -> Dict:
        """Отримання статистики системи"""
        stats = {
            "total_prompts": self.metadata["total_prompts_collected"],
            "retraining_count": self.metadata["retraining_count"],
            "current_model_version": self.metadata["current_model_version"],
            "last_retraining": self.metadata.get("last_retraining_time"),
            "collection_start": self.metadata["collection_start_time"]
        }
        
        # Додаткова статистика з файлів
        if self.prompts_file.exists():
            language_stats = {"english": 0, "ukrainian": 0, "unknown": 0}
            success_rate = {"successful": 0, "failed": 0}
            
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    language_stats[data.get("language", "unknown")] += 1
                    if data["generation_success"]:
                        success_rate["successful"] += 1
                    else:
                        success_rate["failed"] += 1
            
            stats["language_distribution"] = language_stats
            stats["success_rate"] = success_rate
        
        return stats