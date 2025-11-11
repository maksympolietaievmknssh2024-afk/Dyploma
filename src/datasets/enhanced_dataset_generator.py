#!/usr/bin/env python3
"""
Скрипт для поповнення датасету з оптимізацією ваги.
Генерує додаткові дані на основі існуючих, використовуючи варіації та комбінації,
без необхідності зберігати додаткові зображення.
"""

import json
import os
import random
import argparse
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import Counter

# Додаємо шлях до кореневої директорії проекту
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.text_processing import EnhancedTextProcessor
from src.datasets.dataset_processor import DatasetProcessor

class EnhancedDatasetGenerator:
    """Клас для генерації розширеного датасету з оптимізацією ваги"""
    
    def __init__(self, output_dir="enhanced_datasets"):
        """
        Ініціалізація генератора датасетів
        
        Args:
            output_dir: Директорія для збереження результатів
        """
        self.output_dir = output_dir
        self.text_processor = EnhancedTextProcessor()
        self.dataset_processor = DatasetProcessor(output_dir=output_dir)
        
        # Створюємо директорію для результатів, якщо вона не існує
        os.makedirs(output_dir, exist_ok=True)
        
        # Розширені категорії об'єктів для генерації
        self.categories = {
            "animals": [
                "cat", "dog", "horse", "elephant", "lion", "tiger", "bear", "wolf", 
                "fox", "rabbit", "deer", "giraffe", "zebra", "panda", "koala", "penguin",
                "eagle", "owl", "parrot", "flamingo", "peacock", "swan", "dolphin", "whale"
            ],
            "objects": [
                "car", "bicycle", "motorcycle", "bus", "train", "airplane", "boat", "ship",
                "house", "building", "castle", "bridge", "tower", "lighthouse", "windmill",
                "chair", "table", "bed", "sofa", "lamp", "clock", "mirror", "vase"
            ],
            "nature": [
                "tree", "flower", "rose", "sunflower", "tulip", "daisy", "lily", "orchid",
                "mountain", "hill", "valley", "river", "lake", "ocean", "beach", "forest",
                "desert", "waterfall", "rainbow", "sunset", "sunrise", "moon", "stars"
            ],
            "people": [
                "man", "woman", "child", "boy", "girl", "person", "family", "couple",
                "artist", "musician", "dancer", "singer", "athlete", "teacher", "doctor"
            ],
            "abstract": [
                "dream", "idea", "thought", "emotion", "feeling", "memory", "fantasy",
                "imagination", "concept", "vision", "illusion", "reality", "time", "space"
            ],
            "urban": [
                "city", "street", "building", "skyscraper", "apartment", "shop", "store",
                "restaurant", "cafe", "park", "garden", "square", "market", "station"
            ]
        }
        
        # Атрибути для генерації
        self.attributes = {
            "colors": [
                "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
                "black", "white", "gray", "silver", "gold", "turquoise", "violet"
            ],
            "adjectives": [
                "beautiful", "elegant", "majestic", "tiny", "huge", "ancient", "modern",
                "bright", "dark", "shiny", "matte", "smooth", "rough", "soft", "hard"
            ],
            "environments": [
                "in a garden", "in a forest", "on a beach", "in the mountains", "in a city",
                "in a park", "by a lake", "in a field", "in the desert", "in the snow",
                "under the stars", "at sunset", "at sunrise", "in the rain"
            ],
            "styles": [
                "realistic", "artistic", "abstract", "impressionist", "surreal", "minimalist",
                "detailed", "sketch-like", "watercolor", "oil painting", "digital art",
                "vintage", "modern", "futuristic", "fantasy"
            ]
        }
    
    def analyze_dataset(self, dataset_path):
        """
        Аналізує існуючий датасет для розуміння його структури
        
        Args:
            dataset_path: Шлях до датасету
            
        Returns:
            Словник з інформацією про датасет
        """
        print(f"Аналіз датасету: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Якщо датасет має структуру з samples
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
            metadata = data.get("metadata", {})
        else:
            samples = data
            metadata = {}
        
        # Аналіз категорій
        categories = {}
        prompt_lengths = []
        
        for item in samples:
            category = item.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            
            prompt = item.get("prompt", "")
            prompt_lengths.append(len(prompt))
        
        avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        
        print(f"Знайдено {len(samples)} зразків")
        print(f"Середня довжина промпту: {avg_prompt_length:.1f} символів")
        print("Розподіл категорій:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {count} ({count/len(samples)*100:.1f}%)")
        
        return {
            "count": len(samples),
            "categories": categories,
            "avg_prompt_length": avg_prompt_length,
            "metadata": metadata,
            "samples": samples
        }
    
    def generate_variations(self, prompt, category, num_variations=3):
        """
        Генерує варіації промпту
        
        Args:
            prompt: Оригінальний промпт
            category: Категорія промпту
            num_variations: Кількість варіацій для генерації
            
        Returns:
            Список варіацій промпту
        """
        variations = []
        
        # Базова варіація - додавання атрибутів
        for _ in range(num_variations):
            # Вибираємо випадкові атрибути
            color = random.choice(self.attributes["colors"])
            adjective = random.choice(self.attributes["adjectives"])
            environment = random.choice(self.attributes["environments"])
            style = random.choice(self.attributes["styles"])
            
            # Створюємо варіації з різними комбінаціями атрибутів
            variation_type = random.randint(1, 4)
            
            if variation_type == 1:
                # Додаємо колір та прикметник
                new_prompt = f"{adjective} {color} {prompt}"
            elif variation_type == 2:
                # Додаємо оточення
                new_prompt = f"{prompt} {environment}"
            elif variation_type == 3:
                # Додаємо стиль
                new_prompt = f"{prompt}, {style}"
            else:
                # Комбінуємо кілька атрибутів
                new_prompt = f"{adjective} {prompt} {environment}, {style}"
            
            # Додаємо варіацію, якщо вона унікальна
            if new_prompt not in variations and new_prompt != prompt:
                variations.append(new_prompt)
        
        # Якщо категорія відома, додаємо специфічні для категорії варіації
        if category in self.categories:
            # Вибираємо випадковий об'єкт з тієї ж категорії
            related_object = random.choice(self.categories[category])
            
            # Створюємо варіацію з пов'язаним об'єктом
            related_variation = f"{prompt} with {related_object}"
            if related_variation not in variations and related_variation != prompt:
                variations.append(related_variation)
        
        # Використовуємо EnhancedTextProcessor для створення додаткових варіацій
        enhanced_prompt = self.text_processor.enhance_prompt(prompt)
        if enhanced_prompt not in variations and enhanced_prompt != prompt:
            variations.append(enhanced_prompt)
        
        return variations[:num_variations]  # Обмежуємо кількість варіацій
    
    def generate_enhanced_dataset(self, dataset_path, expansion_factor=0.5, max_variations=3):
        """
        Генерує розширений датасет на основі існуючого
        
        Args:
            dataset_path: Шлях до оригінального датасету
            expansion_factor: Коефіцієнт розширення (0.5 = +50% нових зразків)
            max_variations: Максимальна кількість варіацій для кожного зразка
            
        Returns:
            Шлях до згенерованого датасету
        """
        print(f"Генерація розширеного датасету на основі: {dataset_path}")
        print(f"Коефіцієнт розширення: {expansion_factor} (додамо {int(expansion_factor*100)}% нових зразків)")
        
        # Аналізуємо оригінальний датасет
        dataset_info = self.analyze_dataset(dataset_path)
        original_samples = dataset_info["samples"]
        metadata = dataset_info["metadata"]
        
        # Визначаємо кількість нових зразків для генерації
        num_original = len(original_samples)
        num_to_generate = int(num_original * expansion_factor)
        
        print(f"Генеруємо {num_to_generate} нових зразків...")
        
        # Вибираємо випадкові зразки для створення варіацій
        indices_to_vary = random.sample(range(num_original), min(num_to_generate, num_original))
        
        # Генеруємо нові зразки
        new_samples = []
        
        for idx in tqdm(indices_to_vary):
            original = original_samples[idx]
            prompt = original.get("prompt", "")
            category = original.get("category", "unknown")
            
            # Пропускаємо порожні промпти
            if not prompt:
                continue
            
            # Генеруємо варіації промпту
            variations = self.generate_variations(
                prompt, 
                category, 
                num_variations=min(max_variations, int(num_to_generate / len(indices_to_vary)) + 1)
            )
            
            # Створюємо нові зразки на основі варіацій
            for variation in variations:
                # Копіюємо оригінальний зразок і змінюємо промпт
                new_sample = original.copy()
                new_sample["prompt"] = variation
                
                # Додаємо мітку, що це згенерований зразок
                new_sample["generated"] = True
                
                # Додаємо до списку нових зразків
                new_samples.append(new_sample)
                
                # Якщо досягли потрібної кількості, зупиняємося
                if len(new_samples) >= num_to_generate:
                    break
            
            # Якщо досягли потрібної кількості, зупиняємося
            if len(new_samples) >= num_to_generate:
                break
        
        # Об'єднуємо оригінальні та нові зразки
        combined_samples = original_samples + new_samples
        
        # Оновлюємо метадані
        updated_metadata = metadata.copy()
        updated_metadata["description"] = f"Enhanced dataset with {len(new_samples)} additional generated samples"
        updated_metadata["generation_date"] = datetime.now().isoformat()
        updated_metadata["original_dataset"] = os.path.basename(dataset_path)
        updated_metadata["samples_count"] = len(combined_samples)
        
        # Створюємо новий датасет
        enhanced_dataset = {
            "metadata": updated_metadata,
            "samples": combined_samples
        }
        
        # Визначаємо ім'я вихідного файлу
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(self.output_dir, f"{dataset_name}_enhanced.json")
        
        # Зберігаємо датасет
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dataset, f, ensure_ascii=False, indent=None, separators=(',', ':'))
        
        print(f"Згенеровано {len(new_samples)} нових зразків")
        print(f"Загальна кількість зразків: {len(combined_samples)}")
        print(f"Датасет збережено: {output_path}")
        
        # Оптимізуємо датасет для зменшення розміру
        optimized_path = self.optimize_dataset(output_path)
        
        return optimized_path
    
    def optimize_dataset(self, dataset_path):
        """
        Оптимізує датасет для зменшення розміру
        
        Args:
            dataset_path: Шлях до датасету
            
        Returns:
            Шлях до оптимізованого датасету
        """
        print(f"Оптимізація датасету: {dataset_path}")
        
        # Реалізуємо власну оптимізацію замість використання DatasetProcessor
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Визначаємо структуру даних
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
            metadata = data.get("metadata", {})
            has_samples_structure = True
        else:
            samples = data
            metadata = {}
            has_samples_structure = False
        
        original_size = len(samples)
        
        # Видалення дублікатів
        unique_samples = []
        seen_prompts = set()
        
        for item in samples:
            prompt = item.get("prompt", "") if isinstance(item, dict) else ""
            if prompt not in seen_prompts:
                unique_samples.append(item)
                seen_prompts.add(prompt)
        
        samples = unique_samples
        print(f"Видалено {original_size - len(samples)} дублікатів")
        
        # Створюємо оптимізований датасет
        if has_samples_structure:
            optimized_data = {
                "metadata": metadata,
                "samples": samples
            }
        else:
            optimized_data = samples
        
        # Визначаємо ім'я вихідного файлу
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(self.output_dir, f"{dataset_name}_optimized.json")
        
        # Зберігаємо оптимізований датасет
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, ensure_ascii=False, separators=(',', ':'))
        
        original_file_size = os.path.getsize(dataset_path)
        optimized_file_size = os.path.getsize(output_path)
        
        print(f"Розмір датасету зменшено з {original_file_size/1024/1024:.2f} МБ до {optimized_file_size/1024/1024:.2f} МБ")
        return output_path
    
    def analyze_categories(self, dataset_path):
        """
        Аналізує категорії в датасеті та виявляє недостатньо представлені
        
        Args:
            dataset_path: Шлях до датасету
            
        Returns:
            Словник з інформацією про категорії
        """
        dataset_info = self.analyze_dataset(dataset_path)
        categories = dataset_info["categories"]
        
        # Знаходимо найменш представлені категорії
        sorted_categories = sorted(categories.items(), key=lambda x: x[1])
        
        print("Найменш представлені категорії:")
        for category, count in sorted_categories[:3]:
            print(f"  - {category}: {count}")
        
        return {
            "categories": categories,
            "least_represented": sorted_categories[:3]
        }
    
    def balance_categories(self, dataset_path, target_ratio=0.8):
        """
        Балансує категорії в датасеті, додаючи більше зразків для недостатньо представлених категорій
        
        Args:
            dataset_path: Шлях до датасету
            target_ratio: Цільове співвідношення (відносно найбільшої категорії)
            
        Returns:
            Шлях до збалансованого датасету
        """
        print(f"Балансування категорій в датасеті: {dataset_path}")
        
        # Аналізуємо датасет
        dataset_info = self.analyze_dataset(dataset_path)
        samples = dataset_info["samples"]
        metadata = dataset_info["metadata"]
        categories = dataset_info["categories"]
        
        # Знаходимо найбільшу категорію
        max_category, max_count = max(categories.items(), key=lambda x: x[1])
        
        # Визначаємо цільову кількість для кожної категорії
        target_counts = {}
        categories_to_expand = {}
        
        for category, count in categories.items():
            target_count = int(max_count * target_ratio)
            if count < target_count:
                categories_to_expand[category] = target_count - count
            target_counts[category] = target_count
        
        print("Цільові кількості для категорій:")
        for category, target in target_counts.items():
            current = categories[category]
            diff = target - current
            print(f"  - {category}: {current} -> {target} ({diff:+})")
        
        # Якщо немає категорій для розширення, повертаємо оригінальний датасет
        if not categories_to_expand:
            print("Всі категорії вже збалансовані")
            return dataset_path
        
        # Групуємо зразки за категоріями
        samples_by_category = {}
        for sample in samples:
            category = sample.get("category", "unknown")
            if category not in samples_by_category:
                samples_by_category[category] = []
            samples_by_category[category].append(sample)
        
        # Генеруємо додаткові зразки для недостатньо представлених категорій
        new_samples = []
        
        for category, num_to_add in categories_to_expand.items():
            print(f"Генерація {num_to_add} нових зразків для категорії '{category}'")
            
            # Якщо категорія не має зразків, пропускаємо
            if category not in samples_by_category or not samples_by_category[category]:
                continue
            
            # Вибираємо випадкові зразки для створення варіацій
            category_samples = samples_by_category[category]
            
            # Визначаємо, скільки варіацій потрібно для кожного зразка
            variations_per_sample = max(1, int(num_to_add / len(category_samples)) + 1)
            
            # Генеруємо нові зразки
            added_for_category = 0
            
            while added_for_category < num_to_add:
                # Вибираємо випадковий зразок
                original = random.choice(category_samples)
                prompt = original.get("prompt", "")
                
                # Пропускаємо порожні промпти
                if not prompt:
                    continue
                
                # Генеруємо варіації промпту
                variations = self.generate_variations(
                    prompt, 
                    category, 
                    num_variations=variations_per_sample
                )
                
                # Створюємо нові зразки на основі варіацій
                for variation in variations:
                    # Копіюємо оригінальний зразок і змінюємо промпт
                    new_sample = original.copy()
                    new_sample["prompt"] = variation
                    
                    # Додаємо мітку, що це згенерований зразок
                    new_sample["generated"] = True
                    new_sample["balanced"] = True
                    
                    # Додаємо до списку нових зразків
                    new_samples.append(new_sample)
                    added_for_category += 1
                    
                    # Якщо досягли потрібної кількості, зупиняємося
                    if added_for_category >= num_to_add:
                        break
        
        # Об'єднуємо оригінальні та нові зразки
        combined_samples = samples + new_samples
        
        # Оновлюємо метадані
        updated_metadata = metadata.copy()
        updated_metadata["description"] = f"Balanced dataset with {len(new_samples)} additional generated samples"
        updated_metadata["balancing_date"] = datetime.now().isoformat()
        updated_metadata["original_dataset"] = os.path.basename(dataset_path)
        updated_metadata["samples_count"] = len(combined_samples)
        
        # Створюємо новий датасет
        balanced_dataset = {
            "metadata": updated_metadata,
            "samples": combined_samples
        }
        
        # Визначаємо ім'я вихідного файлу
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(self.output_dir, f"{dataset_name}_balanced.json")
        
        # Зберігаємо датасет
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(balanced_dataset, f, ensure_ascii=False, indent=None, separators=(',', ':'))
        
        print(f"Згенеровано {len(new_samples)} нових зразків")
        print(f"Загальна кількість зразків: {len(combined_samples)}")
        print(f"Датасет збережено: {output_path}")
        
        # Оптимізуємо датасет для зменшення розміру
        optimized_path = self.optimize_dataset(output_path)
        
        return optimized_path

def main():
    """Головна функція для запуску генератора датасетів"""
    parser = argparse.ArgumentParser(description="Генерація розширеного датасету з оптимізацією ваги")
    
    # Загальні параметри
    parser.add_argument("--dataset", required=True, help="Шлях до оригінального датасету")
    parser.add_argument("--output_dir", default="enhanced_datasets", help="Директорія для збереження результатів")
    
    # Підкоманди
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")
    
    # Команда для аналізу датасету
    analyze_parser = subparsers.add_parser("analyze", help="Аналіз датасету")
    
    # Команда для генерації розширеного датасету
    enhance_parser = subparsers.add_parser("enhance", help="Генерація розширеного датасету")
    enhance_parser.add_argument("--expansion_factor", type=float, default=0.5, 
                               help="Коефіцієнт розширення (0.5 = +50% нових зразків)")
    enhance_parser.add_argument("--max_variations", type=int, default=3, 
                               help="Максимальна кількість варіацій для кожного зразка")
    
    # Команда для балансування категорій
    balance_parser = subparsers.add_parser("balance", help="Балансування категорій")
    balance_parser.add_argument("--target_ratio", type=float, default=0.8, 
                               help="Цільове співвідношення (відносно найбільшої категорії)")
    
    # Команда для оптимізації датасету
    optimize_parser = subparsers.add_parser("optimize", help="Оптимізація датасету")
    
    args = parser.parse_args()
    
    # Створюємо генератор датасетів
    generator = EnhancedDatasetGenerator(output_dir=args.output_dir)
    
    # Виконуємо відповідну команду
    if args.command == "analyze":
        generator.analyze_dataset(args.dataset)
    
    elif args.command == "enhance":
        generator.generate_enhanced_dataset(
            args.dataset,
            expansion_factor=args.expansion_factor,
            max_variations=args.max_variations
        )
    
    elif args.command == "balance":
        generator.balance_categories(
            args.dataset,
            target_ratio=args.target_ratio
        )
    
    elif args.command == "optimize":
        generator.optimize_dataset(args.dataset)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерація розширеного датасету з оптимізацією ваги")
    
    # Загальні параметри
    parser.add_argument("--dataset", required=True, help="Шлях до оригінального датасету")
    parser.add_argument("--output_dir", default="enhanced_datasets", help="Директорія для збереження результатів")
    
    # Підкоманди
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")
    
    # Команда для аналізу датасету
    analyze_parser = subparsers.add_parser("analyze", help="Аналіз датасету")
    
    # Команда для генерації розширеного датасету
    enhance_parser = subparsers.add_parser("enhance", help="Генерація розширеного датасету")
    enhance_parser.add_argument("--expansion_factor", type=float, default=0.5, 
                               help="Коефіцієнт розширення (0.5 = +50% нових зразків)")
    enhance_parser.add_argument("--max_variations", type=int, default=3, 
                               help="Максимальна кількість варіацій для кожного зразка")
    
    # Команда для балансування категорій
    balance_parser = subparsers.add_parser("balance", help="Балансування категорій")
    balance_parser.add_argument("--target_ratio", type=float, default=0.8, 
                               help="Цільове співвідношення (відносно найбільшої категорії)")
    
    # Команда для оптимізації датасету
    optimize_parser = subparsers.add_parser("optimize", help="Оптимізація датасету")
    
    args = parser.parse_args()
    
    # Створюємо генератор датасетів
    generator = EnhancedDatasetGenerator(output_dir=args.output_dir)
    
    # Виконуємо відповідну команду
    if args.command == "analyze":
        generator.analyze_dataset(args.dataset)
    
    elif args.command == "enhance":
        generator.generate_enhanced_dataset(
            args.dataset,
            expansion_factor=args.expansion_factor,
            max_variations=args.max_variations
        )
    
    elif args.command == "balance":
        generator.balance_categories(
            args.dataset,
            target_ratio=args.target_ratio
        )
    
    elif args.command == "optimize":
        generator.optimize_dataset(args.dataset)
    
    else:
        parser.print_help()