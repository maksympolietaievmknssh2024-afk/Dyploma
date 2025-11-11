"""
Модуль для обробки та підготовки датасетів
"""

import os
import json
import shutil
import random
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class DatasetProcessor:
    """Клас для обробки та підготовки датасетів для навчання моделей"""
    
    def __init__(self, output_dir: str = "processed_datasets"):
        """
        Ініціалізація процесора датасетів
        
        Args:
            output_dir: Директорія для збереження оброблених датасетів
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def combine_datasets(self, dataset_paths: List[str], output_filename: str) -> str:
        """
        Об'єднання декількох датасетів в один
        
        Args:
            dataset_paths: Список шляхів до датасетів для об'єднання
            output_filename: Ім'я вихідного файлу
            
        Returns:
            Шлях до об'єднаного датасету
        """
        combined_data = []
        
        for dataset_path in dataset_paths:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                elif isinstance(data, dict) and "data" in data:
                    combined_data.extend(data["data"])
                else:
                    print(f"Непідтримуваний формат датасету: {dataset_path}")
        
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"Об'єднано {len(combined_data)} записів з {len(dataset_paths)} датасетів")
        return output_path
    
    def split_dataset(self, dataset_path: str, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
                     shuffle: bool = True, seed: Optional[int] = None) -> Tuple[str, str, str]:
        """
        Розділення датасету на тренувальну, валідаційну та тестову частини
        
        Args:
            dataset_path: Шлях до датасету
            train_ratio: Частка даних для тренування
            val_ratio: Частка даних для валідації
            test_ratio: Частка даних для тестування
            shuffle: Чи перемішувати дані
            seed: Seed для відтворюваності
            
        Returns:
            Кортеж шляхів до тренувального, валідаційного та тестового датасетів
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Сума співвідношень має дорівнювати 1"
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(data)
        
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        train_path = os.path.join(self.output_dir, f"{dataset_name}_train.json")
        val_path = os.path.join(self.output_dir, f"{dataset_name}_val.json")
        test_path = os.path.join(self.output_dir, f"{dataset_name}_test.json")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Датасет розділено: {len(train_data)} тренувальних, {len(val_data)} валідаційних, {len(test_data)} тестових записів")
        return train_path, val_path, test_path
    
    def cluster_dataset(self, dataset_path: str, image_dir: str, 
                       output_dir: str = None, n_clusters: int = 7,
                       method: str = "prompt") -> Dict[str, Any]:
        """
        Кластеризація датасету на основі зображень або текстових промптів
        
        Args:
            dataset_path: Шлях до датасету
            image_dir: Директорія з зображеннями
            output_dir: Директорія для збереження кластеризованого датасету
            n_clusters: Кількість кластерів
            method: Метод кластеризації ("prompt" або "image")
            
        Returns:
            Словник з інформацією про кластери
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "clustered_dataset")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Категорії для класифікації
        categories = {
            "nature": ["природа", "ліс", "гори", "море", "озеро", "річка", "пейзаж", "дерева", "квіти", "рослини"],
            "animals": ["тварини", "кіт", "собака", "птах", "риба", "ведмідь", "лев", "тигр", "слон", "корова"],
            "urban": ["місто", "будівля", "архітектура", "вулиця", "дорога", "міський", "будинок", "хмарочос"],
            "people": ["людина", "люди", "портрет", "обличчя", "жінка", "чоловік", "дитина", "дівчина", "хлопець"],
            "objects": ["об'єкт", "предмет", "стіл", "стілець", "телефон", "комп'ютер", "машина", "автомобіль"],
            "abstract": ["абстракція", "абстрактний", "фрактал", "геометричний", "візерунок", "текстура"]
        }
        
        if method == "prompt":
            # Кластеризація на основі текстових промптів
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            prompts = [item.get("prompt", "") for item in dataset]
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(prompts)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
        else:  # method == "image"
            # Кластеризація на основі зображень
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.models import Model
            
            # Завантаження моделі VGG16 без верхніх шарів
            base_model = VGG16(weights='imagenet', include_top=False)
            model = Model(inputs=base_model.input, outputs=base_model.output)
            
            # Отримання ознак зображень
            features = []
            for item in dataset:
                img_path = os.path.join(image_dir, item.get("image", ""))
                if os.path.exists(img_path):
                    try:
                        img = image.load_img(img_path, target_size=(224, 224))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        
                        feature = model.predict(x).flatten()
                        features.append(feature)
                    except Exception as e:
                        print(f"Помилка обробки зображення {img_path}: {e}")
                        features.append(np.zeros(512))  # Заповнення нулями у випадку помилки
                else:
                    features.append(np.zeros(512))  # Заповнення нулями, якщо зображення не знайдено
            
            # Кластеризація
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
        
        # Створення директорій для кластерів
        cluster_dirs = {}
        for i in range(n_clusters):
            cluster_dir = os.path.join(output_dir, f"cluster_{i}")
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_dirs[i] = cluster_dir
        
        # Аналіз кластерів та визначення категорій
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_stats[i] = {
                "count": 0,
                "prompts": [],
                "category": "uncategorized"
            }
        
        # Розподіл даних по кластерам
        for i, (item, cluster_id) in enumerate(zip(dataset, clusters)):
            cluster_stats[cluster_id]["count"] += 1
            cluster_stats[cluster_id]["prompts"].append(item.get("prompt", ""))
            
            # Копіювання зображення в директорію кластера
            if "image" in item and item["image"]:
                src_path = os.path.join(image_dir, item["image"])
                if os.path.exists(src_path):
                    dst_path = os.path.join(cluster_dirs[cluster_id], item["image"])
                    shutil.copy2(src_path, dst_path)
            
            # Збереження інформації про елемент
            item_path = os.path.join(cluster_dirs[cluster_id], f"item_{i}.json")
            with open(item_path, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        # Визначення категорій для кластерів
        for cluster_id, info in cluster_stats.items():
            prompts_text = " ".join(info["prompts"]).lower()
            
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in prompts_text)
                category_scores[category] = score
            
            if max(category_scores.values(), default=0) > 0:
                info["category"] = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Збереження статистики кластерів
        stats_path = os.path.join(output_dir, "cluster_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
        
        # Створення зведеного датасету для кожної категорії
        for category in set(info["category"] for info in cluster_stats.values()):
            category_clusters = [cluster_id for cluster_id, info in cluster_stats.items() 
                               if info["category"] == category]
            
            category_items = []
            for i, (item, cluster_id) in enumerate(zip(dataset, clusters)):
                if cluster_id in category_clusters:
                    category_items.append(item)
            
            if category_items:
                category_path = os.path.join(output_dir, f"{category}_dataset.json")
                with open(category_path, 'w', encoding='utf-8') as f:
                    json.dump(category_items, f, ensure_ascii=False, indent=2)
        
        print(f"Датасет кластеризовано на {n_clusters} кластерів")
        for cluster_id, info in cluster_stats.items():
            print(f"Кластер {cluster_id} ({info['category']}): {info['count']} елементів")
        
        return cluster_stats
    
    def optimize_dataset(self, dataset_path: str, remove_duplicates: bool = True, 
                        compress: bool = True) -> str:
        """
        Оптимізація датасету для зменшення розміру
        
        Args:
            dataset_path: Шлях до датасету
            remove_duplicates: Чи видаляти дублікати
            compress: Чи стискати формат JSON
            
        Returns:
            Шлях до оптимізованого датасету
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_size = len(data)
        
        if remove_duplicates:
            # Видалення дублікатів
            unique_data = []
            seen_prompts = set()
            
            for item in data:
                prompt = item.get("prompt", "")
                if prompt not in seen_prompts:
                    unique_data.append(item)
                    seen_prompts.add(prompt)
            
            data = unique_data
            print(f"Видалено {original_size - len(data)} дублікатів")
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(self.output_dir, f"{dataset_name}_optimized.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if compress:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        original_file_size = os.path.getsize(dataset_path)
        optimized_file_size = os.path.getsize(output_path)
        
        print(f"Розмір датасету зменшено з {original_file_size/1024/1024:.2f} МБ до {optimized_file_size/1024/1024:.2f} МБ")
        return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обробка датасетів")
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")
    
    # Парсер для об'єднання датасетів
    combine_parser = subparsers.add_parser("combine", help="Об'єднання датасетів")
    combine_parser.add_argument("--datasets", nargs="+", required=True, help="Шляхи до датасетів для об'єднання")
    combine_parser.add_argument("--output", required=True, help="Ім'я вихідного файлу")
    combine_parser.add_argument("--output_dir", default="processed_datasets", help="Директорія для збереження")
    
    # Парсер для розділення датасету
    split_parser = subparsers.add_parser("split", help="Розділення датасету")
    split_parser.add_argument("--dataset", required=True, help="Шлях до датасету")
    split_parser.add_argument("--train_ratio", type=float, default=0.8, help="Частка для тренування")
    split_parser.add_argument("--val_ratio", type=float, default=0.1, help="Частка для валідації")
    split_parser.add_argument("--test_ratio", type=float, default=0.1, help="Частка для тестування")
    split_parser.add_argument("--output_dir", default="processed_datasets", help="Директорія для збереження")
    split_parser.add_argument("--seed", type=int, help="Seed для відтворюваності")
    
    # Парсер для кластеризації датасету
    cluster_parser = subparsers.add_parser("cluster", help="Кластеризація датасету")
    cluster_parser.add_argument("--dataset", required=True, help="Шлях до датасету")
    cluster_parser.add_argument("--image_dir", required=True, help="Директорія з зображеннями")
    cluster_parser.add_argument("--output_dir", help="Директорія для збереження")
    cluster_parser.add_argument("--n_clusters", type=int, default=7, help="Кількість кластерів")
    cluster_parser.add_argument("--method", choices=["prompt", "image"], default="prompt", help="Метод кластеризації")
    
    # Парсер для оптимізації датасету
    optimize_parser = subparsers.add_parser("optimize", help="Оптимізація датасету")
    optimize_parser.add_argument("--dataset", required=True, help="Шлях до датасету")
    optimize_parser.add_argument("--no_remove_duplicates", action="store_true", help="Не видаляти дублікати")
    optimize_parser.add_argument("--no_compress", action="store_true", help="Не стискати формат JSON")
    optimize_parser.add_argument("--output_dir", default="processed_datasets", help="Директорія для збереження")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(output_dir=args.output_dir if hasattr(args, "output_dir") else "processed_datasets")
    
    if args.command == "combine":
        processor.combine_datasets(args.datasets, args.output)
    
    elif args.command == "split":
        processor.split_dataset(
            args.dataset, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    
    elif args.command == "cluster":
        processor.cluster_dataset(
            args.dataset,
            args.image_dir,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            method=args.method
        )
    
    elif args.command == "optimize":
        processor.optimize_dataset(
            args.dataset,
            remove_duplicates=not args.no_remove_duplicates,
            compress=not args.no_compress
        )
    
    else:
        parser.print_help()