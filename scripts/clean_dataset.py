#!/usr/bin/env python3
"""
Скрипт для очищення датасету від записів з image_path
"""

import json
import os
import shutil
from pathlib import Path

def clean_dataset(input_file, output_file):
    """
    Очищає датасет від записів з image_path
    """
    print(f"Читаю датасет з {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Загальна кількість записів: {len(data)}")
    
    # Фільтруємо записи без image_path
    cleaned_data = [item for item in data if 'image_path' not in item]
    
    print(f"Записів з image_path видалено: {len(data) - len(cleaned_data)}")
    print(f"Залишилось записів: {len(cleaned_data)}")
    
    # Зберігаємо очищений датасет
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"Очищений датасет збережено в {output_file}")
    
    return len(cleaned_data)

def main():
    # Шляхи до файлів
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "combined_dataset_train.json"
    output_file = base_dir / "combined_dataset_train_cleaned.json"
    
    # Створюємо резервну копію оригінального файлу
    backup_file = base_dir / "combined_dataset_train_backup.json"
    if not backup_file.exists():
        shutil.copy2(input_file, backup_file)
        print(f"Створено резервну копію: {backup_file}")
    
    # Очищаємо датасет
    cleaned_count = clean_dataset(input_file, output_file)
    
    # Замінюємо оригінальний файл очищеним
    shutil.move(output_file, input_file)
    print(f"Оригінальний файл замінено очищеною версією")
    
    print(f"\n✅ Датасет успішно очищено!")
    print(f"📊 Кількість валідних записів: {cleaned_count}")

if __name__ == "__main__":
    main()