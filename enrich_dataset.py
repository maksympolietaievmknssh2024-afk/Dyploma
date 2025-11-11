import json
import os
import random
import copy
from tqdm import tqdm
from collections import defaultdict

def load_dataset(file_path):
    """Завантаження датасету з JSON файлу"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data, file_path):
    """Збереження датасету в JSON файл"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Датасет збережено: {file_path}")
    print(f"Розмір файлу: {os.path.getsize(file_path) / (1024*1024):.2f} МБ")

def analyze_dataset(data):
    """Аналіз датасету"""
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    else:
        samples = data
    
    print(f"Кількість зразків: {len(samples)}")
    
    # Аналіз категорій
    categories = defaultdict(int)
    for sample in samples:
        # Визначення категорії на основі ключових слів у промпті
        prompt = sample.get("prompt", "")
        category = sample.get("category", "unknown")
        
        # Додаткова класифікація за ключовими словами
        if "людина" in prompt or "людей" in prompt or "людини" in prompt:
            categories["human"] += 1
        elif "пейзаж" in prompt or "природа" in prompt:
            categories["landscape"] += 1
        elif "тварина" in prompt or "тварини" in prompt:
            categories["animal"] += 1
        elif "архітектура" in prompt or "будівля" in prompt or "будинок" in prompt:
            categories["architecture"] += 1
        else:
            categories[category] += 1
    
    print("\nРозподіл за категоріями:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(samples) * 100
        print(f"  - {category}: {count} ({percentage:.1f}%)")
    
    return samples, categories

def generate_variations(prompt, n=5):
    """Генерація варіацій промпту з більшою кількістю варіантів"""
    variations = []
    
    # Базові модифікатори для українських промптів
    enhancers = [
        "детально", "яскраво", "реалістично", "високоякісно", 
        "професійно", "чітко", "з деталями", "з високою роздільною здатністю",
        "з гарним освітленням", "з гарною композицією",
        "з драматичним освітленням", "з м'яким освітленням", "з контрастним освітленням",
        "з ефектом глибини", "з ефектом перспективи", "з ефектом боке"
    ]
    
    # Стилі
    styles = [
        "у стилі фотореалізму", "у мінімалістичному стилі", 
        "у стилі акварелі", "у стилі олійного живопису",
        "у стилі цифрового мистецтва", "у стилі аніме",
        "у стилі імпресіонізму", "у стилі експресіонізму", "у стилі кубізму",
        "у стилі сюрреалізму", "у стилі поп-арт", "у стилі графіки"
    ]
    
    # Українські модифікатори
    ukrainian_styles = [
        "у українському стилі", "з українськими мотивами",
        "з елементами петриківського розпису", "у стилі українського бароко",
        "з козацькими мотивами", "з елементами трипільської культури",
        "у карпатському стилі", "з гуцульськими візерунками",
        "з подільськими мотивами", "у стилі української вишиванки"
    ]
    
    # Генеруємо варіації
    for _ in range(n):
        new_prompt = prompt
        modifiers_added = 0
        
        # 60% шанс додати модифікатор
        if random.random() > 0.4 and modifiers_added < 3:
            enhancer = random.choice(enhancers)
            if enhancer not in new_prompt:
                new_prompt = f"{new_prompt}, {enhancer}"
                modifiers_added += 1
        
        # 40% шанс додати стиль
        if random.random() > 0.6 and modifiers_added < 3:
            style = random.choice(styles)
            if style not in new_prompt:
                new_prompt = f"{new_prompt}, {style}"
                modifiers_added += 1
        
        # 30% шанс додати український стиль
        if random.random() > 0.7 and modifiers_added < 3:
            ukr_style = random.choice(ukrainian_styles)
            if ukr_style not in new_prompt:
                new_prompt = f"{new_prompt}, {ukr_style}"
                modifiers_added += 1
        
        if new_prompt != prompt and new_prompt not in variations:
            variations.append(new_prompt)
    
    return variations

def enrich_dataset(input_file, output_file, expansion_factor=0.6):
    """Збагачення датасету з оптимізацією ваги та більшою кількістю варіацій"""
    print(f"Збагачення датасету: {input_file}")
    
    # Завантаження датасету
    data = load_dataset(input_file)
    
    # Аналіз датасету
    samples, categories = analyze_dataset(data)
    
    # Визначаємо кількість нових зразків для генерації
    total_samples = len(samples)
    new_samples_count = int(total_samples * expansion_factor)
    print(f"\nГенерація {new_samples_count} нових зразків...")
    
    # Генеруємо нові зразки
    new_samples = []
    
    # Вибираємо випадкові зразки для генерації варіацій
    selected_indices = random.sample(range(total_samples), min(new_samples_count, total_samples))
    
    for idx in tqdm(selected_indices):
        sample = samples[idx]
        prompt = sample.get("prompt", "")
        
        # Генеруємо варіації промпту з більшою кількістю варіантів
        variations = generate_variations(prompt, n=5)
        
        # Додаємо нові зразки
        for variation in variations:
            if len(new_samples) >= new_samples_count:
                break
                
            new_sample = copy.deepcopy(sample)
            new_sample["prompt"] = variation
            new_samples.append(new_sample)
            
            if len(new_samples) >= new_samples_count:
                break
    
    print(f"Згенеровано {len(new_samples)} нових зразків")
    
    # Об'єднуємо оригінальний датасет з новими зразками
    if isinstance(data, dict) and "samples" in data:
        data["samples"].extend(new_samples)
        enriched_data = data
    else:
        enriched_data = samples + new_samples
    
    # Видаляємо дублікати
    unique_samples = []
    seen_prompts = set()
    
    if isinstance(enriched_data, dict) and "samples" in enriched_data:
        for item in enriched_data["samples"]:
            prompt = item.get("prompt", "")
            if prompt not in seen_prompts:
                unique_samples.append(item)
                seen_prompts.add(prompt)
        
        enriched_data["samples"] = unique_samples
    else:
        for item in enriched_data:
            prompt = item.get("prompt", "")
            if prompt not in seen_prompts:
                unique_samples.append(item)
                seen_prompts.add(prompt)
        
        enriched_data = unique_samples
    
    # Зберігаємо збагачений датасет
    save_dataset(enriched_data, output_file)
    
    # Аналізуємо збагачений датасет
    if isinstance(enriched_data, dict) and "samples" in enriched_data:
        analyze_dataset(enriched_data)
    else:
        analyze_dataset(enriched_data)
    
    return output_file

if __name__ == "__main__":
    # Параметри
    input_file = "combined_dataset_train.json"
    output_file = "enhanced_datasets/combined_dataset_train_super_enriched.json"
    expansion_factor = 0.6  # Збільшення на 60%
    
    # Збагачення датасету
    enrich_dataset(input_file, output_file, expansion_factor)