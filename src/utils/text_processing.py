"""
Модуль для обробки текстових промптів
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Any, Optional

class EnhancedTextProcessor:
    """Клас для обробки та покращення текстових промптів"""
    
    def __init__(self, language: str = "ukrainian"):
        """
        Ініціалізація процесора текстів
        
        Args:
            language: Мова для обробки тексту
        """
        self.language = language
        
        # Завантаження необхідних ресурсів NLTK
        try:
            nltk.data.find(f'tokenizers/punkt')
            nltk.data.find(f'corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Стоп-слова для вибраної мови
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
            print(f"Стоп-слова для мови '{language}' не знайдено. Використовуємо порожній набір.")
    
    def clean_prompt(self, prompt: str) -> str:
        """
        Очищення промпту від зайвих символів та форматування
        
        Args:
            prompt: Вхідний текстовий промпт
            
        Returns:
            Очищений промпт
        """
        # Видалення зайвих пробілів
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        
        # Видалення спеціальних символів
        cleaned = re.sub(r'[^\w\s\.,!?;:\-]', '', cleaned)
        
        return cleaned
    
    def enhance_prompt(self, prompt: str, add_details: bool = True) -> str:
        """
        Покращення промпту для кращої генерації зображень
        
        Args:
            prompt: Вхідний текстовий промпт
            add_details: Чи додавати деталі для покращення якості
            
        Returns:
            Покращений промпт
        """
        # Базове очищення
        enhanced = self.clean_prompt(prompt)
        
        if add_details:
            # Додавання деталей для покращення якості
            quality_terms = [
                "високоякісне зображення", 
                "детальне", 
                "чітке", 
                "професійне фото"
            ]
            
            # Перевірка, чи вже містить промпт якісь з цих термінів
            has_quality_term = any(term in enhanced.lower() for term in quality_terms)
            
            if not has_quality_term:
                enhanced += ", " + quality_terms[0]
        
        return enhanced
    
    def extract_keywords(self, prompt: str, max_keywords: int = 5) -> List[str]:
        """
        Вилучення ключових слів з промпту
        
        Args:
            prompt: Вхідний текстовий промпт
            max_keywords: Максимальна кількість ключових слів
            
        Returns:
            Список ключових слів
        """
        # Токенізація
        tokens = word_tokenize(prompt.lower())
        
        # Видалення стоп-слів та коротких слів
        keywords = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Вибір найбільш важливих слів (за частотою)
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Повернення найчастіших слів
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def categorize_prompt(self, prompt: str) -> str:
        """
        Категоризація промпту за тематикою
        
        Args:
            prompt: Вхідний текстовий промпт
            
        Returns:
            Категорія промпту
        """
        prompt_lower = prompt.lower()
        
        # Категорії та ключові слова
        categories = {
            "природа": ["природа", "ліс", "гори", "море", "озеро", "річка", "пейзаж", "дерева", "квіти", "рослини"],
            "тварини": ["тварини", "кіт", "собака", "птах", "риба", "ведмідь", "лев", "тигр", "слон", "корова"],
            "місто": ["місто", "будівля", "архітектура", "вулиця", "дорога", "міський", "будинок", "хмарочос"],
            "люди": ["людина", "люди", "портрет", "обличчя", "жінка", "чоловік", "дитина", "дівчина", "хлопець"],
            "об'єкти": ["об'єкт", "предмет", "стіл", "стілець", "телефон", "комп'ютер", "машина", "автомобіль"],
            "абстракція": ["абстракція", "абстрактний", "фрактал", "геометричний", "візерунок", "текстура"]
        }
        
        # Підрахунок кількості ключових слів для кожної категорії
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            category_scores[category] = score
        
        # Вибір категорії з найвищим балом
        if max(category_scores.values(), default=0) > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return "інше"
    
    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        """
        Генерація варіацій промпту для різноманітності
        
        Args:
            prompt: Вхідний текстовий промпт
            num_variations: Кількість варіацій
            
        Returns:
            Список варіацій промпту
        """
        variations = [prompt]
        
        # Базові модифікатори для варіацій
        style_modifiers = [
            "у стилі акварелі", 
            "у стилі олійного живопису", 
            "у стилі фотореалізму",
            "у стилі мінімалізму",
            "у стилі імпресіонізму"
        ]
        
        lighting_modifiers = [
            "з м'яким освітленням",
            "з драматичним освітленням",
            "при заході сонця",
            "при ранковому світлі",
            "з контрастним освітленням"
        ]
        
        # Створення варіацій
        for i in range(num_variations - 1):
            if i < len(style_modifiers):
                new_variation = f"{prompt}, {style_modifiers[i]}"
            elif i < len(style_modifiers) + len(lighting_modifiers):
                new_variation = f"{prompt}, {lighting_modifiers[i - len(style_modifiers)]}"
            else:
                # Комбінація модифікаторів
                style_idx = i % len(style_modifiers)
                light_idx = (i // len(style_modifiers)) % len(lighting_modifiers)
                new_variation = f"{prompt}, {style_modifiers[style_idx]}, {lighting_modifiers[light_idx]}"
            
            variations.append(new_variation)
        
        return variations[:num_variations]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обробка текстових промптів")
    parser.add_argument("--prompt", type=str, required=True, help="Текстовий промпт для обробки")
    parser.add_argument("--action", choices=["clean", "enhance", "keywords", "categorize", "variations"], 
                      default="enhance", help="Дія для виконання")
    parser.add_argument("--language", type=str, default="ukrainian", help="Мова для обробки")
    parser.add_argument("--num_variations", type=int, default=3, help="Кількість варіацій для генерації")
    
    args = parser.parse_args()
    
    processor = EnhancedTextProcessor(language=args.language)
    
    if args.action == "clean":
        result = processor.clean_prompt(args.prompt)
        print(f"Очищений промпт: {result}")
    
    elif args.action == "enhance":
        result = processor.enhance_prompt(args.prompt)
        print(f"Покращений промпт: {result}")
    
    elif args.action == "keywords":
        result = processor.extract_keywords(args.prompt)
        print(f"Ключові слова: {', '.join(result)}")
    
    elif args.action == "categorize":
        result = processor.categorize_prompt(args.prompt)
        print(f"Категорія: {result}")
    
    elif args.action == "variations":
        result = processor.generate_variations(args.prompt, num_variations=args.num_variations)
        print("Варіації промпту:")
        for i, variation in enumerate(result, 1):
            print(f"{i}. {variation}")