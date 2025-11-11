import re
import nltk
import spacy
import torch
import numpy as np
from nltk.corpus import wordnet
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

# Глобальна змінна для кешування SentenceTransformer
_GLOBAL_SEMANTIC_MODEL = None

def get_global_semantic_model():
    """Отримання глобальної семантичної моделі (singleton pattern)"""
    global _GLOBAL_SEMANTIC_MODEL
    if _GLOBAL_SEMANTIC_MODEL is None:
        _GLOBAL_SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _GLOBAL_SEMANTIC_MODEL

class EnhancedTextProcessor:
    """
    Покращений клас для обробки та розуміння текстових описів з глибшим семантичним аналізом.
    Використовує сучасні NLP техніки для кращого розуміння контексту та значень слів.
    """
    
    def __init__(self):
        self.context_dict = self._initialize_enhanced_context_dictionary()
        self.semantic_model = None
        self._load_semantic_model()
        
        # Кеш для семантичних ембедингів
        self.embedding_cache = {}
        
        # Розширений список стоп-слів для кращої обробки
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))

        # Базові пресети негативного промпту (можна розширювати)
        self.NEGATIVE_PRESETS = {
            "none": [],
            "draft": [
                "low quality", "blurry", "out of focus", "noise", "overexposed",
                "underexposed", "washed out", "jpeg artifacts", "watermark", "text"
            ],
            "standard": [
                "low quality", "blurry", "motion blur", "out of focus", "noise",
                "oversaturated", "overexposed", "underexposed", "banding",
                "jpeg artifacts", "compression artifacts", "watermark", "text",
                "deformed", "distorted"
            ],
            "high": [
                "low quality", "blurry", "motion blur", "out of focus", "noise",
                "oversaturated", "overexposed", "underexposed", "banding",
                "jpeg artifacts", "compression artifacts", "watermark", "text",
                "deformed", "distorted", "chromatic aberration", "posterization",
                "grainy"
            ],
            "ultra": [
                "low quality", "blurry", "motion blur", "out of focus", "noise",
                "oversaturated", "overexposed", "underexposed", "banding",
                "jpeg artifacts", "compression artifacts", "watermark", "text",
                "deformed", "distorted", "chromatic aberration", "posterization",
                "grainy", "glitch", "artifact"
            ],
        }
        
    def _load_semantic_model(self):
        """Завантажує модель для семантичного аналізу"""
        try:
            # Використовуємо глобальну модель замість створення нової
            self.semantic_model = get_global_semantic_model()
            print("✅ Семантична модель завантажена успішно")
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити семантичну модель: {e}")
            self.semantic_model = None
    
    def _initialize_enhanced_context_dictionary(self):
        """
        Ініціалізує розширений словник контекстів з більш детальними семантичними зв'язками.
        """
        context_dict = defaultdict(list)
        
        # Розширений набір багатозначних слів
        context_dict["bat"] = [
            {
                "context": "animal", 
                "keywords": ["animal", "flying", "nocturnal", "mammal", "cave", "wings", "vampire", "fruit", "insect", "dark", "night", "hanging"],
                "semantic_features": ["живий", "літаючий", "нічний"],
                "related_objects": ["bird", "mouse", "cave", "night"],
                "clarify": "bat (flying nocturnal mammal)"
            },
            {
                "context": "sports", 
                "keywords": ["baseball", "cricket", "sports", "game", "hit", "swing", "wood", "aluminum", "player", "field", "ball"],
                "semantic_features": ["інструмент", "спорт", "дерев'яний"],
                "related_objects": ["ball", "player", "field", "game"],
                "clarify": "baseball bat (sports equipment)"
            }
        ]

        # Додаткові багатозначні слова
        context_dict["bank"] = [
            {
                "context": "finance",
                "keywords": ["money", "finance", "building", "account", "loan", "credit", "deposit", "atm", "teller", "vault"],
                "clarify": "bank building (financial institution)"
            },
            {
                "context": "river",
                "keywords": ["river", "water", "shore", "trees", "sand", "outdoor", "nature", "stream", "fishing"],
                "clarify": "river bank (shore)"
            }
        ]

        context_dict["mouse"] = [
            {
                "context": "animal",
                "keywords": ["small", "animal", "cheese", "rodent", "tail", "fur", "cat", "trap", "hole"],
                "clarify": "mouse (small rodent animal)"
            },
            {
                "context": "device",
                "keywords": ["computer", "desk", "usb", "wireless", "gaming", "keyboard", "click", "cursor"],
                "clarify": "computer mouse (input device)"
            }
        ]

        context_dict["spring"] = [
            {
                "context": "season",
                "keywords": ["flowers", "blooming", "season", "grass", "garden", "sunny", "warm", "april", "may"],
                "clarify": "spring season (flowers blooming)"
            },
            {
                "context": "coil",
                "keywords": ["metal", "mechanism", "coil", "car", "shock", "industrial", "compression", "tension"],
                "clarify": "metal spring coil"
            },
            {
                "context": "water",
                "keywords": ["water", "source", "mountain", "natural", "stream", "fresh", "drinking"],
                "clarify": "natural water spring"
            }
        ]

        # Додаємо більше омонімів для кращого покриття
        context_dict["bark"] = [
            {
                "context": "tree",
                "keywords": ["tree", "wood", "forest", "brown", "rough", "texture", "trunk"],
                "clarify": "tree bark (outer covering)"
            },
            {
                "context": "dog",
                "keywords": ["dog", "sound", "loud", "animal", "noise", "woof"],
                "clarify": "dog bark (sound)"
            }
        ]

        context_dict["bow"] = [
            {
                "context": "weapon",
                "keywords": ["arrow", "archery", "hunting", "string", "target", "medieval", "weapon"],
                "clarify": "bow (archery weapon)"
            },
            {
                "context": "gesture",
                "keywords": ["respect", "greeting", "formal", "head", "polite", "ceremony"],
                "clarify": "bow (respectful gesture)"
            },
            {
                "context": "ship",
                "keywords": ["ship", "boat", "front", "naval", "maritime", "stern"],
                "clarify": "bow (front of ship)"
            }
        ]

        context_dict["rose"] = [
            {
                "context": "flower",
                "keywords": ["flower", "red", "garden", "petals", "thorns", "romantic", "valentine"],
                "clarify": "rose flower (beautiful bloom)"
            },
            {
                "context": "past_tense",
                "keywords": ["stood", "up", "increased", "elevated", "higher", "ascended"],
                "clarify": "rose (past tense of rise)"
            }
        ]

        context_dict["lead"] = [
            {
                "context": "metal",
                "keywords": ["metal", "heavy", "gray", "toxic", "pipe", "bullet", "weight"],
                "clarify": "lead metal (heavy element)"
            },
            {
                "context": "guide",
                "keywords": ["guide", "leader", "front", "direction", "team", "follow"],
                "clarify": "lead (to guide or direct)"
            }
        ]

        context_dict["tear"] = [
            {
                "context": "emotion",
                "keywords": ["cry", "sad", "emotion", "eye", "drop", "weep"],
                "clarify": "tear (from crying)"
            },
            {
                "context": "rip",
                "keywords": ["rip", "break", "fabric", "paper", "damage", "hole"],
                "clarify": "tear (to rip apart)"
            }
        ]

        context_dict["wind"] = [
            {
                "context": "air",
                "keywords": ["air", "breeze", "storm", "blow", "weather", "gust"],
                "clarify": "wind (moving air)"
            },
            {
                "context": "twist",
                "keywords": ["twist", "coil", "wrap", "turn", "spiral", "clock"],
                "clarify": "wind (to twist or coil)"
            }
        ]

        context_dict["close"] = [
            {
                "context": "near",
                "keywords": ["near", "proximity", "distance", "intimate", "nearby"],
                "clarify": "close (near in distance)"
            },
            {
                "context": "shut",
                "keywords": ["shut", "door", "window", "end", "finish", "seal"],
                "clarify": "close (to shut)"
            }
        ]

        context_dict["fair"] = [
            {
                "context": "just",
                "keywords": ["just", "equal", "honest", "right", "balanced", "impartial"],
                "clarify": "fair (just and equal)"
            },
            {
                "context": "event",
                "keywords": ["carnival", "festival", "rides", "games", "food", "entertainment"],
                "clarify": "fair (carnival or festival)"
            },
            {
                "context": "light",
                "keywords": ["light", "pale", "blonde", "skin", "complexion"],
                "clarify": "fair (light colored)"
            }
        ]
        
        return context_dict

    def disambiguate_homonyms(self, text: str) -> str:
        """
        Розрізняє омоніми за контекстом і додає уточнення прямо в промпт.

        Покращений алгоритм:
        - Шукає ключі зі словника контекстів у тексті.
        - Підраховує збіги ключових слів кожного контексту з вагами.
        - Використовує семантичний аналіз для кращого розуміння контексту.
        - Обирає контекст з найбільшою кількістю збігів.
        - Замінює багатозначне слово на уточнене формулювання.
        """
        if not text:
            return text

        original = text
        lower = text.lower()
        words = lower.split()
        
        for ambiguous, variants in self.context_dict.items():
            if ambiguous not in lower:
                continue

            # Оцінка відповідності кожного варіанту з покращеною логікою
            best = None
            best_score = -1
            
            for var in variants:
                score = 0
                keywords = var.get("keywords", [])
                
                # Базовий підрахунок збігів ключових слів
                for kw in keywords:
                    if kw in lower:
                        # Додаємо більшу вагу для точних збігів слів
                        if kw in words:
                            score += 2
                        else:
                            score += 1
                
                # Додаткові бали за семантичну близькість
                if self.semantic_model:
                    try:
                        # Отримуємо ембединги для контексту та тексту
                        context_text = " ".join(keywords[:5])  # Беремо перші 5 ключових слів
                        
                        # Використовуємо кеш для ембедингів
                        if context_text not in self.embedding_cache:
                            self.embedding_cache[context_text] = self.semantic_model.encode([context_text])
                        
                        if text not in self.embedding_cache:
                            self.embedding_cache[text] = self.semantic_model.encode([text])
                        
                        # Обчислюємо семантичну схожість
                        similarity = cosine_similarity(
                            self.embedding_cache[context_text],
                            self.embedding_cache[text]
                        )[0][0]
                        
                        # Додаємо семантичний бонус (максимум 3 бали)
                        semantic_bonus = min(3, similarity * 5)
                        score += semantic_bonus
                        
                    except Exception as e:
                        # Якщо семантичний аналіз не працює, продовжуємо без нього
                        pass
                
                # Бонус за наявність пов'язаних об'єктів
                related_objects = var.get("related_objects", [])
                for obj in related_objects:
                    if obj in lower:
                        score += 1.5
                
                if score > best_score:
                    best_score = score
                    best = var

            # Формуємо уточнення тільки якщо є достатня впевненість
            if best and best_score > 1:  # Мінімальний поріг впевненості
                clarify = best.get("clarify")
                
                if clarify:
                    # Перевіряємо, чи не дублюємо вже наявні уточнення
                    if f"({ambiguous}" not in text and f"{ambiguous} (" not in text:
                        # Спеціальна обробка для складних випадків
                        if ambiguous == "bat" and best.get("context") == "sports":
                            if re.search(r"\bbaseball\s+bat\b", text, flags=re.I):
                                text = re.sub(r"\bbaseball\s+bat\b", "baseball bat (sports equipment)", text, flags=re.I)
                            else:
                                text = re.sub(rf"\b{ambiguous}\b", clarify, text, flags=re.I, count=1)
                        else:
                            # Загальна заміна: перше входження слова на уточнене
                            text = re.sub(rf"\b{ambiguous}\b", clarify, text, flags=re.I, count=1)

        # Нормалізація пробілів
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def enhance_prompt_advanced(self, prompt):
        """
        Покращений метод підсилення промпту з глибшим семантичним аналізом.
        """
        # 1) Легка підтримка акцентів у стилі промпт-редакторів:
        #    [phrase:1.4] або (phrase:1.2) — підсилення, {phrase:0.8} — ослаблення
        def _apply_positive(m):
            phrase = m.group(1).strip()
            weight = float(m.group(2)) if m.group(2) else 1.2
            # Дублюємо фразу залежно від ваги (простий, стабільний варіант)
            repeats = max(1, int(round(1 + (weight - 1) * 2)))
            return (", ".join([phrase] * repeats))

        def _apply_negative(m):
            phrase = m.group(1).strip()
            weight = float(m.group(2)) if m.group(2) else 0.8
            # При зменшенні — залишаємо один раз або видаляємо якщо дуже низько
            return phrase if weight >= 0.7 else ""

        text = prompt
        # Позитивні акценти: квадратні або круглі дужки
        text = re.sub(r"[\(\[]([^\)\]]+?)(?::([0-9\.]+))?[\)\]]", _apply_positive, text)
        # Негативні акценти: фігурні дужки
        text = re.sub(r"\{([^}]+?)(?::([0-9\.]+))?\}", _apply_negative, text)

        # 2) Нормалізація пробілів та ком
        text = re.sub(r"\s+,\s+", ", ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3) Легка семантична підказка: якщо присутні вікові/стильові маркери,
        #    додаємо узгоджені терміни для стабільності (без важких залежностей)
        additions = []
        if re.search(r"\b(hyperrealistic|photorealistic)\b", text, re.I):
            additions.append("high detail, sharp focus")
        if re.search(r"\b(close[- ]?up)\b", text, re.I):
            additions.append("shallow depth of field")
        if re.search(r"\b(cyberpunk|neon)\b", text, re.I):
            additions.append("dramatic lighting")

        if additions:
            text = f"{text}, " + ", ".join(additions)

        return text

    def analyze_semantic_relationships(self, text: str) -> dict:
        """
        Проста семантична розмітка контексту:
        - Виявляє багатозначне слово в тексті
        - Обирає контекст (сенс) на основі збігів ключових слів
        - Повертає службові поля для тестів і візуалізацій
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "ambiguous_word": None,
                "detected_context": None,
                "matched_keywords": [],
                "score": 0,
                "clarify": None,
            }

        lower = text.lower()
        result = {
            "ambiguous_word": None,
            "detected_context": None,
            "matched_keywords": [],
            "score": 0,
            "clarify": None,
        }

        for ambiguous, variants in self.context_dict.items():
            if ambiguous in lower:
                best = None
                best_score = -1
                best_matched = []
                for var in variants:
                    kws = var.get("keywords", [])
                    matched = [kw for kw in kws if kw in lower]
                    score = len(matched)
                    if score > best_score:
                        best_score = score
                        best = var
                        best_matched = matched

                result["ambiguous_word"] = ambiguous
                result["detected_context"] = best.get("context") if best else None
                result["matched_keywords"] = best_matched
                result["score"] = max(best_score, 0)
                result["clarify"] = best.get("clarify") if best else None
                break

        return result

    def get_semantic_embedding(self, text: str):
        """
        Повертає семантичне вбудовування для тексту із кешем.
        Використовує SentenceTransformer ('all-MiniLM-L6-v2').
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            model = self.semantic_model or get_global_semantic_model()
            emb = model.encode([text])[0]
            self.embedding_cache[text] = emb
            return emb
        except Exception as e:
            # Фолбек: нульовий вектор фіксованої довжини
            return np.zeros(384, dtype=np.float32)

    def parse_prompt_structure(self, prompt: str) -> dict:
        """
        Розбирає промпт на базові компоненти: subject, attributes, style, environment, camera, lighting.
        Легка евристика: поєднання spaCy та регулярних правил для стилів/світла/камери.
        """
        structure = {
            'subject': [],
            'attributes': [],
            'style': [],
            'environment': [],
            'camera': [],
            'lighting': []
        }

        try:
            doc = nlp(prompt)
            # Subject: основні іменники/власні назви
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    structure['subject'].append(token.lemma_.lower())
                if token.pos_ == "ADJ":
                    structure['attributes'].append(token.lemma_.lower())
        except Exception:
            # Простий фолбек: слова як суб'єкти
            words = re.findall(r"[A-Za-z]+", prompt)
            structure['subject'] = [w.lower() for w in words[:2]]

        # Лексикони стилів/камери/світла/середовища
        style_lexicon = {
            'photorealistic', 'hyperrealistic', 'cinematic', 'watercolor', 'oil painting', '3d render',
            'isometric', 'vector art', 'line art', 'anime', 'cyberpunk', 'neon', 'vintage', 'black and white',
        }
        camera_lexicon = {
            'macro', 'close-up', 'wide angle', 'telephoto', 'fisheye', 'long exposure', 'tilt-shift',
            'depth of field', 'bokeh', 'shallow depth of field'
        }
        lighting_lexicon = {
            'soft light', 'hard light', 'rim light', 'backlight', 'golden hour', 'dramatic lighting', 'studio lighting'
        }
        environment_lexicon = {
            'indoor', 'outdoor', 'street', 'forest', 'desert', 'ocean', 'night', 'day', 'sunset', 'sunrise', 'cave', 'city', 'studio'
        }

        text_lower = prompt.lower()
        def present(lex):
            return [term for term in lex if term in text_lower]

        structure['style'] = present(style_lexicon)
        structure['camera'] = present(camera_lexicon)
        structure['lighting'] = present(lighting_lexicon)
        structure['environment'] = present(environment_lexicon)

        # Дедуплікація збереженням порядку
        for k in structure:
            seen = set()
            unique = []
            for v in structure[k]:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            structure[k] = unique

        return structure

    def enhance_prompt_with_weights(self, prompt: str):
        """
        Повертає покращений промпт ТА карту ваг для фраз, витягнуту з синтаксису ( ...:w ), [ ...:w ], { ...:w }.
        weight_map: dict[str, float]
        """
        weight_map = {}

        # Знайти всі фрази з вагами ДО трансформацій
        for pattern, default_w in [
            (r"[\(\[]([^\)\]]+?)(?::([0-9\.]+))?[\)\]]", 1.2),
            (r"\{([^}]+?)(?::([0-9\.]+))?\}", 0.8),
        ]:
            for m in re.finditer(pattern, prompt):
                phrase = m.group(1).strip()
                try:
                    w = float(m.group(2)) if m.group(2) else default_w
                except Exception:
                    w = default_w
                # Агрегуємо: середнє якщо повторюється кілька разів
                if phrase in weight_map:
                    weight_map[phrase] = (weight_map[phrase] + w) / 2.0
                else:
                    weight_map[phrase] = w

        enhanced = self.enhance_prompt_advanced(prompt)
        return enhanced, weight_map

    def apply_negative_preset(self, preset: str, negative_prompt: str = "") -> str:
        """Повертає злитий негативний промпт з урахуванням вибраного пресета.

        - `preset`: один із ключів у `NEGATIVE_PRESETS` (наприклад, 'draft', 'standard', 'high', 'ultra').
        - `negative_prompt`: користувацькі негативні терміни; будуть об'єднані та дедупліковані.
        """
        try:
            preset = (preset or "none").strip().lower()
        except Exception:
            preset = "none"

        base_terms = self.NEGATIVE_PRESETS.get(preset, [])
        user_terms = [t.strip() for t in re.split(r",|;|\n", negative_prompt or "") if t.strip()]

        # Дедуплікація зі збереженням порядку
        seen = set()
        merged: list[str] = []
        for term in base_terms + user_terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                merged.append(term)

        return ", ".join(merged)
    
    def enhance_prompt(self, prompt):
        """
        Enhance a prompt by adding contextual information (для зворотної сумісності).
        """
        return self.enhance_prompt_advanced(prompt)

class TextProcessor:
    """
    Backward-compatible wrapper that provides the previous TextProcessor API
    while delegating to EnhancedTextProcessor internally.
    """
    def __init__(self):
        self._enhanced = EnhancedTextProcessor()

    def enhance_prompt(self, prompt: str):
        return self._enhanced.enhance_prompt(prompt)

    def enhance_prompt_advanced(self, prompt: str):
        return self._enhanced.enhance_prompt_advanced(prompt)

    def enhance_prompt_with_weights(self, prompt: str):
        return self._enhanced.enhance_prompt_with_weights(prompt)

    def parse_prompt_structure(self, prompt: str) -> dict:
        return self._enhanced.parse_prompt_structure(prompt)

    def apply_negative_preset(self, preset: str, negative_prompt: str = "") -> str:
        return self._enhanced.apply_negative_preset(preset, negative_prompt)

    def process_negative_prompt(self, negative_prompt: str, preset: str = "standard") -> str:
        """Process negative prompt with preset and custom additions"""
        return self._enhanced.apply_negative_preset(preset, negative_prompt)

    def disambiguate_homonyms(self, text: str) -> str:
        return self._enhanced.disambiguate_homonyms(text)

    def suggest_negative_terms_for_prompt(self, prompt: str):
        """Suggest negative terms based on subject to avoid common confusions.

        - For animal-focused prompts (e.g., dog/retriever), avoid human/portrait features.
        - Keeps implementation lightweight and deterministic without external dependencies.
        """
        try:
            lower = (prompt or "").lower()
        except Exception:
            lower = ""

        terms = []
        # Dog/retriever specific guardrails
        if re.search(r"\b(dog|retriever|canine|puppy)\b", lower):
            terms += [
                "human", "person", "people", "portrait", "face",
                "hands", "arms", "legs", "woman", "man", "girl", "boy",
                # Composition guardrails to reduce head-only crops
                "close-up", "headshot", "selfie"
            ]

        # Generic animal cues
        if re.search(r"\b(animal|wildlife|zoo)\b", lower):
            for t in ["human", "person", "portrait", "face"]:
                if t not in terms:
                    terms.append(t)

        return terms

    def extract_objects_from_prompt(self, text: str):
        """Extract key objects (nouns/proper nouns) from text using spaCy.

        Falls back to a simple regex-based extraction if spaCy fails.
        """
        try:
            doc = nlp(text)
            objects = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
            # Return unique objects preserving order
            seen = set()
            unique = []
            for obj in objects:
                key = obj.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(obj)
            return unique
        except Exception:
            # Fallback: simple word split unique list
            return list({w for w in re.findall(r"[A-Za-z]+", text)})