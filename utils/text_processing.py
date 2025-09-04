import re
import nltk
import spacy
from nltk.corpus import wordnet
from collections import defaultdict

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

class TextProcessor:
    """
    A class for processing and understanding textual descriptions of objects.
    This helps the model interpret what the user means when requesting specific objects.
    """
    
    def __init__(self):
        self.context_dict = self._initialize_context_dictionary()
    
    def _initialize_context_dictionary(self):
        """
        Initialize a dictionary of common objects and their contextual meanings.
        This helps disambiguate objects that could have multiple interpretations.
        
        Returns:
            dict: A dictionary mapping object names to their possible contexts
        """
        context_dict = defaultdict(list)
        
        # Example: "flying saucer" could mean a UFO or a type of dish/tableware
        context_dict["flying saucer"] = [
            {"context": "ufo", "keywords": ["space", "alien", "ufo", "sky", "extraterrestrial", "spacecraft"]},
            {"context": "tableware", "keywords": ["dish", "plate", "tableware", "dining", "kitchen", "meal"]}
        ]
        
        # Add more ambiguous objects as needed
        context_dict["bat"] = [
            {"context": "animal", "keywords": ["animal", "flying", "nocturnal", "mammal", "cave", "wings"]},
            {"context": "sports", "keywords": ["baseball", "cricket", "sports", "game", "hit", "swing"]}
        ]
        
        context_dict["mouse"] = [
            {"context": "animal", "keywords": ["animal", "rodent", "small", "cheese", "pet", "furry"]},
            {"context": "computer", "keywords": ["computer", "click", "cursor", "device", "usb", "wireless"]}
        ]
        
        return context_dict
    
    def expand_context_dictionary(self, object_name, contexts):
        """
        Add a new object and its contexts to the dictionary.
        
        Args:
            object_name (str): The name of the object
            contexts (list): A list of context dictionaries
        """
        self.context_dict[object_name] = contexts
    
    def get_synonyms(self, word):
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word (str): The word to find synonyms for
            
        Returns:
            list: A list of synonyms
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)
    
    def disambiguate_object(self, object_name, description):
        """
        Determine the most likely context for an object based on the description.
        
        Args:
            object_name (str): The name of the object
            description (str): The full description or prompt
            
        Returns:
            str: The most likely context for the object
        """
        # Check if the object is in our context dictionary
        if object_name in self.context_dict:
            contexts = self.context_dict[object_name]
            
            # Calculate scores for each context based on keyword matches
            scores = []
            for context_info in contexts:
                score = 0
                for keyword in context_info["keywords"]:
                    if keyword in description.lower():
                        score += 1
                    
                    # Check for synonyms of the keyword in the description
                    for synonym in self.get_synonyms(keyword):
                        if synonym in description.lower():
                            score += 0.5
                
                scores.append((context_info["context"], score))
            
            # Return the context with the highest score
            if scores:
                return max(scores, key=lambda x: x[1])[0]
        
        # If the object isn't in our dictionary or no context matches well,
        # use spaCy for semantic analysis
        doc = nlp(description)
        
        # Extract key entities and their relationships
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Perform basic semantic analysis
        # This is a simplified approach; a more sophisticated model would be used in production
        if any(word in entities for word in ["space", "sky", "alien", "ufo"]):
            return "space_related"
        elif any(word in entities for word in ["animal", "creature", "wildlife"]):
            return "animal"
        elif any(word in entities for word in ["technology", "device", "gadget"]):
            return "technology"
        else:
            return "general"
    
    def extract_objects_from_prompt(self, prompt):
        """
        Extract the main objects mentioned in a prompt.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            list: A list of objects mentioned in the prompt
        """
        doc = nlp(prompt)
        
        # Extract noun chunks as potential objects
        objects = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Filter out common stop words and determiners
        filtered_objects = []
        for obj in objects:
            obj_doc = nlp(obj)
            # Keep only if it contains at least one noun
            if any(token.pos_ == "NOUN" for token in obj_doc):
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def enhance_prompt(self, prompt):
        """
        Enhance a prompt by adding contextual information for ambiguous objects.
        
        Args:
            prompt (str): The original prompt
            
        Returns:
            str: The enhanced prompt with contextual clarifications
        """
        # Extract objects from the prompt
        objects = self.extract_objects_from_prompt(prompt)
        
        enhanced_prompt = prompt
        for obj in objects:
            # Check if this is an ambiguous object in our dictionary
            if obj in self.context_dict:
                # Determine the most likely context
                context = self.disambiguate_object(obj, prompt)
                
                # Add clarification based on the context
                if context == "ufo" and obj == "flying saucer":
                    enhanced_prompt = enhanced_prompt.replace(
                        obj, f"{obj} (as in the UFO/spacecraft)"
                    )
                elif context == "tableware" and obj == "flying saucer":
                    enhanced_prompt = enhanced_prompt.replace(
                        obj, f"{obj} (as in the dish/plate)"
                    )
                # Add more context-specific replacements as needed
        
        return enhanced_prompt