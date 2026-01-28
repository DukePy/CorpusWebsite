"""
Metaphor Detection Script for "X is Y" structures
Detects potential metaphorical expressions using NOUN+BE+NOUN patterns
Filters out definitional statements using semantic similarity
"""

import spacy
from pathlib import Path
import re
from typing import List, Tuple, Dict
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

class MetaphorDetector:
    """
    Detects potential metaphors in the form "X is Y" where X and Y are nouns.
    Uses semantic similarity to filter out definitions.
    """
    
    def __init__(self, model_name="en_core_web_sm", similarity_threshold=0.6, use_entity_linker=True):
        """
        Initialize the metaphor detector.
        
        Args:
            model_name: spaCy model name (for linguistic parsing only)
            similarity_threshold: Maximum similarity score to consider as metaphor
                                (higher values = more dissimilar = more likely metaphor)
            use_entity_linker: Whether to use entity linker to filter person names
        """
        print(f"Loading spaCy model for linguistic parsing: {model_name}...")
        self.nlp = spacy.load(model_name)
        
        # Set max_length to handle large texts (up to 2 million characters)
        # For texts larger than this, we'll use adaptive chunking
        self.nlp.max_length = 2000000
        
        self.similarity_threshold = similarity_threshold
        self.use_entity_linker = use_entity_linker
        
        # Load sentence transformer model for semantic similarity
        print("Loading Sentence-Transformer model for embeddings: all-MiniLM-L6-v2...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add entity linker for historical figure detection
        if use_entity_linker:
            print("Loading Entity Linker for person name filtering...")
            try:
                self.nlp.add_pipe("entityLinker", last=True)
                print("  Entity Linker loaded successfully")
            except Exception as e:
                print(f"  Warning: Could not load Entity Linker: {e}")
                print("  Continuing without entity linking (all person names will be filtered)")
                self.use_entity_linker = False
        
    def extract_noun_be_noun_patterns(self, doc) -> List[Dict]:
        """
        Extract NOUN+BE+NOUN patterns from a spaCy Doc object.
        
        Returns:
            List of dictionaries containing pattern information
        """
        patterns = []
        
        for sent in doc.sents:
            # Look for copula (be verb) constructions
            for token in sent:
                # Check if token is a form of "be" verb
                if token.lemma_ == "be" and token.pos_ == "AUX":
                    # Look for nominal subject (X)
                    subject_noun = None
                    subject_phrase = []
                    
                    # Find the subject
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Get the head noun of the subject
                            if child.pos_ in ["NOUN", "PROPN"]:
                                subject_noun = child
                                # Collect the full noun phrase
                                subject_phrase = self._get_noun_phrase(child)
                            break
                    
                    # Look for predicate nominative (Y)
                    predicate_noun = None
                    predicate_phrase = []
                    
                    for child in token.children:
                        if child.dep_ in ["attr", "acomp"]:
                            # Get the predicate noun
                            if child.pos_ in ["NOUN", "PROPN"]:
                                predicate_noun = child
                                predicate_phrase = self._get_noun_phrase(child)
                            # Sometimes the predicate is nested deeper
                            elif child.dep_ == "attr":
                                for subchild in child.subtree:
                                    if subchild.pos_ in ["NOUN", "PROPN"]:
                                        predicate_noun = subchild
                                        predicate_phrase = self._get_noun_phrase(subchild)
                                        break
                            break
                    
                    # If we found both subject and predicate nouns, record the pattern
                    if subject_noun and predicate_noun:
                        pattern_info = {
                            'sentence': sent.text.strip(),
                            'subject': ' '.join(subject_phrase),
                            'subject_head': subject_noun.text,
                            'be_verb': token.text,
                            'predicate': ' '.join(predicate_phrase),
                            'predicate_head': predicate_noun.text,
                            'subject_token': subject_noun,
                            'predicate_token': predicate_noun
                        }
                        patterns.append(pattern_info)
        
        return patterns
    
    def _get_noun_phrase(self, noun_token) -> List[str]:
        """
        Extract the full noun phrase centered on a noun token.
        Includes determiners, adjectives, compounds, etc.
        """
        # Get all children and the token itself
        phrase_tokens = []
        
        # Collect tokens that are part of the noun phrase
        for token in noun_token.subtree:
            if token.dep_ in ["det", "amod", "compound", "nummod", "poss", "nmod"]:
                phrase_tokens.append(token)
        
        # Add the head noun itself
        phrase_tokens.append(noun_token)
        
        # Sort by position in sentence
        phrase_tokens.sort(key=lambda t: t.i)
        
        return [t.text for t in phrase_tokens]
    
    def _is_non_historical_person(self, noun_token) -> bool:
        """
        Check if a noun token is a person name (not a historical figure).
        Returns True if it's a modern person name that should be filtered out.
        Returns False if it's a historical figure or not a person name.
        
        Args:
            noun_token: spaCy Token object
            
        Returns:
            bool: True if should be filtered (modern person name), False otherwise
        """
        # Check if the token is part of a PERSON entity
        if noun_token.ent_type_ != "PERSON":
            return False
        
        # Get the full person name entity by finding the entity span this token belongs to
        person_entity = None
        for ent in noun_token.doc.ents:
            if ent.start <= noun_token.i < ent.end and ent.label_ == "PERSON":
                person_entity = ent.text
                break
        
        if not person_entity:
            return False
        
        # If not using entity linker, filter all person names
        if not self.use_entity_linker:
            return True
        
        # Use entity linker to check if it's a historical figure
        doc = self.nlp(person_entity)
        
        for ent in doc.ents:
            if hasattr(ent._, "linkedEntities") and ent._.linkedEntities:
                for entity in ent._.linkedEntities:
                    # Get entity description
                    desc = entity.get_description().lower() if entity.get_description() else ""
                    
                    # Check for historical/notable person indicators
                    historical_indicators = [
                        'writer', 'author', 'poet', 'novelist',
                        'politician', 'president', 'minister', 'king', 'queen', 'emperor',
                        'philosopher', 'thinker', 'scholar', 'scientist',
                        'artist', 'painter', 'musician', 'composer',
                        'soldier', 'general', 'officer',
                        'born', 'died'  # Biographical indicators suggesting historical figure
                    ]
                    
                    # Split description into words
                    desc_words = set(re.findall(r'\b[a-z]+\b', desc))
                    
                    # Check if description contains historical indicators
                    has_historical_indicator = any(indicator in desc_words for indicator in historical_indicators)
                    
                    # Filter out explicit fictional characters
                    if "fictional" in desc or "literary character" in desc:
                        return True  # Filter out fictional characters
                    
                    # If it has historical indicators, keep it (return False = don't filter)
                    if has_historical_indicator:
                        return False  # Don't filter - it's a historical figure
        
        # If no entity link found, assume it's a local/modern person name - filter it
        return True
    
    def calculate_semantic_distance(self, pattern: Dict) -> float:
        """
        Calculate semantic distance between subject and predicate nouns.
        Lower similarity = higher distance = more likely to be metaphorical
        
        Uses transformer embeddings for accurate contextual similarity.
        
        Returns:
            Semantic distance score (1 - similarity)
        """
        subject_token = pattern['subject_token']
        predicate_token = pattern['predicate_token']
        
        # Get transformer embeddings for both tokens
        subject_embedding = self._get_token_embedding(subject_token)
        predicate_embedding = self._get_token_embedding(predicate_token)
        
        # Calculate cosine similarity between embeddings
        similarity = self._cosine_similarity(subject_embedding, predicate_embedding)
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        
        return distance
    
    def _get_token_embedding(self, token) -> np.ndarray:
        """
        Extract the transformer embedding for a specific token using sentence-transformers.
        
        Args:
            token: spaCy Token object
        
        Returns:
            numpy array of the token's embedding
        """
        # Use sentence-transformers to encode the token text
        # This gives us a high-quality semantic embedding
        token_text = token.text
        embedding = self.sbert_model.encode(token_text, convert_to_numpy=True)
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Handle zero vectors
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Ensure the result is between 0 and 1 (sometimes floating point errors can cause slight deviations)
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def is_potential_metaphor(self, pattern: Dict) -> Tuple[bool, float, str]:
        """
        Determine if a pattern is likely a metaphor based on semantic distance.
        Also filters out patterns where X or Y are non-historical person names.
        
        Returns:
            Tuple of (is_metaphor, distance_score, filter_reason)
            filter_reason is empty string if not filtered, otherwise contains reason
        """
        subject_token = pattern['subject_token']
        predicate_token = pattern['predicate_token']
        
        # Check if subject or predicate is a non-historical person name
        if self._is_non_historical_person(subject_token):
            return False, 0.0, f"Subject '{subject_token.text}' is a person name (not historical)"
        
        if self._is_non_historical_person(predicate_token):
            return False, 0.0, f"Predicate '{predicate_token.text}' is a person name (not historical)"
        
        # Calculate semantic distance
        distance = self.calculate_semantic_distance(pattern)
        
        # High distance (low similarity) suggests metaphor
        # Low distance (high similarity) suggests definition
        is_metaphor = distance > (1 - self.similarity_threshold)
        
        return is_metaphor, distance, ""
    
    def process_text(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Process text and extract potential metaphors.
        Filters out patterns with non-historical person names.
        Uses adaptive chunking for large texts to avoid memory issues.
        
        Returns:
            Tuple of (metaphors list, filtered_patterns list)
        """
        # Adaptive chunking: calculate number of chunks based on text length
        # Keep each chunk under 800k characters to be safe (below 1M spaCy limit)
        max_chunk_size = 50000
        text_length = len(text)
        
        if text_length <= max_chunk_size:
            # Text is small enough, process in one go
            doc = self.nlp(text)
            patterns = self.extract_noun_be_noun_patterns(doc)
        else:
            # Text is too large, use adaptive chunking
            num_chunks = (text_length // max_chunk_size) + 1
            print(f"\n>> Text is large ({text_length:,} chars). Processing in {num_chunks} chunks...")
            
            # Split text into sentences to ensure we don't break mid-sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Distribute sentences across chunks as evenly as possible
            chunk_patterns = []
            current_chunk = []
            current_chunk_size = 0
            target_chunk_size = text_length // num_chunks
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                # If adding this sentence would exceed target size and we have content, start new chunk
                if current_chunk_size + sentence_len > target_chunk_size and current_chunk:
                    # Process current chunk
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 0:
                        doc = self.nlp(chunk_text)
                        chunk_patterns.extend(self.extract_noun_be_noun_patterns(doc))
                    
                    # Start new chunk
                    current_chunk = [sentence]
                    current_chunk_size = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_chunk_size += sentence_len
            
            # Process final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 0:
                    doc = self.nlp(chunk_text)
                    chunk_patterns.extend(self.extract_noun_be_noun_patterns(doc))
            
            patterns = chunk_patterns
            print(f"   Processed {num_chunks} chunks, found {len(patterns)} patterns")
        
        metaphors = []
        filtered_patterns = []
        
        for pattern in patterns:
            is_metaphor, distance, filter_reason = self.is_potential_metaphor(pattern)
            
            # If pattern was filtered (person names), add to filtered list
            if filter_reason:
                filtered_info = {
                    'sentence': pattern['sentence'],
                    'subject': pattern['subject'],
                    'be_verb': pattern['be_verb'],
                    'predicate': pattern['predicate'],
                    'filter_reason': filter_reason
                }
                filtered_patterns.append(filtered_info)
                continue
            
            metaphor_info = {
                'sentence': pattern['sentence'],
                'subject': pattern['subject'],
                'be_verb': pattern['be_verb'],
                'predicate': pattern['predicate'],
                'semantic_distance': round(distance, 4),
                'is_potential_metaphor': is_metaphor
            }
            
            metaphors.append(metaphor_info)
        
        if filtered_patterns:
            print(f"  Filtered out {len(filtered_patterns)} patterns with non-historical person names")
        
        return metaphors, filtered_patterns
    
    def process_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a text file and extract potential metaphors.
        
        Returns:
            Tuple of (metaphors list, filtered_patterns list)
        """
        print(f"\nProcessing: {Path(file_path).name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.process_text(text)
    
    def process_directory(self, directory: str, pattern: str = "*.txt") -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Process all text files in a directory.
        
        Returns:
            Tuple of (metaphor_results dict, filtered_results dict)
            Each dict maps filenames to their respective lists
        """
        dir_path = Path(directory)
        metaphor_results = {}
        filtered_results = {}
        
        for file_path in dir_path.glob(pattern):
            if file_path.name != "Enigma of China - Qiu Xiaolong.txt":
                continue
            metaphors, filtered = self.process_file(str(file_path))
            metaphor_results[file_path.name] = metaphors
            filtered_results[file_path.name] = filtered
        
        return metaphor_results, filtered_results


def main():
    """
    Main function to run the metaphor detector on corpus files.
    """
    # Initialize detector
    # Threshold of 0.6 means we keep patterns where similarity < 0.6
    # (i.e., words are not too similar, suggesting metaphor rather than definition)
    # Using en_core_web_sm for fast linguistic parsing
    # Sentence-transformers handles the semantic similarity
    # Entity linker filters out non-historical person names
    detector = MetaphorDetector(
        model_name="en_core_web_sm",
        similarity_threshold=0.5,
        use_entity_linker=True
    )
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Process all txt corpus files
    print("\n" + "="*70)
    print("METAPHOR DETECTION: 'X is Y' patterns")
    print("="*70)
    
    results, filtered_results = detector.process_directory(str(script_dir), pattern="*.txt")
    
    # Display results
    for filename, metaphors in results.items():
        print(f"\n{'='*70}")
        print(f"FILE: {filename}")
        print(f"{'='*70}")
        
        # Separate potential metaphors from definitions
        potential_metaphors = [m for m in metaphors if m['is_potential_metaphor']]
        definitions = [m for m in metaphors if not m['is_potential_metaphor']]
        
        print(f"\nFound {len(metaphors)} total 'X is Y' patterns:")
        print(f"  - {len(potential_metaphors)} potential metaphors (low similarity)")
        print(f"  - {len(definitions)} likely definitions (high similarity)")
        
        # Display potential metaphors
        if potential_metaphors:
            print(f"\n{'-'*70}")
            print("POTENTIAL METAPHORS (sorted by semantic distance):")
            print(f"{'-'*70}")
            
            # Sort by semantic distance (highest first = most metaphorical)
            potential_metaphors.sort(key=lambda x: x['semantic_distance'], reverse=True)
            
            for i, m in enumerate(potential_metaphors, 1):
                print(f"\n{i}. [{m['semantic_distance']:.3f}] {m['subject']} {m['be_verb']} {m['predicate']}")
                print(f"   Sentence: {m['sentence'][:150]}{'...' if len(m['sentence']) > 150 else ''}")
        
        # Display some definitions for comparison
        if definitions:
            print(f"\n{'-'*70}")
            print("DEFINITIONS/NON-METAPHORS (top 5 examples):")
            print(f"{'-'*70}")
            
            # Sort by semantic distance (lowest first = most definitional)
            definitions.sort(key=lambda x: x['semantic_distance'])
            
            for i, m in enumerate(definitions[:5], 1):
                print(f"\n{i}. [{m['semantic_distance']:.3f}] {m['subject']} {m['be_verb']} {m['predicate']}")
                print(f"   Sentence: {m['sentence'][:150]}{'...' if len(m['sentence']) > 150 else ''}")
        
        # Display filtered patterns (person names)
        if filename in filtered_results and filtered_results[filename]:
            filtered_list = filtered_results[filename]
            print(f"\n{'-'*70}")
            print(f"FILTERED PATTERNS (person names): {len(filtered_list)} patterns")
            print(f"{'-'*70}")
            
            for i, f in enumerate(filtered_list, 1):
                print(f"\n{i}. {f['subject']} {f['be_verb']} {f['predicate']}")
                print(f"   Reason: {f['filter_reason']}")
                print(f"   Sentence: {f['sentence'][:150]}{'...' if len(f['sentence']) > 150 else ''}")
    
    # Save results to JSON
    output_file = script_dir / "metaphor_detection_results.json"
    
    # Convert results for JSON serialization (include both metaphors and filtered)
    json_results = {}
    for filename in results.keys():
        json_results[filename] = {
            'metaphors': results[filename],
            'filtered_patterns': filtered_results.get(filename, [])
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")

    print("Done!")


if __name__ == "__main__":
    main()

