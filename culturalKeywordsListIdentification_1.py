import math
import re
from collections import Counter
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Number of parts to split target corpus into for memory-efficient processing
NUM_CORPUS_CHUNKS = 5

# ==========================================
# Set this to True if you have spaCy installed and want to use the real model.
# You must run: pip install spacy && python -m spacy download en_core_web_sm
USE_REAL_SPACY = True 

# Set this to True to use the REAL Wikidata API (requires requests library + internet)
USE_WIKIDATA_API = True

# Set this to True to use the Entity Linker (spacy-entity-linker) 
# Requires: pip install spacy-entity-linker && python -m spacy_entity_linker "download_knowledge_base"
# This provides better entity linking than the basic Wikidata API
USE_ENTITY_LINKER = True

# Set this to True to use coreference resolution to find the best/longest mention of each name
# Requires: pip install fastcoref
# Recommended: True - helps find full names like "Inspector Chen Cao" instead of just "Chen"
RUN_COREF = True

# Corpus file paths (set to None to use dummy corpus)
TARGET_CORPUS_FILE = "Enigma of China - Qiu Xiaolong.txt"
REFERENCE_CORPUS_FILES = [
    #"referenceCorpus.txt",
    "Enigma of China Reference Corpus.txt"
    #"A Lovers Discourse_Remove.txt"  # Uncomment to include this file
]

#TARGET_CORPUS_FILE = None
#REFERENCE_CORPUS_FILES = None

# ==========================================
# PART 1: SETUP & CORPUS LOADING
# ==========================================

def split_text_into_chunks(text, num_chunks=5):
    """
    Split text into roughly equal chunks by sentence boundaries.
    
    Args:
        text: The text to split
        num_chunks: Number of chunks to create
        
    Returns:
        list: List of text chunks
    """
    # Simple sentence splitting (can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Calculate sentences per chunk
    sentences_per_chunk = len(sentences) // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * sentences_per_chunk
        if i == num_chunks - 1:
            # Last chunk gets remaining sentences
            end_idx = len(sentences)
        else:
            end_idx = (i + 1) * sentences_per_chunk
        
        chunk = ' '.join(sentences[start_idx:end_idx])
        chunks.append(chunk)
    
    return chunks


def load_corpus_from_file(filepath):
    """
    Load text corpus from a file.
    
    Args:
        filepath: Path to the text file (relative or absolute)
        
    Returns:
        str: Contents of the file
    """
    import os
    
    try:
        # Try absolute path first
        if os.path.isabs(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Try relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, filepath)
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except FileNotFoundError:
        print(f"Error: Could not find corpus file: {filepath}")
        print(f"Tried path: {full_path if not os.path.isabs(filepath) else filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading corpus file {filepath}: {e}")
        sys.exit(1)


def load_corpora():
    """
    Loads corpora from files (if configured) or creates dummy corpora.
    
    Returns:
        tuple: (target_corpus, reference_corpus)
    """
    # If file paths are configured, load from files
    if TARGET_CORPUS_FILE:
        print(f"Loading target corpus from: {TARGET_CORPUS_FILE}")
        target_corpus = load_corpus_from_file(TARGET_CORPUS_FILE)
        print(f"  -> Loaded {len(target_corpus)} characters")
    else:
        print("Using dummy target corpus")
        target_corpus = """
        Lu Xun is a Chinese writer. Lu Xun is very famous.
        Chief Inspector Chen Cao sat in his office, drinking a cup of Dragon Well tea. 
        It was a symbol of the old days. He looked at the Party Secretary, a high-ranking cadre 
        who spoke of the Red distinctiveness. "Comrade," Chen said, quoting Lu Xun, 
        "Fierce-browed, I coolly defy a thousand pointing fingers." 
        The corruption case was difficult. Unlike Judge Dee, Chen had to navigate modern politics.
        He lit a cigarette. The Shanghai dust settled on the window.
        Chen thought about Confucius and the mandate of heaven.
        quoting Lu Xun.
        The cadre smiled, "You represent the law, Inspector."
        """ * 50
    
    # Load reference corpus
    if REFERENCE_CORPUS_FILES:
        print(f"Loading reference corpus from {len(REFERENCE_CORPUS_FILES)} file(s):")
        reference_parts = []
        for ref_file in REFERENCE_CORPUS_FILES:
            print(f"  - {ref_file}")
            reference_parts.append(load_corpus_from_file(ref_file))
        reference_corpus = "\n\n".join(reference_parts)
        print(f"  -> Total loaded {len(reference_corpus)} characters")
    else:
        print("Using dummy reference corpus")
        reference_corpus = """
        The detective sat in his office, drinking a cup of coffee. 
        It was a sign of the long night ahead. He looked at the sergeant, a high-ranking officer 
        who spoke of the evidence. "Sir," the detective said, looking at the gun.
        "We need to find the fingerprints."
        The murder case was difficult. Unlike Sherlock Holmes, he had to navigate modern rules.
        He lit a cigarette. The London dust settled on the window.
        He thought about the law and the justice system.
        The officer smiled, "You represent the police, Detective."
        """ * 50
    
    return target_corpus, reference_corpus

# ==========================================
# PART 2: THE NLP ENGINES (MOCK & REAL)
# ==========================================

class MockNLPProcessor:
    """
    Simulates spaCy behavior using Regex for users who haven't installed libraries.
    """
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'of', 'in', 'is', 'it', 'was', 'he', 'his', 'to', 'and', 'that', 'with', 'had', 'at', 'i', 'you'}
        self.knowledge_base = self._get_kb()
        
        # Storage for extracted person names
        self.person_names = set()
        self.name_tokens = set()
        
        # MOCK COREFERENCE CLUSTERS
        # This simulates what fastcoref would discover in the text
        self.coref_clusters = {
            "Chen": "Chief Inspector Chen Cao",
            "Inspector": "Chief Inspector Chen Cao",
            "He": "Chief Inspector Chen Cao", # Context dependent in reality
            "Lu": "Lu Xun",
            "Xun": "Lu Xun",
            "Judge": "Judge Dee"
        }

    def _get_kb(self):
        # UPDATED: Simulation now reflects what Wikidata would actually return
        return {
            "Chen": {"id": "AMBIGUOUS", "desc": "Surname (Generic)"},
            "Chief Inspector Chen Cao": {"id": None, "desc": "No KB Match"}, # The resolved name has no entry
            "Yu": {"id": "AMBIGUOUS", "desc": "Surname (Generic)"},
            "Lu Xun": {"id": "Q23114", "desc": "Chinese writer"},
            "Confucius": {"id": "Q4604", "desc": "Chinese philosopher"},
            "Judge Dee": {"id": "Q1234", "desc": "Historical figure"},
            "Mao": {"id": "Q5816", "desc": "Historical figure"},
            "Shanghai": {"id": "Q8686", "desc": "City"}
        }

    def tokenize(self, text):
        """Simple regex tokenizer that mimics spaCy's word separation."""
        words = re.findall(r'\b[A-Za-z-]+\b', text)
        return words
        
    def resolve_coreference(self, word):
        """
        Simulates fastcoref: Returns the 'Representative Mention' for a word.
        e.g., 'Chen' -> 'Chief Inspector Chen Cao'
        """
        capitalized = word.capitalize()
        # If the word is part of a known cluster, return the full name
        return self.coref_clusters.get(capitalized, capitalized)

    def is_cultural_entity(self, word):
        # STEP 1: RESOLVE COREF (Find out who 'Chen' actually is)
        resolved_name = self.resolve_coreference(word)
        
        # STEP 2: LINK RESOLVED NAME (Look up 'Chief Inspector Chen Cao')
        if resolved_name in self.knowledge_base:
            entity_data = self.knowledge_base[resolved_name]
            
            # If resolved name has no ID (None), it's fictional -> Discard
            if entity_data["id"] is None:
                return False
                
            # If it has a Q-ID, it's Historical -> Keep
            if entity_data["id"].startswith("Q"):
                return True
                
        # Fallback: Check the original word if resolution failed
        capitalized = word.capitalize()
        if capitalized in self.knowledge_base:
             entity_data = self.knowledge_base[capitalized]
             if entity_data["id"] == "AMBIGUOUS": return False # Safety net
             if entity_data["id"].startswith("Q"): return True
             
        return False
    
    def extract_person_names(self, text):
        """
        Mock extraction of person names using simple heuristics.
        In real mode, this would use spaCy NER.
        """
        print("Extracting person names (Mock mode - using simple capitalization heuristic)...")
        
        # Simple heuristic: capitalized words that appear in our knowledge base
        words = self.tokenize(text)
        
        for word in words:
            if word and word[0].isupper():
                # Check if it's a known name
                if word in self.knowledge_base or word.capitalize() in self.knowledge_base:
                    self.person_names.add(word)
                    self.name_tokens.add(word.lower())
        
        # Also add known names from coref clusters
        for full_name in self.coref_clusters.values():
            self.person_names.add(full_name)
            for token in full_name.split():
                if token[0].isupper():
                    self.name_tokens.add(token.lower())
        
        print(f"  Found {len(self.person_names)} person names (mock)")
        print(f"  Found {len(self.name_tokens)} name tokens")
        
        return self.person_names, self.name_tokens
    
    def is_part_of_person_name(self, word):
        """
        Check if a word is part of any person name.
        """
        return word.lower() in self.name_tokens


class EntityLinkerFilter:
    """
    THE ADVANCED SOLUTION:
    
    Pipeline:
    1. NER extracts all person names from corpus (e.g., "Chen", "Lu Xun")
    2. Coreference finds the longest/best mention of each name
       (e.g., "Chen" → "Chief Inspector Chen Cao")
    3. Entity linker validates these best mentions against Wikidata
       (e.g., "Inspector Chen Cao" → no match = fictional)
    
    Benefits of using coref:
    - Better entity linking: "Inspector Chen Cao" vs just "Chen"
    - More context: Titles and full names improve Wikidata matches
    - Still efficient: Only runs once on the full document, not per word
    """
    def __init__(self, use_coref=False):
        try:
            import spacy
            
            # Load Spacy with Entity Linker
            print("Loading spaCy + Entity Linker...")
            #self.nlp = spacy.load("en_core_web_sm")
            #self.nlp = spacy.load("en_core_web_md")  # Using medium model to avoid transformer compatibility issues
            self.nlp = spacy.load("en_core_web_lg")  # Using large model to avoid transformer compatibility issues
            #self.nlp = spacy.load("en_core_web_trf")
            
            # Increase max length for processing large texts
            self.nlp.max_length = 2000000  # 2 million characters
            
            # This requires: pip install spacy-entity-linker && python -m spacy_entity_linker "download_knowledge_base"
            self.nlp.add_pipe("entityLinker", last=True)
            
            # Storage for extracted person names
            self.person_names = set()  # Full names found in corpus
            self.name_tokens = set()   # Individual tokens that are part of names
            self.best_mentions = {}    # Maps name token -> best/longest mention from coref
            self.entity_links = {}     # Maps name token -> (entity_label, entity_description) from initial validation
            
            # Optionally load coreference model
            self.use_coref = use_coref
            self.coref_model = None
            self.coref_cache = {}
            self.current_clusters = None
            
            if use_coref:
                from fastcoref import FCoref
                print("Loading FCoref model...")
                self.coref_model = FCoref(device='cpu')
                print("Note: Coref loaded but not needed for name filtering")
            
            self.current_text = None
            
        except ImportError as e:
            print(f"Missing libraries: {e}")
            print("Run: pip install spacy spacy-entity-linker")
            if use_coref:
                print("For coref: pip install fastcoref")
            sys.exit(1)
    
    def extract_person_names(self, text):
        """
        Extract all PERSON entities from the corpus using NER.
        This identifies which words are actually part of person names.
        
        Args:
            text: The corpus text to analyze
            
        Returns:
            tuple: (set of full names, set of individual name tokens)
        """
        print("Extracting person names from corpus using NER...")
        
        # Process text with spaCy (in chunks if too large)
        max_length = 1000000  # spaCy's default max length
        
        person_names = set()
        name_tokens = set()
        
        # Split into chunks if text is too long
        text_length = len(text)
        if text_length > max_length:
            print(f"  Text is large ({text_length} chars), processing in chunks...")
            chunks = [text[i:i+max_length] for i in range(0, text_length, max_length)]
        else:
            chunks = [text]
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            doc = self.nlp(chunk)
            
            # Extract all PERSON entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    full_name = ent.text
                    person_names.add(full_name)
                    
                    # Also add individual tokens from the name
                    for token in ent:
                        if token.is_alpha:  # Only alphabetic tokens
                            name_tokens.add(token.text.lower())
        
        print(f"  Found {len(person_names)} NER candidates")

        print(f"  NER candidates: {person_names}")
        
        # Validate candidates using entity linking
        validated_names, validated_tokens = self._validate_person_names(person_names)
        
        print(f"  After validation: {len(validated_names)} actual person names")
        
        # Show some examples
        if validated_names:
            examples = list(validated_names)[:10]
            print(f"  Examples: {', '.join(examples)}")
        
        self.person_names = validated_names
        self.name_tokens = validated_tokens
        
        return validated_names, validated_tokens
    
    def _validate_person_names(self, candidate_names):
        """
        Validate person name candidates using entity linking.
        Keep only entities that either:
        1. Link to person entities (have person indicators in description)
        2. Don't link to anything (could be fictional/local characters)
        
        Args:
            candidate_names: Set of person name candidates from NER
            
        Returns:
            tuple: (validated names set, validated tokens set)
        """
        print("  Validating person names using entity linking...")
        
        validated_names = set()
        validated_tokens = set()
        removed_count = 0
        
        for candidate in candidate_names:
            print(f"  Validating candidate: {candidate}")
            doc = self.nlp(candidate)
            
            is_valid = True  # Default: keep if no links found
            found_link = False
            
            # Check entity linking results
            for ent in doc.ents:
                if hasattr(ent._, "linkedEntities") and ent._.linkedEntities:
                    for entity in ent._.linkedEntities:
                        found_link = True
                        
                        # Get entity description
                        desc = entity.get_description().lower() if entity.get_description() else ""
                        label = entity.get_label().lower() if entity.get_label() else ""

                        print(f"Entity: {entity.get_label()} ({desc})")
                        
                        # Check if it's a PERSON entity
                        person_indicators = [
                            'person', 'human', 'people',
                            'writer', 'author', 'poet', 'novelist',
                            'politician', 'president', 'minister', 'king', 'queen', 'emperor',
                            'philosopher', 'thinker', 'scholar', 'scientist',
                            'artist', 'painter', 'musician', 'composer',
                            'actor', 'actress', 'director',
                            'soldier', 'general', 'officer',
                            'character',  # Include fictional characters here
                            'born', 'died'  # Biographical indicators
                        ]
                        
                        # Split description into words for accurate matching
                        # Remove punctuation to avoid missing matches like "writer," vs "writer"
                        desc_words = set(re.findall(r'\b[a-z]+\b', desc))
                        
                        # Check for explicit person indicator words
                        has_person_indicator = any(indicator in desc_words for indicator in person_indicators)
                        
                        # Also check for person-indicating suffixes
                        # Examples: journalist, chairman, teacher, actor, musician
                        person_suffixes = ['man', 'woman', 'ist', 'er', 'or', 'ian']
                        
                        # Common false positives to exclude
                        suffix_blacklist = {
                            'manner', 'under', 'after', 'other', 'over', 'never',
                            'either', 'neither', 'whether', 'rather', 'later',
                            'number', 'member', 'gender', 'order', 'border',
                            'for', 'or', 'nor'
                        }
                        
                        has_person_suffix = any(
                            word.endswith(suffix) and word not in suffix_blacklist and len(word) > len(suffix)
                            for word in desc_words for suffix in person_suffixes
                        )
                        
                        # If description contains person indicators, it's valid
                        if has_person_indicator or has_person_suffix:
                            # Valid person entity - store the entity link for later use
                            for token in candidate.split():
                                if token.isalpha():
                                    token_lower = token.lower()
                                    # Store the entity link: token -> (label, description)
                                    if token_lower not in self.entity_links:
                                        self.entity_links[token_lower] = (label, desc)
                            break
                        else:
                            # Linked to something but not a person
                            is_valid = False
                            removed_count += 1
                            print(f"    Removed '{candidate}': {label} ({desc})")
                            break
                    
                    if not is_valid:
                        break
            
            # If no link was found, keep it (could be fictional/local character)
            # If link was found and is_valid=True, keep it (person entity)
            # If link was found and is_valid=False, don't keep it (non-person entity)
            if is_valid:
                validated_names.add(candidate)
                # Add tokens from validated names
                for token in candidate.split():
                    if token.isalpha():
                        validated_tokens.add(token.lower())
        
        if removed_count > 0:
            print(f"  Removed {removed_count} non-person entities")
        
        return validated_names, validated_tokens
    
    def preprocess_document(self, text, run_coref=False):
        """
        Extract all person names from the corpus.
        Optionally run FastCoref (not needed for simple name filtering).
        This should be called once per document before processing individual words.
        
        Args:
            text: The corpus text
            run_coref: If True, also run coreference resolution (optional, adds overhead)
        """
        # Use text hash as cache key to avoid reprocessing same text
        text_hash = hash(text)
        
        if text_hash in self.coref_cache:
            self.current_text = text
            self.current_clusters = self.coref_cache[text_hash]
            return
        
        # Extract person names using NER (this is what we actually need)
        self.extract_person_names(text)
        
        # Optionally run coreference resolution to find best mentions
        if run_coref and self.coref_model:
            print("Running coreference resolution on document...")
            preds = self.coref_model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=True)
            
            # Cache the results
            self.coref_cache[text_hash] = clusters
            self.current_clusters = clusters
            
            print(f"Found {len(clusters)} coreference clusters")
            
            # Find the best (longest) mention for each name
            self._map_names_to_best_mentions(clusters)
        else:
            if run_coref and not self.coref_model:
                print("Warning: Coref requested but model not loaded")
            else:
                print("Note: Using NER names directly (no coref enhancement)")
            self.coref_cache[text_hash] = []
            self.current_clusters = []
        
        self.current_text = text
    
    def preprocess_document_chunked(self, text, num_chunks=5, run_coref=False):
        """
        Process a large document in chunks to avoid memory issues.
        Splits text into chunks, processes each separately, and combines results.
        
        Args:
            text: The full corpus text
            num_chunks: Number of chunks to split into
            run_coref: If True, run coreference resolution on each chunk
        """
        print(f"\n>> Processing document in {num_chunks} chunks to manage memory...")
        
        # Split text into chunks
        from collections import defaultdict
        chunks = split_text_into_chunks(text, num_chunks)
        
        # Store combined results
        all_person_names = set()
        all_name_tokens = set()
        all_clusters = []
        all_best_mentions = {}
        all_entity_links = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            chunk_size = len(chunk.split())
            print(f"\n--- Processing Chunk {i}/{num_chunks} ({chunk_size} tokens) ---")
            
            # Temporarily clear previous state
            self.person_names = set()
            self.name_tokens = set()
            self.best_mentions = {}
            self.entity_links = {}
            
            # Extract person names using NER
            self.extract_person_names(chunk)
            
            # Combine NER results
            all_person_names.update(self.person_names)
            all_name_tokens.update(self.name_tokens)
            
            # Combine entity links (keep first occurrence if duplicate)
            for token, link in self.entity_links.items():
                if token not in all_entity_links:
                    all_entity_links[token] = link
            
            # Optionally run coreference on this chunk
            if run_coref and self.coref_model:
                print(f"  Running coreference on chunk {i}...")
                preds = self.coref_model.predict(texts=[chunk])
                clusters = preds[0].get_clusters(as_strings=True)
                
                all_clusters.extend(clusters)
                print(f"  Found {len(clusters)} clusters in chunk {i}")
                
                # Map names to best mentions for this chunk
                self._map_names_to_best_mentions(clusters)
                
                # Merge best mentions (prefer longer mentions)
                for token, mention in self.best_mentions.items():
                    if token not in all_best_mentions or len(mention) > len(all_best_mentions[token]):
                        all_best_mentions[token] = mention
        
        # Restore combined results
        print(f"\n>> Combining results from all chunks...")
        print(f"  Total person names found: {len(all_person_names)}")
        print(f"  Total unique name tokens: {len(all_name_tokens)}")
        print(f"  Total entity links stored: {len(all_entity_links)}")
        if run_coref:
            print(f"  Total coreference clusters: {len(all_clusters)}")
            print(f"  Best mentions mapped: {len(all_best_mentions)}")
        
        self.person_names = all_person_names
        self.name_tokens = all_name_tokens
        self.entity_links = all_entity_links
        self.current_clusters = all_clusters
        self.best_mentions = all_best_mentions
        self.current_text = text
        
        # Cache the combined results
        text_hash = hash(text)
        self.coref_cache[text_hash] = all_clusters
    
    def _map_names_to_best_mentions(self, clusters):
        """
        For each person name found by NER, find the longest coreference mention
        that includes at least one token from that name.
        
        Example:
            NER found: "Chen"
            Coref cluster: ["Chen", "he", "Inspector", "Chief Inspector Chen Cao"]
            Best mention: "Chief Inspector Chen Cao" (longest that contains "Chen")
        
        Args:
            clusters: List of coreference clusters (each cluster is a list of mention strings)
        """
        print("Mapping names to their best mentions from coreference...")
        
        self.best_mentions = {}
        mapped_count = 0
        
        # For each extracted person name
        for person_name in self.person_names:
            name_tokens_set = set(t.lower() for t in person_name.split() if t.isalpha())
            
            best_mention = person_name  # Default to NER name
            max_length = len(person_name)
            
            # Search through all coref clusters
            for cluster in clusters:
                for mention in cluster:
                    mention_tokens = set(t.lower() for t in mention.split() if t.isalpha())
                    
                    # Check if ALL tokens from our name are in this mention (subset check)
                    if name_tokens_set <= mention_tokens:  # Subset: all name tokens must be present
                        if len(mention) > max_length:
                            best_mention = mention
                            max_length = len(mention)
            
            # Map each individual token from the name to the best mention
            for token in name_tokens_set:
                self.best_mentions[token] = best_mention
            
            if best_mention != person_name:
                mapped_count += 1
                print(f"  '{person_name}' → '{best_mention}'")
        
        print(f"Enhanced {mapped_count} names with longer coreference mentions")
    
    def is_part_of_person_name(self, word):
        """
        Check if a word is part of any person name found in the corpus.
        
        Args:
            word: The word to check (lowercase)
            
        Returns:
            bool: True if word is part of a person name, False otherwise
        """
        return word.lower() in self.name_tokens

    def get_resolved_entity_status(self, word, context_text=None):
        """
        Determines if a word refers to a historical figure or fictional character.
        First checks if we already have an entity link from initial validation.
        Falls back to coreference resolution if needed.
        
        Args:
            word: The word to check (already known to be part of a name)
            context_text: Optional, for backward compatibility (ignored, NER already done)
        """
        word_lower = word.lower()
        
        # STEP 0: Check if we already have an entity link from initial validation
        # This is more reliable than re-linking coreference mentions
        if word_lower in self.entity_links:
            label, desc = self.entity_links[word_lower]
            
            # Apply the same filtering logic as before
            if "fictional" in desc or "literary character" in desc:
                return False, f"Direct Link: {label} ({desc}) -> Filtered (Fictional)"
            
            # If we have a valid historical entity link, use it!
            return True, f"Direct Link: {label} ({desc})"
        
        # STEP 1: Get the best mention for this word
        # If we have coref results, use the longest mention that includes this word
        if word_lower in self.best_mentions:
            resolved_name = self.best_mentions[word_lower]
        else:
            # Fall back to finding the name from NER
            matching_names = []
            for full_name in self.person_names:
                name_tokens = [t.lower() for t in full_name.split() if t.isalpha()]
                if word_lower in name_tokens:
                    matching_names.append(full_name)
            
            if not matching_names:
                return False, f"'{word}' marked as name but no full name found"
            
            resolved_name = matching_names[0]
        
        # STEP 2: Entity Linking on the resolved name
        doc = self.nlp(resolved_name)
        
        # Check all entities found in the resolved name
        found_link = False
        for ent in doc.ents:
            # The entityLinker pipe adds a ._.linkedEntities attribute
            if hasattr(ent._, "linkedEntities") and ent._.linkedEntities:
                for entity in ent._.linkedEntities:
                    found_link = True
                    # Logic: If it has a valid Wikidata Q-ID and isn't a fictional category
                    desc = entity.get_description().lower()
                    
                    # Filtering Logic
                    if "fictional" in desc: continue
                    if "literary character" in desc: continue
                    
                    # If we found a valid historical match for the RESOLVED name
                    return True, f"Resolved to '{resolved_name}' -> Linked: {entity.get_label()} ({desc})"
                    
        if found_link:
             return False, f"Resolved to '{resolved_name}' -> Linked but filtered (Fictional)"
        return False, f"Resolved to '{resolved_name}' -> No Valid Link (Likely Local Character)"


class WikidataFilter:
    """
    THE REAL SOLUTION: 
    Connects to Wikidata API to automatically determine if a name is 
    a real historical figure or a fictional/unknown character.
    """
    def __init__(self):
        try:
            import requests
            self.requests = requests
        except ImportError:
            print("Error: requests library not found. Please run 'pip install requests'")
            sys.exit(1)
            
        self.cache = {}

    def check_entity(self, query):
        """
        Queries Wikidata for the entity name.
        Returns: True (Keep - Cultural/Historical), False (Discard - Fictional/Common)
        """
        if query in self.cache:
            return self.cache[query]

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
            "limit": 1
        }
        
        try:
            # Note: This requires internet access. 
            response = self.requests.get(url, params=params)
            data = response.json()
            
            # LOGIC 1: THE VOID CHECK
            # If Wikidata returns no results, it likely doesn't exist in history.
            if not data.get("search"):
                self.cache[query] = False
                return False
            
            # LOGIC 2: THE TYPE CHECK
            item = data["search"][0]
            description = item.get("description", "").lower()
            
            # Filter out explicit fictional characters or generic surnames
            if "fictional character" in description or "surname" in description:
                self.cache[query] = False
                return False
                
            # If it exists and isn't fictional/surname, it's likely a real concept/person
            self.cache[query] = True
            return True
            
        except Exception as e:
            print(f"API Error for {query}: {e}")
            return False


class SpacyNLPProcessor:
    """
    Wraps the REAL spaCy library to match the API expected by the pipeline.
    """
    def __init__(self):
        try:
            import spacy
            # Load the small English model
            #self.nlp = spacy.load("en_core_web_sm")
            #self.nlp = spacy.load("en_core_web_md")
            self.nlp = spacy.load("en_core_web_lg")  # Using large model to avoid transformer compatibility issues
            #self.nlp = spacy.load("en_core_web_trf")
            
            # Increase max length for tokenization (reference corpus may be large)
            # We only need tokenization for the reference corpus, not NER
            self.nlp.max_length = 2000000  # 2 million characters
        except ImportError:
            print("Error: spaCy is not installed. Please run 'pip install spacy'")
            sys.exit(1)
        except OSError:
            print("Error: Model not found. Please run 'python -m spacy download en_core_web_sm'")
            sys.exit(1)
            
        # Use spaCy's built-in stop words
        self.stopwords = self.nlp.Defaults.stop_words
        
        # Storage for extracted person names
        self.person_names = set()
        self.name_tokens = set()
        
        # Initialize the Real Wikidata Filter if configured
        if USE_WIKIDATA_API:
            self.wikidata = WikidataFilter()
        elif USE_ENTITY_LINKER:
             # Just a placeholder, actual logic is passed in run_pipeline
             pass
        else:
            self.wikidata = MockNLPProcessor() # Fallback to mock logic if offline

    def extract_person_names(self, text):
        """
        Extract all PERSON entities from the corpus using NER.
        
        Args:
            text: The corpus text to analyze
            
        Returns:
            tuple: (set of full names, set of individual name tokens)
        """
        print("Extracting person names from corpus using NER...")
        
        max_length = 1000000
        person_names = set()
        name_tokens = set()
        
        # Split into chunks if needed
        text_length = len(text)
        if text_length > max_length:
            print(f"  Text is large ({text_length} chars), processing in chunks...")
            chunks = [text[i:i+max_length] for i in range(0, text_length, max_length)]
        else:
            chunks = [text]
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            doc = self.nlp(chunk)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    full_name = ent.text
                    person_names.add(full_name)
                    
                    for token in ent:
                        if token.is_alpha:
                            name_tokens.add(token.text.lower())
        
        print(f"  Found {len(person_names)} unique person names")
        print(f"  Found {len(name_tokens)} unique name tokens")
        
        if person_names:
            examples = list(person_names)[:10]
            print(f"  Examples: {', '.join(examples)}")
        
        self.person_names = person_names
        self.name_tokens = name_tokens
        
        return person_names, name_tokens
    
    def is_part_of_person_name(self, word):
        """
        Check if a word is part of any person name found in the corpus.
        
        Args:
            word: The word to check (lowercase)
            
        Returns:
            bool: True if word is part of a person name, False otherwise
        """
        return word.lower() in self.name_tokens

    def tokenize(self, text):
        """
        Wraps spaCy's processing to return a list of strings, 
        matching the expected input of the statistical engine.
        """
        doc = self.nlp(text)
        # Return text of tokens if they are alphabetic (removes punctuation)
        return [token.text for token in doc if token.is_alpha]

    def is_cultural_entity(self, word):
        """
        Hybrid logic: Uses Wikidata if enabled, otherwise Mock.
        """
        if USE_WIKIDATA_API:
            return self.wikidata.check_entity(word)
        # Note: Advanced Coref is context-dependent, so it's handled inside the loop
        # inside run_phase_1_pipeline, not here in the simple filter.
        else:
            return self.wikidata.is_cultural_entity(word)

# ==========================================
# PART 3: THE STATISTICAL ENGINE (KEYNESS)
# ==========================================

def calculate_log_likelihood(target_tokens, ref_tokens):
    """
    Implements the G^2 formula.
    """
    target_counts = Counter(target_tokens)
    ref_counts = Counter(ref_tokens)
    
    c = len(target_tokens)
    d = len(ref_tokens)
    N = c + d
    
    keywords = []
    
    for word, a in target_counts.items():
        b = ref_counts[word]
        if a < 3: continue 
        
        E1 = c * (a + b) / N
        E2 = d * (a + b) / N
        
        term1 = a * math.log(a / E1) if a > 0 else 0
        term2 = b * math.log(b / E2) if b > 0 else 0
        
        g2 = 2 * (term1 + term2)
        
        if a > E1:
            keywords.append({
                "word": word,
                "freq_target": a,
                "freq_ref": b,
                "G2": round(g2, 2)
            })
            
    return keywords

# ==========================================
# PART 4: THE PIPELINE RUNNER
# ==========================================

def run_phase_1_pipeline():
    print("--- Phase 1: Automated Cultural Keyword Extraction ---\n")
    
    # 1. Select Engine
    entity_filter = None
    
    if USE_ENTITY_LINKER:
        print(">> Using Entity Linker Pipeline (spacy-entity-linker)")
        entity_filter = EntityLinkerFilter(use_coref=RUN_COREF)
        # We still need basic tokenizer
        nlp = SpacyNLPProcessor() if USE_REAL_SPACY else MockNLPProcessor()
    elif USE_REAL_SPACY:
        print(">> Using REAL spaCy model (en_core_web_sm)")
        nlp = SpacyNLPProcessor()
    else:
        print(">> Using MOCK NLP Processor (Regex only)")
        nlp = MockNLPProcessor()
    
    # 2. Ingest Data
    t_txt, r_txt = load_corpora()
    
    print(f"Target Corpus Size: {len(t_txt.split())} tokens")
    print(f"Reference Corpus Size: {len(r_txt.split())} tokens\n")
    
    # 2.5 Extract person names from corpus using NER (with chunking for memory efficiency)
    if USE_ENTITY_LINKER and entity_filter:
        # Use entity linker's NER to extract person names
        # Process in chunks to avoid memory issues
        if NUM_CORPUS_CHUNKS > 1:
            entity_filter.preprocess_document_chunked(t_txt, num_chunks=NUM_CORPUS_CHUNKS, run_coref=RUN_COREF)
        else:
            # Single pass processing (original method)
            entity_filter.preprocess_document(t_txt, run_coref=RUN_COREF)
    elif USE_REAL_SPACY or not USE_ENTITY_LINKER:
        # Extract person names using basic spaCy NER
        if hasattr(nlp, 'extract_person_names'):
            nlp.extract_person_names(t_txt)
        else:
            print("Note: Name extraction not available in Mock mode")
    
    # 3. Tokenization
    print("Step 1: Tokenization & Stopword Removal...")
    t_tokens_raw = nlp.tokenize(t_txt)
    r_tokens_raw = nlp.tokenize(r_txt)
    
    t_tokens_norm = [w.lower() for w in t_tokens_raw if w.lower() not in nlp.stopwords]
    r_tokens_norm = [w.lower() for w in r_tokens_raw if w.lower() not in nlp.stopwords]
    
    # 4. Statistical Calculation
    print("Step 2: Calculating Log-Likelihood (G^2)...")
    raw_keywords = calculate_log_likelihood(t_tokens_norm, r_tokens_norm)
    raw_keywords.sort(key=lambda x: x["G2"], reverse=True)
    
    print(f" -> Found {len(raw_keywords)} statistical candidates.\n")
    
    # 5. The Semantic Sieve
    G2_THRESHOLD = 6.63  # Lowered from 10.83 to capture more candidates
    validated_keywords = []
    
    print("Step 3: Applying Semantic Sieve (Thresholding + Entity Linking)...")
    print(f"Processing top 200 candidates to find at least 150 valid keywords...")
    print(f"{'WORD':<15} {'G2 SCORE':<10} {'STATUS':<20} {'REASON'}")
    print("-" * 65)
    
    for k in raw_keywords[:200]:  # Increased from 15 to 200 
        word = k['word']
        score = k['G2']
        
        if score < G2_THRESHOLD:
            print(f"{word:<15} {score:<10} DISCARDED {'Below Stat Threshold'}")
            continue
        
        # FILTER 1: Single letters (e.g., "c", "e")
        if len(word) == 1:
            print(f"{word:<15} {score:<10} DISCARDED {'Single Letter'}")
            continue
        
        # FILTER 2: POS tagging - filter out adjectives and common non-cultural words
        # Use spaCy for POS tagging
        if USE_ENTITY_LINKER or USE_REAL_SPACY:
            nlp_for_pos = entity_filter.nlp if USE_ENTITY_LINKER else nlp.nlp
            doc = nlp_for_pos(word)
            if doc and len(doc) > 0:
                pos_tag = doc[0].pos_
                # Filter out adjectives (ADJ), determiners (DET), pronouns (PRON)
                if pos_tag in ['ADJ', 'DET', 'PRON', 'AUX', 'CCONJ', 'SCONJ']:
                    print(f"{word:<15} {score:<10} DISCARDED {f'POS: {pos_tag}'}")
                    continue
            
        capitalized = word.capitalize()
        
        # Main Logic: Check if word is a person name, then validate if historical or fictional
        # Check which processor we're using
        processor_to_check = entity_filter if USE_ENTITY_LINKER and entity_filter else nlp
        
        # Determine if this word is part of a person name
        is_name = False
        if hasattr(processor_to_check, 'is_part_of_person_name'):
            is_name = processor_to_check.is_part_of_person_name(word)
        
        if is_name:
            # This word is part of a person name - check if historical or fictional
            if USE_ENTITY_LINKER and entity_filter:
                # Use entity linker to check
                is_cultural, adv_reason = entity_filter.get_resolved_entity_status(word)
                
                if is_cultural:
                    status = "KEPT (Historical)"
                    reason = adv_reason
                    validated_keywords.append(k)
                else:
                    status = "DISCARDED (Name)"
                    reason = adv_reason
            elif hasattr(nlp, 'knowledge_base') and capitalized in nlp.knowledge_base:
                # Using Mock NLP with knowledge base
                if nlp.is_cultural_entity(word):
                    status = "KEPT (Historical)"
                    reason = f"Historical Figure: {nlp.knowledge_base[capitalized]['desc']}"
                    validated_keywords.append(k)
                else:
                    status = "DISCARDED (Name)"
                    reason = "Fictional/Plot Character"
            else:
                # Using real spaCy + Wikidata
                result = nlp.is_cultural_entity(capitalized)
                if result is True:
                    status = "KEPT (Historical)"
                    reason = "Historical Figure (Wikidata)"
                    validated_keywords.append(k)
                elif result is False:
                    status = "DISCARDED (Name)"
                    reason = "Fictional/Unknown Person"
                else:
                    # API error - be conservative, discard names we can't verify
                    status = "DISCARDED (Name)"
                    reason = "Could not verify (API unavailable)"
        else:
            # Not a person name - keep as lexical cultural term
            status = "KEPT (Lexical)"
            reason = "Cultural Term (not a name)"
            validated_keywords.append(k)
            
        print(f"{word:<15} {score:<10} {status:<20} {reason}")

    # Final Output
    print("\n--- Final Validated Cultural Keywords ---")
    semantic_map = {
        "tea": "Food/Drink", "dragon": "Mythology", "cadre": "Social Politics",
        "red": "Social Politics", "confucius": "Philosophy", "shanghai": "Geography",
        "lu": "Literature", "xun": "Literature"
    }
    
    for k in validated_keywords:
        word = k['word']
        category = semantic_map.get(word, "General Cultural")
        print(f"Keyword: {word.upper():<12} | G2: {k['G2']} | Category: {category}")
    
    # Save results to file
    import json
    from datetime import datetime
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "target_corpus": TARGET_CORPUS_FILE,
            "reference_corpus": REFERENCE_CORPUS_FILES,
            "target_corpus_size": len(t_tokens_norm),
            "reference_corpus_size": len(r_tokens_norm),
            "g2_threshold": G2_THRESHOLD,
            "total_candidates": len(raw_keywords),
            "validated_keywords_count": len(validated_keywords)
        },
        "validated_keywords": [
            {
                "keyword": k['word'],
                "g2_score": k['G2'],
                "category": semantic_map.get(k['word'], "General Cultural"),
                "target_frequency": k.get('freq_target', 0),
                "reference_frequency": k.get('freq_ref', 0)
            }
            for k in validated_keywords
        ]
    }
    
    output_file = "cultural_keywords_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving results: {e}")

if __name__ == "__main__":
    run_phase_1_pipeline()