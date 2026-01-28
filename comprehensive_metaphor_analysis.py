"""
Comprehensive Metaphor and Cultural Analysis Tool

Combines three analysis methods:
1. Cultural Keywords Identification (from culturalKeywordsListIdentification_1.py)
2. Signaling Marker Search (from signalingMarkerSearch.py)
3. X is Y Metaphor Detection (from metaphorDetector.py)

Provides a unified analysis of a corpus with all three methods.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# Import functionality from existing scripts
import culturalKeywordsListIdentification_1 as ckl
from culturalKeywordsListIdentification_1 import (
    load_corpora,
    run_phase_1_pipeline,
    EntityLinkerFilter,
    SpacyNLPProcessor,
    MockNLPProcessor,
    calculate_log_likelihood
)

from signalingMarkerSearch import SignalingMarkerSearch

from metaphorDetector import MetaphorDetector


class ComprehensiveMetaphorAnalyzer:
    """
    Unified analyzer that combines cultural keywords, signaling markers, and X-is-Y metaphor detection.
    """
    
    def __init__(self, corpus_file: str, reference_corpus_files: List[str] = None):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            corpus_file: Path to the target corpus file
            reference_corpus_files: List of reference corpus files (for cultural keywords)
        """
        self.corpus_file = corpus_file
        self.reference_corpus_files = reference_corpus_files or []
        
        print("\n" + "="*80)
        print("COMPREHENSIVE METAPHOR ANALYSIS TOOL")
        print("="*80)
        print(f"\nTarget Corpus: {corpus_file}")
        if reference_corpus_files:
            print(f"Reference Corpus: {', '.join(reference_corpus_files)}")
        print()
        
        # Load corpus text
        self.corpus_text = self._load_corpus(corpus_file)
        print(f"Loaded corpus: {len(self.corpus_text)} characters, {len(self.corpus_text.split())} words\n")
        
        # Results storage
        self.cultural_keywords_results = None
        self.signaling_markers_results = None
        self.metaphor_structures_results = None
    
    def _load_corpus(self, filepath: str) -> str:
        """Load text from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return ""
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        Uses improved sentence splitting to handle various punctuation.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def find_keyword_occurrences(self, keyword: str, text: str) -> List[Dict]:
        """
        Find all sentence occurrences of a keyword in the text.
        
        Args:
            keyword: The keyword to search for
            text: The corpus text
            
        Returns:
            List of dictionaries containing sentence information
        """
        sentences = self.extract_sentences(text)
        occurrences = []
        
        # Create case-insensitive pattern with word boundaries
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        
        for idx, sentence in enumerate(sentences, 1):
            if pattern.search(sentence):
                occurrences.append({
                    'sentence_id': idx,
                    'sentence': sentence
                })
        
        return occurrences
    
    def analyze_cultural_keywords(self) -> Dict:
        """
        Run cultural keyword identification and find all occurrences.
        
        Returns:
            Dictionary with cultural keywords and their occurrences
        """
        print("\n" + "="*80)
        print("PHASE 1: CULTURAL KEYWORDS IDENTIFICATION")
        print("="*80 + "\n")
        
        # Import necessary functions and run the pipeline
        # We need to temporarily modify the configuration
        import culturalKeywordsListIdentification_1 as ckl
        
        # Store original values
        original_target = ckl.TARGET_CORPUS_FILE
        original_ref = ckl.REFERENCE_CORPUS_FILES
        
        # Set to our corpus files
        ckl.TARGET_CORPUS_FILE = self.corpus_file
        ckl.REFERENCE_CORPUS_FILES = self.reference_corpus_files if self.reference_corpus_files else [
            "Enigma of China Reference Corpus.txt"
        ]
        
        try:
            # Run the pipeline - this will print its own output
            # We need to capture the validated keywords
            
            # Load corpora
            t_txt, r_txt = load_corpora()
            
            # Initialize NLP processor
            entity_filter = None
            
            if ckl.USE_ENTITY_LINKER:
                print(">> Using Entity Linker Pipeline")
                entity_filter = EntityLinkerFilter(use_coref=ckl.RUN_COREF)
                nlp = SpacyNLPProcessor() if ckl.USE_REAL_SPACY else MockNLPProcessor()
            elif ckl.USE_REAL_SPACY:
                print(">> Using REAL spaCy model")
                nlp = SpacyNLPProcessor()
            else:
                print(">> Using MOCK NLP Processor")
                nlp = MockNLPProcessor()
            
            # Extract person names with adaptive chunking
            if ckl.USE_ENTITY_LINKER and entity_filter:
                # Adaptive chunking: calculate number of chunks based on text length
                # Keep each chunk under 800k characters to avoid memory issues
                max_chunk_size = 50000
                text_length = len(t_txt)
                adaptive_chunks = max(1, (text_length // max_chunk_size) + 1)
                
                if adaptive_chunks > 1:
                    print(f"\n>> Using adaptive chunking: {adaptive_chunks} chunks for {text_length:,} characters")
                    entity_filter.preprocess_document_chunked(t_txt, num_chunks=adaptive_chunks, run_coref=ckl.RUN_COREF)
                else:
                    entity_filter.preprocess_document(t_txt, run_coref=ckl.RUN_COREF)
            elif ckl.USE_REAL_SPACY or not ckl.USE_ENTITY_LINKER:
                if hasattr(nlp, 'extract_person_names'):
                    nlp.extract_person_names(t_txt)
            
            # Tokenization
            print("\nStep 1: Tokenization & Stopword Removal...")
            t_tokens_raw = nlp.tokenize(t_txt)
            r_tokens_raw = nlp.tokenize(r_txt)
            
            t_tokens_norm = [w.lower() for w in t_tokens_raw if w.lower() not in nlp.stopwords]
            r_tokens_norm = [w.lower() for w in r_tokens_raw if w.lower() not in nlp.stopwords]
            
            # Calculate log-likelihood
            print("Step 2: Calculating Log-Likelihood (G^2)...")
            raw_keywords = calculate_log_likelihood(t_tokens_norm, r_tokens_norm)
            raw_keywords.sort(key=lambda x: x["G2"], reverse=True)
            
            print(f" -> Found {len(raw_keywords)} statistical candidates.\n")
            
            # Apply semantic sieve
            G2_THRESHOLD = 6.63
            validated_keywords = []
            
            print("Step 3: Applying Semantic Sieve...")
            print(f"{'WORD':<15} {'G2 SCORE':<10} {'STATUS':<20} {'REASON'}")
            print("-" * 65)
            
            for k in raw_keywords[:200]:
                word = k['word']
                score = k['G2']
                
                if score < G2_THRESHOLD:
                    print(f"{word:<15} {score:<10} DISCARDED {'Below Stat Threshold'}")
                    continue
                
                # Filter single letters
                if len(word) == 1:
                    print(f"{word:<15} {score:<10} DISCARDED {'Single Letter'}")
                    continue
                
                # POS tagging filter
                if ckl.USE_ENTITY_LINKER or ckl.USE_REAL_SPACY:
                    nlp_for_pos = entity_filter.nlp if ckl.USE_ENTITY_LINKER else nlp.nlp
                    doc = nlp_for_pos(word)
                    if doc and len(doc) > 0:
                        pos_tag = doc[0].pos_
                        if pos_tag in ['ADJ', 'DET', 'PRON', 'AUX', 'CCONJ', 'SCONJ']:
                            print(f"{word:<15} {score:<10} DISCARDED {f'POS: {pos_tag}'}")
                            continue
                
                capitalized = word.capitalize()
                
                processor_to_check = entity_filter if ckl.USE_ENTITY_LINKER and entity_filter else nlp
                
                is_name = False
                if hasattr(processor_to_check, 'is_part_of_person_name'):
                    is_name = processor_to_check.is_part_of_person_name(word)
                
                if is_name:
                    if ckl.USE_ENTITY_LINKER and entity_filter:
                        is_cultural, adv_reason = entity_filter.get_resolved_entity_status(word)
                        
                        if is_cultural:
                            status = "KEPT (Historical)"
                            reason = adv_reason
                            validated_keywords.append(k)
                        else:
                            status = "DISCARDED (Name)"
                            reason = adv_reason
                    elif hasattr(nlp, 'knowledge_base') and capitalized in nlp.knowledge_base:
                        if nlp.is_cultural_entity(word):
                            status = "KEPT (Historical)"
                            reason = f"Historical Figure: {nlp.knowledge_base[capitalized]['desc']}"
                            validated_keywords.append(k)
                        else:
                            status = "DISCARDED (Name)"
                            reason = "Fictional/Plot Character"
                    else:
                        result = nlp.is_cultural_entity(capitalized)
                        if result is True:
                            status = "KEPT (Historical)"
                            reason = "Historical Figure (Wikidata)"
                            validated_keywords.append(k)
                        elif result is False:
                            status = "DISCARDED (Name)"
                            reason = "Fictional/Unknown Person"
                        else:
                            status = "DISCARDED (Name)"
                            reason = "Could not verify"
                else:
                    status = "KEPT (Lexical)"
                    reason = "Cultural Term (not a name)"
                    validated_keywords.append(k)
                    
                print(f"{word:<15} {score:<10} {status:<20} {reason}")
            
            # Now find all occurrences of each keyword in sentences
            print(f"\n\nFound {len(validated_keywords)} validated cultural keywords.")
            print("Finding all occurrences in corpus sentences...\n")
            
            keywords_with_occurrences = []
            
            for k in validated_keywords:
                keyword = k['word']
                occurrences = self.find_keyword_occurrences(keyword, self.corpus_text)
                
                keyword_info = {
                    'keyword': keyword,
                    'g2_score': k['G2'],
                    'target_frequency': k.get('freq_target', 0),
                    'reference_frequency': k.get('freq_ref', 0),
                    'occurrence_count': len(occurrences),
                    'occurrences': occurrences
                }
                
                keywords_with_occurrences.append(keyword_info)
                print(f"  {keyword}: {len(occurrences)} occurrences")
            
            # Sort by frequency (occurrence count)
            keywords_with_occurrences.sort(key=lambda x: x['occurrence_count'], reverse=True)
            
            results = {
                'total_keywords': len(validated_keywords),
                'keywords': keywords_with_occurrences
            }
            
            self.cultural_keywords_results = results
            
            return results
            
        finally:
            # Restore original values
            ckl.TARGET_CORPUS_FILE = original_target
            ckl.REFERENCE_CORPUS_FILES = original_ref
    
    def analyze_signaling_markers(self, signaling_words: List[str] = None) -> Dict:
        """
        Run signaling marker search.
        
        Args:
            signaling_words: Optional list of signaling words to search for
            
        Returns:
            Dictionary with signaling marker results
        """
        print("\n" + "="*80)
        print("PHASE 2: SIGNALING MARKER SEARCH")
        print("="*80 + "\n")
        
        # Default signaling words if none provided
        if signaling_words is None:
            signaling_words = [
                "metaphor", "metaphorical", "metaphorically", 
                "literally", "actually", "virtually", "practically", "essentially", 
                "really", "indeed", "somewhat", "pretty much", "more or less", "almost", 
                "similar to", "in a sense", "in more than one sense", "in both senses of",
                "import", "mean", "meaning of", "in every sense of", "in all senses of",
                "imitation", "model", "pretending",
                "token", "sign", "symbol", "epitome", 
                "appears", "resembles", "represents", "symbolizes", "embodies", 
                "became", "becomes", "become", "turned into", "transformed into",
                "a kind of", "sort of", "type of", "kind of",
                "like", "as",
                "seem", "appear",
                "can", "could", "may", "might", "must", "shall", "should", 
                "will", "would", "have to", "ought to", "need to"
            ]
        
        # Initialize signaling marker searcher
        searcher = SignalingMarkerSearch(signaling_words)
        
        # Search for signaling words
        print(f"Searching for {len(signaling_words)} signaling words...")
        results = searcher.search_signaling_words(self.corpus_text)
        
        # Get statistics
        stats = searcher.get_statistics(results)
        
        print(f"\nFound {stats['total_sentence_matches']} total matches")
        print(f"{stats['words_with_matches']} signaling words found in corpus\n")
        
        # Convert results to frequency-sorted format
        markers_with_occurrences = []
        
        for word in sorted(results.keys(), key=lambda w: len(results[w]), reverse=True):
            matches = results[word]
            if matches:
                marker_info = {
                    'marker': word,
                    'occurrence_count': len(matches),
                    'occurrences': matches
                }
                markers_with_occurrences.append(marker_info)
                print(f"  {word}: {len(matches)} occurrences")
        
        output = {
            'statistics': stats,
            'markers': markers_with_occurrences
        }
        
        self.signaling_markers_results = output
        
        return output
    
    def analyze_metaphor_structures(self, similarity_threshold: float = 0.5, 
                                   use_entity_linker: bool = True) -> Dict:
        """
        Run X is Y metaphor structure detection.
        
        Args:
            similarity_threshold: Threshold for semantic similarity
            use_entity_linker: Whether to use entity linker for filtering
            
        Returns:
            Dictionary with metaphor structure results
        """
        print("\n" + "="*80)
        print("PHASE 3: 'X IS Y' METAPHOR STRUCTURE DETECTION")
        print("="*80 + "\n")
        
        # Initialize metaphor detector
        print(f"Initializing metaphor detector (threshold: {similarity_threshold})...")
        detector = MetaphorDetector(
            model_name="en_core_web_sm",
            similarity_threshold=similarity_threshold,
            use_entity_linker=use_entity_linker
        )
        
        # Process the corpus
        print(f"\nProcessing corpus: {self.corpus_file}")
        metaphors, filtered = detector.process_text(self.corpus_text)
        
        # Separate potential metaphors from definitions
        potential_metaphors = [m for m in metaphors if m['is_potential_metaphor']]
        definitions = [m for m in metaphors if not m['is_potential_metaphor']]
        
        print(f"\nFound {len(metaphors)} total 'X is Y' patterns:")
        print(f"  - {len(potential_metaphors)} potential metaphors (low similarity)")
        print(f"  - {len(definitions)} likely definitions (high similarity)")
        print(f"  - {len(filtered)} filtered patterns (person names)\n")
        
        # Sort by semantic distance
        potential_metaphors.sort(key=lambda x: x['semantic_distance'], reverse=True)
        definitions.sort(key=lambda x: x['semantic_distance'])
        
        results = {
            'total_patterns': len(metaphors),
            'potential_metaphors_count': len(potential_metaphors),
            'definitions_count': len(definitions),
            'filtered_count': len(filtered),
            'potential_metaphors': potential_metaphors,
            'definitions': definitions,
            'filtered_patterns': filtered
        }
        
        self.metaphor_structures_results = results
        
        return results
    
    def run_complete_analysis(self, signaling_words: List[str] = None,
                            similarity_threshold: float = 0.5) -> Dict:
        """
        Run all three analysis phases.
        
        Returns:
            Dictionary containing all results
        """
        # Phase 1: Cultural Keywords
        cultural_results = self.analyze_cultural_keywords()
        
        # Phase 2: Signaling Markers
        signaling_results = self.analyze_signaling_markers(signaling_words)
        
        # Phase 3: Metaphor Structures
        metaphor_results = self.analyze_metaphor_structures(similarity_threshold)
        
        # Combine all results
        complete_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'corpus_file': self.corpus_file,
                'reference_corpus_files': self.reference_corpus_files,
                'corpus_length': len(self.corpus_text),
                'corpus_word_count': len(self.corpus_text.split())
            },
            'cultural_keywords': cultural_results,
            'signaling_markers': signaling_results,
            'metaphor_structures': metaphor_results
        }
        
        return complete_results
    
    def display_results(self):
        """Display all results in a formatted manner."""
        if not all([self.cultural_keywords_results, 
                   self.signaling_markers_results, 
                   self.metaphor_structures_results]):
            print("Please run complete analysis first!")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print("="*80)
        
        # Cultural Keywords
        print("\n" + "-"*80)
        print("1. CULTURAL KEYWORDS (sorted by frequency)")
        print("-"*80)
        
        ck_results = self.cultural_keywords_results
        print(f"\nTotal Cultural Keywords: {ck_results['total_keywords']}")
        
        for i, kw in enumerate(ck_results['keywords'][:20], 1):  # Show top 20
            print(f"\n{i}. {kw['keyword'].upper()} (G²={kw['g2_score']:.2f}, Occurrences={kw['occurrence_count']})")
            print(f"   Target Freq: {kw['target_frequency']}, Reference Freq: {kw['reference_frequency']}")
            
            # Show first 3 sentences
            for j, occ in enumerate(kw['occurrences'][:3], 1):
                sentence = occ['sentence'][:150]
                if len(occ['sentence']) > 150:
                    sentence += "..."
                print(f"   {j}. {sentence}")
            
            if kw['occurrence_count'] > 3:
                print(f"   ... and {kw['occurrence_count'] - 3} more occurrences")
        
        if len(ck_results['keywords']) > 20:
            print(f"\n... and {len(ck_results['keywords']) - 20} more keywords")
        
        # Signaling Markers
        print("\n" + "-"*80)
        print("2. SIGNALING MARKERS (sorted by frequency)")
        print("-"*80)
        
        sm_results = self.signaling_markers_results
        stats = sm_results['statistics']
        print(f"\nTotal Markers Found: {stats['words_with_matches']}/{stats['total_signaling_words']}")
        print(f"Total Sentence Matches: {stats['total_sentence_matches']}")
        
        for i, marker in enumerate(sm_results['markers'][:20], 1):  # Show top 20
            print(f"\n{i}. '{marker['marker'].upper()}' (Occurrences={marker['occurrence_count']})")
            
            # Show first 3 sentences
            for j, occ in enumerate(marker['occurrences'][:3], 1):
                sentence = occ['sentence'][:150]
                if len(occ['sentence']) > 150:
                    sentence += "..."
                print(f"   {j}. {sentence}")
            
            if marker['occurrence_count'] > 3:
                print(f"   ... and {marker['occurrence_count'] - 3} more occurrences")
        
        if len(sm_results['markers']) > 20:
            print(f"\n... and {len(sm_results['markers']) - 20} more markers")
        
        # Metaphor Structures
        print("\n" + "-"*80)
        print("3. 'X IS Y' METAPHOR STRUCTURES")
        print("-"*80)
        
        ms_results = self.metaphor_structures_results
        print(f"\nTotal Patterns Found: {ms_results['total_patterns']}")
        print(f"  Potential Metaphors: {ms_results['potential_metaphors_count']}")
        print(f"  Definitions: {ms_results['definitions_count']}")
        print(f"  Filtered (person names): {ms_results['filtered_count']}")
        
        print("\nPOTENTIAL METAPHORS (sorted by semantic distance):")
        for i, m in enumerate(ms_results['potential_metaphors'], 1):
            print(f"\n{i}. [{m['semantic_distance']:.3f}] {m['subject']} {m['be_verb']} {m['predicate']}")
            sentence = m['sentence'][:200]
            if len(m['sentence']) > 200:
                sentence += "..."
            print(f"   {sentence}")
    
    def save_results(self, output_file: str = None):
        """Save all results to a JSON file."""
        if output_file is None:
            output_file = f"comprehensive_analysis_{Path(self.corpus_file).stem}.json"
        
        if not all([self.cultural_keywords_results, 
                   self.signaling_markers_results, 
                   self.metaphor_structures_results]):
            print("Please run complete analysis first!")
            return
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'corpus_file': self.corpus_file,
                'reference_corpus_files': self.reference_corpus_files
            },
            'cultural_keywords': self.cultural_keywords_results,
            'signaling_markers': self.signaling_markers_results,
            'metaphor_structures': self.metaphor_structures_results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n{'='*80}")
            print(f"✓ Complete results saved to: {output_file}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function to run comprehensive analysis."""
    
    # Configuration
    corpus_file = "Enigma of China - Qiu Xiaolong.txt"
    reference_corpus_files = ["Enigma of China Reference Corpus.txt"]
    
    # Initialize analyzer
    analyzer = ComprehensiveMetaphorAnalyzer(
        corpus_file=corpus_file,
        reference_corpus_files=reference_corpus_files
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        similarity_threshold=0.5
    )
    
    # Display results
    analyzer.display_results()
    
    # Save results
    analyzer.save_results("comprehensive_analysis_results.json")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

