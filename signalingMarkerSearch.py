"""
Signaling Marker Search for Metaphor Detection
This script searches for sentences containing metaphor signaling words in a given corpus.
"""

import re
import json
from typing import List, Dict, Set
from collections import defaultdict


class SignalingMarkerSearch:
    """
    A class for detecting sentences with metaphor signaling words in text.
    """
    
    def __init__(self, signaling_words: List[str] = None):
        """
        Initialize the SignalingMarkerSearch with a list of signaling words.
        
        Args:
            signaling_words: List of metaphor signaling words to search for.
                           If None, uses a default dummy list.
        """
        if signaling_words is None:
            # Dummy list of common metaphor signaling words
            self.signaling_words = [
                # Simile markers
                "like", "as", "as if", "as though", "similar to",
                # Metaphor markers
                "is", "are", "was", "were", "seems", "appears",
                "resembles", "represents", "symbolizes", "embodies",
                # Comparison markers
                "compared to", "in contrast to", "just as",
                # Figurative language markers
                "metaphorically", "figuratively", "symbolically",
                "virtually", "practically", "essentially",
                # Being/becoming markers
                "became", "becomes", "turned into", "transformed into"
            ]
        else:
            self.signaling_words = signaling_words
        
        # Compile regex patterns for efficient searching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for each signaling word."""
        self.patterns = {}
        for word in self.signaling_words:
            # Use word boundaries to match whole words/phrases
            pattern = r'\b' + re.escape(word) + r'\b'
            self.patterns[word] = re.compile(pattern, re.IGNORECASE)
    
    def read_corpus(self, filepath: str) -> str:
        """
        Read the corpus text from a file.
        
        Args:
            filepath: Path to the corpus file
            
        Returns:
            The text content of the corpus
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return ""
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: The text to extract sentences from
            
        Returns:
            List of sentences
        """
        # Split on sentence boundaries (., !, ?)
        # More aggressive splitting to avoid very long sentences
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences and remove very short ones
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def search_signaling_words(self, text: str) -> Dict[str, List[Dict]]:
        """
        Search for sentences containing signaling words in the text.
        
        Args:
            text: The corpus text to search
            
        Returns:
            Dictionary mapping signaling words to lists of sentence matches
        """
        sentences = self.extract_sentences(text)
        results = defaultdict(list)
        
        for idx, sentence in enumerate(sentences, 1):
            for word in self.signaling_words:
                pattern = self.patterns[word]
                if pattern.search(sentence):
                    results[word].append({
                        'sentence_id': idx,
                        'sentence': sentence,
                        'length': len(sentence.split())
                    })
        
        return dict(results)
    
    def get_statistics(self, results: Dict[str, List[Dict]]) -> Dict:
        """
        Generate statistics about the search results.
        
        Args:
            results: Search results from search_signaling_words
            
        Returns:
            Dictionary with statistics
        """
        total_matches = sum(len(matches) for matches in results.values())
        words_with_matches = len([w for w in results if results[w]])
        
        stats = {
            'total_signaling_words': len(self.signaling_words),
            'words_with_matches': words_with_matches,
            'total_sentence_matches': total_matches,
            'matches_per_word': {}
        }
        
        for word in sorted(results.keys(), key=lambda w: len(results[w]), reverse=True):
            stats['matches_per_word'][word] = len(results[word])
        
        return stats
    
    def _truncate_sentence(self, sentence: str, word: str, max_length: int = 150) -> str:
        """
        Truncate sentence around the signaling word for display.
        
        Args:
            sentence: The full sentence
            word: The signaling word to center around
            max_length: Maximum length of the displayed snippet
            
        Returns:
            Truncated sentence with context around the signaling word
        """
        if len(sentence) <= max_length:
            return sentence
        
        # Find the position of the signaling word
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        match = pattern.search(sentence)
        
        if not match:
            return sentence[:max_length] + "..."
        
        pos = match.start()
        
        # Calculate context window
        context_size = max_length // 2
        start = max(0, pos - context_size)
        end = min(len(sentence), pos + context_size)
        
        snippet = sentence[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(sentence):
            snippet = snippet + "..."
        
        return snippet
    
    def display_results(self, results: Dict[str, List[Dict]], max_per_word: int = 5):
        """
        Display search results in a readable format.
        
        Args:
            results: Search results from search_signaling_words
            max_per_word: Maximum number of examples to display per word
        """
        print("\n" + "="*80)
        print("METAPHOR SIGNALING WORD SEARCH RESULTS")
        print("="*80)
        
        stats = self.get_statistics(results)
        
        print(f"\nStatistics:")
        print(f"  Total signaling words searched: {stats['total_signaling_words']}")
        print(f"  Words found in corpus: {stats['words_with_matches']}")
        print(f"  Total sentence matches: {stats['total_sentence_matches']}")
        
        print(f"\n" + "-"*80)
        print("MATCHES BY SIGNALING WORD")
        print("-"*80)
        
        for word in sorted(results.keys(), key=lambda w: len(results[w]), reverse=True):
            matches = results[word]
            if matches:
                print(f"\n[{word.upper()}] - {len(matches)} matches")
                print("-" * 40)
                
                for i, match in enumerate(matches[:max_per_word], 1):
                    sentence = match['sentence']
                    # Truncate long sentences
                    truncated = self._truncate_sentence(sentence, word, max_length=150)
                    # Highlight the signaling word
                    highlighted = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        f"**{word.upper()}**",
                        truncated,
                        flags=re.IGNORECASE
                    )
                    print(f"  {i}. {highlighted}")
                
                if len(matches) > max_per_word:
                    print(f"  ... and {len(matches) - max_per_word} more matches")
    
    def save_results(self, results: Dict[str, List[Dict]], output_file: str):
        """
        Save search results to a JSON file.
        
        Args:
            results: Search results from search_signaling_words
            output_file: Path to output JSON file
        """
        stats = self.get_statistics(results)
        
        output_data = {
            'statistics': stats,
            'results': results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function to run the signaling marker search."""

    signaling_words = ["metaphor", "metaphorical", "metaphorically", "literally", "actually", "virtually", "practically", "essentially", "really", "indeed", 
        "somewhat", "pretty much", "more or less", "almost", "similar to",
        "In alone sense", "in more than one sense", "in both senses of",
        "import", "mean", "meaning of", "in every sense of", "in all senses of",
        "imitation", "model", "pretending",
        "token", "sign", "symbol", "epitome", "appears", "resembles", "represents", "symbolizes", "embodies", "became", "becomes", "become", "turned into", "transformed into",
        "a kind of", "sort of", "type of", "kind of",
        "like", "as",
        "seem", "appear",
        "can", "could", "may", "might", "must", "shall", "should", "will", "would", "have to", "ought to", "need to"]
    
    # Initialize the searcher with dummy signaling words when no signaling words are provided
    searcher = SignalingMarkerSearch(signaling_words)
    
    # Set the corpus file path
    corpus_file = "Enigma of China - Qiu Xiaolong.txt"
    
    print(f"Reading corpus from: {corpus_file}")
    text = searcher.read_corpus(corpus_file)
    
    if not text:
        print("Error: Could not read corpus file.")
        return
    
    print(f"Corpus loaded: {len(text)} characters, {len(text.split())} words")
    
    # Search for signaling words
    print("\nSearching for metaphor signaling words...")
    results = searcher.search_signaling_words(text)
    
    # Display results
    searcher.display_results(results, max_per_word=5)
    
    # Save results to JSON
    output_file = "signaling_word_results.json"
    searcher.save_results(results, output_file)
    
    print("\n" + "="*80)
    print("Search complete!")
    print("="*80)


if __name__ == "__main__":
    main()
