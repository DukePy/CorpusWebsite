from flask import Flask, render_template, request, jsonify
import re
from collections import Counter
import math
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
import os
import tempfile
from pathlib import Path
import json

# Download required NLTK data (will only download once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Configuration
SHOW_EDUCATORS_HUB = False  # Set to False to hide the Educator's Hub page

# Path to corpora directory
CORPORA_DIR = os.path.join(os.path.dirname(__file__), 'corpora')

# Path to publications file
PUBLICATIONS_FILE = os.path.join(os.path.dirname(__file__), 'publicationList.txt')

def load_corpora():
    """Load all corpus files from the corpora directory"""
    corpora = {}
    if not os.path.exists(CORPORA_DIR):
        os.makedirs(CORPORA_DIR)
        return corpora
    
    for filename in os.listdir(CORPORA_DIR):
        if filename.endswith('.txt'):
            if filename == 'README.txt':
                continue
            filepath = os.path.join(CORPORA_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Use filename without extension as the key
                    corpus_id = os.path.splitext(filename)[0]
                    corpora[corpus_id] = {
                        'title': corpus_id,
                        'filename': filename,
                        'content': content
                    }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return corpora

def get_corpus_text(corpus_id):
    """Get the text content of a specific corpus"""
    corpora = load_corpora()
    if corpus_id in corpora:
        return corpora[corpus_id]['content']
    return None

def load_publications():
    """Load and parse the publications list from publicationList.txt"""
    publications = {
        'Journal article': [],
        'Conference article': [],
        'Book chapter': [],
        'Under review': [],
        'In preparation': []
    }
    
    if not os.path.exists(PUBLICATIONS_FILE):
        return publications
    
    try:
        with open(PUBLICATIONS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        current_category = None
        current_entry = []
        
        # Pattern to detect a new entry starting with author names
        # Matches patterns like "Ma, Q., & Su, X." or "Su, X., Ma, Q.,"
        new_entry_pattern = re.compile(r'^[A-Z][a-z]+,\s+[A-Z]\.,')
        
        for line in content.split('\n'):
            line_stripped = line.strip()
            
            # Check if this is a category header
            if line_stripped.endswith(':') and line_stripped[:-1] in publications:
                # Save previous entry if exists
                if current_category and current_entry:
                    entry_text = ' '.join(current_entry).strip()
                    if entry_text:
                        publications[current_category].append(entry_text)
                    current_entry = []
                
                current_category = line_stripped[:-1]
            elif line_stripped and current_category:
                # Check if this line starts a new entry (author pattern)
                if new_entry_pattern.match(line_stripped) and current_entry:
                    # Save previous entry
                    entry_text = ' '.join(current_entry).strip()
                    if entry_text:
                        publications[current_category].append(entry_text)
                    current_entry = [line_stripped]
                else:
                    # Continue current entry
                    current_entry.append(line_stripped)
            elif not line_stripped and current_category and current_entry:
                # Empty line indicates end of current entry
                entry_text = ' '.join(current_entry).strip()
                if entry_text:
                    publications[current_category].append(entry_text)
                current_entry = []
        
        # Don't forget the last entry
        if current_category and current_entry:
            entry_text = ' '.join(current_entry).strip()
            if entry_text:
                publications[current_category].append(entry_text)
        
        return publications
    except Exception as e:
        print(f"Error loading publications: {e}")
        return publications

# Cultural insights database
CULTURAL_INSIGHTS = [
    {
        "term": "falling leaves return to their roots",
        "type": "Cultural Metaphor",
        "insight": "Embodies the Confucian value of filial piety and ancestral reverence."
    },
    {
        "term": "marriage is a gamble",
        "type": "Cultural Metaphor",
        "insight": "Highlights the uncertainty inherent in arranged marriages in the cultural context."
    },
    {
        "term": "shuanggui",
        "type": "Linguistic Innovation (Borrowing)",
        "insight": "A politically charged term for an extra-judicial detention process, untranslatable in English."
    },
    {
        "term": "connections",
        "type": "Linguistic Innovation (Semantic)",
        "insight": "Refers to the complex Chinese concept of 'guānxi' (关系), a network of relationships and mutual obligations."
    },
    {
        "term": "red envelopes",
        "type": "Linguistic Innovation (Semantic)",
        "insight": "Translates 'hóngbāo' (红包), but used here in its modern, euphemistic sense for bribery."
    },
    {
        "term": "bamboo shoots after a spring rain",
        "type": "Linguistic Innovation (Phrasal)",
        "insight": "A translated idiom used to describe rapid, widespread growth."
    },
    {
        "term": "dragon well tea",
        "type": "Linguistic Innovation (Transliteration)",
        "insight": "A direct translation of 'Lóngjǐng chá' (龙井茶), grounding the scene in a specific Chinese cultural practice."
    },
    {
        "term": "fish swimming in a cauldron",
        "type": "Cultural Metaphor",
        "insight": "Symbolizes a feeling of being trapped and powerless within oppressive familial or social structures."
    },
    {
        "term": "hearts reduced to ashes",
        "type": "Cultural Metaphor",
        "insight": "Alludes to Daoist philosophy, where ashes symbolize detachment and emotional numbness."
    },
    {
        "term": "chicken talking to a duck",
        "type": "Cultural Metaphor",
        "insight": "Reflects challenges in mutual understanding due to linguistic or cultural differences."
    }
]

# ============ CORPUS ANALYSIS FUNCTIONS ============

def tokenize_text(text):
    """Tokenize and clean text"""
    try:
        tokens = word_tokenize(text.lower())
    except:
        # Fallback to simple split if NLTK fails
        tokens = text.lower().split()
    
    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]
    return tokens

def get_word_frequencies(tokens):
    """Calculate word frequencies"""
    return Counter(tokens)

def generate_concordance(text, search_term, context_window=5):
    """Generate concordance lines for a search term"""
    words = text.split()
    results = []
    search_lower = search_term.lower()
    
    for i, word in enumerate(words):
        if search_lower == word.lower():
            left = ' '.join(words[max(0, i-context_window):i])
            center = word
            right = ' '.join(words[i+1:min(len(words), i+context_window+1)])
            results.append({
                'left': left,
                'keyword': center,
                'right': right,
                'position': i  # Add position information
            })
    
    return results

def calculate_collocates(tokens, search_term, window=5):
    """Find collocates of a search term"""
    collocates = []
    search_lower = search_term.lower()
    
    for i, token in enumerate(tokens):
        if token == search_lower:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            context = tokens[start:i] + tokens[i+1:end]
            collocates.extend(context)
    
    if not collocates:
        return []
    
    return Counter(collocates).most_common(20)

def extract_ngrams(tokens, n=3):
    """Extract n-grams from tokens"""
    if len(tokens) < n:
        return []
    
    n_grams = list(ngrams(tokens, n))
    ngram_freq = Counter([' '.join(gram) for gram in n_grams])
    return ngram_freq.most_common(100)

def calculate_keyness(text, reference_corpus=None, sort_by='keyness', sort_order='desc', 
                      start_rank=1, end_rank=100):
    """
    Calculate keywords using G² (log-likelihood) score comparing target and reference corpus.
    Similar to the cultural keywords function but without semantic filtering.
    
    Args:
        text: Target corpus text
        reference_corpus: Reference corpus text (optional)
        sort_by: Sort criterion - 'keyness' (G² score) or 'freq' (frequency)
        sort_order: 'desc' (descending) or 'asc' (ascending)
        start_rank: Starting rank for pagination (1-based)
        end_rank: Ending rank for pagination (inclusive)
    
    Returns:
        dict: Contains 'keywords' list and metadata
    """
    # Tokenize target corpus
    target_tokens = tokenize_text(text)
    target_counts = Counter(target_tokens)
    
    # Common English words to filter out (simplified stopword list)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he',
                 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that', 'these',
                 'those', 'my', 'your', 'his', 'her', 'its', 'our'}
    
    # If no reference corpus provided, use simplified frequency-based scoring
    if not reference_corpus or not reference_corpus.strip():
        keywords = []
        for word, count in target_counts.items():
            if len(word) > 2 and word not in stopwords:
                # Simplified keyness: frequency * word length (rewards longer, unique words)
                keyness_score = count * (len(word) / 3.0) * 5
                keywords.append({
                    'word': word,
                    'freq': count,
                    'freq_ref': 0,
                    'keyness': round(keyness_score, 1)
                })
        
        # Sort by specified criterion
        reverse = (sort_order == 'desc')
        if sort_by == 'freq':
            keywords.sort(key=lambda x: x['freq'], reverse=reverse)
        else:
            keywords.sort(key=lambda x: x['keyness'], reverse=reverse)
        
        # Apply pagination
        total_keywords = len(keywords)
        start_idx = max(0, start_rank - 1)
        end_idx = min(total_keywords, end_rank)
        paginated_keywords = keywords[start_idx:end_idx]
        
        # Add rank to each keyword
        for i, kw in enumerate(paginated_keywords):
            kw['rank'] = start_idx + i + 1
        
        return {
            'keywords': paginated_keywords,
            'total_keywords': total_keywords,
            'start_rank': start_rank,
            'end_rank': end_rank,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'has_reference_corpus': False
        }
    
    # Tokenize reference corpus
    ref_tokens = tokenize_text(reference_corpus)
    ref_counts = Counter(ref_tokens)
    
    # Calculate corpus sizes
    c = len(target_tokens)  # Target corpus size
    d = len(ref_tokens)     # Reference corpus size
    N = c + d               # Total corpus size
    
    # Calculate G² (log-likelihood) for each word
    keywords = []
    
    for word, a in target_counts.items():
        # Skip stopwords and short words
        if word in stopwords or len(word) <= 2:
            continue
            
        # Skip very infrequent words (appears less than 3 times)
        if a < 3:
            continue
        
        b = ref_counts.get(word, 0)  # Frequency in reference corpus
        
        # Calculate expected frequencies
        E1 = c * (a + b) / N
        E2 = d * (a + b) / N
        
        # Calculate G² (log-likelihood) score
        term1 = a * math.log(a / E1) if a > 0 and E1 > 0 else 0
        term2 = b * math.log(b / E2) if b > 0 and E2 > 0 else 0
        
        g2_score = 2 * (term1 + term2)
        
        # Only keep words that are overused in target corpus (a > E1)
        if a > E1:
            keywords.append({
                'word': word,
                'freq': a,
                'freq_ref': b,
                'keyness': round(g2_score, 2)
            })
    
    # Sort by specified criterion
    reverse = (sort_order == 'desc')
    if sort_by == 'freq':
        keywords.sort(key=lambda x: x['freq'], reverse=reverse)
    else:  # sort by keyness
        keywords.sort(key=lambda x: x['keyness'], reverse=reverse)
    
    # Apply pagination
    total_keywords = len(keywords)
    start_idx = max(0, start_rank - 1)
    end_idx = min(total_keywords, end_rank)
    paginated_keywords = keywords[start_idx:end_idx]
    
    # Add rank to each keyword
    for i, kw in enumerate(paginated_keywords):
        kw['rank'] = start_idx + i + 1
    
    return {
        'keywords': paginated_keywords,
        'total_keywords': total_keywords,
        'start_rank': start_rank,
        'end_rank': end_rank,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'has_reference_corpus': True,
        'target_corpus_size': c,
        'reference_corpus_size': d
    }

def detect_cultural_insights(text):
    """Detect cultural insights in the text"""
    text_lower = text.lower()
    found_insights = []
    
    for insight in CULTURAL_INSIGHTS:
        if insight['term'].lower() in text_lower:
            found_insights.append(insight)
    
    return found_insights

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', show_educators_hub=SHOW_EDUCATORS_HUB)

@app.route('/api/publications', methods=['GET'])
def get_publications():
    """Get list of publications from publicationList.txt"""
    try:
        publications = load_publications()
        return jsonify({
            'success': True,
            'publications': publications
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpora', methods=['GET'])
def get_corpora_list():
    """Get list of available corpora"""
    try:
        corpora = load_corpora()
        corpus_list = [
            {
                'id': corpus_id,
                'title': info['title'],
                'filename': info['filename']
            }
            for corpus_id, info in corpora.items()
        ]
        return jsonify({
            'success': True,
            'corpora': corpus_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return the text content"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not file.filename.endswith('.txt'):
            return jsonify({'error': 'Only .txt files are supported'}), 400
        
        # Read file content
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            file.seek(0)
            try:
                content = file.read().decode('latin-1')
            except:
                file.seek(0)
                content = file.read().decode('cp1252', errors='ignore')
        
        if not content.strip():
            return jsonify({'error': 'File is empty'}), 400
        
        return jsonify({
            'success': True,
            'content': content,
            'filename': file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Main analysis endpoint"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        ref_corpus_id = data.get('ref_corpus_id', None)
        custom_ref_text = data.get('ref_text', '')
        
        # Keywords sorting and pagination parameters
        keywords_sort_by = data.get('keywords_sort_by', 'keyness')  # 'keyness' or 'freq'
        keywords_sort_order = data.get('keywords_sort_order', 'desc')  # 'desc' or 'asc'
        keywords_start_rank = data.get('keywords_start_rank', 1)
        keywords_end_rank = data.get('keywords_end_rank', 100)
        
        # Validate sorting parameters
        if keywords_sort_by not in ['keyness', 'freq']:
            keywords_sort_by = 'keyness'
        if keywords_sort_order not in ['desc', 'asc']:
            keywords_sort_order = 'desc'
        
        # Validate and limit rank ranges (max 500 keywords per request)
        keywords_start_rank = max(1, int(keywords_start_rank))
        keywords_end_rank = min(int(keywords_end_rank), keywords_start_rank + 499)
        
        # Get target text
        if corpus_id == 'custom' or corpus_id == 'upload':
            # For custom or uploaded text, use the provided text
            text = custom_text
        else:
            # For pre-loaded corpora, load from file
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Get reference corpus (optional)
        reference_corpus = None
        if ref_corpus_id:
            if ref_corpus_id == 'custom' or ref_corpus_id == 'upload':
                # Use provided reference text
                reference_corpus = custom_ref_text
            else:
                # Load reference corpus from file
                reference_corpus = get_corpus_text(ref_corpus_id)
                if reference_corpus is None:
                    return jsonify({'error': f'Reference corpus "{ref_corpus_id}" not found'}), 404
        elif custom_ref_text:
            # If no ref_corpus_id but ref_text is provided, use it
            reference_corpus = custom_ref_text
        
        # Perform analysis
        tokens = tokenize_text(text)
        freq = get_word_frequencies(tokens)
        
        # Calculate keywords with sorting and pagination
        keywords_result = calculate_keyness(
            text, 
            reference_corpus,
            sort_by=keywords_sort_by,
            sort_order=keywords_sort_order,
            start_rank=keywords_start_rank,
            end_rank=keywords_end_rank
        )
        
        return jsonify({
            'success': True,
            'word_count': len(tokens),
            'unique_words': len(freq),
            'wordlist': [{'word': w, 'freq': f} for w, f in freq.most_common(50)],
            'keywords': keywords_result['keywords'],  # Extract just the array for backward compatibility
            'keywords_metadata': {  # Add metadata separately
                'total_keywords': keywords_result['total_keywords'],
                'start_rank': keywords_result['start_rank'],
                'end_rank': keywords_result['end_rank'],
                'sort_by': keywords_result['sort_by'],
                'sort_order': keywords_result['sort_order'],
                'has_reference_corpus': keywords_result.get('has_reference_corpus', False),
                'target_corpus_size': keywords_result.get('target_corpus_size', 0),
                'reference_corpus_size': keywords_result.get('reference_corpus_size', 0)
            },
            'ngrams': [{'ngram': ng, 'freq': f} for ng, f in extract_ngrams(tokens)],
            'cultural_insights': detect_cultural_insights(text)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/keywords', methods=['POST'])
def get_keywords():
    """
    Dedicated endpoint for fetching keywords with sorting and pagination.
    This allows users to change sorting/pagination without re-running the entire analysis.
    """
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        ref_corpus_id = data.get('ref_corpus_id', None)
        custom_ref_text = data.get('ref_text', '')
        
        # Keywords sorting and pagination parameters
        sort_by = data.get('sort_by', 'keyness')  # 'keyness' or 'freq'
        sort_order = data.get('sort_order', 'desc')  # 'desc' or 'asc'
        start_rank = data.get('start_rank', 1)
        end_rank = data.get('end_rank', 100)
        
        # Validate sorting parameters
        if sort_by not in ['keyness', 'freq']:
            sort_by = 'keyness'
        if sort_order not in ['desc', 'asc']:
            sort_order = 'desc'
        
        # Validate and limit rank ranges (max 500 keywords per request)
        start_rank = max(1, int(start_rank))
        end_rank = min(int(end_rank), start_rank + 499)
        
        # Get target text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Get reference corpus (optional)
        reference_corpus = None
        if ref_corpus_id:
            if ref_corpus_id == 'custom' or ref_corpus_id == 'upload':
                reference_corpus = custom_ref_text
            else:
                reference_corpus = get_corpus_text(ref_corpus_id)
                if reference_corpus is None:
                    return jsonify({'error': f'Reference corpus "{ref_corpus_id}" not found'}), 404
        elif custom_ref_text:
            reference_corpus = custom_ref_text
        
        # Calculate keywords with sorting and pagination
        keywords_result = calculate_keyness(
            text, 
            reference_corpus,
            sort_by=sort_by,
            sort_order=sort_order,
            start_rank=start_rank,
            end_rank=end_rank
        )
        
        return jsonify({
            'success': True,
            'keywords': keywords_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/concordance', methods=['POST'])
def get_concordance():
    """Get concordance lines for a search term"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        search_term = data.get('search_term', '')
        
        # Get text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text or not search_term:
            return jsonify({'error': 'Missing text or search term'}), 400
        
        results = generate_concordance(text, search_term)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/collocates', methods=['POST'])
def get_collocates():
    """Get collocates for a search term"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        search_term = data.get('search_term', '')
        
        # Get text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text or not search_term:
            return jsonify({'error': 'Missing text or search term'}), 400
        
        tokens = tokenize_text(text)
        collocates = calculate_collocates(tokens, search_term.lower())
        
        return jsonify({
            'success': True,
            'collocates': [{'word': w, 'freq': f} for w, f in collocates]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ngrams', methods=['POST'])
def get_ngrams():
    """Get n-grams with specified size"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        n = data.get('n', 3)
        
        # Validate n value
        if not isinstance(n, int) or n < 2 or n > 10:
            return jsonify({'error': 'n must be an integer between 2 and 10'}), 400
        
        # Get text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        tokens = tokenize_text(text)
        ngrams_result = extract_ngrams(tokens, n)
        
        return jsonify({
            'success': True,
            'ngrams': [{'ngram': ng, 'freq': f} for ng, f in ngrams_result],
            'n': n
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/context')
def show_context():
    """Display full context for a concordance occurrence"""
    return render_template('context.html')

@app.route('/api/context', methods=['POST'])
def get_context():
    """Get full corpus with highlighted sentence or keyword"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        sentence = data.get('sentence', '')
        highlight_keyword = data.get('highlight_keyword', '')
        highlight_mode = data.get('highlight_mode', 'keyword')  # 'keyword' or 'sentence'
        position = data.get('position', None)  # For backwards compatibility
        
        # Get text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Handle old position-based requests (backwards compatibility)
        if position is not None and not sentence:
            words = text.split()
            
            # Validate position
            if position < 0 or position >= len(words):
                return jsonify({'error': 'Invalid position'}), 400
            
            # Return full corpus with position info
            keyword = words[position]
            before_text = ' '.join(words[:position])
            after_text = ' '.join(words[position+1:])
            
            # Get corpus title if available
            corpus_title = corpus_id
            if corpus_id not in ['custom', 'upload']:
                corpora = load_corpora()
                if corpus_id in corpora:
                    corpus_title = corpora[corpus_id]['title']
            
            return jsonify({
                'success': True,
                'before': before_text,
                'keyword': keyword,
                'after': after_text,
                'position': position,
                'total_words': len(words),
                'corpus_title': corpus_title
            })
        
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400
        
        # Find the sentence in the corpus
        # Try to find exact match first
        sentence_index = text.find(sentence.strip())
        
        # If not found, try with first 50 characters (in case of variations)
        if sentence_index == -1:
            sentence_substr = sentence.strip()[:50]
            sentence_index = text.find(sentence_substr)
        
        # If still not found, try with some flexibility (removing extra whitespace)
        if sentence_index == -1:
            # Normalize whitespace for comparison
            normalized_sentence = ' '.join(sentence.split())
            normalized_text = ' '.join(text.split())
            normalized_index = normalized_text.find(normalized_sentence[:50])
            
            if normalized_index >= 0:
                # Approximate the position in original text
                sentence_index = text.find(normalized_sentence[:20])
        
        if sentence_index == -1:
            return jsonify({'error': 'Could not locate sentence in corpus'}), 404
        
        # Get the text before and after the sentence
        before_text = text[:sentence_index]
        sentence_in_text = text[sentence_index:sentence_index + len(sentence)]
        after_text = text[sentence_index + len(sentence):]
        
        # For keyword mode, find and highlight the keyword within the sentence
        if highlight_mode == 'keyword' and highlight_keyword:
            # Find the keyword in the sentence (case-insensitive)
            keyword_pattern = re.compile(r'\b' + re.escape(highlight_keyword) + r'\b', re.IGNORECASE)
            match = keyword_pattern.search(sentence_in_text)
            
            if match:
                # Split the sentence around the keyword
                keyword_start = match.start()
                keyword_end = match.end()
                keyword_text = sentence_in_text[keyword_start:keyword_end]
                sentence_before_keyword = sentence_in_text[:keyword_start]
                sentence_after_keyword = sentence_in_text[keyword_end:]
                
                # Combine with before text
                before_text = before_text + sentence_before_keyword
                after_text = sentence_after_keyword + after_text
                highlighted_text = keyword_text
            else:
                # Keyword not found, highlight whole sentence
                highlighted_text = sentence_in_text
                after_text = after_text
        else:
            # Sentence mode - highlight the whole sentence
            highlighted_text = sentence_in_text
        
        # Get corpus title if available
        corpus_title = corpus_id
        if corpus_id not in ['custom', 'upload']:
            corpora = load_corpora()
            if corpus_id in corpora:
                corpus_title = corpora[corpus_id]['title']
        
        return jsonify({
            'success': True,
            'before': before_text,
            'keyword': highlighted_text,
            'after': after_text,
            'corpus_title': corpus_title,
            'highlight_mode': highlight_mode
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/metaphor_analysis', methods=['POST'])
def run_metaphor_analysis():
    """Run comprehensive metaphor analysis on uploaded corpus"""
    try:
        data = request.json
        text = data.get('text', '')
        ref_text = data.get('ref_text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # IMPORTANT: Patch SQLite FIRST to allow cross-thread usage
        # The entity linker's knowledge base uses SQLite, which by default doesn't allow
        # connections to be used across threads. We patch it to enable this.
        import sqlite3
        
        # Store original connect function
        original_sqlite_connect = sqlite3.connect
        
        # Patch sqlite3.connect to add check_same_thread=False
        def patched_connect(database, timeout=5.0, detect_types=0, isolation_level='DEFERRED', 
                          check_same_thread=True, factory=sqlite3.Connection, cached_statements=128, uri=False):
            # Force check_same_thread=False for thread safety in Flask
            return original_sqlite_connect(database, timeout=timeout, detect_types=detect_types,
                                         isolation_level=isolation_level, check_same_thread=False,
                                         factory=factory, cached_statements=cached_statements, uri=uri)
        
        sqlite3.connect = patched_connect
        
        # Now configure the analysis settings
        import culturalKeywordsListIdentification_1 as ckl
        
        # Store original values
        original_use_entity_linker = ckl.USE_ENTITY_LINKER
        original_run_coref = ckl.RUN_COREF
        original_use_real_spacy = ckl.USE_REAL_SPACY
        
        # Enable entity linker with cross-thread SQLite support
        ckl.USE_ENTITY_LINKER = True
        ckl.RUN_COREF = False  # Disable coref to speed up analysis
        ckl.USE_REAL_SPACY = True
        
        # Now import the comprehensive metaphor analyzer
        # The analyzer module now references ckl module directly, so our changes above will apply
        try:
            from comprehensive_metaphor_analysis import ComprehensiveMetaphorAnalyzer
        except ImportError as e:
            # Restore settings before returning
            ckl.USE_ENTITY_LINKER = original_use_entity_linker
            ckl.RUN_COREF = original_run_coref
            ckl.USE_REAL_SPACY = original_use_real_spacy
            sqlite3.connect = original_sqlite_connect
            return jsonify({'error': f'Failed to import metaphor analyzer: {str(e)}'}), 500
        
        # Create temporary files for the corpus text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(text)
            tmp_corpus_path = tmp_file.name
        
        # Create temporary file for reference corpus if provided
        tmp_ref_path = None
        if ref_text and ref_text.strip():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_ref_file:
                tmp_ref_file.write(ref_text)
                tmp_ref_path = tmp_ref_file.name
        
        try:
            # Use custom reference corpus if provided, otherwise use default
            if tmp_ref_path:
                reference_corpus = tmp_ref_path
            else:
                reference_corpus = os.path.join(os.path.dirname(__file__), 'Enigma of China Reference Corpus.txt')
                
                if not os.path.exists(reference_corpus):
                    return jsonify({'error': 'Default reference corpus not found'}), 500
            
            try:
                # Initialize analyzer (flags already set at the beginning of function)
                # NOTE: The entity linker (with SQLite) will be created inside analyze_cultural_keywords(),
                # which runs in THIS request thread. This avoids threading issues because the SQLite
                # connection is created and used in the same thread.
                analyzer = ComprehensiveMetaphorAnalyzer(
                    corpus_file=tmp_corpus_path,
                    reference_corpus_files=[reference_corpus]
                )
                
                # Run complete analysis
                # First request will be slow (~10-15 seconds) due to entity linker initialization
                results = analyzer.run_complete_analysis(similarity_threshold=0.5)
                
            finally:
                # Restore original settings
                ckl.USE_ENTITY_LINKER = original_use_entity_linker
                ckl.RUN_COREF = original_run_coref
                ckl.USE_REAL_SPACY = original_use_real_spacy
                
                # Restore original SQLite connect function
                sqlite3.connect = original_sqlite_connect
            
            # Clean up the results to make them JSON serializable
            # Format the results to be more frontend-friendly
            formatted_results = {
                'cultural_keywords': {
                    'total': results['cultural_keywords']['total_keywords'],
                    'items': []
                },
                'signaling_markers': {
                    'total': len(results['signaling_markers']['markers']),
                    'items': []
                },
                'metaphor_structures': {
                    'total': results['metaphor_structures']['potential_metaphors_count'],
                    'items': []
                }
            }
            
            # Format cultural keywords (top 20)
            for kw in results['cultural_keywords']['keywords'][:20]:
                formatted_results['cultural_keywords']['items'].append({
                    'keyword': kw['keyword'],
                    'frequency': kw['occurrence_count'],
                    'g2_score': round(kw['g2_score'], 2),
                    'occurrences': [
                        {
                            'sentence': occ['sentence'],
                            'sentence_id': occ['sentence_id']
                        }
                        for occ in kw['occurrences'][:10]  # Limit to 10 occurrences per keyword
                    ]
                })
            
            # Format signaling markers (top 20)
            for marker in results['signaling_markers']['markers'][:20]:
                formatted_results['signaling_markers']['items'].append({
                    'marker': marker['marker'],
                    'frequency': marker['occurrence_count'],
                    'occurrences': [
                        {
                            'sentence': occ['sentence'],
                            'sentence_id': occ.get('sentence_id', 0)
                        }
                        for occ in marker['occurrences'][:10]  # Limit to 10 occurrences per marker
                    ]
                })
            
            # Format metaphor structures (all potential metaphors)
            # Create a mapping of sentences to IDs from cultural keywords and signaling markers
            sentence_to_id = {}
            for kw in results['cultural_keywords']['keywords']:
                for occ in kw['occurrences']:
                    sentence_to_id[occ['sentence']] = occ['sentence_id']
            
            for marker in results['signaling_markers']['markers']:
                for occ in marker['occurrences']:
                    if 'sentence_id' in occ:
                        sentence_to_id[occ['sentence']] = occ['sentence_id']
            
            for metaphor in results['metaphor_structures']['potential_metaphors']:
                # Try to find sentence_id by matching the sentence
                sentence_id = sentence_to_id.get(metaphor['sentence'], 0)
                
                formatted_results['metaphor_structures']['items'].append({
                    'subject': metaphor['subject'],
                    'be_verb': metaphor['be_verb'],
                    'predicate': metaphor['predicate'],
                    'sentence': metaphor['sentence'],
                    'semantic_distance': round(metaphor['semantic_distance'], 3),
                    'sentence_id': sentence_id
                })
            
            return jsonify({
                'success': True,
                'results': formatted_results
            })
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(tmp_corpus_path)
            except:
                pass
            
            if tmp_ref_path:
                try:
                    os.unlink(tmp_ref_path)
                except:
                    pass
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Make sure to restore settings even on error
        try:
            if 'original_use_entity_linker' in locals():
                import culturalKeywordsListIdentification_1 as ckl
                ckl.USE_ENTITY_LINKER = original_use_entity_linker
                ckl.RUN_COREF = original_run_coref
                ckl.USE_REAL_SPACY = original_use_real_spacy
            
            if 'original_sqlite_connect' in locals():
                import sqlite3
                sqlite3.connect = original_sqlite_connect
        except:
            pass
            
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

