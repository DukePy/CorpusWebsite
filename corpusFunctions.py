from flask import Flask, render_template, request, jsonify
import re
from collections import Counter
import math
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
import os
from pathlib import Path

# Download required NLTK data (will only download once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Path to corpora directory
CORPORA_DIR = os.path.join(os.path.dirname(__file__), 'corpora')

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

def calculate_keyness(text, reference_corpus=None):
    """Calculate keywords using simplified keyness score"""
    tokens = tokenize_text(text)
    freq = Counter(tokens)
    
    # Common English words to filter out (simplified stopword list)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he',
                 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that', 'these',
                 'those', 'my', 'your', 'his', 'her', 'its', 'our'}
    
    keywords = []
    for word, count in freq.most_common(50):
        if len(word) > 2 and word not in stopwords:
            # Simplified keyness: frequency * word length (rewards longer, unique words)
            keyness_score = count * (len(word) / 3.0) * 5
            keywords.append({
                'word': word,
                'freq': count,
                'keyness': round(keyness_score, 1)
            })
    
    # Sort by keyness and return top 15
    keywords.sort(key=lambda x: x['keyness'], reverse=True)
    return keywords[:15]

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
    return render_template('index.html')

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
        
        # Get text
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
        
        # Perform analysis
        tokens = tokenize_text(text)
        freq = get_word_frequencies(tokens)
        
        return jsonify({
            'success': True,
            'word_count': len(tokens),
            'unique_words': len(freq),
            'wordlist': [{'word': w, 'freq': f} for w, f in freq.most_common(50)],
            'keywords': calculate_keyness(text),
            'ngrams': [{'ngram': ng, 'freq': f} for ng, f in extract_ngrams(tokens)],
            'cultural_insights': detect_cultural_insights(text)
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
    """Get full corpus with highlighted word position"""
    try:
        data = request.json
        corpus_id = data.get('corpus_id', 'custom')
        custom_text = data.get('text', '')
        position = data.get('position', 0)
        
        # Get text
        if corpus_id == 'custom' or corpus_id == 'upload':
            text = custom_text
        else:
            text = get_corpus_text(corpus_id)
            if text is None:
                return jsonify({'error': f'Corpus "{corpus_id}" not found'}), 404
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

