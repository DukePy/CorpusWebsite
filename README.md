# Corpus Website - Digital Gateway to Asian Literature

A Flask-based web application for analyzing Asian literature in English using corpus linguistics methods.

## Features

- **Corpus Analysis Suite**: Analyze pre-loaded corpora or your own custom text
- **Word List**: View word frequencies in your corpus
- **Keywords**: Identify statistically significant words
- **N-Grams**: Find frequent multi-word clusters
- **Concordance**: Search for specific terms and see them in context
- **Collocates**: Find words that frequently appear near your search term
- **Cultural Insights**: Automatic detection of cultural metaphors and linguistic innovations

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download NLTK data** (will happen automatically on first run):
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Running the Application

1. **Start the Flask server**:
```bash
python corpusFunctions.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. The application will be running and ready to use!

## Adding Corpus Files

The application dynamically loads corpus files from the `corpora` directory. To add new corpora:

1. **Create a text file** in the `corpora` directory:
   ```
   corpora/Your Corpus Title - Author Name.txt
   ```

2. **Name convention**: The filename (without .txt extension) will be used as the title in the dropdown
   - Example: `Pride and Prejudice - Jane Austen.txt` → appears as "Pride and Prejudice - Jane Austen"

3. **File content**: Plain text (UTF-8 encoding)
   - Can be any length
   - Will be analyzed automatically when selected

4. **Refresh the application**: The new corpus will appear in the dropdown immediately after page refresh

### Example Corpus Structure

```
CorpusWebsite/
├── corpora/
│   ├── Falling Leaves Return to Their Roots - Adeline Yen Mah.txt
│   ├── Enigma of China - Qiu Xiaolong.txt
│   ├── Your New Corpus - Author Name.txt
│   └── Another Corpus.txt
```

## Usage

### Quick Search from Homepage

1. On the homepage, enter a word or phrase in the search box
2. Click "Search" or press Enter
3. The system will automatically:
   - Navigate to "Explore the Gateway"
   - Select and analyze the first corpus ("Falling Leaves Return to Their Roots")
   - Show concordance lines for your search term
   - Display collocates (words that appear near your search term)

### Analyzing Text

You have three ways to analyze text:

#### Option 1: Use Pre-loaded Corpus
1. Go to "Explore the Gateway" section
2. Select a corpus from the dropdown menu
3. Click "Analyze Text" button

#### Option 2: Upload Your Own File
1. Go to "Explore the Gateway" section
2. Select "Upload Text File (.txt)" from the dropdown
3. Click the upload area or drag and drop your `.txt` file
4. The file will be automatically analyzed after upload
5. Maximum file size: 10MB
6. Supported encoding: UTF-8, Latin-1, CP1252

#### Option 3: Paste Custom Text
1. Go to "Explore the Gateway" section
2. Select "Paste Custom Text" from the dropdown
3. Paste your text in the text area
4. Click "Analyze Text" button

#### Exploring Results
After analysis, explore the results in different tabs:
- **Word List**: See all words sorted by frequency
- **Keywords**: View statistically significant words
- **N-Grams**: Explore frequent multi-word phrases
  - Use the dropdown to select n-gram size (2-10 words)
  - Click "Generate" to view n-grams of the selected size
- **Concordance**: Enter a search term to see it in context
- **Collocates**: Enter a search term to find associated words

### Cultural Insights

The system automatically detects:
- Cultural metaphors (e.g., "falling leaves return to their roots")
- Linguistic innovations (e.g., "shuanggui", "red envelopes")
- Translanguaging strategies specific to Asian literature in English

## Project Structure

```
CorpusWebsite/
├── corpusFunctions.py          # Main Flask application
├── templates/
│   └── index.html              # Main HTML template
├── static/                     # Static files (CSS, JS, images)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## API Endpoints

The application provides six REST API endpoints:

- `GET /api/corpora` - Get list of available corpus files
- `POST /api/upload` - Upload a text file and return its content
- `POST /api/analyze` - Analyze text and return all statistics
- `POST /api/concordance` - Get concordance lines for a search term
- `POST /api/collocates` - Find collocates of a search term
- `POST /api/ngrams` - Generate n-grams of specified size (2-10 words)

## Technologies Used

- **Backend**: Flask (Python web framework)
- **NLP**: NLTK (Natural Language Toolkit)
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **UI Design**: Modern, responsive design with Tailwind CSS

## Troubleshooting

### Error: "An error occurred during analysis"

If you see this error, check the following:

1. **Make sure Flask server is running**: You should see output like:
   ```
   * Running on http://127.0.0.1:5000
   ```

2. **Check browser console**: Press F12 or right-click → Inspect → Console tab to see detailed error messages

3. **Verify dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Clear browser cache**: Sometimes old JavaScript files cause issues

5. **Check server terminal**: Look for any Python errors in the terminal where Flask is running

### Server won't start

If you get `ModuleNotFoundError`, install the missing package:
```bash
pip3 install flask nltk
```

### NLTK data not found

If you get NLTK data errors, download the required data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Future Enhancements

- File upload functionality
- Advanced statistical measures (TF-IDF, mutual information)
- Visualization (word clouds, frequency plots)
- Export results as CSV/JSON
- User authentication and saved analyses
- AI-powered cultural insight detection
- Integration with larger reference corpora

## Credits

Research project by The Education University of Hong Kong
- Department of Linguistics and Modern Language Studies
- Principal Project Supervisor: Dr Ma Qing, Angel

## License

Educational and research use only.

