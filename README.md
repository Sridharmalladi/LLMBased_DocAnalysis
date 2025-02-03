# Document Analysis System using LangChain and Hugging Face

A powerful document analysis system that combines LangChain's document processing capabilities with Hugging Face's state-of-the-art transformer models for comprehensive text analysis.

## 🌟 Features

- Multi-format document support (PDF, DOCX, TXT, Excel)
- Automatic file type detection
- Intelligent text chunking
- Sentiment analysis with confidence scores
- Batch processing for large documents
- Detailed analytics and insights
- Comprehensive logging system

## 🛠️ Technologies Used

- **LangChain**: For document loading and text processing
- **Hugging Face Transformers**: For advanced text analysis
- **DistilBERT**: Pre-trained model fine-tuned for sentiment analysis
- **Pandas**: For data manipulation and analysis
- **Python-Magic**: For file type detection
- **PyTorch**: For deep learning operations

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Hugging Face API key

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-analysis-system.git
cd document-analysis-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face API key:
```bash
export HUGGINGFACE_API_KEY="your_api_key_here"
```

## 💡 Usage

### Basic Usage

```python
from src import DocumentAnalyzer
from config.config import Config

# Initialize analyzer
analyzer = DocumentAnalyzer(vars(Config))

# Analyze a document
results = analyzer.analyze_document("path/to/your/document.pdf")

# Get analysis summary
summary = analyzer.get_summary(results)
```

### Command Line Interface

```bash
python main.py --input path/to/document.pdf --output analysis_results.csv
```

## 📊 Analysis Capabilities

### 1. Sentiment Analysis
- Document-level sentiment scoring
- Chunk-wise sentiment analysis
- Confidence scores for predictions
- Sentiment consistency measurement

### 2. Document Processing
- Automatic format detection
- Smart text chunking
- Batch processing for efficiency
- Support for multiple file formats

### 3. Analysis Insights
- Overall document sentiment
- Sentiment distribution
- Key sections identification
- Confidence metrics

## 🎯 Use Cases

### 1. Business Document Analysis
- Contract sentiment analysis
- Customer feedback processing
- Business report analysis
- Email sentiment tracking

### 2. Academic Research
- Research paper analysis
- Literature review assistance
- Thesis sentiment analysis
- Academic document processing

### 3. Content Analysis
- Blog post sentiment analysis
- Article tone evaluation
- Social media content analysis
- Marketing material assessment

### 4. Legal Document Processing
- Legal document sentiment analysis
- Contract clause evaluation
- Legal report processing
- Compliance document analysis

## 📁 Project Structure

```
doc_analysis/
├── config/
│   └── config.py             # Configuration settings
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading functionality
│   ├── text_processor.py     # Text processing operations
│   ├── model_handler.py      # Model management
│   └── analyzer.py           # Main analysis logic
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py     # Unit tests
├── requirements.txt
└── main.py                  # Entry point
```

## 🔄 Model Details

The system uses the `distilbert-base-uncased-finetuned-sst-2-english` model, which offers:
- Fast inference times
- Good accuracy for sentiment analysis
- Memory efficiency
- Pre-trained on sentiment analysis tasks
- Fine-tuned on the Stanford Sentiment Treebank dataset

## 📈 Performance

- Processing Speed: ~100 pages/minute (dependent on hardware)
- Memory Usage: ~2GB RAM for standard operations
- Accuracy: ~92% on sentiment classification tasks
- Batch Processing: Configurable batch size for optimization

## 🛡️ Error Handling

The system includes comprehensive error handling for:
- File loading errors
- Processing errors
- Model inference errors
- Invalid file formats
- Memory constraints

## 📝 Logging

Detailed logging system that tracks:
- Document processing steps
- Analysis progress
- Error messages
- Performance metrics
- System status
