import pytest
from src.analyzer import DocumentAnalyzer
from config.config import Config
import os

def test_document_analyzer():
    config = vars(Config)
    analyzer = DocumentAnalyzer(config)
    
    # Create a test document
    test_file = "test_document.txt"
    with open(test_file, "w") as f:
        f.write("This is a test document for analysis.")
    
    try:
        results = analyzer.analyze_document(test_file)
        assert len(results) > 0
        assert 'chunk_text' in results.columns
        assert 'sentiment_score' in results.columns
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)