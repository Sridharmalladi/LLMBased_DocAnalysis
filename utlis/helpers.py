import logging
from typing import List, Dict, Any
import json
from pathlib import Path
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging configuration with file output
    """
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary data to JSON file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into dictionary
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        raise

def batch_generator(items: List[Any], batch_size: int):
    """
    Generate batches from a list of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def ensure_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_file_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all file paths in a directory with specified extensions
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters if needed
    # Add more text cleaning operations as needed
    return text

def calculate_stats(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from analysis results
    """
    if not analysis_results:
        return {}
    
    stats = {
        'total_chunks': len(analysis_results),
        'average_score': sum(r['sentiment_score'] for r in analysis_results) / len(analysis_results),
        'max_score': max(r['sentiment_score'] for r in analysis_results),
        'min_score': min(r['sentiment_score'] for r in analysis_results),
    }
    
    return stats