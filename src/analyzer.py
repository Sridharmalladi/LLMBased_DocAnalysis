from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .model_handler import ModelHandler
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    A class for analyzing documents using transformer-based models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DocumentAnalyzer with configuration settings.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model settings
                                   and API credentials.
        """
        self.config = config
        self.document_loader = DocumentLoader()
        
        # Using DistilBERT model fine-tuned for sentiment analysis
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        self.text_processor = TextProcessor(
            model_name=self.model_name,
            max_length=config['MAX_LENGTH']
        )
        
        self.model_handler = ModelHandler(
            model_name=self.model_name,
            api_key=config['HUGGINGFACE_API_KEY']
        )

    def analyze_document(self, file_path: str) -> pd.DataFrame:
        """
        Analyze a document and return sentiment analysis results.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            pd.DataFrame: DataFrame containing analysis results with columns:
                         - chunk_text: The text segment
                         - sentiment_score: Sentiment score (0-1)
                         - sentiment_label: Sentiment label (Positive/Negative)
                         - confidence: Model confidence score
                         - chunk_index: Index of the chunk
        """
        try:
            logger.info(f"Starting analysis of document: {file_path}")
            
            # Load document
            documents = self.document_loader.load_document(file_path)
            logger.info(f"Successfully loaded document with {len(documents)} pages")
            
            # Process text into chunks
            text_chunks = self.text_processor.process_documents(documents)
            logger.info(f"Document processed into {len(text_chunks)} chunks")
            
            # Analyze chunks with progress bar
            chunk_texts = [chunk.page_content for chunk in text_chunks]
            analysis_results = self.model_handler.analyze_text(
                chunk_texts,
                batch_size=self.config['BATCH_SIZE']
            )
            
            # Convert raw scores to probabilities and labels
            positive_scores = analysis_results[:, 1]  # Positive sentiment scores
            sentiment_labels = ['Positive' if score >= 0.5 else 'Negative' 
                              for score in positive_scores]
            confidence_scores = np.max(analysis_results, axis=1)
            
            # Prepare results DataFrame
            results_df = pd.DataFrame({
                'chunk_text': chunk_texts,
                'sentiment_score': positive_scores,
                'sentiment_label': sentiment_labels,
                'confidence': confidence_scores,
                'chunk_index': range(len(chunk_texts))
            })
            
            # Add summary statistics
            self._add_summary_stats(results_df)
            
            logger.info("Document analysis completed successfully")
            return results_df
            
        except Exception as e:
            logger.error(f"Error during document analysis: {str(e)}")
            raise

    def _add_summary_stats(self, df: pd.DataFrame) -> None:
        """
        Add summary statistics to the analysis results.
        
        Args:
            df (pd.DataFrame): DataFrame containing analysis results
        """
        try:
            summary_stats = {
                'average_sentiment': df['sentiment_score'].mean(),
                'sentiment_std': df['sentiment_score'].std(),
                'positive_chunks': (df['sentiment_label'] == 'Positive').sum(),
                'negative_chunks': (df['sentiment_label'] == 'Negative').sum(),
                'average_confidence': df['confidence'].mean()
            }
            
            # Add summary stats as DataFrame attributes
            for key, value in summary_stats.items():
                setattr(df, key, value)
                
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")

    def get_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the analysis results.
        
        Args:
            results_df (pd.DataFrame): Analysis results DataFrame
            
        Returns:
            Dict[str, Any]: Summary statistics and insights
        """
        return {
            'document_summary': {
                'total_chunks': len(results_df),
                'average_sentiment': getattr(results_df, 'average_sentiment', 0),
                'sentiment_std': getattr(results_df, 'sentiment_std', 0),
                'positive_chunks': getattr(results_df, 'positive_chunks', 0),
                'negative_chunks': getattr(results_df, 'negative_chunks', 0),
                'average_confidence': getattr(results_df, 'average_confidence', 0)
            },
            'key_insights': {
                'overall_sentiment': 'Positive' if getattr(results_df, 'average_sentiment', 0) >= 0.5 else 'Negative',
                'sentiment_consistency': 'High' if getattr(results_df, 'sentiment_std', 1) < 0.2 else 'Mixed',
                'confidence_level': 'High' if getattr(results_df, 'average_confidence', 0) > 0.8 else 'Moderate'
            }
        }