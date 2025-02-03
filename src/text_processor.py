from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=50,
            length_function=lambda text: len(self.tokenizer.encode(text))
        )

    def process_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        chunks = self.text_splitter.create_documents(texts)
        return chunks

# src/model_handler.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List
import numpy as np

class ModelHandler:
    def __init__(self, model_name: str, api_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def analyze_text(self, texts: List[str], batch_size: int = 8):
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)
                results.extend(predictions.cpu().numpy())
        
        return np.array(results)