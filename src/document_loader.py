from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from pathlib import Path
import magic

class DocumentLoader:
    def __init__(self):
        self.supported_formats = {
            'application/pdf': PyPDFLoader,
            'text/plain': TextLoader,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': Docx2txtLoader,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': UnstructuredExcelLoader
        }

    def detect_file_type(self, file_path):
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type

    def load_document(self, file_path):
        file_type = self.detect_file_type(file_path)
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        loader_class = self.supported_formats[file_type]
        loader = loader_class(file_path)
        return loader.load()