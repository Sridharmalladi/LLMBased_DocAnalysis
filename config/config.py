from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "hf_VoErqWBezNltWtANldNfiSBR**********")
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
