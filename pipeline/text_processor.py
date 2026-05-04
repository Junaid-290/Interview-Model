# pipeline/text_processor.py
# Loads pretrained model FROM models/ folder (not from internet every time)

import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_PATH = "models/text_model"
FALLBACK = "all-MiniLM-L6-v2"

def load_model():
    if os.path.exists(MODEL_PATH):
        print("  Loading text model from models/text_model/")
        return SentenceTransformer(MODEL_PATH)
    print("  text_model not found locally, downloading...")
    return SentenceTransformer(FALLBACK)

model = load_model()

def load_text(path: str) -> str:
    if path.endswith(".pdf"):
        doc = fitz.open(path)
        return " ".join(page.get_text() for page in doc)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_embedding(path: str) -> np.ndarray:
    text = load_text(path)[:512]
    return model.encode(text)