# setup_models.py - Downloads ALL pretrained models into models/ folder
# Run this ONCE before anything else

import os
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    T5ForConditionalGeneration, T5Tokenizer,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("DOWNLOADING ALL PRETRAINED MODELS")
print("=" * 60)

# ── MODEL 1: Text Scoring ──────────────────────────────────────
print("\n[1/4] Text scoring model (Sentence Transformer)...")
m = SentenceTransformer("all-MiniLM-L6-v2")
m.save(os.path.join(MODELS_DIR, "text_model"))
print("✓ models/text_model/")

# ── MODEL 2: Speech to Text ────────────────────────────────────
print("\n[2/4] Speech-to-text model (Whisper tiny)...")
proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
wmodel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

whisper_dir = os.path.join(MODELS_DIR, "whisper_model")

proc.save_pretrained(whisper_dir)
wmodel.save_pretrained(whisper_dir)

print("✓ models/whisper_model/")

# ── MODEL 3: Question Generation ──────────────────────────────
print("\n[3/4] Question generation model (Flan-T5 small)...")

t5 = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small"
)

t5tok = T5Tokenizer.from_pretrained(
    "google/flan-t5-small"
)

question_dir = os.path.join(MODELS_DIR, "question_model")

t5.save_pretrained(question_dir)
t5tok.save_pretrained(question_dir)

print("✓ models/question_model/")

# ── MODEL 4: Text to Speech ───────────────────────────────────
print("\n[4/4] Text-to-speech model (SpeechT5)...")

tts_proc = SpeechT5Processor.from_pretrained(
    "microsoft/speecht5_tts"
)

tts_model = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts"
)

vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan"
)

tts_dir = os.path.join(MODELS_DIR, "tts_model")

tts_proc.save_pretrained(tts_dir)
tts_model.save_pretrained(tts_dir)

vocoder_dir = os.path.join(tts_dir, "vocoder")
os.makedirs(vocoder_dir, exist_ok=True)

vocoder.save_pretrained(vocoder_dir)

# ── Speaker Embedding ─────────────────────────────────────────
print("  Downloading speaker embedding...")

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation"
)

xvector = embeddings_dataset[7306]["xvector"]

torch.save(
    torch.tensor(xvector).unsqueeze(0),
    os.path.join(tts_dir, "speaker_embedding.pt")
)

print("✓ models/tts_model/ + speaker_embedding.pt")

print("\n" + "=" * 60)
print("ALL MODELS DOWNLOADED")
print("models/ folder now contains:")

for item in os.listdir(MODELS_DIR):
    print(f"  ├── {item}")

print("=" * 60)