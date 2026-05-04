# setup_on_deploy.py
# Railway runs this before starting the server

import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Downloading models on first deploy...")

# Text model
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
m.save(os.path.join(MODELS_DIR, "text_model"))
print("✓ text_model")

# Whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
proc.save_pretrained(os.path.join(MODELS_DIR, "whisper_model"))
wm.save_pretrained(os.path.join(MODELS_DIR, "whisper_model"))
print("✓ whisper_model")

# TTS
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch, zipfile, numpy as np
from huggingface_hub import hf_hub_download

tts_proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
tts_proc.save_pretrained(os.path.join(MODELS_DIR, "tts_model"))
tts_model.save_pretrained(os.path.join(MODELS_DIR, "tts_model"))
vocoder.save_pretrained(os.path.join(MODELS_DIR, "tts_model/vocoder"))

zip_path = hf_hub_download(
    repo_id="Matthijs/cmu-arctic-xvectors",
    filename="spkrec-xvect.zip",
    repo_type="dataset"
)
with zipfile.ZipFile(zip_path, "r") as z:
    npy_files = [n for n in z.namelist() if n.endswith(".npy")]
    with z.open(npy_files[0]) as f:
        xvector = np.load(f)
torch.save(
    torch.tensor(xvector).unsqueeze(0),
    os.path.join(MODELS_DIR, "tts_model/speaker_embedding.pt")
)
print("✓ tts_model")
print("All models ready.")