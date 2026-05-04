from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, librosa, os

MODEL_PATH = "models/whisper_model"

def load_model():
    proc = WhisperProcessor.from_pretrained(MODEL_PATH)
    mdl = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    return proc, mdl

processor, model = load_model()

def speech_to_text(audio_path: str) -> str:
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    transcript = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return transcript.strip()