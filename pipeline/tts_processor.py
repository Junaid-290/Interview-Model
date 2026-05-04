from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch, soundfile as sf, io, os

MODEL_PATH = "models/tts_model"
VOCODER_PATH = "models/tts_model/vocoder"
SPEAKER_EMB = "models/tts_model/speaker_embedding.pt"

def load_tts():
    proc = SpeechT5Processor.from_pretrained(MODEL_PATH)
    mdl = SpeechT5ForTextToSpeech.from_pretrained(MODEL_PATH)
    voc = SpeechT5HifiGan.from_pretrained(VOCODER_PATH)
    spk = torch.load(SPEAKER_EMB, weights_only=True)
    return proc, mdl, voc, spk

processor, tts_model, vocoder, speaker_emb = load_tts()

def text_to_speech(text: str) -> bytes:
    inputs = processor(text=text[:300], return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"],
            speaker_emb,
            vocoder=vocoder
        )
    buf = io.BytesIO()
    sf.write(buf, speech.numpy(), samplerate=16000, format="WAV")
    buf.seek(0)
    return buf.read()