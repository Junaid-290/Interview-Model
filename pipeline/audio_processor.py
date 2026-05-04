# pipeline/audio_processor.py
# Uses Librosa for audio feature extraction - no download needed

import librosa
import numpy as np

def extract_audio_embedding(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=16000, duration=60)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)

    return np.concatenate([
        np.mean(mfcc, axis=1),    # 40-dim
        [np.mean(pitch)],         # 1-dim
        [np.mean(energy)]         # 1-dim
    ])  # shape: (42,)