# train.py - Generic pipeline for ANY session type
# Supports: interview, presentation, lecture, meeting, or any custom type

import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pipeline.text_processor import extract_text_embedding
from pipeline.video_processor import extract_video_embedding
from pipeline.audio_processor import extract_audio_embedding
from pipeline.fusion_model import FusionModel

TEXT_DIR = "data/text"
VIDEO_DIR = "data/video"
AUDIO_DIR = "data/audio"
MODEL_SAVE_PATH = "models/fusion_model.pt"
LABELS_FILE = "labels.json"

def load_labels():
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found.")
        exit()
    with open(LABELS_FILE, "r") as f:
        data = json.load(f)
    # Support both old format and new generic format
    if "sessions" in data:
        return data["sessions"]
    # backward compat: old dict format
    return [{"name": k, "type": "general", "score": v} 
            for k, v in data.items()]

def find_file(directory, name, extensions):
    for ext in extensions:
        path = os.path.join(directory, f"{name}{ext}")
        if os.path.exists(path):
            return path
    return None

def build_dataset(sessions):
    X, y, meta = [], [], []

    for session in sessions:
        name = session["name"]
        stype = session.get("type", "general")
        score = session["score"]

        print(f"\nProcessing: {name} (type: {stype})")

        text_path  = find_file(TEXT_DIR,  name, [".pdf", ".txt"])
        video_path = find_file(VIDEO_DIR, name, [".mp4", ".avi", ".mov"])
        audio_path = find_file(AUDIO_DIR, name, [".wav", ".mp3"])

        missing = []
        if not text_path:  missing.append("text")
        if not video_path: missing.append("video")
        if not audio_path: missing.append("audio")

        if missing:
            print(f"  SKIP — missing: {', '.join(missing)}")
            continue

        try:
            t = extract_text_embedding(text_path)
            v = extract_video_embedding(video_path)
            a = extract_audio_embedding(audio_path)
            X.append(np.concatenate([t, v, a]))
            y.append(float(score))
            meta.append({"name": name, "type": stype})
            print(f"  ✓ loaded (score: {score})")
        except Exception as e:
            print(f"  ERROR: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta

def train():
    print("=" * 50)
    print("AI ANALYSIS PIPELINE — TRAINING")
    print("=" * 50)

    sessions = load_labels()
    types = set(s.get("type", "general") for s in sessions)
    print(f"\nSession types found: {', '.join(types)}")
    print(f"Total sessions: {len(sessions)}")

    X, y, meta = build_dataset(sessions)

    if len(X) == 0:
        print("\nNo valid data found. Check your data/ folders.")
        return

    print(f"\nTraining on {len(X)} sessions...")
    print("-" * 50)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = FusionModel()

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH,
                              weights_only=True))
        print("✓ Loaded existing model — continuing fine-tuning\n")
    else:
        print("No existing model — training from scratch\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")

    for epoch in range(30):
        epoch_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        bar = "█" * int((1 - min(avg_loss, 1)) * 20)
        print(f"Epoch {epoch+1:02d}/30 | Loss: {avg_loss:.4f} |{bar}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("-" * 50)
    print(f"✓ Best model saved → {MODEL_SAVE_PATH}")
    print(f"✓ Best loss: {best_loss:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    train()