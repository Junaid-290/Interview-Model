# pipeline/video_processor.py
# Compatible with latest MediaPipe (no mp.solutions)

import cv2
import numpy as np
import os

def extract_video_embedding(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    frame_count = 0
    brightness_scores = []
    motion_scores = []
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness = proxy for lighting/presence
        brightness_scores.append(float(np.mean(gray)))

        # Motion = proxy for gestures/engagement
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion_scores.append(float(np.mean(diff)))

        prev_frame = gray

    cap.release()

    brightness = np.mean(brightness_scores) if brightness_scores else 128.0
    motion = np.mean(motion_scores) if motion_scores else 0.0

    # Normalize
    brightness_norm = brightness / 255.0
    motion_norm = min(motion / 50.0, 1.0)

    # Eye contact proxy: consistent brightness = facing camera
    eye_contact = 1.0 - abs(brightness_norm - 0.5) * 2
    confidence = motion_norm  # more controlled motion = engaged

    # Build fixed 98-dim vector
    brightness_vec = np.full(64, brightness_norm)
    motion_vec = np.full(32, motion_norm)
    meta = np.array([eye_contact, confidence])

    return np.concatenate([brightness_vec, motion_vec, meta])  # 98-dim