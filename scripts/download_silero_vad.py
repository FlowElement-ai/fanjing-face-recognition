#!/usr/bin/env python3
"""
Download Silero VAD model for ambient voice activity detection.

The model is used by AudioVAD to detect if anyone is speaking in the environment.
This helps prevent false positive speaking detections when the environment is silent.

Usage:
    python scripts/download_silero_vad.py
"""

import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_half.onnx"
MODEL_PATH = Path("models/speaking/silero_vad_half.onnx")
EXPECTED_SIZE = 1280395  # ~1.2MB


def download_silero_vad():
    """Download Silero VAD ONNX model."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        size = MODEL_PATH.stat().st_size
        if abs(size - EXPECTED_SIZE) < 1000:
            print(f"Model already exists: {MODEL_PATH} ({size:,} bytes)")
            return
        print(f"Existing model has unexpected size ({size:,} bytes), re-downloading...")

    print(f"Downloading Silero VAD model from: {MODEL_URL}")
    print(f"Target path: {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size = MODEL_PATH.stat().st_size
        print(f"Downloaded successfully: {size:,} bytes")
    except Exception as e:
        print(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    download_silero_vad()
