"""
AudioVAD — Silero VAD for ambient voice activity detection.

Runs in a background thread, continuously monitors the system default microphone.
Provides has_voice() to check if anyone is speaking in the environment.

Integration logic:
  - If has_voice() returns False (environment is silent), the main pipeline
    can override all visual speaking detections to "not_speaking".
  - If microphone is unavailable, VAD silently degrades and has_voice()
    always returns True (no interference with visual detection).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioVAD:
    """
    Silero VAD wrapper for real-time ambient voice detection.
    
    Uses ONNX Runtime for inference. Captures audio from system default
    microphone in a background thread. If microphone is unavailable,
    silently degrades without affecting visual detection.
    """

    def __init__(
        self,
        model_path: str = "models/speaking/silero_vad_half.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.3,
        history_size: int = 6,
    ):
        """
        Initialize AudioVAD.

        Args:
            model_path: Path to Silero VAD ONNX model.
            sample_rate: Audio sample rate (16000 Hz for Silero VAD).
            threshold: VAD probability threshold (0.0-1.0).
            history_size: Number of recent frames to keep for smoothing.
                          Each frame is ~32ms, so 6 frames ≈ 200ms.
        """
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._history_size = history_size

        self._session: Optional[object] = None
        self._state: Optional[np.ndarray] = None
        self._context: Optional[np.ndarray] = None

        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[object] = None
        self._available = False

    def start(self) -> None:
        """Start background audio capture and VAD detection."""
        if self._running:
            return

        try:
            self._init_model()
            self._init_audio()
            self._available = True
        except Exception as e:
            logger.warning("AudioVAD unavailable (microphone or model error): %s", e)
            self._available = False
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="audio-vad"
        )
        self._thread.start()
        logger.info("AudioVAD started (sample_rate=%d, threshold=%.2f)",
                    self._sample_rate, self._threshold)

    def stop(self) -> None:
        """Stop background thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._session = None
        self._available = False
        logger.info("AudioVAD stopped")

    def has_voice(self) -> bool:
        """
        Check if voice was detected in recent audio.

        Returns True if:
          - VAD is unavailable (no interference with visual detection)
          - Any of the last N frames detected voice above threshold

        Returns False only when:
          - VAD is available AND environment is confirmed silent
        """
        if not self._available:
            return True

        with self._lock:
            if not self._history:
                return True
            return any(p > self._threshold for p in self._history)

    def _init_model(self) -> None:
        """Initialize ONNX Runtime session for Silero VAD."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset VAD internal state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def _init_audio(self) -> None:
        """Initialize audio input stream from system default microphone."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            latency="low",
        )
        self._stream.start()

    def _capture_loop(self) -> None:
        """Background thread: capture audio and run VAD inference."""
        while self._running:
            try:
                if self._stream is None:
                    break

                audio, overflowed = self._stream.read(512)
                if overflowed:
                    logger.debug("AudioVAD: buffer overflow")

                prob = self._infer(audio.flatten())

                with self._lock:
                    self._history.append(prob)

            except Exception as e:
                logger.debug("AudioVAD capture error: %s", e)
                break

        self._available = False

    def _infer(self, audio_chunk: np.ndarray) -> float:
        """
        Run VAD inference on a single audio chunk.

        Args:
            audio_chunk: 512 samples of audio (32ms at 16kHz).

        Returns:
            Speech probability (0.0-1.0).
        """
        if self._session is None:
            return 0.0

        x = audio_chunk.reshape(1, -1).astype(np.float32)
        x = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
        }
        out, state = self._session.run(None, ort_inputs)

        self._state = state
        self._context = x[:, -64:]

        return float(out[0, 0])
