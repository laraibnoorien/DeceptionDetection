#!/usr/bin/env python3
#python3 speech_v2.py --mic 
"""
speech_v2.py — Acoustic Speech Analysis Pipeline
=================================================
speech modality for real-time deception and stress inference.

Pipeline (single file, no broken relative imports):

  MicrophoneCaptureThread  (daemon, sounddevice InputStream, 16 kHz)
        │  raw PCM float32 chunks → thread-safe RingBuffer
        ▼
  VoiceActivityDetector    (dual-threshold: energy + ZCR)
        │  voiced speech segments only → chunk queue
        ▼
  SpeechPreprocessor       (bandpass, silence trim, normalise, pre-emphasis)
        │  cleaned waveform
        ▼
  SpeechFeatureExtractor   (94-d: MFCC×3 + F0 + energy + spectral + prosody)
        │  + HNR computed here (autocorrelation)
        │  + SNR computed here (percentile noise-floor method)
        ▼
  SMModelManager           (loads deception_rf.pkl + stress_rf.pkl)
        │  RF stress probability, RF deception probability
        ▼
  DeceptionFusionEngine    (stress × 0.20 + RF-decep × 0.40 +
        │                   prosody × 0.15 + hesitation × 0.15 +
        │                   voice-quality × 0.10)
        ▼
  SMOutput  (thread-safe, timestamped, confidence-gated)

Standalone run (from __main__):
  python speech_v2.py [--mic | --file path.wav] [--model-dir path]
  → Live terminal display: wearable-style rolling readout

Pipeline integration (from main.py):
  processor = StreamingSMProcessor(model_dir=..., input_source="mic")
  processor.start()
  output: SMOutput = processor.get_last_output()
  processor.stop()

  # OR feed external audio array:
  output = processor.process_audio_chunk(features_94d, audio_quality=0.9)
  output = processor.process_raw_audio(waveform_np, sr=16000)

Model paths (default, override via model_dir):
  stress_rf.pkl      → SMModelManager.stress_model_path
  deception_rf.pkl   → SMModelManager.deception_model_path
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Optional heavy deps — imported lazily so the module loads even without them ──
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

# ── Local support modules (absolute imports — works standalone AND as package) ──
def _try_import_support():
    """
    Attempt to import feature_extraction and preprocessing from three locations:
      1. Same directory as this file (standalone run)
      2. Package path O3_SpeechModality (when imported as part of the project)
      3. Any parent package that exposes them
    Returns (SpeechFeatureExtractor, extract_speech_features,
             SpeechPreprocessor, FeatureConfig) or raises ImportError.
    """
    # Try same-dir first (standalone)
    _dir = str(Path(__file__).parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    try:
        from feature_extraction import (          # type: ignore
            SpeechFeatureExtractor, extract_speech_features, FeatureConfig)
        from preprocessing import SpeechPreprocessor  # type: ignore
        return SpeechFeatureExtractor, extract_speech_features, SpeechPreprocessor, FeatureConfig
    except ImportError:
        pass

    try:
        from O3_SpeechModality.feature_extraction import (   # type: ignore
            SpeechFeatureExtractor, extract_speech_features, FeatureConfig)
        from O3_SpeechModality.preprocessing import SpeechPreprocessor  # type: ignore
        return SpeechFeatureExtractor, extract_speech_features, SpeechPreprocessor, FeatureConfig
    except ImportError:
        pass

    raise ImportError(
        "Cannot find feature_extraction.py / preprocessing.py. "
        "Place them in the same directory as speech_v2.py, or ensure "
        "O3_SpeechModality is on PYTHONPATH.")


(SpeechFeatureExtractor,
 extract_speech_features,
 SpeechPreprocessor,
 FeatureConfig) = _try_import_support()


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_SR:    int   = 16_000   # Hz — must match FeatureConfig.target_sr
CHUNK_MS:     int   = 500      # ms per microphone callback block
N_FEATURES:   int   = 94       # feature vector dimension (39+39+16)
MODEL_DIR_ENV:str   = "SM_MODEL_DIR"

STRESS_MODEL_NAME    = "stress_rf.pkl"
DECEPTION_MODEL_NAME = "deception_rf.pkl"


# ============================================================================
# 1.  OUTPUT DATACLASS
# ============================================================================

@dataclass
class SMOutput:
    """
    Thread-safe speech modality output.
    All probability fields are in [0, 1].
    confidence reflects audio/signal quality — separate from the probabilities.
    """
    # Core inference
    stress_prob:      float
    deception_prob:   float         # fused: RF + prosody + hesitation + quality
    stress_level:     str           # LOW | MODERATE | HIGH
    deception_level:  str

    # Signal quality
    status:           str           # OK | LOW_QUALITY | INSUFFICIENT_AUDIO | ERROR
    voice_quality:    float         # [0,1] — SNR-derived audio quality score
    confidence:       float         # [0,1] — inference confidence (quality-gated)
    snr_db:           float         # raw SNR in dB

    # Interpretation
    interpretation:   str

    # Diagnostics
    timestamp:        float = field(default_factory=time.time)
    feature_count:    int   = 0
    hnr_db:           float = 0.0   # Harmonics-to-Noise Ratio
    prosody_risk:     float = 0.0
    hesitation_risk:  float = 0.0
    speech_rate:      float = 0.0
    voiced_ratio:     float = 0.0
    pause_ratio:      float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stress_prob":      round(self.stress_prob, 4),
            "deception_prob":   round(self.deception_prob, 4),
            "stress_level":     self.stress_level,
            "deception_level":  self.deception_level,
            "status":           self.status,
            "voice_quality":    round(self.voice_quality, 4),
            "confidence":       round(self.confidence, 4),
            "snr_db":           round(self.snr_db, 2),
            "hnr_db":           round(self.hnr_db, 2),
            "prosody_risk":     round(self.prosody_risk, 4),
            "hesitation_risk":  round(self.hesitation_risk, 4),
            "speech_rate":      round(self.speech_rate, 4),
            "voiced_ratio":     round(self.voiced_ratio, 4),
            "pause_ratio":      round(self.pause_ratio, 4),
            "interpretation":   self.interpretation,
            "timestamp":        self.timestamp,
            "feature_count":    self.feature_count,
        }


# ============================================================================
# 2.  MODEL MANAGER
# ============================================================================

class NeutralModel:
    """
    Picklable placeholder model returning 0.5/0.5 priors.
    Used only when real .pkl files are absent.
    """
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full((n, 2), 0.5, dtype=np.float32)


class SMModelManager:
    """
    Loads and manages deception_rf.pkl and stress_rf.pkl.
    Falls back to NeutralModel if files are missing.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            env = os.environ.get(MODEL_DIR_ENV)
            model_dir = Path(env) if env else Path(__file__).parent / "models"
        self.model_dir             = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.stress_model_path     = self.model_dir / STRESS_MODEL_NAME
        self.deception_model_path  = self.model_dir / DECEPTION_MODEL_NAME
        self.stress_model:     Any = None
        self.deception_model:  Any = None
        self._using_placeholders   = False

    def load_models(self) -> bool:
        ok = True
        for attr, path, label in (
            ("stress_model",    self.stress_model_path,    "stress"),
            ("deception_model", self.deception_model_path, "deception"),
        ):
            if path.exists() and joblib is not None:
                try:
                    setattr(self, attr, joblib.load(path))
                    logger.info(f"Loaded {label} model: {path}")
                except Exception as exc:
                    logger.error(f"Failed to load {label} model: {exc}")
                    setattr(self, attr, NeutralModel())
                    self._using_placeholders = True
                    ok = False
            else:
                logger.warning(f"{label} model not found at {path} — using neutral placeholder")
                setattr(self, attr, NeutralModel())
                self._using_placeholders = True
                ok = False
        return ok

    def get_models(self) -> Tuple[Any, Any]:
        if self.stress_model is None or self.deception_model is None:
            self.load_models()
        return self.stress_model, self.deception_model

    @property
    def using_placeholders(self) -> bool:
        return self._using_placeholders


# ============================================================================
# 3.  AUDIO QUALITY & SNR
# ============================================================================

class AudioQualityAnalyzer:
    """
    Computes SNR, HNR, and quality score from raw waveform.
    No synthetic data — all computations from real audio.
    """

    def __init__(self, noise_floor_percentile: float = 10.0):
        self.noise_floor_percentile = noise_floor_percentile

    def compute_snr(self, audio: np.ndarray) -> float:
        """
        SNR = 10 * log10(mean_frame_power / noise_floor_power).
        Noise floor = bottom-percentile of per-frame power.
        Returns dB, clipped to [-20, 60].
        """
        if len(audio) < 64:
            return 0.0
        audio = np.asarray(audio, dtype=np.float32)
        # Normalize to prevent scale dependency
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / (peak + 1e-12)
        frame_size = max(128, len(audio) // 40)
        powers = np.array([
            np.mean(audio[i:i + frame_size] ** 2)
            for i in range(0, len(audio) - frame_size, frame_size)
        ], dtype=np.float32)
        if len(powers) == 0:
            return 0.0
        noise  = float(np.percentile(powers, self.noise_floor_percentile))
        signal = float(np.mean(powers))
        noise  = max(noise,  1e-12)
        signal = max(signal, 1e-12)
        return float(np.clip(10.0 * np.log10(signal / noise), -20.0, 60.0))

    def compute_hnr(self, audio: np.ndarray, sr: int = TARGET_SR) -> float:
        """
        Harmonics-to-Noise Ratio via autocorrelation peak method.
        HNR = 10 * log10(r_peak / (1 - r_peak)), in dB.
        Returns 0.0 if audio is too short or unvoiced.
        """
        if len(audio) < sr // 10:   # need at least 100 ms
            return 0.0
        audio = np.asarray(audio, dtype=np.float64)
        # Use a short central window to avoid edge effects
        win_len = min(len(audio), int(0.04 * sr))  # 40 ms
        hop     = max(1, int(0.01 * sr))            # 10 ms
        hnr_vals: List[float] = []
        for start in range(0, len(audio) - win_len, hop):
            frame = audio[start:start + win_len]
            frame -= frame.mean()
            energy = np.sum(frame ** 2)
            if energy < 1e-10:
                continue
            # Normalised autocorrelation
            ac = np.correlate(frame, frame, mode="full")
            ac = ac[len(ac) // 2:]   # keep lags ≥ 0
            ac /= (ac[0] + 1e-12)    # normalise by zero-lag (= 1.0)
            # Search for peak in range corresponding to 50–500 Hz
            lo = max(1, int(sr / 500))
            hi = min(len(ac) - 1, int(sr / 50))
            if lo >= hi:
                continue
            r_peak = float(np.max(ac[lo:hi]))
            r_peak = float(np.clip(r_peak, 1e-6, 1.0 - 1e-6))
            hnr_vals.append(10.0 * np.log10(r_peak / (1.0 - r_peak)))
        return float(np.mean(hnr_vals)) if hnr_vals else 0.0

    def snr_to_quality(self, snr_db: float) -> float:
        """
        Map SNR to [0, 1] quality score using a piecewise linear ramp.
          < 5 dB  → 0.15–0.20   (very low)
          5–15 dB → 0.20–0.65   (moderate)
          15–25 dB→ 0.65–0.90   (good)
          > 25 dB → 0.90–0.98   (excellent)
        """
        if snr_db < 5.0:
            return float(np.clip(0.15 + 0.01 * snr_db, 0.05, 0.20))
        elif snr_db < 15.0:
            return 0.20 + 0.045 * (snr_db - 5.0)
        elif snr_db < 25.0:
            return 0.65 + 0.025 * (snr_db - 15.0)
        else:
            return float(np.clip(0.90 + 0.004 * (snr_db - 25.0), 0.90, 0.98))

    def analyze(self, audio: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
        snr    = self.compute_snr(audio)
        hnr    = self.compute_hnr(audio, sr)
        qual   = self.snr_to_quality(snr)
        rms    = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak   = float(np.max(np.abs(audio)))
        return {
            "snr_db":        snr,
            "hnr_db":        hnr,
            "quality_score": qual,
            "rms_energy":    rms,
            "peak_amplitude":peak,
            "is_clipped":    float(peak > 0.99),
        }


# ============================================================================
# 4.  VOICE ACTIVITY DETECTOR
# ============================================================================

class VoiceActivityDetector:
    """
    Dual-threshold VAD: energy level + zero-crossing rate.
    Segments incoming audio into voiced chunks ready for feature extraction.

    Logic (per frame):
      voiced = energy > energy_threshold AND zcr < zcr_threshold
    Hanging: once speech starts, hold for `hang_frames` after it goes silent
    to avoid chopping words.
    """

    def __init__(
        self,
        sr:               int   = TARGET_SR,
        frame_ms:         int   = 20,       # analysis frame length
        energy_threshold: float = 3e-4,     # RMS² — tune to room noise floor
        zcr_threshold:    float = 0.15,     # fraction of zero crossings
        hang_frames:      int   = 8,        # hold-off after silence
        min_speech_ms:    int   = 300,      # discard segments shorter than this
        max_segment_ms:   int   = 8_000,    # flush if segment grows too long
    ):
        self.sr               = sr
        self.frame_len        = int(frame_ms * sr / 1000)
        self.energy_thresh    = energy_threshold
        self.zcr_thresh       = zcr_threshold
        self.hang_frames      = hang_frames
        self.min_speech_samp  = int(min_speech_ms * sr / 1000)
        self.max_speech_samp  = int(max_segment_ms * sr / 1000)

        self._speech_buf: List[np.ndarray] = []
        self._buf_len:    int              = 0
        self._hang_ctr:   int              = 0
        self._in_speech:  bool             = False

    def _is_voiced_frame(self, frame: np.ndarray) -> bool:
        energy = float(np.mean(frame ** 2))
        zcr    = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
        return energy > self.energy_thresh and zcr < self.zcr_thresh

    def feed(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Feed one raw audio chunk (any length).
        Returns a list of complete speech segments ready for processing.
        May return [] if no segment completed.
        """
        chunk    = np.asarray(chunk, dtype=np.float32)
        segments: List[np.ndarray] = []
        pos = 0

        while pos + self.frame_len <= len(chunk):
            frame  = chunk[pos: pos + self.frame_len]
            voiced = self._is_voiced_frame(frame)
            pos   += self.frame_len

            if voiced:
                self._hang_ctr = self.hang_frames
                if not self._in_speech:
                    self._in_speech = True
                self._speech_buf.append(frame)
                self._buf_len += len(frame)

                # Force-flush if segment exceeds maximum length
                if self._buf_len >= self.max_speech_samp:
                    seg = np.concatenate(self._speech_buf)
                    segments.append(seg)
                    self._speech_buf = []
                    self._buf_len    = 0

            else:
                if self._in_speech:
                    if self._hang_ctr > 0:
                        self._hang_ctr -= 1
                        self._speech_buf.append(frame)
                        self._buf_len += len(frame)
                    else:
                        # End of speech segment
                        self._in_speech = False
                        if self._buf_len >= self.min_speech_samp:
                            seg = np.concatenate(self._speech_buf)
                            segments.append(seg)
                        self._speech_buf = []
                        self._buf_len    = 0

        # Handle leftover partial frame
        if pos < len(chunk):
            leftover = chunk[pos:]
            if self._in_speech:
                self._speech_buf.append(leftover)
                self._buf_len += len(leftover)

        return segments

    def flush(self) -> Optional[np.ndarray]:
        """
        Flush any accumulated speech at pipeline shutdown.
        Returns segment or None if too short.
        """
        if self._buf_len >= self.min_speech_samp:
            seg = np.concatenate(self._speech_buf)
            self._speech_buf = []
            self._buf_len    = 0
            self._in_speech  = False
            return seg
        self._speech_buf = []
        self._buf_len    = 0
        self._in_speech  = False
        return None

    def reset(self) -> None:
        self._speech_buf = []
        self._buf_len    = 0
        self._hang_ctr   = 0
        self._in_speech  = False


# ============================================================================
# 5.  THREAD-SAFE RING BUFFER
# ============================================================================

class RingBuffer:
    """
    Fixed-capacity FIFO of numpy arrays.
    push() drops oldest when full (no blocking).
    """

    def __init__(self, maxlen: int = 30):
        self._buf:  deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, arr: np.ndarray) -> None:
        with self._lock:
            self._buf.append(arr)

    def pop(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._buf.popleft() if self._buf else None

    def pop_all(self) -> List[np.ndarray]:
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# ============================================================================
# 6.  MICROPHONE CAPTURE DAEMON THREAD
# ============================================================================

class MicrophoneCaptureThread:
    """
    Continuous microphone capture via sounddevice InputStream.
    Runs as a daemon thread — exits automatically when the main process ends.
    Raw float32 chunks pushed to a RingBuffer consumed by the VAD.
    """

    def __init__(
        self,
        sr:         int = TARGET_SR,
        chunk_ms:   int = CHUNK_MS,
        ring_buf:   Optional[RingBuffer] = None,
    ):
        if sd is None:
            raise ImportError(
                "sounddevice is required for microphone capture. "
                "Install with: pip install sounddevice")
        self.sr         = sr
        self.blocksize  = int(chunk_ms * sr / 1000)
        self.ring_buf   = ring_buf or RingBuffer(maxlen=40)
        self._stream:   Optional[Any] = None
        self._stop_evt  = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self.chunks_captured: int = 0
        self.overrun_count:   int = 0

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info: Any, status: Any) -> None:
        if status:
            logger.debug(f"Mic stream status: {status}")
            if status.input_overflow:
                self.overrun_count += 1
        chunk = indata[:, 0].copy().astype(np.float32)
        self.ring_buf.push(chunk)
        self.chunks_captured += 1

    def _run(self) -> None:
        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                blocksize=self.blocksize,
                dtype=np.float32,
                callback=self._callback,
            ):
                logger.info(
                    f"Mic capture started: {self.sr} Hz, "
                    f"block={self.blocksize} samples ({CHUNK_MS} ms)")
                while not self._stop_evt.is_set():
                    time.sleep(0.05)
        except Exception as exc:
            logger.error(f"Mic capture thread crashed: {exc}", exc_info=True)

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SM-Mic")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ============================================================================
# 7.  DECEPTION FUSION ENGINE
# ============================================================================

class DeceptionFusionEngine:
    """
    Fuses RF model outputs with acoustic prosody and hesitation features
    into a single deception probability.

    Fusion weights:
      RF deception model:  0.40
      RF stress model:     0.20
      Prosody risk:        0.15   (jitter, shimmer, HNR, pitch variance)
      Hesitation risk:     0.15   (pause ratio, pause count, voiced_ratio)
      Voice quality:       0.10   (SNR-derived)

    Quality gating:
      confidence = quality_score  (output separately — does NOT scale probabilities)
      status = LOW_QUALITY        when quality < 0.35
    """

    # Fusion weights (must sum to 1.0)
    W_DECEP    = 0.40
    W_STRESS   = 0.20
    W_PROSODY  = 0.15
    W_HESIT    = 0.15
    W_QUALITY  = 0.10

    # Prosody reference values (from literature / empirical calibration)
    # These define "normal" — deviation drives risk upward
    JITTER_NORMAL   = 0.01    # ~1% cycle-to-cycle variation
    SHIMMER_NORMAL  = 0.03    # ~3% amplitude variation
    HNR_NORMAL_DB   = 15.0   # dB — healthy voice
    PITCH_VAR_NORM  = 400.0  # Hz² — moderate pitch variation

    def compute_prosody_risk(self, features: Dict[str, Any],
                             hnr_db: float) -> float:
        """
        Prosody-based deception risk in [0, 1].

        Elevated jitter + shimmer + low HNR + high pitch variance
        are associated with vocal stress and deception cues.

        pitch_stability from feature_extraction = 1/(std+eps), range [0,∞).
        We use pitch_var directly for a bounded computation.
        """
        jitter      = float(features.get("jitter",         0.0))
        shimmer     = float(features.get("shimmer",        0.0))
        pitch_var   = float(features.get("pitch_var",      0.0))

        # Jitter: excess above normal as fraction of 3×normal
        jitter_risk = float(np.clip(
            (jitter - self.JITTER_NORMAL) / (3.0 * self.JITTER_NORMAL + 1e-9),
            0.0, 1.0))

        # Shimmer: same treatment
        shimmer_risk = float(np.clip(
            (shimmer - self.SHIMMER_NORMAL) / (3.0 * self.SHIMMER_NORMAL + 1e-9),
            0.0, 1.0))

        # HNR: lower than normal → higher risk (mapped via tanh-like curve)
        hnr_deficit  = max(0.0, self.HNR_NORMAL_DB - hnr_db)
        hnr_risk     = float(np.clip(hnr_deficit / (self.HNR_NORMAL_DB + 1e-9),
                                     0.0, 1.0))

        # Pitch variance: scaled by reference value
        pitch_risk   = float(np.clip(
            pitch_var / (self.PITCH_VAR_NORM * 3.0 + 1e-9),
            0.0, 1.0))

        prosody_risk = (0.30 * jitter_risk +
                        0.30 * shimmer_risk +
                        0.25 * hnr_risk +
                        0.15 * pitch_risk)
        return float(np.clip(prosody_risk, 0.0, 1.0))

    def compute_hesitation_risk(self, features: Dict[str, Any]) -> float:
        """
        Hesitation-based deception risk in [0, 1].

        Uses pause_ratio, pause_count, and voiced_ratio.
        speech_rate (frames/sec) is NOT used directly because its absolute
        range depends on frame hop — instead we use voiced_ratio which
        is naturally normalised to [0, 1].
        """
        pause_ratio  = float(np.clip(features.get("pause_ratio",  0.0), 0.0, 1.0))
        pause_count  = float(features.get("pause_count",  0))
        voiced_ratio = float(np.clip(features.get("voiced_ratio", 1.0), 0.0, 1.0))

        # High pause_ratio → hesitation
        pause_ratio_risk = float(np.clip(pause_ratio / 0.50, 0.0, 1.0))

        # Many pauses per utterance → fragmented speech
        pause_count_risk = float(np.clip(pause_count / 10.0, 0.0, 1.0))

        # Low voiced_ratio → lots of silence/whispering
        unvoiced_risk = float(np.clip(1.0 - voiced_ratio, 0.0, 1.0))

        hesitation = (0.40 * pause_ratio_risk +
                      0.30 * pause_count_risk +
                      0.30 * unvoiced_risk)
        return float(np.clip(hesitation, 0.0, 1.0))

    def fuse(
        self,
        rf_stress_prob:   float,
        rf_decep_prob:    float,
        prosody_risk:     float,
        hesitation_risk:  float,
        voice_quality:    float,
    ) -> Tuple[float, float]:
        """
        Compute fused deception probability and confidence.

        Returns:
            (fused_deception_prob, confidence)

        confidence = voice_quality   [0,1]
        Probabilities are NOT scaled by quality — they represent the best
        estimate given the available signal. confidence tells the consumer
        how much to trust them.
        """
        fused = (
            self.W_DECEP   * float(np.clip(rf_decep_prob,    0.0, 1.0)) +
            self.W_STRESS  * float(np.clip(rf_stress_prob,   0.0, 1.0)) +
            self.W_PROSODY * float(np.clip(prosody_risk,     0.0, 1.0)) +
            self.W_HESIT   * float(np.clip(hesitation_risk,  0.0, 1.0)) +
            self.W_QUALITY * (1.0 - float(np.clip(voice_quality, 0.0, 1.0)))
            # Quality term: lower quality → slight upward nudge in uncertainty
        )
        confidence = float(np.clip(voice_quality, 0.0, 1.0))
        return float(np.clip(fused, 0.0, 1.0)), confidence


# ============================================================================
# 8.  STREAMING SM PROCESSOR  (main API class)
# ============================================================================

class StreamingSMProcessor:
    """
    Full acoustic speech analysis pipeline.

    Modes:
      input_source="mic"      → starts MicrophoneCaptureThread + VAD loop
      input_source="file"     → call process_raw_audio(waveform, sr) directly
      input_source="external" → call process_audio_chunk(feature_vec) directly

    Thread safety:
      _last_output protected by _output_lock.
      All public getters acquire the lock.
    """

    def __init__(
        self,
        model_dir:    Optional[str] = None,
        input_source: str           = "mic",    # "mic" | "file" | "external"
        sr:           int           = TARGET_SR,
        chunk_ms:     int           = CHUNK_MS,
        vad_energy_threshold: float = 3e-4,
        vad_zcr_threshold:    float = 0.15,
        min_speech_ms:        int   = 300,
        callback: Optional[Callable[["SMOutput"], None]] = None,
    ):
        if input_source not in ("mic", "file", "external"):
            raise ValueError("input_source must be 'mic', 'file', or 'external'")

        self.input_source = input_source
        self.sr           = sr
        self._callback    = callback

        # Model loading
        model_dir_p  = Path(model_dir) if model_dir else None
        self._manager = SMModelManager(model_dir_p)
        self._stress_model, self._decep_model = self._manager.get_models()

        # Signal processing
        self._preprocessor    = SpeechPreprocessor()
        self._feature_extractor = SpeechFeatureExtractor()
        self._quality_analyzer  = AudioQualityAnalyzer()
        self._fusion            = DeceptionFusionEngine()

        # VAD
        self._vad = VoiceActivityDetector(
            sr=sr,
            energy_threshold=vad_energy_threshold,
            zcr_threshold=vad_zcr_threshold,
            min_speech_ms=min_speech_ms,
        )

        # Mic capture (only for "mic" mode)
        self._ring_buf:   Optional[RingBuffer]          = None
        self._mic_thread: Optional[MicrophoneCaptureThread] = None

        # Processing loop
        self._running      = False
        self._proc_thread: Optional[threading.Thread]   = None

        # Thread-safe output buffer
        self._output_lock  = threading.Lock()
        self._last_output: Optional[SMOutput] = None

        if self._manager.using_placeholders:
            logger.warning(
                "Real model files not found. "
                "NeutralModel (0.5 priors) will be used until .pkl files are loaded.")

    # ── Probability helpers ──────────────────────────────────────────────────

    @staticmethod
    def _prob_to_level(p: float) -> str:
        if p < 0.33:
            return "LOW"
        elif p < 0.67:
            return "MODERATE"
        return "HIGH"

    @staticmethod
    def _interpret(sp: float, dp: float, quality: float,
                   confidence: float, prosody: float, hesit: float) -> str:
        if quality < 0.20:
            return "Audio quality too low for reliable analysis"
        if confidence < 0.30:
            return f"Low confidence (SNR-based) — treat results with caution"
        parts: List[str] = []
        if dp > 0.70:
            parts.append(f"HIGH deception risk ({dp:.0%})")
        elif dp > 0.50:
            parts.append(f"Moderate deception indicators ({dp:.0%})")
        if sp > 0.70:
            parts.append(f"High stress ({sp:.0%})")
        elif sp > 0.50:
            parts.append(f"Moderate stress ({sp:.0%})")
        if prosody > 0.60:
            parts.append("Vocal instability detected")
        if hesit > 0.60:
            parts.append("Hesitation patterns detected")
        if not parts:
            return (f"Speech within normal range "
                    f"(decep={dp:.0%}, stress={sp:.0%}, conf={confidence:.0%})")
        return "; ".join(parts)

    # ── Core inference (feature vector in) ──────────────────────────────────

    def _predict(self, vec: np.ndarray) -> Tuple[float, float]:
        """Run RF classifiers on a 94-d feature vector."""
        x = vec.reshape(1, -1)
        try:
            sp = float(self._stress_model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.error(f"Stress predict: {exc}")
            sp = 0.5
        try:
            dp = float(self._decep_model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.error(f"Deception predict: {exc}")
            dp = 0.5
        return sp, dp

    def _pad_or_trim(self, vec: np.ndarray) -> np.ndarray:
        """Ensure feature vector is exactly N_FEATURES long."""
        if vec.size < N_FEATURES:
            logger.debug(f"Padding feature vector {vec.size} → {N_FEATURES}")
            return np.pad(vec, (0, N_FEATURES - vec.size))
        if vec.size > N_FEATURES:
            logger.debug(f"Trimming feature vector {vec.size} → {N_FEATURES}")
            return vec[:N_FEATURES]
        return vec

    # ── Public API: process pre-extracted features ───────────────────────────

    def process_audio_chunk(
        self,
        features:      Any,
        audio_quality: float = 1.0,
        snr_db:        float = 20.0,
        hnr_db:        float = 15.0,
        raw_features:  Optional[Dict[str, Any]] = None,
    ) -> SMOutput:
        """
        Classify a pre-extracted 94-d feature vector.

        Args:
            features:      array-like of length 94
            audio_quality: SNR-derived quality score [0,1]
            snr_db:        raw SNR in dB
            hnr_db:        HNR in dB
            raw_features:  full feature dict from SpeechFeatureExtractor.extract()
                           (enables prosody/hesitation fusion)
        Returns:
            SMOutput (thread-safe)
        """
        vec = self._pad_or_trim(np.asarray(features, dtype=np.float32).ravel())
        sp, dp_rf = self._predict(vec)

        # Prosody / hesitation (require raw feature dict)
        prosody_risk   = 0.0
        hesitation_risk = 0.0
        speech_rate    = 0.0
        voiced_ratio   = 0.0
        pause_ratio    = 0.0
        if raw_features:
            prosody_risk    = self._fusion.compute_prosody_risk(raw_features, hnr_db)
            hesitation_risk = self._fusion.compute_hesitation_risk(raw_features)
            speech_rate     = float(raw_features.get("speech_rate",  0.0))
            voiced_ratio    = float(raw_features.get("voiced_ratio", 0.0))
            pause_ratio     = float(raw_features.get("pause_ratio",  0.0))

        dp_fused, confidence = self._fusion.fuse(
            rf_stress_prob  = sp,
            rf_decep_prob   = dp_rf,
            prosody_risk    = prosody_risk,
            hesitation_risk = hesitation_risk,
            voice_quality   = audio_quality,
        )

        if audio_quality < 0.20:
            status = "LOW_QUALITY"
        elif confidence < 0.25:
            status = "LOW_CONFIDENCE"
        else:
            status = "OK"

        out = SMOutput(
            stress_prob     = sp,
            deception_prob  = dp_fused,
            stress_level    = self._prob_to_level(sp),
            deception_level = self._prob_to_level(dp_fused),
            status          = status,
            voice_quality   = audio_quality,
            confidence      = confidence,
            snr_db          = snr_db,
            interpretation  = self._interpret(
                sp, dp_fused, audio_quality, confidence,
                prosody_risk, hesitation_risk),
            timestamp       = time.time(),
            feature_count   = int(vec.size),
            hnr_db          = hnr_db,
            prosody_risk    = prosody_risk,
            hesitation_risk = hesitation_risk,
            speech_rate     = speech_rate,
            voiced_ratio    = voiced_ratio,
            pause_ratio     = pause_ratio,
        )
        self._set_output(out)
        return out

    # ── Public API: process raw waveform ────────────────────────────────────

    def process_raw_audio(
        self,
        audio: np.ndarray,
        sr:    int = TARGET_SR,
    ) -> SMOutput:
        """
        Full pipeline from raw waveform: preprocess → features → classify.
        Use for file-mode or when external code provides raw audio.
        """
        if len(audio) < sr // 10:
            return self._insufficient_output()

        audio = np.asarray(audio, dtype=np.float32)

        # Quality analysis on raw (pre-processed) audio
        qa = self._quality_analyzer.analyze(audio, sr)

        # Preprocess: bandpass, VAD trim, normalise, pre-emphasis
        try:
            prep_result  = self._preprocessor.preprocess(audio, sr)
            clean_audio  = prep_result["waveform"]
            clean_sr     = int(prep_result["sample_rate"])
        except Exception as exc:
            logger.error(f"Preprocessing failed: {exc}")
            clean_audio = audio
            clean_sr    = sr

        if len(clean_audio) < sr // 10:
            return self._insufficient_output()

        # Feature extraction
        try:
            feat_dict = self._feature_extractor.extract(clean_audio, clean_sr)
        except Exception as exc:
            logger.error(f"Feature extraction failed: {exc}")
            return self._error_output(str(exc))

        vec = np.asarray(feat_dict.get("feature_vector", []), dtype=np.float32)

        return self.process_audio_chunk(
            features      = vec,
            audio_quality = qa["quality_score"],
            snr_db        = qa["snr_db"],
            hnr_db        = qa["hnr_db"],
            raw_features  = feat_dict,
        )

    def process_wav_file(self, path: str) -> SMOutput:
        """Load a WAV file and run the full pipeline."""
        if not _LIBROSA_OK:
            raise ImportError("librosa required: pip install librosa")
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            audio, sr = librosa.load(path, sr=None, mono=True)
        return self.process_raw_audio(np.asarray(audio, dtype=np.float32), int(sr))

    # ── Mic streaming API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start continuous mic capture + processing loop (mic mode only)."""
        if self.input_source != "mic":
            raise RuntimeError(
                "start() is only valid for input_source='mic'.")
        if self._running:
            logger.warning("Already running")
            return

        self._ring_buf  = RingBuffer(maxlen=40)
        self._mic_thread = MicrophoneCaptureThread(
            sr=self.sr, chunk_ms=CHUNK_MS, ring_buf=self._ring_buf)
        self._mic_thread.start()

        self._running    = True
        self._proc_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="SM-Proc")
        self._proc_thread.start()
        logger.info("StreamingSMProcessor started (mic mode)")

    def stop(self) -> None:
        """Stop mic capture and processing loop."""
        self._running = False
        if self._mic_thread:
            self._mic_thread.stop()
        # Flush remaining VAD buffer
        seg = self._vad.flush()
        if seg is not None:
            self.process_raw_audio(seg, self.sr)
        if self._proc_thread:
            self._proc_thread.join(timeout=5)
        logger.info("StreamingSMProcessor stopped")

    def _processing_loop(self) -> None:
        """
        Daemon loop: drain RingBuffer → VAD → preprocess → classify.
        Runs in SM-Proc thread.
        """
        try:
            while self._running:
                if self._ring_buf is None:
                    time.sleep(0.05)
                    continue

                chunk = self._ring_buf.pop()
                if chunk is None:
                    time.sleep(0.02)
                    continue

                # VAD: get voiced segments
                segments = self._vad.feed(chunk)
                for seg in segments:
                    out = self.process_raw_audio(seg, self.sr)
                    if self._callback:
                        try:
                            self._callback(out)
                        except Exception as exc:
                            logger.error(f"SM callback error: {exc}")

        except Exception as exc:
            logger.error(f"SM processing loop crashed: {exc}", exc_info=True)

    # ── Output helpers ───────────────────────────────────────────────────────

    def _set_output(self, out: SMOutput) -> None:
        with self._output_lock:
            self._last_output = out

    def get_last_output(self) -> Optional[SMOutput]:
        """Thread-safe read of latest SMOutput."""
        with self._output_lock:
            return self._last_output

    def _insufficient_output(self) -> SMOutput:
        out = SMOutput(
            stress_prob     = 0.0,
            deception_prob  = 0.0,
            stress_level    = "LOW",
            deception_level = "LOW",
            status          = "INSUFFICIENT_AUDIO",
            voice_quality   = 0.0,
            confidence      = 0.0,
            snr_db          = 0.0,
            interpretation  = "Insufficient audio for analysis",
            timestamp       = time.time(),
            feature_count   = 0,
        )
        self._set_output(out)
        return out

    def _error_output(self, msg: str) -> SMOutput:
        out = SMOutput(
            stress_prob     = 0.0,
            deception_prob  = 0.0,
            stress_level    = "LOW",
            deception_level = "LOW",
            status          = "ERROR",
            voice_quality   = 0.0,
            confidence      = 0.0,
            snr_db          = 0.0,
            interpretation  = f"Processing error: {msg}",
            timestamp       = time.time(),
            feature_count   = 0,
        )
        self._set_output(out)
        return out


# ============================================================================
# 9.  LIVE TERMINAL DISPLAY
# ============================================================================

class LiveTerminalDisplay:
    """
    Prints a rolling live readout to the terminal.
    Updates in-place using ANSI escape codes.
    Falls back to plain newline printing if terminal doesn't support it.
    """

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _CYAN   = "\033[36m"
    _GREY   = "\033[90m"

    def __init__(self, use_ansi: bool = True):
        self._ansi = use_ansi and sys.stdout.isatty()
        self._lines_printed = 0

    def _col(self, prob: float) -> str:
        if not self._ansi:
            return ""
        if prob < 0.40:
            return self._GREEN
        elif prob < 0.65:
            return self._YELLOW
        return self._RED

    def _bar(self, prob: float, width: int = 20) -> str:
        filled = int(prob * width)
        return "█" * filled + "░" * (width - filled)

    def _clear_lines(self, n: int) -> None:
        if self._ansi and n > 0:
            sys.stdout.write(f"\033[{n}A\033[J")

    def print(self, out: Optional[SMOutput], source: str = "mic") -> None:
        lines: List[str] = []
        b = self._BOLD   if self._ansi else ""
        r = self._RESET  if self._ansi else ""
        c = self._CYAN   if self._ansi else ""
        g = self._GREY   if self._ansi else ""

        lines.append(f"{b}{c}╔══ SPEECH MODALITY — LIVE ({source.upper()}) ══{r}")

        if out is None:
            lines.append("  Waiting for speech…")
        elif out.status in ("INSUFFICIENT_AUDIO", "ERROR"):
            lines.append(f"  {g}[{out.status}] {out.interpretation}{r}")
        else:
            ts = time.strftime("%H:%M:%S", time.localtime(out.timestamp))
            lines.append(f"  {g}{ts}  status={out.status}  conf={out.confidence:.0%}{r}")

            dc = self._col(out.deception_prob)
            sc = self._col(out.stress_prob)
            lines.append(
                f"  {b}DECEPTION{r}  "
                f"{dc}{self._bar(out.deception_prob)}{r}  "
                f"{dc}{out.deception_prob:.3f}{r}  [{out.deception_level}]")
            lines.append(
                f"  {b}STRESS   {r}  "
                f"{sc}{self._bar(out.stress_prob)}{r}  "
                f"{sc}{out.stress_prob:.3f}{r}  [{out.stress_level}]")
            lines.append(
                f"  SNR={out.snr_db:+.1f} dB  "
                f"HNR={out.hnr_db:+.1f} dB  "
                f"Quality={out.voice_quality:.0%}  "
                f"Voiced={out.voiced_ratio:.0%}")
            lines.append(
                f"  Prosody={out.prosody_risk:.3f}  "
                f"Hesitation={out.hesitation_risk:.3f}  "
                f"PauseRatio={out.pause_ratio:.0%}")
            lines.append(f"  {g}{out.interpretation}{r}")

        lines.append(f"{b}{c}╚{'═'*42}{r}")

        self._clear_lines(self._lines_printed)
        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._lines_printed = len(lines)


# ============================================================================
# 10.  STANDALONE ENTRY POINT
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Speech Modality v2 — acoustic deception/stress analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mic analysis
  python speech_v2.py --mic

  # Analyse a WAV file
  python speech_v2.py --file interview.wav

  # Custom model directory + longer VAD segments
  python speech_v2.py --mic --model-dir /path/to/models --min-speech-ms 500
""")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic",  action="store_true",
                     help="Capture audio from default microphone (continuous)")
    src.add_argument("--file", metavar="PATH",
                     help="Analyse a WAV file")

    ap.add_argument("--model-dir", default=None, metavar="DIR",
                    help=f"Directory containing {STRESS_MODEL_NAME} and "
                         f"{DECEPTION_MODEL_NAME} "
                         f"(default: ./models or ${MODEL_DIR_ENV})")
    ap.add_argument("--min-speech-ms", type=int, default=300, metavar="MS",
                    help="Minimum voiced segment length for VAD (default: 300 ms)")
    ap.add_argument("--energy-threshold", type=float, default=3e-4,
                    help="VAD energy threshold (default: 3e-4)")
    ap.add_argument("--zcr-threshold", type=float, default=0.15,
                    help="VAD zero-crossing rate threshold (default: 0.15)")
    ap.add_argument("--no-ansi", action="store_true",
                    help="Disable ANSI colour codes in terminal output")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG logging")
    return ap


def _run_mic_mode(args: argparse.Namespace) -> None:
    """Continuous mic capture → VAD → classify → live display."""
    display   = LiveTerminalDisplay(use_ansi=not args.no_ansi)
    processor = StreamingSMProcessor(
        model_dir            = args.model_dir,
        input_source         = "mic",
        vad_energy_threshold = args.energy_threshold,
        vad_zcr_threshold    = args.zcr_threshold,
        min_speech_ms        = args.min_speech_ms,
        callback             = lambda out: display.print(out, source="mic"),
    )

    processor.start()
    display.print(None, source="mic")

    print("\nListening… speak into your microphone. Press Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        processor.stop()
        out = processor.get_last_output()
        if out:
            print("\n── Final output ──")
            import json
            print(json.dumps(out.to_dict(), indent=2))


def _run_file_mode(args: argparse.Namespace) -> None:
    """Load WAV file → full pipeline → display result."""
    display   = LiveTerminalDisplay(use_ansi=not args.no_ansi)
    processor = StreamingSMProcessor(
        model_dir    = args.model_dir,
        input_source = "file",
    )

    path = args.file
    print(f"Analysing: {path}")
    out = processor.process_wav_file(path)
    display.print(out, source="file")

    import json
    print("\n── Full output ──")
    print(json.dumps(out.to_dict(), indent=2))


if __name__ == "__main__":
    ap   = _build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mic:
        _run_mic_mode(args)
    else:
        _run_file_mode(args)