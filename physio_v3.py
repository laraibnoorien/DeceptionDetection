#!/usr/bin/env python3
"""
PHYSIO V3 — PHYSIOLOGICAL MODALITY SYSTEM
==========================================
Production-ready physiological modality for real-time deception inference.

Data sources:
  • Timex smartwatch via direct BLE (bleak) — HR, SpO₂
  • Gadgetbridge Android export files (CSV / JSON / NDJSON)
  • Camera-based rPPG (OpenCV + MediaPipe) — forehead + bilateral cheek ROI

rPPG pipeline (real signal, no simulation):
  • MediaPipe FaceMesh landmark detection
  • Forehead ROI + left/right cheek ROI green-channel extraction
  • Per-ROI bandpass-filtered signal (0.7–4.0 Hz)
  • FFT peak detection → HR in BPM
  • SNR = peak PSD / total band PSD (true ratio, no magic scaling)
  • Per-ROI rolling waveform exposed for ECG-style HUD display
  • CHROM-inspired cross-ROI fusion for illumination robustness

Outputs (PMOutput):
  Fused HR, HRV, RMSSD, RSA, SpO₂, EDA metrics,
  rPPG HR, rPPG SNR, per-ROI waveforms,
  stress probability, anomaly score, deception probability,
  signal quality, baseline deviation.

Standalone HUD (run as __main__):
  Left  panel — live camera feed with annotated ROI overlays
  Right panel:
    ├─ WEARABLE  — BLE HR, SpO₂, HRV, quality bar
    ├─ rPPG      — camera HR, SNR, face-detected indicator, per-ROI values
    ├─ WAVEFORM  — ECG-style rolling green-channel plot
    └─ VITALS    — fused HR, SpO₂, stress, deception gauge bars

Integration:
  pipeline.py  →  PMPipelineAdapter(cfg)
  websocket    →  PMSystem.register_callback(cb)   → PMOutput
  session mgr  →  PMSystem.connect() / .start() / .stop()
  frontend HUD →  PMOutput.to_dict() / .to_json()

No simulation, no fake data, no hardcoded signals.
All thresholds configurable via PMConfig.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import warnings
import os
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy import signal as sp_signal
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1.  CONFIGURATION
# ============================================================================

@dataclass
class PMConfig:
    """
    All tunable parameters in one dataclass.
    Pass an instance to PMSystem; override only what differs from defaults.
    """
    # ── Device ───────────────────────────────────────────────────────────────
    timex_mac:             Optional[str]  = None
    gadgetbridge_dir:      Optional[str]  = None
    camera_id:             Optional[int]  = None

    # ── Physiological validity ranges ────────────────────────────────────────
    hr_min_bpm:            float = 30.0
    hr_max_bpm:            float = 220.0
    spo2_min_pct:          float = 70.0
    spo2_max_pct:          float = 100.0

    # ── Calibration ──────────────────────────────────────────────────────────
    calib_seconds:         float = 15.0
    baseline_hr_mad_k:     float = 2.5
    baseline_spo2_mad_k:   float = 2.5
    baseline_hr_lo:        float = 40.0
    baseline_hr_hi:        float = 100.0
    baseline_spo2_warn:    float = 95.0

    # ── Signal buffers ────────────────────────────────────────────────────────
    ble_fs:                float = 2.0
    cam_fs:                float = 30.0
    window_seconds:        float = 30.0

    # ── rPPG ─────────────────────────────────────────────────────────────────
    rppg_hr_min_hz:        float = 0.7    # 42 bpm lower bound
    rppg_hr_max_hz:        float = 4.0    # 240 bpm upper bound
    rppg_warmup_seconds:   float = 4.0    # minimum data before emitting HR
    rppg_gap_factor:       float = 2.0    # frame gap > factor*expected → reset
    rppg_min_snr:          float = 0.05   # minimum SNR to emit a reading
    rppg_waveform_seconds: float = 6.0    # seconds of waveform kept for HUD

    # ── RSA / respiration ────────────────────────────────────────────────────
    rsa_band_lo_hz:        float = 0.15
    rsa_band_hi_hz:        float = 0.40
    rsa_interp_fs:         float = 4.0
    resp_band_lo_hz:       float = 0.12
    resp_band_hi_hz:       float = 0.50

    # ── EDA decomposition ────────────────────────────────────────────────────
    eda_scr_hp_hz:         float = 0.05
    scr_peak_std_k:        float = 0.5
    scr_min_dist_s:        float = 1.0

    # ── Anomaly detector ─────────────────────────────────────────────────────
    iforest_estimators:    int   = 100
    iforest_contamination: float = 0.01
    iforest_retrain_every: int   = 100
    iforest_min_samples:   int   = 20
    iforest_max_history:   int   = 2000

    # ── Stress / deception thresholds ────────────────────────────────────────
    hr_stress_threshold_frac:  float = 0.20
    spo2_stress_delta:         float = 2.0
    scr_deception_min:         float = 2.0
    scr_deception_scale:       float = 8.0
    anomaly_deception_thresh:  float = 0.30

    # ── Emotion thresholds ───────────────────────────────────────────────────
    emo_fearful_hr:      float = 100.0
    emo_fearful_oxy:     float = 0.30
    emo_fearful_arousal: float = 0.70
    emo_angry_hr:        float = 90.0
    emo_angry_tension:   float = 0.55
    emo_angry_eda_var:   float = 1.50
    emo_disgust_hr:      float = 85.0
    emo_disgust_zygo:    float = 20.0
    emo_disgust_eda_var: float = 1.00
    emo_sad_arousal:     float = 0.35
    emo_sad_tension:     float = 0.30
    emo_sad_hr:          float = 80.0
    emo_happy_stress:    float = 0.35
    emo_happy_spo2:      float = 96.0
    emo_happy_hr:        float = 85.0

    # ── Cognitive load ───────────────────────────────────────────────────────
    cog_stress_exp:      float = 0.80
    cog_arousal_exp:     float = 0.70
    cog_component_min:   float = 0.10

    # ── BLE ──────────────────────────────────────────────────────────────────
    ble_notify_timeout:  float = 10.0
    ble_max_retries:     int   = 3
    ble_signal_buf:      int   = 2000
    ble_quality_buf:     int   = 120

    # ── Process loop ─────────────────────────────────────────────────────────
    process_interval:    float = 0.5
    watchdog_timeout:    float = 10.0

    # ── HUD ──────────────────────────────────────────────────────────────────
    hud_width:           int   = 1280
    hud_height:          int   = 720
    hud_cam_width:       int   = 640   # left panel width


# ============================================================================
# 2.  ENUMERATIONS
# ============================================================================

class EmotionEnum(Enum):
    HAPPY     = "HAPPY"
    SAD       = "SAD"
    ANGRY     = "ANGRY"
    FEARFUL   = "FEARFUL"
    DISGUSTED = "DISGUSTED"
    NEUTRAL   = "NEUTRAL"


class SignalType(Enum):
    HR   = "HR"
    HRV  = "HRV"
    SPO2 = "SPO2"
    EMG  = "EMG"
    EDA  = "EDA"
    TEMP = "TEMP"
    RPPG = "RPPG"
    RSA  = "RSA"


# ============================================================================
# 3.  DATA STRUCTURES
# ============================================================================

@dataclass
class PhysioSignal:
    signal_type: SignalType
    value:       float
    timestamp:   float
    confidence:  float = 1.0
    source:      str   = "unknown"


@dataclass
class RPPGState:
    """Live rPPG state exposed to HUD and pipeline consumers."""
    hr_bpm:          float          # estimated HR from camera
    snr:             float          # signal quality 0–1
    face_detected:   bool
    forehead_signal: List[float]    # rolling green-channel waveform (forehead)
    cheek_l_signal:  List[float]    # rolling green-channel waveform (left cheek)
    cheek_r_signal:  List[float]    # rolling green-channel waveform (right cheek)
    fused_signal:    List[float]    # CHROM-fused waveform used for HR estimation
    forehead_value:  float          # latest raw green mean (forehead)
    cheek_l_value:   float          # latest raw green mean (left cheek)
    cheek_r_value:   float          # latest raw green mean (right cheek)
    timestamp:       float


@dataclass
class PhysioFeatures:
    # SpO₂
    spo2_mean:         float = 0.0
    spo2_variance:     float = 0.0
    spo2_delta:        float = 0.0
    oxygen_stress:     float = 0.0
    # HR / HRV
    hr_mean:           float = 0.0
    hr_variance:       float = 0.0
    hr_elevation:      float = 0.0
    hrv_rmssd:         float = 0.0
    hrv_sdnn:          float = 0.0
    rsa_power:         float = 0.0
    rsa_index:         float = 0.0
    respiration_freq:  float = 0.0
    respiration_rate:  float = 0.0
    # EMG
    emg_zygo_mean:     float = 0.0
    emg_corr_mean:     float = 0.0
    emg_asym:          float = 0.0
    muscle_tension:    float = 0.0
    # EDA
    eda_mean:          float = 0.0
    eda_variance:      float = 0.0
    eda_peak_freq:     float = 0.0
    scl_mean:          float = 0.0
    scl_slope:         float = 0.0
    scr_freq:          float = 0.0
    scr_amp:           float = 0.0
    # Temperature
    temp_mean:         float = 0.0
    temp_delta:        float = 0.0
    temp_gradient:     float = 0.0
    temp_gradient_vel: float = 0.0
    # rPPG
    rppg_hr:           float = 0.0
    rppg_snr:          float = 0.0
    rppg_quality:      float = 0.0
    # Composite
    arousal_level:     float = 0.0

    def is_valid(self) -> bool:
        return self.hr_mean > 0.0 or self.rppg_hr > 0.0


@dataclass
class PMOutput:
    stress_probability:    float
    emotion:               EmotionEnum
    deception_probability: float
    cognitive_load:        float
    anomaly_score:         float
    arousal_level:         float
    oxygen_level:          float
    heart_rate:            float          # fused (wearable preferred, rPPG fallback)
    hrv_rmssd:             float
    rppg_hr:               float          # camera-only HR
    rppg_snr:              float          # camera signal quality
    rppg_face_detected:    bool
    signal_quality:        float
    features:              PhysioFeatures
    rppg_state:            Optional[RPPGState]
    timestamp:             float
    calibrated:            bool
    device:                str

    def to_dict(self) -> Dict:
        d = {
            "stress_probability":    round(self.stress_probability, 4),
            "emotion":               self.emotion.value,
            "deception_probability": round(self.deception_probability, 4),
            "cognitive_load":        round(self.cognitive_load, 4),
            "anomaly_score":         round(self.anomaly_score, 4),
            "arousal_level":         round(self.arousal_level, 4),
            "oxygen_level":          round(self.oxygen_level, 2),
            "heart_rate":            round(self.heart_rate, 1),
            "hrv_rmssd":             round(self.hrv_rmssd, 2),
            "rppg_hr":               round(self.rppg_hr, 1),
            "rppg_snr":              round(self.rppg_snr, 4),
            "rppg_face_detected":    self.rppg_face_detected,
            "signal_quality":        round(self.signal_quality, 4),
            "timestamp":             self.timestamp,
            "calibrated":            self.calibrated,
            "device":                self.device,
            "features": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self.features).items()
            },
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ============================================================================
# 4.  GATT UUID REGISTRY
# ============================================================================

class TimexGATT:
    HR_SERVICE      = "0000180d-0000-1000-8000-00805f9b34fb"
    HR_MEASUREMENT  = "00002a37-0000-1000-8000-00805f9b34fb"
    BODY_SENSOR_LOC = "00002a38-0000-1000-8000-00805f9b34fb"
    SPO2_SERVICE    = "00001822-0000-1000-8000-00805f9b34fb"
    PLX_SPOT_CHECK  = "00002a5e-0000-1000-8000-00805f9b34fb"
    PLX_CONTINUOUS  = "00002a5f-0000-1000-8000-00805f9b34fb"
    BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb"
    BATTERY_LEVEL   = "00002a19-0000-1000-8000-00805f9b34fb"


# ============================================================================
# 5.  GATT PACKET PARSERS
# ============================================================================

def _parse_hr_measurement(data: bytes, cfg: PMConfig) -> float:
    if len(data) < 2:
        raise ValueError(f"HR packet too short: {len(data)} B")
    flags = data[0]
    hr = int.from_bytes(data[1:3], "little") if (flags & 0x01) else data[1]
    hr = float(hr)
    if not (cfg.hr_min_bpm < hr < cfg.hr_max_bpm):
        raise ValueError(f"HR {hr} outside valid range")
    return hr


def _parse_plx_continuous(data: bytes, cfg: PMConfig) -> Tuple[float, float]:
    if len(data) < 6:
        raise ValueError(f"PLX packet too short: {len(data)} B")

    def _sfloat(raw: int) -> float:
        exp = (raw >> 12) & 0xF
        man = raw & 0x0FFF
        if exp >= 8:
            exp -= 16
        if man >= 0x800:
            man -= 0x1000
        return man * (10 ** exp)

    spo2 = _sfloat(int.from_bytes(data[2:4], "little"))
    pr   = _sfloat(int.from_bytes(data[4:6], "little"))
    if not (cfg.spo2_min_pct <= spo2 <= cfg.spo2_max_pct):
        raise ValueError(f"SpO2 {spo2} outside valid range")
    if not (cfg.hr_min_bpm <= pr <= cfg.hr_max_bpm):
        raise ValueError(f"PLX PR {pr} outside valid range")
    return spo2, pr


# ============================================================================
# 6.  TIMEX BLE DEVICE
# ============================================================================

class TimexWearableDevice:
    """
    Timex smartwatch via BLE GATT notifications.
    Exponential backoff reconnect. Dynamic per-sample confidence.
    """

    def __init__(self, mac: str, cfg: PMConfig):
        self.mac       = mac
        self.cfg       = cfg
        self.connected = False
        self._signal_queue: deque[PhysioSignal] = deque(maxlen=cfg.ble_signal_buf)
        self._quality_buf:  deque[float]        = deque(maxlen=cfg.ble_quality_buf)
        self._hr_recent:    deque[float]        = deque(maxlen=10)
        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None

    def _hr_confidence(self, hr: float) -> float:
        self._hr_recent.append(hr)
        if len(self._hr_recent) < 3:
            return 0.70
        arr = np.array(list(self._hr_recent))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        dev = abs(hr - med) / (mad * 1.4826)
        return float(np.clip(1.0 - dev / 6.0, 0.50, 1.0))

    def _spo2_confidence(self, spo2: float) -> float:
        return float(np.clip((spo2 - 80.0) / 20.0, 0.50, 1.0))

    def _on_hr(self, _handle: int, data: bytes) -> None:
        try:
            hr   = _parse_hr_measurement(data, self.cfg)
            conf = self._hr_confidence(hr)
            sig  = PhysioSignal(SignalType.HR, hr, time.time(), conf, "Timex-BLE")
            self._signal_queue.append(sig)
            self._quality_buf.append(conf)
        except Exception as exc:
            logger.debug(f"HR notify: {exc}")

    def _on_spo2(self, _handle: int, data: bytes) -> None:
        try:
            spo2, pr = _parse_plx_continuous(data, self.cfg)
            now      = time.time()
            conf_s   = self._spo2_confidence(spo2)
            self._signal_queue.append(
                PhysioSignal(SignalType.SPO2, spo2, now, conf_s, "Timex-BLE"))
            self._quality_buf.append(conf_s)
            conf_p = self._hr_confidence(pr)
            self._signal_queue.append(
                PhysioSignal(SignalType.HR, pr, now, conf_p, "Timex-BLE-PR"))
        except Exception as exc:
            logger.debug(f"SpO2 notify: {exc}")

    async def _run_async(self) -> None:
        try:
            from bleak import BleakClient
        except ImportError:
            raise RuntimeError("pip install bleak")

        async with BleakClient(self.mac, timeout=15.0) as client:
            self.connected = True
            logger.info(f"BLE connected → {self.mac}")
            try:
                await client.start_notify(TimexGATT.HR_MEASUREMENT, self._on_hr)
                logger.info("  ✓ HR notifications")
            except Exception as e:
                logger.warning(f"  HR notify failed: {e}")
            spo2_ok = False
            for uuid, label in (
                (TimexGATT.PLX_CONTINUOUS, "PLX-continuous"),
                (TimexGATT.PLX_SPOT_CHECK, "PLX-spot-check"),
            ):
                try:
                    await client.start_notify(uuid, self._on_spo2)
                    logger.info(f"  ✓ SpO2 ({label})")
                    spo2_ok = True
                    break
                except Exception:
                    continue
            if not spo2_ok:
                logger.warning("  SpO2 notifications unavailable")
            while not self._stop_event.is_set() and client.is_connected:
                await asyncio.sleep(0.2)
        self.connected = False

    def _thread_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        retry = 0
        while retry < self.cfg.ble_max_retries and not self._stop_event.is_set():
            try:
                loop.run_until_complete(self._run_async())
                break
            except Exception as exc:
                retry += 1
                self.connected = False
                logger.error(f"BLE error (retry {retry}/{self.cfg.ble_max_retries}): {exc}")
                if retry < self.cfg.ble_max_retries and not self._stop_event.is_set():
                    time.sleep(2 ** retry)
        self.connected = False
        loop.close()

    def connect(self) -> bool:
        if self._thread and self._thread.is_alive():
            return self.connected
        self._stop_event.clear()
        self._signal_queue.clear()
        self._quality_buf.clear()
        self._thread = threading.Thread(
            target=self._thread_target, daemon=True, name="BLE-Timex")
        self._thread.start()
        deadline = time.time() + self.cfg.ble_notify_timeout
        while time.time() < deadline:
            if self.connected:
                return True
            time.sleep(0.2)
        logger.error("BLE connect timed out")
        return False

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.connected = False

    def get_signals(self, max_count: Optional[int] = None) -> List[PhysioSignal]:
        n = max_count if max_count is not None else len(self._signal_queue)
        out = []
        for _ in range(min(n, len(self._signal_queue))):
            out.append(self._signal_queue.popleft())
        return out

    @property
    def mean_quality(self) -> float:
        return float(np.mean(self._quality_buf)) if self._quality_buf else 0.0


# ============================================================================
# 7.  GADGETBRIDGE FILE INGEST
# ============================================================================

class GadgetbridgeIngest:
    """
    Watches a Gadgetbridge export directory for CSV / JSON / NDJSON files.
    Parses only new bytes on each poll cycle.
    Validates all values against PMConfig physiological ranges.
    """

    def __init__(self, export_dir: str, cfg: PMConfig):
        self.export_dir    = os.fspath(export_dir)
        self.cfg           = cfg
        self._signal_queue: deque[PhysioSignal] = deque(maxlen=cfg.ble_signal_buf)
        self._quality_buf:  deque[float]        = deque(maxlen=cfg.ble_quality_buf)
        self._file_positions: Dict[str, int]    = {}
        self._stop_event   = threading.Event()
        self._thread:       Optional[threading.Thread] = None
        self.connected     = False

    def _emit(self, stype: SignalType, value: float,
              ts: float, confidence: float, source: str) -> None:
        if stype == SignalType.HR:
            if not (self.cfg.hr_min_bpm <= value <= self.cfg.hr_max_bpm):
                return
        elif stype == SignalType.SPO2:
            if not (self.cfg.spo2_min_pct <= value <= self.cfg.spo2_max_pct):
                return
        sig = PhysioSignal(stype, value, ts, confidence, source)
        self._signal_queue.append(sig)
        self._quality_buf.append(confidence)

    def _parse_item(self, item: Dict) -> None:
        ts   = float(item.get("timestamp") or item.get("time") or time.time())
        conf = float(item.get("confidence", 1.0))
        src  = str(item.get("source", "gadgetbridge"))
        for key, stype in (("hr", SignalType.HR), ("spo2", SignalType.SPO2)):
            v = item.get(key)
            if v is not None and v != "":
                try:
                    self._emit(stype, float(v), ts, conf, src)
                except (TypeError, ValueError):
                    pass

    def _process_file(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(self._file_positions.get(path, 0))
                text = fh.read()
                self._file_positions[path] = fh.tell()
        except Exception:
            return
        if not text.strip():
            return
        # JSON array
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        self._parse_item(item)
                return
            if isinstance(obj, dict):
                self._parse_item(obj)
                return
        except json.JSONDecodeError:
            pass
        # NDJSON
        parsed = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    self._parse_item(item)
                    parsed = True
            except json.JSONDecodeError:
                break
        if parsed:
            return
        # CSV
        import csv as _csv
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return
        first = lines[0].lower()
        if not any(h in first for h in ("timestamp", "time", "hr", "spo2")):
            return
        reader = _csv.reader(lines)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            return
        for row in reader:
            self._parse_item(dict(zip(header, [v.strip() for v in row])))

    def _watch_loop(self) -> None:
        self.connected = True
        while not self._stop_event.is_set():
            try:
                for fname in os.listdir(self.export_dir):
                    if not fname.lower().endswith((".csv", ".json", ".ndjson")):
                        continue
                    self._process_file(os.path.join(self.export_dir, fname))
            except Exception as exc:
                logger.debug(f"GadgetbridgeIngest: {exc}")
            time.sleep(1.0)
        self.connected = False

    def connect(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        if not os.path.isdir(self.export_dir):
            logger.error(f"Gadgetbridge dir not found: {self.export_dir}")
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="GB-Ingest")
        self._thread.start()
        time.sleep(0.3)
        return True

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.connected = False

    def get_signals(self, max_count: Optional[int] = None) -> List[PhysioSignal]:
        n = max_count if max_count is not None else len(self._signal_queue)
        out = []
        for _ in range(min(n, len(self._signal_queue))):
            out.append(self._signal_queue.popleft())
        return out

    @property
    def mean_quality(self) -> float:
        return float(np.mean(self._quality_buf)) if self._quality_buf else 0.0


# ============================================================================
# 8.  rPPG EXTRACTOR  (real signal — no simulation)
# ============================================================================

class RPPGExtractor:
    """
    Camera-based HR estimation from forehead + bilateral cheek ROIs.

    Signal pipeline (per frame):
      1. MediaPipe FaceMesh → 468 landmarks
      2. Per-ROI convex-hull mask → mean green-channel value
      3. Three independent rolling buffers (forehead, cheek-L, cheek-R)
      4. Zero-mean + bandpass filter (Butterworth 2nd order, 0.7–4.0 Hz)
      5. CHROM-inspired fusion: weight ROIs by inverse variance for
         illumination robustness (no fixed coefficients)
      6. FFT on fused signal → peak frequency → HR in BPM
      7. SNR = peak_psd / total_band_psd (true ratio)

    All intermediate per-ROI waveforms are kept in RPPGState for the HUD
    ECG-style display. No synthetic data is ever injected.

    Thread-safe: process_frame() is called from the camera thread;
    get_state() is called from the process loop / HUD thread.
    A threading.Lock protects shared state.
    """

    # MediaPipe FaceMesh landmark indices
    FOREHEAD_IDS = [10,338,297,332,284,251,389,356,454,323,361,288,
                    397,365,379,378,400,377,152,148,176,149,150,136,
                    172,58,132,93,234,127,162,21,54,103,67,109]
    CHEEK_L_IDS  = [116,117,118,119,120,121,126,209,49,48]
    CHEEK_R_IDS  = [345,346,347,348,349,350,355,429,279,278]

    def __init__(self, cfg: PMConfig):
        self.cfg = cfg
        fps      = cfg.cam_fs
        win      = int(cfg.window_seconds * fps)
        wav_win  = int(cfg.rppg_waveform_seconds * fps)

        # Per-ROI green-channel rolling buffers (for HR computation)
        self._fore_buf:  deque[float] = deque(maxlen=win)
        self._cl_buf:    deque[float] = deque(maxlen=win)
        self._cr_buf:    deque[float] = deque(maxlen=win)

        # Per-ROI waveform buffers exposed to HUD (shorter, for display)
        self._fore_wav:  deque[float] = deque(maxlen=wav_win)
        self._cl_wav:    deque[float] = deque(maxlen=wav_win)
        self._cr_wav:    deque[float] = deque(maxlen=wav_win)
        self._fused_wav: deque[float] = deque(maxlen=wav_win)

        self._frame_times: deque[float] = deque(maxlen=win)

        # MediaPipe FaceMesh (guarded: some mediapipe builds expose different API)
        try:
            mp_fm = getattr(mp, 'solutions', None)
            if mp_fm is None:
                raise AttributeError('mediapipe.solutions not available')
            self._mesh = mp_fm.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mesh_available = True
        except Exception as e:
            logger.warning(f"mediapipe FaceMesh unavailable ({e}); rPPG disabled")
            self._mesh = None
            self._mesh_available = False

        # Bandpass filter coefficients (guarded)
        nyq = fps / 2.0
        try:
            if nyq <= 0:
                raise ValueError(f"Invalid sampling rate: {fps}")
            low = float(cfg.rppg_hr_min_hz) / nyq
            high = float(cfg.rppg_hr_max_hz) / nyq
            eps = 1e-6
            # Clamp into (0,1) and ensure low < high
            low = max(eps, min(0.999, low))
            high = max(low + eps, min(0.999, high))
            self._b, self._a = sp_signal.butter(
                2,
                [low, high],
                btype="band",
            )
        except Exception as e:
            logger.warning(f"rPPG: Could not create bandpass filter ({e}); using passthrough filter")
            # Passthrough filter coefficients
            self._b, self._a = np.array([1.0]), np.array([1.0])

        # Latest ROI raw values (for HUD readout)
        self._fore_val: float = 0.0
        self._cl_val:   float = 0.0
        self._cr_val:   float = 0.0

        # Latest HR / SNR estimate
        self._last_hr:  float = 0.0
        self._last_snr: float = 0.0
        self._face_det: bool  = False

        self._lock = threading.Lock()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _roi_mean_green(self, rgb: np.ndarray, lm,
                        ids: List[int]) -> Optional[float]:
        """Extract mean green-channel value inside the convex hull of landmarks."""
        h, w = rgb.shape[:2]
        pts  = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in ids],
            dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        if mask.sum() == 0:
            return None
        return float(rgb[:, :, 1][mask > 0].mean())

    def _roi_polygon_pts(self, rgb: np.ndarray, lm,
                         ids: List[int]) -> Optional[np.ndarray]:
        """Return convex hull points for ROI overlay drawing."""
        h, w = rgb.shape[:2]
        pts  = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in ids],
            dtype=np.int32)
        hull = cv2.convexHull(pts)
        return hull

    def _bandpass_filter(self, signal_arr: np.ndarray) -> np.ndarray:
        """Zero-mean then bandpass filter. Returns filtered array."""
        s = signal_arr - signal_arr.mean()
        if len(s) < 15:
            return s
        try:
            return sp_signal.filtfilt(self._b, self._a, s)
        except Exception:
            return s

    def _fft_hr(self, fused: np.ndarray, n: int) -> Tuple[float, float]:
        """
        Compute HR from FFT of fused signal.
        Returns (hr_bpm, snr) where snr = peak_psd / total_band_psd.
        """
        freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.cam_fs)
        psd   = np.abs(np.fft.rfft(fused)) ** 2
        band  = ((freqs >= self.cfg.rppg_hr_min_hz) &
                 (freqs <= self.cfg.rppg_hr_max_hz))
        if not band.any():
            return 0.0, 0.0
        band_psd  = psd[band]
        peak_i    = int(np.argmax(band_psd))
        hr_bpm    = float(freqs[band][peak_i]) * 60.0
        total_psd = float(band_psd.sum())
        snr       = float(band_psd[peak_i]) / max(total_psd, 1e-12)
        return hr_bpm, float(np.clip(snr, 0.0, 1.0))

    def _chrom_fuse(self, fore: np.ndarray, cl: np.ndarray,
                    cr: np.ndarray) -> np.ndarray:
        """
        CHROM-inspired fusion: weight each ROI signal by inverse variance.
        Higher-variance (noisier) channels contribute less.
        Falls back to mean if all variances are near zero.
        """
        signals = [fore, cl, cr]
        vars_   = np.array([float(np.var(s)) if len(s) > 1 else 1.0
                             for s in signals])
        if vars_.sum() < 1e-12:
            # All signals identical or flat — equal weight
            weights = np.ones(3) / 3.0
        else:
            inv_var = 1.0 / (vars_ + 1e-12)
            weights = inv_var / inv_var.sum()

        # Align lengths to shortest
        min_len = min(len(s) for s in signals)
        if min_len < 2:
            return np.zeros(1)
        stacked = np.stack([s[-min_len:] for s in signals], axis=0)
        fused   = (stacked * weights[:, np.newaxis]).sum(axis=0)
        return fused

    # ── Public interface ─────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray,
                      timestamp: Optional[float] = None,
                      draw_rois: bool = False) -> Tuple[float, float, np.ndarray]:
        """
        Process one camera frame.

        Args:
            frame_bgr:  Raw BGR frame from cv2.VideoCapture.read()
            timestamp:  Capture timestamp (seconds). Uses time.time() if ≤ 0.
            draw_rois:  If True, annotate the frame with ROI overlays.

        Returns:
            (hr_bpm, snr, annotated_frame_bgr)
            hr_bpm and snr are 0.0 until warmup completes or face lost.
        """
        if timestamp is None or timestamp <= 0.0:
            timestamp = time.time()

        annotated = frame_bgr.copy() if draw_rois else frame_bgr

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # If mediapipe mesh unavailable, skip rPPG processing
        if not getattr(self, '_mesh_available', False) or self._mesh is None:
            return self._last_hr, 0.0, annotated

        res = self._mesh.process(rgb)

        if not getattr(res, 'multi_face_landmarks', None):
            with self._lock:
                self._face_det = False
            return self._last_hr, 0.0, annotated

        lm = res.multi_face_landmarks[0].landmark

        # ── Frame gap detection ──────────────────────────────────────────────
        if self._frame_times:
            gap      = timestamp - self._frame_times[-1]
            expected = 1.0 / self.cfg.cam_fs
            if gap > expected * self.cfg.rppg_gap_factor:
                logger.warning(f"rPPG: frame gap {gap*1000:.0f} ms → buffer reset")
                self._fore_buf.clear()
                self._cl_buf.clear()
                self._cr_buf.clear()
                self._frame_times.clear()
                with self._lock:
                    self._last_hr  = 0.0
                    self._last_snr = 0.0
                    self._face_det = False
                return 0.0, 0.0, annotated

        self._frame_times.append(timestamp)

        # ── Extract per-ROI green means ──────────────────────────────────────
        fore_g = self._roi_mean_green(rgb, lm, self.FOREHEAD_IDS)
        cl_g   = self._roi_mean_green(rgb, lm, self.CHEEK_L_IDS)
        cr_g   = self._roi_mean_green(rgb, lm, self.CHEEK_R_IDS)

        if fore_g is None and cl_g is None and cr_g is None:
            with self._lock:
                self._face_det = False
            return self._last_hr, 0.0, annotated

        # Use available values; substitute mean of others if one ROI is missing
        vals   = [v for v in (fore_g, cl_g, cr_g) if v is not None]
        mean_g = float(np.mean(vals))
        fore_g = fore_g if fore_g is not None else mean_g
        cl_g   = cl_g   if cl_g   is not None else mean_g
        cr_g   = cr_g   if cr_g   is not None else mean_g

        self._fore_buf.append(fore_g)
        self._cl_buf.append(cl_g)
        self._cr_buf.append(cr_g)
        self._fore_wav.append(fore_g)
        self._cl_wav.append(cl_g)
        self._cr_wav.append(cr_g)

        with self._lock:
            self._fore_val = fore_g
            self._cl_val   = cl_g
            self._cr_val   = cr_g
            self._face_det = True

        # ── Draw ROI overlays ────────────────────────────────────────────────
        if draw_rois:
            for ids, colour in (
                (self.FOREHEAD_IDS, (0,  220,  0)),    # green
                (self.CHEEK_L_IDS,  (220, 100, 0)),    # blue-ish
                (self.CHEEK_R_IDS,  (220, 100, 0)),
            ):
                hull = self._roi_polygon_pts(rgb, lm, ids)
                if hull is not None:
                    cv2.polylines(annotated, [hull], True, colour, 2)
                    # Semi-transparent fill
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [hull], colour)
                    cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # ── Check warmup ────────────────────────────────────────────────────
        n_min = int(self.cfg.cam_fs * self.cfg.rppg_warmup_seconds)
        n     = min(len(self._fore_buf), len(self._cl_buf), len(self._cr_buf))
        if n < n_min:
            return 0.0, 0.0, annotated

        # ── Bandpass filter each ROI ─────────────────────────────────────────
        fore_f = self._bandpass_filter(np.array(list(self._fore_buf)))
        cl_f   = self._bandpass_filter(np.array(list(self._cl_buf)))
        cr_f   = self._bandpass_filter(np.array(list(self._cr_buf)))

        # ── CHROM-inspired fusion ────────────────────────────────────────────
        fused  = self._chrom_fuse(fore_f, cl_f, cr_f)
        if len(fused) < n_min:
            return self._last_hr, self._last_snr, annotated

        self._fused_wav.append(float(fused[-1]))

        # ── FFT HR estimation ────────────────────────────────────────────────
        hr_bpm, snr = self._fft_hr(fused, len(fused))

        if snr < self.cfg.rppg_min_snr:
            # Signal too noisy — keep previous estimate
            return self._last_hr, self._last_snr, annotated

        with self._lock:
            self._last_hr  = hr_bpm
            self._last_snr = snr

        return hr_bpm, snr, annotated

    def get_state(self) -> RPPGState:
        """Thread-safe snapshot of all rPPG state for HUD / pipeline."""
        with self._lock:
            return RPPGState(
                hr_bpm         = self._last_hr,
                snr            = self._last_snr,
                face_detected  = self._face_det,
                forehead_signal= list(self._fore_wav),
                cheek_l_signal = list(self._cl_wav),
                cheek_r_signal = list(self._cr_wav),
                fused_signal   = list(self._fused_wav),
                forehead_value = self._fore_val,
                cheek_l_value  = self._cl_val,
                cheek_r_value  = self._cr_val,
                timestamp      = time.time(),
            )

    def as_signal(self) -> Optional[PhysioSignal]:
        """Emit as PhysioSignal for the SignalProcessor if SNR is sufficient."""
        with self._lock:
            hr  = self._last_hr
            snr = self._last_snr
        if hr > 0 and snr >= self.cfg.rppg_min_snr:
            return PhysioSignal(
                SignalType.RPPG, hr, time.time(), snr, "rPPG")
        return None

    def close(self) -> None:
        self._mesh.close()


# ============================================================================
# 9.  SIGNAL PROCESSOR
# ============================================================================

class SignalProcessor:
    """
    Fuses wearable + rPPG signals.
    Extracts HR, HRV, RSA, EDA (SCL/SCR), respiration rate.
    Baseline is locked from real data only.
    """

    def __init__(self, cfg: PMConfig):
        self.cfg = cfg
        mb = int(cfg.window_seconds * cfg.ble_fs)
        mc = int(cfg.window_seconds * cfg.cam_fs)

        self.hr_buf:   deque[float] = deque(maxlen=mb)
        self.hrv_buf:  deque[float] = deque(maxlen=mb)
        self.spo2_buf: deque[float] = deque(maxlen=mb)
        self.rppg_buf: deque[float] = deque(maxlen=mc)
        self.emg_buf:  deque[float] = deque(maxlen=mb)
        self.eda_buf:  deque[float] = deque(maxlen=mb)
        self.temp_buf: deque[float] = deque(maxlen=mb)
        self.qual_buf: deque[float] = deque(maxlen=mb)

        self.baseline_hr:         Optional[float] = None
        self.baseline_spo2:       Optional[float] = None
        self.baseline_hrv_rmssd:  Optional[float] = None

    def add_signal(self, sig: PhysioSignal) -> None:
        self.qual_buf.append(sig.confidence)
        buf_map = {
            SignalType.HR:   self.hr_buf,
            SignalType.HRV:  self.hrv_buf,
            SignalType.SPO2: self.spo2_buf,
            SignalType.RPPG: self.rppg_buf,
            SignalType.EMG:  self.emg_buf,
            SignalType.EDA:  self.eda_buf,
            SignalType.TEMP: self.temp_buf,
        }
        buf = buf_map.get(sig.signal_type)
        if buf is not None:
            buf.append(float(sig.value))

    @staticmethod
    def _rmssd(rr_ms: np.ndarray) -> float:
        if len(rr_ms) < 2:
            return 0.0
        return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

    def _extract_rsa(self, hr_array: np.ndarray) -> Tuple[float, float, float]:
        cfg = self.cfg
        if len(hr_array) < 20:
            return 0.0, 0.0, 0.0
        rr    = 60_000.0 / np.clip(hr_array, 1, 300)
        n_uni = max(20, int(len(rr) * cfg.rsa_interp_fs))
        rr_uni = np.interp(
            np.linspace(0, len(rr) - 1, n_uni),
            np.arange(len(rr)), rr)
        freqs = np.fft.rfftfreq(len(rr_uni), d=1.0 / cfg.rsa_interp_fs)
        psd   = np.abs(np.fft.rfft(rr_uni)) ** 2
        band  = (freqs >= cfg.rsa_band_lo_hz) & (freqs <= cfg.rsa_band_hi_hz)
        rsa_power = float(psd[band].sum())
        rsa_index = rsa_power / (float(psd.sum()) + 1e-9)
        resp_freq = float(freqs[band][np.argmax(psd[band])]) if band.any() else 0.0
        return rsa_power, rsa_index, resp_freq

    def _estimate_respiration_rate(self, hr_array: np.ndarray) -> float:
        cfg = self.cfg
        if len(hr_array) < 30:
            return 0.0
        rr    = 60_000.0 / np.clip(hr_array, 1, 300)
        freqs = np.fft.rfftfreq(len(rr), d=1.0 / cfg.ble_fs)
        psd   = np.abs(np.fft.rfft(rr - rr.mean())) ** 2
        band  = (freqs >= cfg.resp_band_lo_hz) & (freqs <= cfg.resp_band_hi_hz)
        if not band.any():
            return 0.0
        return float(freqs[band][np.argmax(psd[band])]) * 60.0

    def _decompose_eda(self, eda: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        if len(eda) < 4:
            return {"scl_mean": 0.0, "scl_slope": 0.0,
                    "scr_freq": 0.0, "scr_amp": 0.0}
        nyq  = cfg.ble_fs / 2.0
        b, a = sp_signal.butter(2, cfg.eda_scr_hp_hz / nyq, btype="high")
        scr  = sp_signal.filtfilt(b, a, eda)
        scl  = eda - scr
        t    = np.arange(len(scl))
        scl_slope = float(np.polyfit(t, scl, 1)[0])
        peaks, props = sp_signal.find_peaks(
            scr,
            height=np.std(scr) * cfg.scr_peak_std_k,
            distance=int(cfg.ble_fs * cfg.scr_min_dist_s),
        )
        duration_min = len(eda) / cfg.ble_fs / 60.0
        scr_freq = float(len(peaks) / max(duration_min, 1e-6))
        scr_amp  = float(np.mean(props.get("peak_heights", [0.0])))
        return {
            "scl_mean":  float(np.mean(scl)),
            "scl_slope": scl_slope,
            "scr_freq":  scr_freq,
            "scr_amp":   scr_amp,
        }

    def extract_features(self) -> PhysioFeatures:
        f   = PhysioFeatures()
        cfg = self.cfg

        if len(self.spo2_buf) >= 4:
            s               = np.array(list(self.spo2_buf))
            f.spo2_mean     = float(np.mean(s))
            f.spo2_variance = float(np.var(s))
            f.spo2_delta    = float(s[-1] - s[0])
            f.oxygen_stress = float(np.clip((95.0 - f.spo2_mean) / 5.0, 0.0, 1.0))

        if len(self.hr_buf) >= 4:
            h              = np.array(list(self.hr_buf))
            f.hr_mean      = float(np.mean(h))
            f.hr_variance  = float(np.var(h))
            f.hr_elevation = max(0.0,
                f.hr_mean - (self.baseline_hr or f.hr_mean))
            rr = 60_000.0 / np.clip(h, 1.0, 300.0)
            f.hrv_rmssd    = self._rmssd(rr)
            f.hrv_sdnn     = float(np.std(rr))
            (f.rsa_power,
             f.rsa_index,
             f.respiration_freq) = self._extract_rsa(h)
            f.respiration_rate  = self._estimate_respiration_rate(h)

        if len(self.rppg_buf) >= 4:
            r             = np.array(list(self.rppg_buf))
            f.rppg_hr     = float(np.median(r))
            hr_range      = cfg.hr_max_bpm - cfg.hr_min_bpm
            f.rppg_quality = float(np.clip(
                1.0 - np.std(r) / (hr_range * 0.15), 0.0, 1.0))

        if len(self.emg_buf) >= 4:
            e               = np.array(list(self.emg_buf))
            mid             = len(e) // 2
            f.emg_zygo_mean = float(np.mean(np.abs(e[:mid])))
            f.emg_corr_mean = float(np.mean(np.abs(e[mid:])))
            f.emg_asym      = float(abs(f.emg_zygo_mean - f.emg_corr_mean))
            f.muscle_tension = float(np.clip(
                np.sqrt(np.mean(e ** 2)) / 100.0, 0.0, 1.0))

        if len(self.eda_buf) >= 4:
            e      = np.array(list(self.eda_buf))
            eda_d  = self._decompose_eda(e)
            f.eda_mean     = float(np.mean(e))
            f.eda_variance = float(np.var(e))
            f.scl_mean     = eda_d["scl_mean"]
            f.scl_slope    = eda_d["scl_slope"]
            f.scr_freq     = eda_d["scr_freq"]
            f.scr_amp      = eda_d["scr_amp"]

        if len(self.temp_buf) >= 2:
            t            = np.array(list(self.temp_buf))
            f.temp_mean  = float(np.mean(t))
            f.temp_delta = float(t[-1] - t[0])

        arousal_parts: List[float] = []
        if f.hr_mean > 0:
            arousal_parts.append(float(np.clip(
                (f.hr_mean - 60.0) / 40.0, 0.0, 1.0)))
        if f.eda_variance > 0:
            arousal_parts.append(float(np.clip(
                f.eda_variance / 2.0, 0.0, 1.0)))
        if f.oxygen_stress > 0:
            arousal_parts.append(f.oxygen_stress * 0.5)
        f.arousal_level = float(np.mean(arousal_parts)) if arousal_parts else 0.0

        return f

    def baseline_ready(self) -> bool:
        target = int(self.cfg.calib_seconds * self.cfg.ble_fs)
        return (len(self.hr_buf) >= target and
                len(self.spo2_buf) >= max(4, target // 2))

    def lock_baseline(self) -> bool:
        cfg = self.cfg
        if not self.baseline_ready():
            return False
        hr_arr    = np.array(list(self.hr_buf))
        hr_med    = float(np.median(hr_arr))
        hr_mad    = float(np.median(np.abs(hr_arr - hr_med))) + 1e-6
        valid_hr  = hr_arr[np.abs(hr_arr - hr_med) < cfg.baseline_hr_mad_k * hr_mad]
        if len(valid_hr) < 5:
            logger.error("HR baseline: too many outliers")
            return False
        self.baseline_hr = float(np.mean(valid_hr))
        if not (cfg.baseline_hr_lo <= self.baseline_hr <= cfg.baseline_hr_hi):
            logger.warning(f"Unusual baseline HR: {self.baseline_hr:.1f} bpm")

        s_arr    = np.array(list(self.spo2_buf))
        s_med    = float(np.median(s_arr))
        s_mad    = float(np.median(np.abs(s_arr - s_med))) + 1e-6
        valid_s  = s_arr[np.abs(s_arr - s_med) < cfg.baseline_spo2_mad_k * s_mad]
        self.baseline_spo2 = float(np.mean(valid_s)) if len(valid_s) >= 4 else s_med
        if self.baseline_spo2 < cfg.baseline_spo2_warn:
            logger.warning(f"Low baseline SpO2: {self.baseline_spo2:.1f}%")

        rr = 60_000.0 / np.clip(hr_arr, 1, 300)
        self.baseline_hrv_rmssd = SignalProcessor._rmssd(rr)

        logger.info(
            f"Baseline locked → HR={self.baseline_hr:.1f} bpm  "
            f"SpO2={self.baseline_spo2:.1f}%  "
            f"HRV-RMSSD={self.baseline_hrv_rmssd:.1f} ms")
        return True

    @property
    def mean_quality(self) -> float:
        return float(np.mean(self.qual_buf)) if self.qual_buf else 0.0


# ============================================================================
# 10.  PHYSIOLOGICAL CLASSIFIER
# ============================================================================

class PhysioClassifier:
    """
    Rule-based stress/emotion + IsolationForest anomaly classifier.
    All thresholds driven by PMConfig.
    Returns neutral / zero until baseline is set from real data.
    """

    def __init__(self, cfg: PMConfig):
        self.cfg      = cfg
        self._baseline: Optional[PhysioFeatures] = None
        self._iforest = IsolationForest(
            n_estimators=cfg.iforest_estimators,
            contamination=cfg.iforest_contamination,
            random_state=42,
        )
        self._history:  deque[List[float]] = deque(maxlen=cfg.iforest_max_history)
        self._trained   = False
        self._n_updates = 0

    def set_baseline(self, f: PhysioFeatures) -> None:
        if not f.is_valid():
            logger.warning("set_baseline: features not yet valid — skipping")
            return
        if self._baseline is None:
            self._baseline = replace(f)
            logger.info(
                f"Classifier baseline → HR={f.hr_mean:.1f}  "
                f"SpO2={f.spo2_mean:.1f}  rPPG-HR={f.rppg_hr:.1f}  "
                f"RMSSD={f.hrv_rmssd:.1f}")

    def _fvec(self, f: PhysioFeatures) -> List[float]:
        return [
            f.spo2_mean, f.spo2_variance,
            f.hr_mean, f.hr_variance, f.hr_elevation,
            f.hrv_rmssd, f.hrv_sdnn,
            f.emg_zygo_mean, f.emg_corr_mean, f.emg_asym,
            f.eda_mean, f.eda_variance, f.scr_freq,
            f.muscle_tension, f.arousal_level, f.oxygen_stress,
            f.rppg_hr, f.rppg_snr,
        ]

    def _anomaly(self, vec: List[float]) -> float:
        self._history.append(vec)
        self._n_updates += 1
        min_s = self.cfg.iforest_min_samples
        if len(self._history) >= min_s and (
                not self._trained or
                self._n_updates % self.cfg.iforest_retrain_every == 0):
            recent = list(self._history)[-500:]
            if len(recent) >= min_s:
                try:
                    self._iforest.fit(np.array(recent))
                    self._trained = True
                except Exception as exc:
                    logger.error(f"IsolationForest fit: {exc}")
                    return 0.0
        if not self._trained:
            return 0.0
        try:
            score = self._iforest.score_samples(
                np.array(vec).reshape(1, -1))[0]
            return float(np.clip(-score, 0.0, 1.0))
        except Exception as exc:
            logger.error(f"IsolationForest score: {exc}")
            return 0.0

    def classify(
        self, f: PhysioFeatures
    ) -> Tuple[EmotionEnum, float, float, float, float]:
        if not f.is_valid():
            return EmotionEnum.NEUTRAL, 0.0, 0.0, 0.0, 0.0

        cfg      = self.cfg
        baseline = self._baseline
        anomaly  = self._anomaly(self._fvec(f))

        # ── Stress ───────────────────────────────────────────────────────────
        sp: List[float] = []
        if f.spo2_mean > 0 and baseline and baseline.spo2_mean > 0:
            delta = baseline.spo2_mean - f.spo2_mean
            sp.append(float(np.clip(max(0, delta) / cfg.spo2_stress_delta,
                                    0.0, 1.0)))
        if f.hr_mean > 0 and baseline and baseline.hr_mean > 0:
            delta     = f.hr_mean - baseline.hr_mean
            threshold = baseline.hr_mean * cfg.hr_stress_threshold_frac
            sp.append(float(np.clip(max(0, delta) / max(threshold, 1.0),
                                    0.0, 1.0)))
        if (f.hrv_rmssd > 0 and baseline and
                baseline.hrv_rmssd is not None and baseline.hrv_rmssd > 0):
            sp.append(float(np.clip(
                1.0 - f.hrv_rmssd / baseline.hrv_rmssd, 0.0, 1.0)))
        if f.eda_variance > 0:
            sp.append(float(np.clip(f.eda_variance / 2.0, 0.0, 1.0)))
        stress_prob = float(np.mean(sp)) if sp else 0.0

        # ── Emotion ──────────────────────────────────────────────────────────
        c = cfg
        if (f.hr_mean > c.emo_fearful_hr
                and f.oxygen_stress > c.emo_fearful_oxy
                and f.arousal_level > c.emo_fearful_arousal):
            emotion = EmotionEnum.FEARFUL
        elif (f.hr_mean > c.emo_angry_hr
              and f.muscle_tension > c.emo_angry_tension
              and f.eda_variance > c.emo_angry_eda_var):
            emotion = EmotionEnum.ANGRY
        elif (f.hr_mean > c.emo_disgust_hr
              and f.emg_zygo_mean < c.emo_disgust_zygo
              and f.eda_variance > c.emo_disgust_eda_var):
            emotion = EmotionEnum.DISGUSTED
        elif (f.arousal_level < c.emo_sad_arousal
              and f.muscle_tension < c.emo_sad_tension
              and f.hr_mean < c.emo_sad_hr):
            emotion = EmotionEnum.SAD
        elif (stress_prob < c.emo_happy_stress
              and f.spo2_mean > c.emo_happy_spo2
              and f.hr_mean < c.emo_happy_hr):
            emotion = EmotionEnum.HAPPY
        else:
            emotion = EmotionEnum.NEUTRAL

        # ── Deception ────────────────────────────────────────────────────────
        dp: List[float] = []
        if f.hr_variance > 0 and baseline and baseline.hr_variance > 0:
            var_ratio = f.hr_variance / (baseline.hr_variance + 1e-6)
            dp.append(float(np.clip((var_ratio - 1.0) / 0.5, 0.0, 0.7)))
        if (f.hrv_rmssd > 0 and baseline and
                baseline.hrv_rmssd is not None and baseline.hrv_rmssd > 0):
            hrv_ratio = f.hrv_rmssd / baseline.hrv_rmssd
            dp.append(float(np.clip(1.0 - hrv_ratio, 0.0, 0.7)))
        if f.scr_freq > cfg.scr_deception_min:
            dp.append(float(np.clip(
                (f.scr_freq - cfg.scr_deception_min) / cfg.scr_deception_scale,
                0.0, 0.6)))
        if anomaly > cfg.anomaly_deception_thresh:
            dp.append(anomaly * 0.5)
        deception_prob = float(np.mean(dp)) if dp else 0.0

        # ── Cognitive load ───────────────────────────────────────────────────
        components: List[float] = []
        if stress_prob > 0:
            components.append(stress_prob ** cfg.cog_stress_exp)
        if f.arousal_level > 0:
            components.append(f.arousal_level ** cfg.cog_arousal_exp)
        if (baseline and baseline.hrv_rmssd is not None
                and baseline.hrv_rmssd > 0 and f.hrv_rmssd > 0):
            components.append(
                max(0.0, 1.0 - f.hrv_rmssd / baseline.hrv_rmssd))
        components = [c for c in components if c > cfg.cog_component_min]
        if len(components) >= 2:
            cog_load = float(np.power(np.prod(components), 1.0 / len(components)))
        else:
            cog_load = float(np.mean(components)) if components else 0.0
        cog_load = float(np.clip(cog_load, 0.0, 1.0))

        return emotion, stress_prob, deception_prob, cog_load, anomaly


# ============================================================================
# 11.  HUD RENDERER
# ============================================================================

class HUDRenderer:
    """
    Renders the live OpenCV HUD window.

    Layout (hud_width × hud_height):
    ┌─────────────────────────┬──────────────────────────────────┐
    │  Camera feed            │  WEARABLE panel                  │
    │  + ROI overlays         │  rPPG panel                      │
    │  (hud_cam_width wide)   │  WAVEFORM (ECG-style)            │
    │                         │  VITALS bars                     │
    └─────────────────────────┴──────────────────────────────────┘

    No data is generated here — it only visualises what PMSystem provides.
    """

    # Colour palette (BGR)
    C_BG      = (18,  18,  18)
    C_GREEN   = (0,   200, 80)
    C_CYAN    = (0,   220, 220)
    C_YELLOW  = (0,   220, 180)
    C_RED     = (60,  60,  220)
    C_BLUE    = (220, 140, 40)
    C_ORANGE  = (0,   140, 255)
    C_WHITE   = (230, 230, 230)
    C_GREY    = (100, 100, 100)
    C_PANEL   = (28,  28,  28)
    C_DIVIDER = (60,  60,  60)

    def __init__(self, cfg: PMConfig):
        self.cfg       = cfg
        self.W         = cfg.hud_width
        self.H         = cfg.hud_height
        self.CAM_W     = cfg.hud_cam_width
        self.PANEL_X   = self.CAM_W + 4
        self.PANEL_W   = self.W - self.PANEL_X

    def _blank(self) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        canvas[:] = self.C_BG
        # Panel background
        canvas[:, self.CAM_W:self.CAM_W+2] = self.C_DIVIDER
        canvas[:, self.PANEL_X:] = self.C_PANEL
        return canvas

    @staticmethod
    def _put(img: np.ndarray, text: str, x: int, y: int,
             colour=(230, 230, 230), scale: float = 0.55,
             thickness: int = 1) -> None:
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness,
                    cv2.LINE_AA)

    def _bar(self, img: np.ndarray, x: int, y: int, w: int, h: int,
             value: float, colour, label: str = "", show_pct: bool = True) -> None:
        """Draw a filled progress bar."""
        value = float(np.clip(value, 0.0, 1.0))
        cv2.rectangle(img, (x, y), (x + w, y + h), self.C_GREY, 1)
        fill = int(value * (w - 2))
        if fill > 0:
            cv2.rectangle(img, (x + 1, y + 1),
                          (x + 1 + fill, y + h - 1), colour, -1)
        if label:
            self._put(img, label, x, y - 4, self.C_WHITE, 0.42)
        if show_pct:
            self._put(img, f"{value*100:.0f}%",
                      x + w + 4, y + h - 2, colour, 0.42)

    def _waveform(self, img: np.ndarray, signal: List[float],
                  x: int, y: int, w: int, h: int,
                  colour, label: str = "") -> None:
        """Draw ECG-style rolling waveform."""
        if len(signal) < 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.C_GREY, 1)
            if label:
                self._put(img, label, x + 2, y + 14, self.C_GREY, 0.42)
            return
        sig = np.array(signal, dtype=np.float64)
        mn, mx = sig.min(), sig.max()
        span = mx - mn if (mx - mn) > 1e-6 else 1.0
        norm = (sig - mn) / span

        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.C_GREY, 1)

        n     = len(norm)
        xs    = np.linspace(x, x + w - 1, n).astype(int)
        ys    = (y + h - 1 - norm * (h - 2)).astype(int)
        ys    = np.clip(ys, y, y + h - 1)

        for i in range(1, n):
            cv2.line(img, (xs[i-1], ys[i-1]), (xs[i], ys[i]), colour, 1,
                     cv2.LINE_AA)

        if label:
            self._put(img, label, x + 4, y + 14, colour, 0.40)

    def render(self,
               cam_frame:    Optional[np.ndarray],
               pm_output:    Optional[PMOutput],
               rppg_state:   Optional[RPPGState],
               ble_quality:  float,
               calibrating:  bool) -> np.ndarray:
        """
        Compose one full HUD frame.

        Args:
            cam_frame:   Annotated BGR camera frame (with ROI overlays).
                         None → blank left panel.
            pm_output:   Latest PMOutput (or None during calibration).
            rppg_state:  Latest RPPGState (or None).
            ble_quality: Wearable signal quality 0–1.
            calibrating: True while baseline not yet locked.
        """
        canvas = self._blank()
        px     = self.PANEL_X + 10   # right panel text X origin
        W      = self.PANEL_W - 20   # usable right panel width

        # ── Left panel: camera ───────────────────────────────────────────────
        if cam_frame is not None:
            try:
                resized = cv2.resize(cam_frame, (self.CAM_W, self.H))
                canvas[:, :self.CAM_W] = resized
            except Exception:
                pass

        # ── Status banner ────────────────────────────────────────────────────
        status_txt = "CALIBRATING..." if calibrating else "LIVE"
        status_col = self.C_YELLOW if calibrating else self.C_GREEN
        self._put(canvas, f"PHYSIO V3  |  {status_txt}",
                  px, 24, status_col, 0.65, 2)
        cv2.line(canvas, (self.PANEL_X, 32),
                 (self.W, 32), self.C_DIVIDER, 1)

        y = 52

        
        # SECTION A — WEARABLE
        
        self._put(canvas, "WEARABLE  (BLE)", px, y, self.C_CYAN, 0.55, 1)
        y += 4
        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 16

        hr_ble   = pm_output.heart_rate  if pm_output else 0.0
        spo2_ble = pm_output.oxygen_level if pm_output else 0.0
        hrv_ble  = pm_output.hrv_rmssd   if pm_output else 0.0

        # HR value — large
        hr_txt   = f"{hr_ble:.1f}" if hr_ble > 0 else "--.-"
        self._put(canvas, "HR", px, y + 20, self.C_GREY, 0.45)
        self._put(canvas, hr_txt + " bpm", px + 30, y + 20,
                  self.C_GREEN, 0.80, 2)

        # SpO2
        spo2_txt = f"{spo2_ble:.1f}" if spo2_ble > 0 else "--.-"
        self._put(canvas, "SpO2", px + W // 2, y + 20, self.C_GREY, 0.45)
        spo2_col = self.C_GREEN if spo2_ble >= 95 else (
                   self.C_YELLOW if spo2_ble >= 90 else self.C_RED)
        self._put(canvas, spo2_txt + " %", px + W // 2 + 38, y + 20,
                  spo2_col, 0.72, 1)
        y += 30

        # HRV
        hrv_txt  = f"{hrv_ble:.1f} ms" if hrv_ble > 0 else "-- ms"
        self._put(canvas, f"HRV-RMSSD  {hrv_txt}", px, y, self.C_WHITE, 0.48)
        y += 18

        # BLE quality bar
        self._bar(canvas, px, y, W - 50, 10, ble_quality,
                  self.C_CYAN, label="BLE quality")
        y += 26

        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 10

        
        # SECTION B — rPPG
        
        self._put(canvas, "rPPG  (CAMERA)", px, y, self.C_ORANGE, 0.55, 1)
        y += 4
        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 16

        if rppg_state:
            face_col = self.C_GREEN if rppg_state.face_detected else self.C_RED
            face_txt = "FACE DETECTED" if rppg_state.face_detected else "NO FACE"
            self._put(canvas, face_txt, px, y, face_col, 0.50)
            y += 18

            rppg_hr_txt = (f"{rppg_state.hr_bpm:.1f} bpm"
                           if rppg_state.hr_bpm > 0 else "-- bpm")
            self._put(canvas, "rPPG HR", px, y + 18, self.C_GREY, 0.45)
            self._put(canvas, rppg_hr_txt, px + 58, y + 18,
                      self.C_ORANGE, 0.72, 1)

            snr_txt = f"{rppg_state.snr*100:.1f}%"
            self._put(canvas, "SNR", px + W // 2, y + 18, self.C_GREY, 0.45)
            snr_col = (self.C_GREEN if rppg_state.snr > 0.2 else
                       self.C_YELLOW if rppg_state.snr > 0.08 else self.C_RED)
            self._put(canvas, snr_txt, px + W // 2 + 30, y + 18,
                      snr_col, 0.60)
            y += 28

            # Per-ROI raw green values
            self._put(canvas,
                f"Fore: {rppg_state.forehead_value:.1f}  "
                f"CkL: {rppg_state.cheek_l_value:.1f}  "
                f"CkR: {rppg_state.cheek_r_value:.1f}",
                px, y, self.C_GREY, 0.42)
            y += 16

            # SNR quality bar
            self._bar(canvas, px, y, W - 50, 10, rppg_state.snr,
                      self.C_ORANGE, label="rPPG SNR")
            y += 26
        else:
            self._put(canvas, "Camera not active", px, y, self.C_GREY, 0.48)
            y += 40

        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 10

        
        # SECTION C — WAVEFORM (ECG-style)
        
        self._put(canvas, "WAVEFORM", px, y, self.C_WHITE, 0.50, 1)
        y += 6

        wav_h   = 72
        wav_top = y

        if rppg_state and len(rppg_state.fused_signal) > 4:
            self._waveform(canvas, rppg_state.fused_signal,
                           px, wav_top, W, wav_h,
                           self.C_GREEN, "Fused rPPG")
        elif rppg_state and len(rppg_state.forehead_signal) > 4:
            self._waveform(canvas, rppg_state.forehead_signal,
                           px, wav_top, W, wav_h,
                           self.C_ORANGE, "Forehead (raw)")
        else:
            cv2.rectangle(canvas, (px, wav_top),
                          (px + W, wav_top + wav_h), (40, 40, 40), -1)
            cv2.rectangle(canvas, (px, wav_top),
                          (px + W, wav_top + wav_h), self.C_GREY, 1)
            self._put(canvas, "Awaiting signal...",
                      px + W // 4, wav_top + wav_h // 2,
                      self.C_GREY, 0.48)

        y = wav_top + wav_h + 6

        # Per-ROI mini waveforms side by side
        mini_h = 36
        mini_w = (W - 4) // 3
        if rppg_state:
            for i, (sig, col, lbl) in enumerate([
                (rppg_state.forehead_signal, self.C_GREEN,  "Fore"),
                (rppg_state.cheek_l_signal,  self.C_BLUE,   "Ck-L"),
                (rppg_state.cheek_r_signal,  self.C_BLUE,   "Ck-R"),
            ]):
                mx = px + i * (mini_w + 2)
                self._waveform(canvas, sig, mx, y, mini_w, mini_h,
                               col, lbl)
        y += mini_h + 8

        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 10

        
        # SECTION D — VITALS / INFERENCE
        
        self._put(canvas, "VITALS & INFERENCE", px, y, self.C_WHITE, 0.50, 1)
        y += 4
        cv2.line(canvas, (px, y), (px + W, y), self.C_DIVIDER, 1)
        y += 14

        if pm_output and not calibrating:
            # Fused HR (wearable if available, else rPPG)
            fused_hr = pm_output.heart_rate if pm_output.heart_rate > 0 else pm_output.rppg_hr
            fused_txt = f"{fused_hr:.1f} bpm" if fused_hr > 0 else "--"
            self._put(canvas, f"Fused HR: {fused_txt}", px, y,
                      self.C_GREEN, 0.55, 1)
            y += 20

            # SpO2
            spo2_val = pm_output.oxygen_level
            spo2_col = (self.C_GREEN if spo2_val >= 95 else
                        self.C_YELLOW if spo2_val >= 90 else self.C_RED)
            spo2_disp = f"{spo2_val:.1f} %" if spo2_val > 0 else "--"
            self._put(canvas, f"SpO2:     {spo2_disp}", px, y, spo2_col, 0.55)
            y += 20

            # Emotion
            emo_col = {
                EmotionEnum.HAPPY:     self.C_GREEN,
                EmotionEnum.FEARFUL:   self.C_RED,
                EmotionEnum.ANGRY:     self.C_RED,
                EmotionEnum.SAD:       self.C_BLUE,
                EmotionEnum.DISGUSTED: self.C_ORANGE,
                EmotionEnum.NEUTRAL:   self.C_WHITE,
            }.get(pm_output.emotion, self.C_WHITE)
            self._put(canvas, f"Emotion:  {pm_output.emotion.value}",
                      px, y, emo_col, 0.55)
            y += 22

            # Gauge bars
            bar_h = 12
            for label, value, colour in [
                ("Stress",    pm_output.stress_probability,    self.C_ORANGE),
                ("Deception", pm_output.deception_probability, self.C_RED),
                ("Cog.Load",  pm_output.cognitive_load,        self.C_YELLOW),
                ("Anomaly",   pm_output.anomaly_score,         self.C_BLUE),
                ("Sig.Qual",  pm_output.signal_quality,        self.C_CYAN),
            ]:
                self._bar(canvas, px, y, W - 55, bar_h,
                          value, colour, label=label)
                y += bar_h + 10

        else:
            # Calibrating state
            self._put(canvas, "Calibrating baseline...",
                      px, y + 20, self.C_YELLOW, 0.55)
            y += 40

        # ── Timestamp footer ────────────────────────────────────────────────
        ts_str = time.strftime("%H:%M:%S")
        self._put(canvas, ts_str,
                  px, self.H - 10, self.C_GREY, 0.42)

        return canvas


# ============================================================================
# 12.  PM SYSTEM
# ============================================================================

class PMSystem:
    """
    Physiological Modality System — production entry point.

    Usage:
        cfg = PMConfig(timex_mac="AA:BB:CC:DD:EE:FF", camera_id=0)
        pm  = PMSystem(cfg)
        pm.register_callback(my_ws_sender)
        pm.connect()
        pm.start()
        ...
        pm.stop()

    Callbacks receive PMOutput on every inference cycle.
    PMOutput.to_dict() / .to_json() are safe for websocket / HUD streaming.
    """

    def __init__(self, cfg: Optional[PMConfig] = None):
        self.cfg = cfg or PMConfig()
        c        = self.cfg

        # ── Select wearable source ───────────────────────────────────────────
        gb_dir = c.gadgetbridge_dir or os.environ.get("GADGETBRIDGE_DIR")
        if gb_dir:
            logger.info(f"PMSystem: Gadgetbridge → {gb_dir}")
            self._ble: object = GadgetbridgeIngest(gb_dir, c)
            self._device_name = "Gadgetbridge"
        elif c.timex_mac:
            logger.info(f"PMSystem: Timex BLE → {c.timex_mac}")
            self._ble = TimexWearableDevice(c.timex_mac, c)
            self._device_name = f"Timex({c.timex_mac})"
        else:
            raise ValueError(
                "PMConfig requires timex_mac or gadgetbridge_dir "
                "(or set GADGETBRIDGE_DIR env-var). "
                "No simulated/fake data in production mode.")

        # ── Camera / rPPG ────────────────────────────────────────────────────
        self._rppg: Optional[RPPGExtractor]    = None
        self._cap:  Optional[cv2.VideoCapture] = None
        if c.camera_id is not None:
            cap = cv2.VideoCapture(c.camera_id)
            if cap.isOpened():
                raw_fps  = cap.get(cv2.CAP_PROP_FPS)
                c.cam_fs = float(raw_fps) if raw_fps > 0 else c.cam_fs
                self._cap  = cap
                self._rppg = RPPGExtractor(c)
                self._device_name += "+rPPG"
                logger.info(f"rPPG camera {c.camera_id} @ {c.cam_fs:.0f} fps")
            else:
                logger.warning(
                    f"Camera {c.camera_id} not available — rPPG disabled")

        self._proc = SignalProcessor(c)
        self._clf  = PhysioClassifier(c)

        self._running          = False
        self._calib_done       = False
        self._last_output:    Optional[PMOutput]   = None
        self._last_rppg_state: Optional[RPPGState]  = None
        self._last_cam_frame:  Optional[np.ndarray] = None
        self._last_proc_time:  float = 0.0
        self._frame_lock = threading.Lock()

        self._proc_thread:     Optional[threading.Thread] = None
        self._cam_thread:      Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        self._callbacks: List[Callable[[PMOutput], None]] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def register_callback(self, cb: Callable[[PMOutput], None]) -> None:
        self._callbacks.append(cb)

    def connect(self) -> bool:
        return self._ble.connect()

    def start(self) -> bool:
        if not self._ble.connected:
            logger.error("PMSystem.start(): device not connected")
            return False
        self._running         = True
        self._last_proc_time  = time.time()

        self._proc_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="PM-Proc")
        self._proc_thread.start()

        if self._cap and self._rppg:
            self._cam_thread = threading.Thread(
                target=self._camera_loop, daemon=True, name="PM-Cam")
            self._cam_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="PM-Watchdog")
        self._watchdog_thread.start()

        logger.info("PM System started")
        return True

    def stop(self) -> None:
        self._running = False
        for t in (self._proc_thread, self._cam_thread, self._watchdog_thread):
            if t:
                t.join(timeout=5)
        self._ble.disconnect()
        if self._cap:
            self._cap.release()
        if self._rppg:
            self._rppg.close()
        logger.info("PM System stopped")

    def get_output(self) -> Optional[PMOutput]:
        return self._last_output

    def get_rppg_state(self) -> Optional[RPPGState]:
        return self._rppg.get_state() if self._rppg else None

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Latest annotated camera frame (thread-safe copy)."""
        with self._frame_lock:
            return self._last_cam_frame.copy() \
                if self._last_cam_frame is not None else None

    def get_ble_quality(self) -> float:
        return self._ble.mean_quality  # type: ignore[union-attr]

    def get_status(self) -> Dict:
        out = self._last_output
        return {
            "connected":   self._ble.connected,
            "calibrated":  self._calib_done,
            "rppg_active": self._rppg is not None,
            "device":      self._device_name,
            "emotion":     out.emotion.value            if out else None,
            "hr_bpm":      out.heart_rate               if out else None,
            "rppg_hr":     out.rppg_hr                  if out else None,
            "spo2_pct":    out.oxygen_level             if out else None,
            "stress":      out.stress_probability       if out else None,
            "deception":   out.deception_probability    if out else None,
            "quality":     out.signal_quality           if out else None,
        }

    # ── Internal loops ───────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        try:
            while self._running:
                # Drain wearable
                for sig in self._ble.get_signals():
                    self._proc.add_signal(sig)

                # Drain rPPG
                if self._rppg:
                    rsig = self._rppg.as_signal()
                    if rsig:
                        self._proc.add_signal(rsig)
                    self._last_rppg_state = self._rppg.get_state()

                if len(self._proc.hr_buf) < 4 and len(self._proc.rppg_buf) < 4:
                    time.sleep(self.cfg.process_interval)
                    continue

                features = self._proc.extract_features()
                self._last_proc_time = time.time()

                if not self._calib_done:
                    if self._proc.baseline_ready():
                        ok = self._proc.lock_baseline()
                        if ok:
                            self._clf.set_baseline(features)
                            self._calib_done = True
                            logger.info("PM calibration complete")
                        else:
                            logger.warning("Baseline lock failed — retrying")
                    time.sleep(self.cfg.process_interval)
                    continue

                emotion, stress, deception, cog, anomaly = \
                    self._clf.classify(features)

                rs = self._last_rppg_state

                # Fused HR: prefer wearable; fall back to rPPG
                fused_hr = features.hr_mean if features.hr_mean > 0 \
                    else (rs.hr_bpm if rs else 0.0)

                out = PMOutput(
                    stress_probability    = stress,
                    emotion               = emotion,
                    deception_probability = deception,
                    cognitive_load        = cog,
                    anomaly_score         = anomaly,
                    arousal_level         = features.arousal_level,
                    oxygen_level          = features.spo2_mean,
                    heart_rate            = fused_hr,
                    hrv_rmssd             = features.hrv_rmssd,
                    rppg_hr               = rs.hr_bpm if rs else 0.0,
                    rppg_snr              = rs.snr    if rs else 0.0,
                    rppg_face_detected    = rs.face_detected if rs else False,
                    signal_quality        = self._proc.mean_quality,
                    features              = features,
                    rppg_state            = rs,
                    timestamp             = time.time(),
                    calibrated            = True,
                    device                = self._device_name,
                )
                self._last_output = out

                for cb in self._callbacks:
                    try:
                        cb(out)
                    except Exception as exc:
                        logger.error(f"PM callback error: {exc}")

                time.sleep(self.cfg.process_interval)

        except Exception as exc:
            logger.error(f"_process_loop crashed: {exc}", exc_info=True)
            self._last_proc_time = 0.0

    def _camera_loop(self) -> None:
        """Capture frames, run rPPG, store annotated frame for HUD."""
        try:
            while self._running and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue
                ts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                ts    = ts_ms / 1000.0 if ts_ms > 0 else time.time()

                _, _, annotated = self._rppg.process_frame(
                    frame, timestamp=ts, draw_rois=True)

                with self._frame_lock:
                    self._last_cam_frame = annotated

        except Exception as exc:
            logger.error(f"_camera_loop crashed: {exc}", exc_info=True)

    def _watchdog_loop(self) -> None:
        timeout = self.cfg.watchdog_timeout
        while self._running:
            time.sleep(timeout / 2)
            if not self._running:
                break
            if (self._proc_thread and not self._proc_thread.is_alive()
                    and self._running):
                logger.warning("Watchdog: restarting _process_loop")
                self._proc_thread = threading.Thread(
                    target=self._process_loop, daemon=True, name="PM-Proc")
                self._proc_thread.start()


# ============================================================================
# 13.  PIPELINE INTEGRATION HELPER
# ============================================================================

class PMPipelineAdapter:
    """
    Thin adapter so pipeline.py can treat PMSystem uniformly.

        adapter = PMPipelineAdapter(cfg)
        adapter.pm.register_callback(ws_send_func)
        adapter.start()
        output = adapter.latest()   # PMOutput or None
        adapter.stop()
    """

    def __init__(self, cfg: Optional[PMConfig] = None):
        self.pm  = PMSystem(cfg)
        self._ok = False

    def start(self) -> bool:
        if not self.pm.connect():
            logger.error("PMPipelineAdapter: connect failed")
            return False
        if not self.pm.start():
            logger.error("PMPipelineAdapter: start failed")
            return False
        self._ok = True
        return True

    def stop(self) -> None:
        self.pm.stop()
        self._ok = False

    def latest(self) -> Optional[PMOutput]:
        return self.pm.get_output() if self._ok else None

    def status(self) -> Dict:
        return self.pm.get_status()


# ============================================================================
# 14.  STANDALONE ENTRY POINT WITH LIVE HUD
# ============================================================================

def _run_hud(args) -> None:
    """
    Run PMSystem with a live OpenCV HUD window.
    Camera opens automatically when --camera is specified.
    Press 'q' to quit.
    """
    gb_dir = args.gbdir or os.environ.get("GADGETBRIDGE_DIR")
    if not args.mac and not gb_dir:
        raise SystemExit(
            "Specify --mac <BLE_MAC> or --gbdir <path> "
            "(or set GADGETBRIDGE_DIR env-var). "
            "No simulation mode.")

    cfg = PMConfig(
        timex_mac        = args.mac,
        gadgetbridge_dir = gb_dir,
        camera_id        = args.camera,
        calib_seconds    = args.calib_secs,
        hud_width        = args.hud_w,
        hud_height       = args.hud_h,
        hud_cam_width    = args.cam_panel_w,
    )

    pm  = PMSystem(cfg)
    hud = HUDRenderer(cfg)

    logger.info("Connecting to wearable...")
    if not pm.connect():
        raise SystemExit("Wearable connect failed")
    if not pm.start():
        raise SystemExit("PMSystem start failed")

    win_name = "Physio V3 — Live Monitor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cfg.hud_width, cfg.hud_height)

    logger.info(f"HUD running — press 'q' to quit  "
                f"({cfg.hud_width}x{cfg.hud_height})")

    try:
        while True:
            out        = pm.get_output()
            rs         = pm.get_rppg_state()
            cam_frame  = pm.get_camera_frame()
            ble_qual   = pm.get_ble_quality()
            calibrating = not (out and out.calibrated)

            frame = hud.render(
                cam_frame   = cam_frame,
                pm_output   = out,
                rppg_state  = rs,
                ble_quality = ble_qual,
                calibrating = calibrating,
            )

            cv2.imshow(win_name, frame)

            # Also print to console if requested
            if args.verbose and out:
                logger.info(
                    f"HR={out.heart_rate:.1f}  rPPG={out.rppg_hr:.1f}  "
                    f"SpO2={out.oxygen_level:.1f}  "
                    f"Stress={out.stress_probability:.3f}  "
                    f"Decep={out.deception_probability:.3f}  "
                    f"SNR={out.rppg_snr:.3f}")

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pm.stop()
        cv2.destroyAllWindows()
        logger.info("Physio V3 stopped.")


def _run_cli(args) -> None:
    """Headless CLI mode — prints a live table (no OpenCV window)."""
    gb_dir = args.gbdir or os.environ.get("GADGETBRIDGE_DIR")
    if not args.mac and not gb_dir:
        raise SystemExit(
            "Specify --mac <BLE_MAC> or --gbdir <path>.")

    cfg = PMConfig(
        timex_mac        = args.mac,
        gadgetbridge_dir = gb_dir,
        camera_id        = args.camera,
        calib_seconds    = args.calib_secs,
    )

    pm = PMSystem(cfg)
    if not pm.connect():
        raise SystemExit("Wearable connect failed")
    if not pm.start():
        raise SystemExit("PMSystem start failed")

    hdr = (f"{'t':>4} | {'Cal':3} | {'Emotion':10} | "
           f"{'HR':>6} | {'rPPG':>6} | {'SpO2':>5} | {'HRV':>6} | "
           f"{'SNR':>5} | {'Stress':>7} | {'Decep':>7} | {'Qual':>5}")
    print(hdr)
    print("─" * len(hdr))

    for i in range(args.secs):
        time.sleep(1)
        out = pm.get_output()
        if args.json_out and out:
            print(out.to_json())
        elif out:
            print(
                f"{i+1:4d} | {'Y':^3} | {out.emotion.value:10} | "
                f"{out.heart_rate:6.1f} | {out.rppg_hr:6.1f} | "
                f"{out.oxygen_level:5.1f} | {out.hrv_rmssd:6.1f} | "
                f"{out.rppg_snr:5.3f} | "
                f"{out.stress_probability:7.3f} | "
                f"{out.deception_probability:7.3f} | "
                f"{out.signal_quality:5.3f}"
            )
        else:
            print(f"{i+1:4d} | {'N':^3} | calibrating...")

    pm.stop()
    print("Done.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Physio V3 — Physiological Modality System")

    # Source
    ap.add_argument("--mac",          default=None,
                    help="Timex BLE MAC address")
    ap.add_argument("--gbdir",        default=None,
                    help="Gadgetbridge export directory")
    ap.add_argument("--camera",       type=int, default=None,
                    help="Camera device index (enables rPPG + HUD)")

    # Mode
    ap.add_argument("--no-hud",       action="store_true",
                    help="Headless CLI mode (no OpenCV window)")
    ap.add_argument("--secs",         type=int, default=120,
                    help="CLI mode duration in seconds")
    ap.add_argument("--json-out",     action="store_true",
                    help="CLI mode: print PMOutput as JSON lines")
    ap.add_argument("--verbose",      action="store_true",
                    help="Print live readings to console alongside HUD")

    # Calibration
    ap.add_argument("--calib-secs",   type=float, default=15.0,
                    help="Calibration window in seconds")

    # HUD geometry
    ap.add_argument("--hud-w",        type=int, default=1280,
                    help="HUD total width in pixels")
    ap.add_argument("--hud-h",        type=int, default=720,
                    help="HUD total height in pixels")
    ap.add_argument("--cam-panel-w",  type=int, default=640,
                    help="Left camera panel width in pixels")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.no_hud:
        _run_cli(args)
    else:
        _run_hud(args)