#!/usr/bin/env python3

# MM (MULTIMODAL - VIDEO) SYSTEM — PRODUCTION V7 (FIXED)
# MM = microexpression + macro expression + fatigue + forced smile + head pose + gaze + flow

import cv2
import numpy as np
import mediapipe as mp
import logging
import math
import time
import json
import datetime
from collections import deque, Counter
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
import os
from scipy.special import expit

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# OPTICAL FLOW PROCESSOR - Farneback + EVM for motion detection
# ============================================================================

class OpticalFlowProcessor:
    """
    Computes optical flow using Farneback algorithm.
    Used for motion-based micro-expression detection and apex detection.
    """
    
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                 n_poly=5, poly_n_sigma=1.2, flags=0):
        """
        Initialize Farneback optical flow parameters.
        
        Args:
            pyr_scale: Image scale (<1) for pyramid
            levels: Number of pyramid levels
            winsize: Averaging window size
            iterations: Iterations per level
            n_poly: Size of pixel neighborhood
            poly_n_sigma: Gaussian sigma for poly expansion
            flags: cv2.OPTFLOW_FARNEBACK_GAUSSIAN (0) or custom
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.n_poly = n_poly
        self.poly_n_sigma = poly_n_sigma
        self.flags = flags
        self.prev_gray = None
        self.flow_history = deque(maxlen=30)
        
    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow between current and previous frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            Optical flow field (magnitude, angle) or None if first frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,  # flow output
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.n_poly,
            self.poly_n_sigma,
            self.flags
        )
        
        self.prev_gray = gray
        
        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return mag
    
    def get_motion_magnitude(self) -> float:
        """Average optical flow magnitude."""
        if not self.flow_history:
            return 0.0
        return float(np.mean(list(self.flow_history)))
    
    def get_motion_intensity(self, mag: Optional[np.ndarray]) -> float:
        """
        Compute motion intensity (0-1) from optical flow magnitude.
        High values indicate rapid motion (macro expression).
        """
        if mag is None:
            return 0.0
        
        intensity = float(np.mean(mag))  # Average magnitude
        self.flow_history.append(intensity)
        return np.clip(intensity / 5.0, 0.0, 1.0)  # Normalize to [0,1]
    
    def reset(self):
        """Reset optical flow state."""
        self.prev_gray = None
        self.flow_history.clear()


class MacroExpressionDetector:
    """
    Detects macro (visible) expressions using optical flow.
    Maps macro expressions to micro-expression confidence.
    
    Pipeline:
    1. Extract optical flow (motion magnitude)
    2. Classify motion intensity (static, micro, macro)
    3. Detect apex (peak motion) for forced expression detection
    """
    
    def __init__(self, static_threshold=0.5, micro_threshold=2.0, 
                 apex_window=15, min_apex_duration=3):
        """
        Initialize macro expression detector.
        
        Args:
            static_threshold: Below this = static/neutral face
            micro_threshold: Between thresholds = micro-expression range
            apex_window: Frames to consider for apex detection
            min_apex_duration: Minimum frames for valid apex
        """
        self.static_threshold = static_threshold
        self.micro_threshold = micro_threshold
        self.apex_window = apex_window
        self.min_apex_duration = min_apex_duration
        
        self.flow_processor = OpticalFlowProcessor()
        self.motion_history = deque(maxlen=apex_window)
        self.apex_detector = None
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame for macro expression detection.
        
        Returns:
            {
                'motion_intensity': float,     # [0, 1]
                'expression_type': str,        # 'static', 'micro', 'macro'
                'is_apex': bool,               # Detected apex
                'apex_confidence': float,      # Confidence of forced expression
                'optical_flow_mean': float,    # Average flow magnitude
            }
        """
        # Compute optical flow
        mag = self.flow_processor.compute_flow(frame)
        motion = self.flow_processor.get_motion_intensity(mag)
        
        self.motion_history.append(motion)
        
        # Classify expression type
        if motion < self.static_threshold:
            expr_type = "static"
        elif motion < self.micro_threshold:
            expr_type = "micro"
        else:
            expr_type = "macro"
        
        # Detect apex (peak motion in time window)
        is_apex = False
        apex_conf = 0.0
        if len(self.motion_history) >= self.min_apex_duration:
            motion_seq = list(self.motion_history)
            peak_idx = np.argmax(motion_seq)
            peak_val = motion_seq[peak_idx]
            
            # Apex detected if peak is significantly higher than neighbors
            if peak_idx > 0 and peak_idx < len(motion_seq) - 1:
                neighbors = [motion_seq[peak_idx - 1], motion_seq[peak_idx + 1]]
                if peak_val > max(neighbors) * 1.3:  # 30% higher than neighbors
                    is_apex = True
                    apex_conf = min(1.0, (peak_val - self.micro_threshold) / 5.0)
        
        return {
            'motion_intensity': motion,
            'expression_type': expr_type,
            'is_apex': is_apex,
            'apex_confidence': apex_conf,
            'optical_flow_mean': float(np.mean(list(self.motion_history)) if self.motion_history else 0.0),
        }
    
    def reset(self):
        """Reset detector state."""
        self.flow_processor.reset()
        self.motion_history.clear()


# ============================================================================
# Micro-expression model & apex detector (safe stubs if unavailable)

try:
    from O4_MicroexpressionModality.microexpression_model_v2 import MicroExpressionModel, ApexDetector
    _MICRO_AVAILABLE = True
except ImportError:
    _MICRO_AVAILABLE = False
    logger.warning("microexpression_model not available; using stub classes")

    class MicroExpressionModel:
        def __init__(self, weights_path=None, device=None, num_classes=7, label_map=None):
            self.is_trained_for_micro = False
        def infer(self, frame):
            return "UNKNOWN", 0.0

    class ApexDetector:
        def __init__(self, *args, **kwargs):
            self._capturing = False
        def feed(self, frame, flow_mean):
            return None
        def reset(self):
            pass


# ENUMS & COLORS

class EmotionEnum(Enum):
    HAPPY      = "HAPPY"
    SAD        = "SAD"
    ANGRY      = "ANGRY"
    SURPRISED  = "SURPRISED"
    FEARFUL    = "FEARFUL"
    DISGUSTED  = "DISGUSTED"
    NEUTRAL    = "NEUTRAL"
    FAKE_SMILE = "FAKE_SMILE"
    OFFENDED   = "OFFENDED"
    REPRESSED  = "REPRESSED"
    CONTEMPT   = "CONTEMPT"
    FROWN      = "FROWN"

EMOTION_COLORS: Dict[EmotionEnum, Tuple[int, int, int]] = {
    EmotionEnum.HAPPY:      (0, 220, 80),
    EmotionEnum.SAD:        (200, 100, 50),
    EmotionEnum.ANGRY:      (0, 50, 220),
    EmotionEnum.SURPRISED:  (0, 200, 255),
    EmotionEnum.FEARFUL:    (200, 0, 200),
    EmotionEnum.DISGUSTED:  (0, 160, 80),
    EmotionEnum.NEUTRAL:    (160, 160, 160),
    EmotionEnum.FAKE_SMILE: (20, 210, 210),
    EmotionEnum.OFFENDED:   (80, 80, 220),
    EmotionEnum.REPRESSED:  (120, 100, 180),
    EmotionEnum.CONTEMPT:   (30, 190, 210),
    EmotionEnum.FROWN:      (120, 80, 200),
}

CHART_EMOTIONS: List[EmotionEnum] = [
    EmotionEnum.HAPPY, EmotionEnum.SURPRISED, EmotionEnum.FEARFUL,
    EmotionEnum.DISGUSTED, EmotionEnum.ANGRY, EmotionEnum.SAD,
    EmotionEnum.CONTEMPT, EmotionEnum.FROWN, EmotionEnum.REPRESSED,
    EmotionEnum.OFFENDED, EmotionEnum.FAKE_SMILE, EmotionEnum.NEUTRAL,
]

MICRO_BLEND_THR: float = 0.55

_MICRO_LABEL_TO_ENUM: Dict[str, EmotionEnum] = {
    "HAPPY":     EmotionEnum.HAPPY,
    "SAD":       EmotionEnum.SAD,
    "ANGRY":     EmotionEnum.ANGRY,
    "SURPRISED": EmotionEnum.SURPRISED,
    "FEARFUL":   EmotionEnum.FEARFUL,
    "DISGUSTED": EmotionEnum.DISGUSTED,
    "NEUTRAL":   EmotionEnum.NEUTRAL,
    "CONTEMPT":  EmotionEnum.CONTEMPT,
}

def _ensure_bright(color: Tuple[int, int, int], min_sum: int = 310) -> Tuple[int, int, int]:
    b, g, r = color
    s = b + g + r
    if s >= min_sum or s == 0:
        return (int(b), int(g), int(r))
    scale = min_sum / float(s)
    return (min(255, int(b*scale)), min(255, int(g*scale)), min(255, int(r*scale)))


# LANDMARK INDICES

class LM:
    NOSE_TIP           = 1
    CHIN               = 152
    LEFT_EYE_OUTER     = 33
    LEFT_EYE_INNER     = 133
    RIGHT_EYE_OUTER    = 263
    RIGHT_EYE_INNER    = 362
    LEFT_MOUTH_CORNER  = 61
    RIGHT_MOUTH_CORNER = 291
    MOUTH_TOP          = 13
    MOUTH_BOTTOM       = 14
    LEFT_EYEBROW_INNER = 65
    LEFT_EYEBROW_OUTER = 55
    RIGHT_EYEBROW_INNER= 295
    RIGHT_EYEBROW_OUTER= 285
    LEFT_CHEEK         = 117
    RIGHT_CHEEK        = 347
    NOSE_LEFT_ALAR     = 129
    NOSE_RIGHT_ALAR    = 358

    LEFT_EYE   = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

    # Cheek region for AU6 (cheek lift) detection
    LEFT_CHEEK_REGION  = [116, 117, 118, 119, 120, 121]
    RIGHT_CHEEK_REGION = [346, 347, 348, 349, 350, 351]

    UPPER_FACE_MICRO = [33, 133, 263, 362, 65, 55, 295, 285,
                        160, 158, 153, 144, 385, 387, 373, 380]
    LOWER_FACE_MICRO = [61, 291, 13, 14, 129, 358, 152]


# DATA CLASSES

@dataclass
class AUActivations:
    """Action Unit activations (0-1 range)"""
    AU1:  float = 0.0
    AU2:  float = 0.0
    AU4:  float = 0.0
    AU5:  float = 0.0
    AU6:  float = 0.0
    AU7:  float = 0.0
    AU9:  float = 0.0
    AU10: float = 0.0
    AU12: float = 0.0
    AU14: float = 0.0
    AU15: float = 0.0
    AU17: float = 0.0
    AU20: float = 0.0
    AU23: float = 0.0
    AU24: float = 0.0
    AU26: float = 0.0
    AU45: float = 0.0
    AU_corner_asym: float = 0.0
    AU12_vel: float = 0.0
    AU26_vel: float = 0.0
    AU6_vel:  float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()
                if k.startswith('AU') and not k.endswith('_vel')}

@dataclass
class MMOutput:
    emotion:         EmotionEnum
    confidence:      float
    aus:             AUActivations
    ear_l:           float
    ear_r:           float
    mar:             float
    pitch:           Optional[float]
    yaw:             Optional[float]
    roll:            Optional[float]
    gaze_h:          float
    gaze_v:          float
    flow_mag:        Optional[float]
    is_fatigued:     bool
    is_forced:       bool
    stability_score: float
    calibrating:     bool = False
    micro_emotion:   Optional[str] = None
    micro_confidence: float = 0.0
    emotion_scores:  Dict = field(default_factory=dict)
    emotion_scores_macro: Dict = field(default_factory=dict)  # backup for micro blending
    deception_prob:  float = 0.0               # <-- BUG-MM1 FIX: added field

    def __post_init__(self):
        if not self.emotion_scores:
            self.emotion_scores = {e: 0.0 for e in EmotionEnum}
        if not self.emotion_scores_macro:
            self.emotion_scores_macro = {e: 0.0 for e in EmotionEnum}

    def to_dict(self) -> Dict[str, any]:
        """Convert MMOutput to a serializable dict for fusion/logging."""
        # Convert enums to strings and AUActivations to dict
        out = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, EmotionEnum):
                out[k] = v.value
            elif hasattr(v, 'to_dict') and callable(getattr(v, 'to_dict')):
                try:
                    out[k] = v.to_dict()
                except Exception:
                    out[k] = None
            else:
                out[k] = v
        return out


# GEOMETRY HELPERS

def dist2(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))

def inter_ocular_distance_2d(lm: np.ndarray) -> float:
    return dist2(lm[LM.LEFT_EYE_OUTER], lm[LM.RIGHT_EYE_OUTER])

def eye_aspect_ratio(lm: np.ndarray, idx: List[int]) -> float:
    p = lm[idx]
    v1 = dist2(p[1], p[5])
    v2 = dist2(p[2], p[4])
    h  = dist2(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def mouth_aspect_ratio(lm: np.ndarray) -> float:
    v = dist2(lm[LM.MOUTH_TOP], lm[LM.MOUTH_BOTTOM])
    h = dist2(lm[LM.LEFT_MOUTH_CORNER], lm[LM.RIGHT_MOUTH_CORNER])
    return v / (h + 1e-6)

def _crop_face(frame: np.ndarray, lm: np.ndarray, padding: float = 0.05) -> np.ndarray:
    """Return 224×224 BGR crop with tighter padding."""
    H, W = frame.shape[:2]
    x1, y1 = lm[:, 0].min(), lm[:, 1].min()
    x2, y2 = lm[:, 0].max(), lm[:, 1].max()
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = int(max(0, x1 - pw))
    y1 = int(max(0, y1 - ph))
    x2 = int(min(W, x2 + pw))
    y2 = int(min(H, y2 + ph))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


# EULERIAN VIDEO MAGNIFICATION

class EulerianMagnifier:
    def __init__(self, alpha: float = 0.95, amplification: float = 2.5, enabled: bool = True):
        self.alpha = alpha
        self.amp = amplification
        self._low = None
        self.enabled = enabled

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        f = frame.astype(np.float32) / 255.0
        if self._low is None:
            self._low = f.copy()
        self._low = self.alpha * self._low + (1.0 - self.alpha) * f
        high = f - self._low
        return np.clip((f + self.amp * high) * 255.0, 0, 255).astype(np.uint8)


# OPTICAL FLOW TRACKER

class OpticalFlowTracker:
    def __init__(self, window: int = 30, micro: bool = False):
        self._prev_gray = None
        self.stats_history: deque = deque(maxlen=window)
        
        if micro:
            self._fb_kwargs = dict(
                pyr_scale=0.5, levels=4, winsize=15,
                iterations=5, poly_n=5, poly_sigma=1.2, flags=0
            )
        else:
            self._fb_kwargs = dict(
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

    def compute(self, gray: np.ndarray,
                lm_2d: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        if gray.size == 0:
            return None
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **self._fb_kwargs)
        self._prev_gray = gray.copy()

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        H, W = gray.shape
        vals = [mag[int(y), int(x)]
                for x, y in lm_2d if 0 <= int(x) < W and 0 <= int(y) < H]

        if not vals:
            return None

        s = {'mean': float(np.mean(vals)),
             'std':  float(np.std(vals)),
             'max':  float(np.max(vals))}
        self.stats_history.append(s)
        return s


# HEAD POSE ESTIMATOR

class HeadPoseEstimator:
    _MODEL_PTS = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ], dtype=np.float32)

    def __init__(self, w: int = 800, h: int = 600):
        self.cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        self.dist = np.zeros((4, 1), dtype=np.float32)
        self._f = None
        self._a = 0.55

    def reset_filter(self):
        self._f = None

    def estimate(self, lm: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        pts = lm[LM.HEAD_POSE_IDX, :2].astype(np.float32)
        ok, rvec, _ = cv2.solvePnP(self._MODEL_PTS, pts, self.cam, self.dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None

        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

        if sy > 1e-6:
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
            roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
            roll  = math.degrees(math.atan2(R[0, 1], R[0, 0]))

        if self._f is None:
            self._f = (pitch, yaw, roll)
        else:
            a = self._a
            self._f = (a*pitch + (1-a)*self._f[0],
                       a*yaw   + (1-a)*self._f[1],
                       a*roll  + (1-a)*self._f[2])
        return self._f


# BLINK DETECTOR (STATE MACHINE)

class BlinkDetector:
    def __init__(self, ear_close_thresh: float = 0.14, 
                 ear_open_thresh: float = 0.20,
                 frames_to_confirm: int = 3):
        self.state = "open"
        self.frame_count = 0
        self.t_close_threshold = frames_to_confirm
        self.ear_close_threshold = ear_close_thresh
        self.ear_open_threshold = ear_open_thresh
        
    def update(self, ear: float) -> float:
        if self.state == "open" and ear < self.ear_close_threshold:
            self.state = "closing"
            self.frame_count = 0
        elif self.state == "closing":
            self.frame_count += 1
            if self.frame_count >= self.t_close_threshold:
                self.state = "closed"
                self.frame_count = 0
        elif self.state == "closed" and ear > self.ear_open_threshold:
            self.state = "opening"
            self.frame_count = 0
        elif self.state == "opening":
            self.frame_count += 1
            if self.frame_count >= self.t_close_threshold:
                self.state = "open"
        
        if self.state == "open":
            return 0.0
        elif self.state == "closing":
            return float(self.frame_count / self.t_close_threshold) * 0.5
        elif self.state == "closed":
            return 1.0
        else:  # opening
            return 0.5 - float(self.frame_count / self.t_close_threshold) * 0.5


# AU CALCULATOR

class AUCalculator:
    def __init__(self, warmup_frames: int = 150, ema_alpha: float = 0.015):
        self._wf = warmup_frames
        self._ema_a = ema_alpha
        self._au_ema_a = 0.45
        self._frame = 0
        self._baseline: Optional[Dict[str, float]] = None
        self._buf: List[Dict] = []
        self._au_ema: Optional[Dict[str, float]] = None
        self._prev_au: Optional[Dict[str, float]] = None
        self.blink_detector = BlinkDetector()

    def _default_baseline(self) -> Dict[str, float]:
        return {'brow_inner': 0.0, 'brow_outer': 0.0, 'cheek_fullness': 0.35,
                'alar': 0.18, 'ul_gap': 0.05, 'mouth_w': 0.45,
                'corner_drop': -0.05, 'chin_raise': 0.12, 'jaw_open': 0.08,
                'corner_asym': 0.0, 'nose_ridge': 0.0}

    def _should_update(self, raw: Dict) -> bool:
        if self._baseline is None:
            return False
        delta = sum(abs(raw.get(k, 0) - self._baseline.get(k, 0)) for k in raw)
        return delta < 0.08

    def _extract(self, lm: np.ndarray, iod: float, face_bbox_area: float = None) -> Dict[str, float]:
        def sv(a, b): return (b[1] - a[1]) / iod

        el  = lm[LM.LEFT_EYE_INNER]
        er  = lm[LM.RIGHT_EYE_INNER]
        bl  = lm[LM.LEFT_EYEBROW_INNER]
        br  = lm[LM.RIGHT_EYEBROW_INNER]
        ol  = lm[LM.LEFT_EYEBROW_OUTER]
        or_ = lm[LM.RIGHT_EYEBROW_OUTER]
        ml  = lm[LM.LEFT_MOUTH_CORNER]
        mr  = lm[LM.RIGHT_MOUTH_CORNER]
        nt  = lm[LM.NOSE_TIP]
        al  = lm[LM.NOSE_LEFT_ALAR]
        ar  = lm[LM.NOSE_RIGHT_ALAR]
        mt  = lm[LM.MOUTH_TOP]
        mb  = lm[LM.MOUTH_BOTTOM]
        ch  = lm[LM.CHIN]
        mid_x = nt[0]

        cheek_l_pts = lm[LM.LEFT_CHEEK_REGION, :2]
        cheek_r_pts = lm[LM.RIGHT_CHEEK_REGION, :2]
        
        def bbox_area(pts):
            if len(pts) == 0:
                return 0.0
            return float((pts[:, 0].max() - pts[:, 0].min()) * 
                        (pts[:, 1].max() - pts[:, 1].min()))
        
        cheek_l_area = bbox_area(cheek_l_pts)
        cheek_r_area = bbox_area(cheek_r_pts)
        normalization = face_bbox_area if face_bbox_area and face_bbox_area > 0 else iod ** 2
        cheek_fullness = (cheek_l_area + cheek_r_area) / (normalization + 1e-6)

        nose_ridge_pts = lm[[4, 131, 360], 1]
        nose_ridge = float(np.std(nose_ridge_pts))

        return {
            'brow_inner':  (sv(el, bl) + sv(er, br)) / 2,
            'brow_outer':  (sv(el, ol) + sv(er, or_)) / 2,
            'cheek_fullness': cheek_fullness,
            'alar':        dist2(al, ar) / iod,
            'ul_gap':      sv(nt, mt),
            'mouth_w':     dist2(ml, mr) / iod,
            'corner_drop': (sv(ch, ml) + sv(ch, mr)) / 2,
            'chin_raise':  sv(mb, ch),
            'jaw_open':    dist2(mt, mb) / iod,
            'corner_asym': abs(abs(ml[0]-mid_x) - abs(mr[0]-mid_x)) / iod,
            'nose_ridge':  nose_ridge / iod,
        }

    def compute(self, lm: np.ndarray) -> AUActivations:
        self._frame += 1
        iod = inter_ocular_distance_2d(lm)

        if iod < 1.0:
            return AUActivations()

        x_min, y_min = lm[:, 0].min(), lm[:, 1].min()
        x_max, y_max = lm[:, 0].max(), lm[:, 1].max()
        face_bbox_area = (x_max - x_min) * (y_max - y_min)

        raw = self._extract(lm, iod, face_bbox_area)

        if self._frame <= self._wf:
            self._buf.append(raw)
            if self._frame == self._wf:
                self._baseline = self._compute_baseline_with_rejection(self._buf)
            return AUActivations()

        if self._baseline is None:
            self._baseline = self._default_baseline()

        if self._should_update(raw):
            for k, v in raw.items():
                self._baseline[k] = ((1 - self._ema_a) * self._baseline.get(k, v)
                                     + self._ema_a * v)

        bl = self._baseline
        aus = AUActivations()

        d_in  = raw['brow_inner'] - bl['brow_inner']
        d_out = raw['brow_outer'] - bl['brow_outer']
        aus.AU1 = float(np.clip(-d_in  / 0.035, 0, 1))
        aus.AU2 = float(np.clip(-d_out / 0.045, 0, 1))
        aus.AU4 = float(np.clip( d_in  / 0.035, 0, 1))

        ear_l   = eye_aspect_ratio(lm, LM.LEFT_EYE)
        ear_r   = eye_aspect_ratio(lm, LM.RIGHT_EYE)
        avg_ear = (ear_l + ear_r) / 2.0

        EAR_MID = 0.27
        aus.AU5 = float(np.clip((avg_ear - EAR_MID) / 0.07, 0, 1))
        
        if 0.15 < avg_ear < 0.25:
            aus.AU7 = float(np.clip((EAR_MID - avg_ear) / 0.05, 0, 1))
        else:
            aus.AU7 = 0.0

        cheek_lift = bl.get('cheek_fullness', 0) - raw['cheek_fullness']
        aus.AU6  = float(np.clip(cheek_lift / 0.15, 0, 1))
        
        alar_width = raw['alar'] - bl.get('alar', raw['alar'])
        ridge_crease = raw['nose_ridge'] - bl.get('nose_ridge', raw['nose_ridge'])
        aus.AU9  = float(np.clip((alar_width * 0.7 + ridge_crease * 0.3) / 0.025, 0, 1))
        
        aus.AU10 = float(np.clip(-(raw['ul_gap'] - bl.get('ul_gap', raw['ul_gap'])) / 0.035, 0, 1))

        d_mw    = raw['mouth_w'] - bl.get('mouth_w', raw['mouth_w'])
        aus.AU12 = float(np.clip(d_mw / 0.035, 0, 1))

        ld = lm[LM.LEFT_MOUTH_CORNER][2]
        rd = lm[LM.RIGHT_MOUTH_CORNER][2]
        aus.AU14 = float(np.clip(abs(ld - rd) / (iod * 0.08 + 1e-6), 0, 1))

        aus.AU15 = float(np.clip(-(raw['corner_drop'] - bl.get('corner_drop', raw['corner_drop'])) / 0.028, 0, 1))
        aus.AU17 = float(np.clip(-(raw['chin_raise']  - bl.get('chin_raise',  raw['chin_raise']))  / 0.028, 0, 1))
        aus.AU20 = float(np.clip((raw['mouth_w'] - 0.38) / 0.18, 0, 1))

        mar      = mouth_aspect_ratio(lm)
        aus.AU23 = float(np.clip((0.14 - mar) / 0.09, 0, 1)) if mar < 0.14 else 0.0
        aus.AU24 = float(np.clip((0.07 - mar) / 0.05, 0, 1)) if mar < 0.07 else 0.0

        aus.AU26 = float(np.clip((raw['jaw_open'] - bl.get('jaw_open', raw['jaw_open'])) / 0.035, 0, 1))
        aus.AU_corner_asym = float(np.clip(raw.get('corner_asym', 0) / 0.045, 0, 1))
        
        aus.AU45 = self.blink_detector.update(avg_ear)

        raw_d = aus.to_dict()
        if self._au_ema is None:
            self._au_ema = dict(raw_d)
            self._prev_au = dict(raw_d)
        else:
            a = self._au_ema_a
            for k, v in raw_d.items():
                if k == 'AU45':
                    self._au_ema[k] = v
                else:
                    self._au_ema[k] = a * v + (1 - a) * self._au_ema.get(k, v)

        for k, v in self._au_ema.items():
            setattr(aus, k, float(v))

        if self._prev_au is not None:
            aus.AU12_vel = float(self._au_ema.get('AU12', 0) - self._prev_au.get('AU12', 0))
            aus.AU26_vel = float(self._au_ema.get('AU26', 0) - self._prev_au.get('AU26', 0))
            aus.AU6_vel  = float(self._au_ema.get('AU6',  0) - self._prev_au.get('AU6',  0))

        self._prev_au = dict(self._au_ema)
        return aus

    def reset_baseline(self):
        self._frame    = 0
        self._baseline = None
        self._buf.clear()
        self._au_ema   = None
        self._prev_au  = None
        self.blink_detector = BlinkDetector()

    @staticmethod
    def _compute_baseline_with_rejection(buf: List[Dict], outlier_std: float = 2.0) -> Dict[str, float]:
        if not buf:
            return {}
        baseline = {}
        keys = buf[0].keys()
        for key in keys:
            values = np.array([b[key] for b in buf])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            valid = values[np.abs(values - median) < outlier_std * (mad + 1e-6)]
            if len(valid) > 0:
                baseline[key] = float(np.mean(valid))
            else:
                baseline[key] = median
        return baseline


# EMOTION CLASSIFIER

class EmotionClassifier:
    def __init__(self, smooth_window: int = 7):
        self._history: deque = deque(maxlen=smooth_window)
        self._conf_h:  deque = deque(maxlen=smooth_window)
        self._vote_prev: Optional[EmotionEnum] = None
        self._vote_t: float = time.time()
        self._response_ms: float = 0.0

    def _soft_entropy_penalty(self, scores: Dict[EmotionEnum, float]) -> float:
        pos = {k: v for k, v in scores.items() if v > 0}
        if len(pos) <= 1:
            return 1.0
        total = sum(pos.values())
        probs = [v / total for v in pos.values()]
        H = -sum(p * math.log(p + 1e-9) for p in probs)
        H_max = math.log(len(probs))
        if H_max <= 0:
            return 1.0
        return math.sqrt(max(0.0, 1.0 - H / H_max))

    def classify(self, aus: AUActivations
                 ) -> Tuple[EmotionEnum, float, float, Dict[EmotionEnum, float]]:
        s: Dict[EmotionEnum, float] = {}

        vel_bonus = min(0.15, max(0.0, aus.AU12_vel * 2.0))
        s[EmotionEnum.HAPPY] = (
            0.35*aus.AU6 + 0.65*aus.AU12 + vel_bonus
        ) if aus.AU6 > 0.15 and aus.AU12 > 0.15 else 0.0

        s[EmotionEnum.FAKE_SMILE] = (
            0.70*aus.AU12 + 0.30*(1-aus.AU6)
        ) if aus.AU12 > 0.35 and aus.AU6 < 0.12 else 0.0

        s[EmotionEnum.SAD] = (
            0.28*aus.AU1 + 0.22*aus.AU4 + 0.32*aus.AU15 + 0.18*aus.AU17
        ) if aus.AU15 > 0.18 and (aus.AU1 > 0.10 or aus.AU4 > 0.10) else 0.0

        s[EmotionEnum.FROWN] = (
            0.70*aus.AU15 + 0.30*(1-aus.AU12)
        ) if aus.AU15 > 0.22 and aus.AU12 < 0.12 and aus.AU1 < 0.10 and aus.AU4 < 0.10 else 0.0

        s[EmotionEnum.ANGRY] = (
            0.40*aus.AU4 + 0.30*aus.AU9 + 0.20*aus.AU23 + 0.10*aus.AU5
        ) if aus.AU4 > 0.20 and (aus.AU9 > 0.15 or aus.AU23 > 0.18) and aus.AU7 < 0.15 and aus.AU9 > 0.25 else 0.0

        s[EmotionEnum.SURPRISED] = (
            0.18*aus.AU1 + 0.18*aus.AU2 + 0.25*aus.AU5 + 0.39*aus.AU26
        ) if aus.AU26 > 0.25 and aus.AU5 > 0.20 and aus.AU4 < 0.12 else 0.0

        s[EmotionEnum.FEARFUL] = (
            0.18*aus.AU1 + 0.15*aus.AU2 + 0.18*aus.AU4 +
            0.15*aus.AU5 + 0.15*aus.AU20 + 0.19*aus.AU26
        ) if aus.AU4 > 0.15 and aus.AU26 > 0.15 and aus.AU20 > 0.10 else 0.0

        s[EmotionEnum.DISGUSTED] = (
            0.45*aus.AU9 + 0.25*aus.AU15 + 0.30*aus.AU10
        ) if aus.AU9 > 0.18 and aus.AU15 > 0.15 else 0.0

        s[EmotionEnum.CONTEMPT] = (
            0.40*aus.AU14 + 0.35*aus.AU_corner_asym + 0.25*min(aus.AU15, aus.AU17)
        ) if aus.AU14 > 0.25 and aus.AU_corner_asym > 0.25 else 0.0

        s[EmotionEnum.OFFENDED] = (
            0.40*aus.AU4 + 0.40*aus.AU15 + 0.20*aus.AU23
        ) if aus.AU4 > 0.20 and aus.AU15 > 0.18 else 0.0

        s[EmotionEnum.REPRESSED] = (
            0.35*aus.AU7 + 0.35*aus.AU23 + 0.15*aus.AU24 + 0.15*(1-aus.AU26)
        ) if aus.AU7 > 0.16 and (aus.AU23 > 0.12 or aus.AU24 > 0.12) else 0.0

        best_score = max(s.values()) if s else 0.0

        if best_score < 0.12:
            raw_em   = EmotionEnum.NEUTRAL
            raw_conf = max(0.35, 1.0 - best_score * 4.0)
        else:
            raw_em   = max(s, key=s.get)
            raw_conf = min(1.0, s[raw_em])

        raw_conf *= self._soft_entropy_penalty(s)
        if raw_em != EmotionEnum.NEUTRAL:
            raw_conf = max(0.40, raw_conf)

        raw_scores = dict(s)
        raw_scores[EmotionEnum.NEUTRAL] = (
            raw_conf if raw_em == EmotionEnum.NEUTRAL else (1.0 - best_score)
        )

        self._history.append(raw_em)
        self._conf_h.append(raw_conf)

        h = list(self._history)
        c = list(self._conf_h)
        if not h:
            return EmotionEnum.NEUTRAL, 0.0, 0.0, raw_scores

        decay = 0.65
        wmap: Dict[EmotionEnum, float] = {}
        cmap: Dict[EmotionEnum, list] = {}
        w = 1.0
        for e, ci in zip(reversed(h), reversed(c)):
            wmap[e] = wmap.get(e, 0.0) + w
            cmap.setdefault(e, []).append(ci)
            w *= decay

        vote  = max(wmap, key=wmap.get)
        vconf = float(np.mean(cmap[vote]))

        if vote != self._vote_prev:
            self._vote_t  = time.time()
            self._vote_prev = vote
        self._response_ms = (time.time() - self._vote_t) * 1000.0

        return vote, vconf, self._response_ms, raw_scores


# GAZE ESTIMATION

def estimate_gaze_2d(lm: np.ndarray) -> Tuple[float, float]:
    if lm.shape[0] < 478:
        return 0.0, 0.0

    iod = inter_ocular_distance_2d(lm) + 1e-6

    il = lm[LM.LEFT_IRIS[0], :2]
    rl = (lm[LM.LEFT_EYE_INNER, :2] + lm[LM.LEFT_EYE_OUTER, :2]) / 2.0

    ir = lm[LM.RIGHT_IRIS[0], :2]
    rr = (lm[LM.RIGHT_EYE_INNER, :2] + lm[LM.RIGHT_EYE_OUTER, :2]) / 2.0

    g = ((il - rl) + (ir - rr)) / (2.0 * iod)
    return float(np.clip(g[0], -1, 1)), float(np.clip(g[1], -1, 1))


# FATIGUE DETECTOR

class FatigueDetector:
    def __init__(self, fps=30.0, blink_thr=20.0, yawn_thr=2.0):
        self.fps       = fps
        self.blink_thr = blink_thr
        self.yawn_thr  = yawn_thr
        self._bt: deque = deque(maxlen=200)
        self._yt: deque = deque(maxlen=100)
        self._ib = self._iy = False
        self._f  = 0

    def update(self, au45: float, mar: float):
        self._f += 1
        now = self._f / self.fps
        if au45 > 0.5:
            if not self._ib:
                self._bt.append(now)
                self._ib = True
        else:
            self._ib = False

        if mar > 0.53:
            if not self._iy:
                self._yt.append(now)
                self._iy = True
        else:
            self._iy = False

    def _rpm(self, t: deque) -> float:
        if len(t) < 2:
            return 0.0
        now = self._f / self.fps
        return float(len([x for x in t if now - x <= 60.0]))

    @property
    def blink_rate(self) -> float: return self._rpm(self._bt)

    @property
    def yawn_rate(self) -> float:  return self._rpm(self._yt)

    @property
    def is_fatigued(self) -> bool:
        return self.blink_rate > self.blink_thr or self.yawn_rate >= self.yawn_thr


# PATTERN ANALYSER

class PatternAnalyser:
    def __init__(self, window: int = 45):
        self._em: deque = deque(maxlen=window)
        self._co: deque = deque(maxlen=window)
        self._au: deque = deque(maxlen=window)
        self._fl: deque = deque(maxlen=window)
        self._scores_history: deque = deque(maxlen=window)  # Track confidence scores for variance

    def update(self, emotion, conf, aus, flow):
        self._em.append(emotion)
        self._co.append(conf)
        self._au.append(aus)
        if flow:
            self._fl.append(flow['mean'])
        self._scores_history.append(conf)

    def _rapid_switch(self) -> bool:
        if len(self._em) < 15:
            return False
        r  = list(self._em)[-15:]
        flips = sum(1 for i in range(1, len(r)) if r[i] != r[i-1])
        ac = float(np.mean(list(self._co)[-15:])) if len(self._co) >= 15 else 1.0
        return flips > 6 and ac < 0.6

    def _au_contradiction(self) -> bool:
        if len(self._au) < 8:
            return False
        n = 0
        for a in list(self._au)[-8:]:
            # Duchenne marker check for forced smile: AU12 (lip corner) > 0.6 AND AU6 (cheek raiser) < 0.4
            # Real smile has both AU12 and AU6; fake smile has AU12 without AU6
            if a.AU12 > 0.6 and a.AU6 < 0.4:
                n += 1
            # Conflicting AU combinations that indicate forced expression
            if a.AU12 > 0.5 and a.AU4 > 0.4 and a.AU7 > 0.3:
                n += 1
            if a.AU12 > 0.4 and a.AU15 > 0.4:
                n += 1
        return n >= 3

    def _low_conf(self) -> bool:
        if len(self._co) < 10 or len(self._em) < 10:
            return False
        rc  = list(self._co)[-10:]
        re  = list(self._em)[-10:]
        dom = Counter(re).most_common(1)[0][1] / len(re)
        return float(np.mean(rc)) < 0.4 and dom < 0.5

    @property
    def is_forced(self) -> bool:
        return self._rapid_switch() or self._au_contradiction() or self._low_conf()

    @property
    def stability_score(self) -> float:
        """
        Compute emotional stability from variance of confidence scores.
        - High variance = unstable = low stability_score
        - Low variance = stable = high stability_score
        Formula: stability = 1.0 - normalized_variance, clipped to [0, 1]
        """
        if len(self._scores_history) < 15:
            return 0.5
        
        scores = np.array(list(self._scores_history)[-15:])
        variance = float(np.var(scores))
        # Empirical max_variance: ~0.25 for max swing [0 -> 1]
        # Normalize to [0, 1] range where 0.25 variance → 1.0 instability
        max_variance = 0.25
        normalized_var = np.clip(variance / max_variance, 0, 1)
        return max(0.0, 1.0 - normalized_var)


# EMOTION RADAR CHART

class EmotionRadarChart:
    def __init__(self, n_history: int = 30):
        self._history: deque = deque(maxlen=n_history)
        self._n = len(CHART_EMOTIONS)
        self._angles = [
            -math.pi/2 + 2*math.pi*i/self._n
            for i in range(self._n)
        ]

    def update(self, scores: Dict[EmotionEnum, float]):
        vec = [float(scores.get(e, 0.0)) for e in CHART_EMOTIONS]
        m   = max(vec) if max(vec) > 0 else 1.0
        self._history.append([v/m for v in vec])

    def draw(self, frame: np.ndarray, cx: int, cy: int, radius: int):
        n      = self._n
        angles = self._angles

        ov = frame.copy()
        cv2.circle(ov, (cx, cy), radius + 28, (12, 12, 18), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        for ring in [0.25, 0.50, 0.75, 1.0]:
            pts = np.array([
                [int(cx + ring*radius*math.cos(a)),
                 int(cy + ring*radius*math.sin(a))]
                for a in angles], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (45, 45, 55), 1, cv2.LINE_AA)

        cv2.circle(frame, (cx, cy), 2, (80, 80, 90), -1)

        for i, (e, a) in enumerate(zip(CHART_EMOTIONS, angles)):
            ex = int(cx + radius * math.cos(a))
            ey = int(cy + radius * math.sin(a))
            cv2.line(frame, (cx, cy), (ex, ey), (50, 50, 60), 1, cv2.LINE_AA)

            lx  = int(cx + (radius + 14) * math.cos(a))
            ly  = int(cy + (radius + 14) * math.sin(a))
            col = _ensure_bright(EMOTION_COLORS.get(e, (160, 160, 160)), 260)
            lbl = e.value[:4]
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            cv2.putText(frame, lbl, (lx - tw//2, ly + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)

        hist   = list(self._history)
        n_hist = len(hist)
        step   = max(1, n_hist // 20) if n_hist > 20 else 1
        for idx in range(0, n_hist, step):
            frame_vec = hist[idx]
            alpha = 0.15 + 0.55 * (idx / max(n_hist - 1, 1))
            pts   = np.array([
                [int(cx + frame_vec[i]*radius*math.cos(angles[i])),
                 int(cy + frame_vec[i]*radius*math.sin(angles[i]))]
                for i in range(n)], dtype=np.int32)

            dom_i   = int(np.argmax(frame_vec))
            tc      = EMOTION_COLORS.get(CHART_EMOTIONS[dom_i], (100, 200, 100))
            bc, gc, rc = int(tc[0]*alpha), int(tc[1]*alpha), int(tc[2]*alpha)

            ov2 = frame.copy()
            cv2.fillPoly(ov2, [pts], (bc, gc, rc))
            cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [pts], True, (bc*2, gc*2, rc*2), 1, cv2.LINE_AA)

        if hist:
            cur   = hist[-1]
            pts   = np.array([
                [int(cx + cur[i]*radius*math.cos(angles[i])),
                 int(cy + cur[i]*radius*math.sin(angles[i]))]
                for i in range(n)], dtype=np.int32)
            dom_i = int(np.argmax(cur))
            tc    = _ensure_bright(EMOTION_COLORS.get(CHART_EMOTIONS[dom_i], (100, 200, 100)))
            cv2.polylines(frame, [pts], True, tc, 2, cv2.LINE_AA)
            for i in range(n):
                px = int(cx + cur[i]*radius*math.cos(angles[i]))
                py = int(cy + cur[i]*radius*math.sin(angles[i]))
                cv2.circle(frame, (px, py), 3, tc, -1, cv2.LINE_AA)

        cv2.putText(frame, "Emotion Deviation",
                   (cx - 52, cy + radius + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 180), 1, cv2.LINE_AA)


# HUD RENDERER

class HUDRenderer:
    _FONT     = cv2.FONT_HERSHEY_SIMPLEX
    _BG_ALPHA = 0.62

    @staticmethod
    def _t(frame, txt, x, y, sc=0.48, col=(230, 230, 230), th=1):
        cv2.putText(frame, txt, (x, y), HUDRenderer._FONT,
                   sc, col, th, cv2.LINE_AA)

    @classmethod
    def draw(cls, frame: np.ndarray, out: MMOutput, fps: float,
             chart: EmotionRadarChart):
        H, W = frame.shape[:2]

        ov = frame.copy()
        cv2.rectangle(ov, (4, 4), (330, 265), (12, 12, 18), -1)

        AU_N  = 10
        AU_PW, AU_PH = 248, AU_N*22 + 28
        AU_PX, AU_PY = W - AU_PW - 4, 4
        cv2.rectangle(ov, (AU_PX, AU_PY), (AU_PX+AU_PW, AU_PY+AU_PH), (12, 12, 18), -1)
        cv2.addWeighted(ov, cls._BG_ALPHA, frame, 1-cls._BG_ALPHA, 0, frame)

        em_col = _ensure_bright(EMOTION_COLORS.get(out.emotion, (200, 200, 200)))
        cls._t(frame, "EMOTION", 10, 24, 0.42, (170, 170, 190))
        cls._t(frame, out.emotion.value, 10, 56, 0.88, em_col, 2)

        conf = max(0.0, min(1.0, out.confidence))
        bw   = int(308 * conf)
        cv2.rectangle(frame, (10, 62), (318, 72), (45, 45, 55), -1)
        cv2.rectangle(frame, (10, 62), (10+bw, 72), em_col, -1)
        cls._t(frame, f"{conf*100:.0f}%", 282, 72, 0.36)

        stab = out.stability_score
        slbl = "STABLE" if stab >= 0.65 else "UNSTABLE"
        scol = (70, 210, 120) if stab >= 0.65 else (70, 70, 200)
        cv2.rectangle(frame, (10, 78), (10+len(slbl)*8+8, 94), scol, -1)
        cls._t(frame, slbl, 14, 92, 0.34, (255, 255, 255), 1)

        cls._t(frame, f"EAR  L:{out.ear_l:.2f}  R:{out.ear_r:.2f}", 10, 116, 0.38)
        cls._t(frame, f"MAR  {out.mar:.2f}", 10, 134, 0.38)

        if out.pitch is not None:
            cls._t(frame, f"Pose  P:{out.pitch:.0f}  Y:{out.yaw:.0f}  R:{out.roll:.0f}",
                  10, 152, 0.37)

        cls._t(frame, f"Gaze  H:{out.gaze_h:+.2f}  V:{out.gaze_v:+.2f}",
              10, 170, 0.37)

        if out.flow_mag is not None:
            cls._t(frame, f"Motion  {out.flow_mag:.2f}", 10, 188, 0.37)

        cls._t(frame, f"FPS  {fps:.1f}", 10, 206, 0.37)

        if out.micro_emotion and out.micro_emotion != "UNKNOWN" and out.micro_confidence > MICRO_BLEND_THR:
            cls._t(frame, f"Micro  {out.micro_emotion} ({out.micro_confidence:.2f})",
                  10, 224, 0.37, (80, 230, 230))

        # Display deception probability (BUG-MM1)
        deception = out.deception_prob
        deception_col = (0, 0, 255) if deception > 0.6 else (0, 200, 0) if deception < 0.3 else (0, 165, 255)
        cls._t(frame, f"Deception  {deception:.2f}", 10, 242, 0.42, deception_col, 2)

        ay = 275
        if out.calibrating:
            cv2.rectangle(frame, (4, ay), (330, ay+26), (60, 60, 0), -1)
            cls._t(frame, "[CALIBRATING] Hold neutral", 10, ay+19, 0.40, (255, 240, 80))
            ay += 32

        if out.is_forced:
            cv2.rectangle(frame, (4, ay), (330, ay+26), (130, 15, 15), -1)
            cls._t(frame, "[!] FORCED / SUPPRESSED", 10, ay+19, 0.40, (255, 255, 255))
            ay += 32

        if out.is_fatigued:
            cv2.rectangle(frame, (4, ay), (330, ay+26), (0, 80, 0), -1)
            cls._t(frame, "[!] FATIGUE DETECTED", 10, ay+19, 0.40, (160, 255, 160))

        AU_LIST = ['AU1','AU2','AU4','AU6','AU7','AU9','AU12','AU15','AU20','AU26']
        BAR_W   = AU_PW - 90
        cls._t(frame, "Action Units", AU_PX+6, AU_PY+16, 0.42, (170, 170, 190))

        for i, nm in enumerate(AU_LIST):
            val = max(0.0, min(1.0, float(getattr(out.aus, nm, 0.0))))
            y0  = AU_PY + 26 + i*22
            cls._t(frame, nm, AU_PX+4, y0+13, 0.34)
            bx  = AU_PX + 50
            bw  = int(BAR_W * val)
            cv2.rectangle(frame, (bx, y0+2), (bx+BAR_W, y0+14), (40, 40, 50), -1)
            r_c = int(min(255, val*2*255))
            g_c = int(min(255, (1-val)*2*255))
            if bw > 0:
                cv2.rectangle(frame, (bx, y0+2), (bx+bw, y0+14), (18, g_c, r_c), -1)
            cls._t(frame, f"{val:.2f}", bx+BAR_W+4, y0+13, 0.31, (190, 190, 210))

        chart_r  = 72
        chart_cx = W - chart_r - 10
        chart_cy = H - chart_r - 30
        chart.draw(frame, chart_cx, chart_cy, chart_r)


# MAIN MM SYSTEM (FIXED: no internal camera, BUG-MM2 removed)

class MMSystem:
    _LOST_RESET_THRESHOLD: int = 60

    def __init__(self, frame_size: Tuple[int, int] = (800, 600),
                 enable_evm: bool = True,
                 micro_weights_path: Optional[str] = None):
        """
        Parameters:
            frame_size: (width, height) for pose estimation and display.
            enable_evm: bool, enable Eulerian motion magnification.
            micro_weights_path: optional path to .pth micro-expression model.
                                If None, uses a default or stub.
        """
        self.frame_size = frame_size
        W, H = frame_size

        try:
            mp_fm = mp.solutions.face_mesh
            self.face_mesh = mp_fm.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        except Exception as e:
            logger.error(f"FaceMesh init: {e}")
            self.face_mesh = None

        # No internal camera capture – frames must be passed to process_frame()

        self.magnifier = EulerianMagnifier(alpha=0.95, amplification=2.5, enabled=enable_evm)
        self.flow_track_macro = OpticalFlowTracker(window=30, micro=False)
        self.flow_track_micro = OpticalFlowTracker(window=8, micro=True)

        self.head_pose = HeadPoseEstimator(W, H)
        self.au_calc   = AUCalculator()
        self.clf       = EmotionClassifier()
        self.fatigue   = FatigueDetector(fps=30.0)  # fps will be updated dynamically if needed
        self.pattern   = PatternAnalyser()
        self.hud       = HUDRenderer()
        self.chart     = EmotionRadarChart(n_history=30)

        # Default micro model path – use environment variable or relative path
        if micro_weights_path is None:
            micro_weights_path = os.environ.get(
                "MICRO_EXPRESSION_WEIGHTS",
                "MICRO_EXPRESSION_WEIGHTS=/Users/laraibnoorien/deception_detection_project/models/model_a_best.pt"
            )
        try:
            self.micro_model = MicroExpressionModel(
                weights_path=micro_weights_path,
                num_classes=7,
            )
            self.apex_detector = ApexDetector(
                window=16,
                capture_len=12,
                spike_thresh=3.0,
                min_mean=0.50,
                min_onset_frames=3,
                refractory_frames=20,   # now accepted via **kwargs
            )
            logger.info("Micro-expression model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load micro-expression model: {e}. Using stub.")
            self.micro_model   = MicroExpressionModel()
            self.apex_detector = ApexDetector()

        self._pose_bad  = False
        self._last_out: Optional[MMOutput] = None
        self._lost      = 0
        self.fps_buf: deque = deque(maxlen=30)
        self._t_last    = time.time()

        logger.info("MM System ready (PRODUCTION V7 - FIXED, no internal camera)")

    @staticmethod
    def _neutral_out(calibrating: bool = False) -> MMOutput:
        return MMOutput(
            emotion=EmotionEnum.NEUTRAL, confidence=0.0,
            aus=AUActivations(),
            ear_l=0.0, ear_r=0.0, mar=0.0,
            pitch=None, yaw=None, roll=None,
            gaze_h=0.0, gaze_v=0.0, flow_mag=None,
            is_fatigued=False, is_forced=False,
            stability_score=0.5, calibrating=calibrating,
            emotion_scores={e: 0.0 for e in EmotionEnum},
            emotion_scores_macro={e: 0.0 for e in EmotionEnum},
            deception_prob=0.0,
        )

    def _compute_micro_flow(self, evm_gray: np.ndarray,
                            lm2d: np.ndarray,
                            H: int, W: int) -> float:
        upper_pts = [(int(lm2d[i, 0]), int(lm2d[i, 1]))
                     for i in LM.UPPER_FACE_MICRO
                     if i < len(lm2d)]
        lower_pts = [(int(lm2d[i, 0]), int(lm2d[i, 1]))
                     for i in LM.LOWER_FACE_MICRO
                     if i < len(lm2d)]
        all_pts   = upper_pts + lower_pts

        if not all_pts:
            return 0.0

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x1 = max(0, min(xs) - 4)
        y1 = max(0, min(ys) - 4)
        x2 = min(W, max(xs) + 4)
        y2 = min(H, max(ys) + 4)

        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return 0.0

        roi      = evm_gray[y1:y2, x1:x2]
        roi_lm   = [(int(x - x1), int(y - y1)) for x, y in all_pts]
        result   = self.flow_track_micro.compute(roi, roi_lm)
        return float(result['mean']) if result else 0.0

    def _integrate_micro(
        self,
        raw_scores: Dict[EmotionEnum, float],
        emotion: EmotionEnum,
        confidence: float,
        micro_emotion: Optional[str],
        micro_confidence: float,
    ) -> Tuple[EmotionEnum, float, Dict[EmotionEnum, float]]:
        if not micro_emotion or micro_emotion == "UNKNOWN":
            return emotion, confidence, raw_scores

        micro_enum = _MICRO_LABEL_TO_ENUM.get(micro_emotion)
        if micro_enum is None:
            return emotion, confidence, raw_scores

        blend_w  = micro_confidence * 0.6
        current  = raw_scores.get(micro_enum, 0.0)
        raw_scores[micro_enum] = min(1.0, current + blend_w * micro_confidence)

        if micro_confidence >= MICRO_BLEND_THR:
            blended_conf = confidence * (1.0 - blend_w) + micro_confidence * blend_w
            promoted_emotion = micro_enum if micro_confidence > confidence else emotion
            logger.info(
                f"Micro blend: {micro_emotion} ({micro_confidence:.2f}) → "
                f"promoted={promoted_emotion.value}, conf={blended_conf:.2f}"
            )
            return promoted_emotion, blended_conf, raw_scores

        return emotion, confidence, raw_scores

    def _compute_deception_prob(self, out: MMOutput) -> float:
        """
        Compute deception probability from MM features with weighted risk fusion.
        Weights:
        - is_forced (50%): Hard indicator of forced/suppressed expressions
        - stability_score (30%): Emotional instability indicates deception
        - micro_confidence (20%): Low micro-expression confidence suggests concealment
        
        Formula: risk = 0.5×is_forced + 0.3×(1-stability) + 0.2×(1-micro_conf)
        Then apply sigmoid: deception = sigmoid(risk×5 - 0.5) for smooth [0,1] mapping
        """
        # Component 1: Forced expression detection (0 or 1 → 0.0 or 1.0)
        forced = 1.0 if out.is_forced else 0.0
        
        # Component 2: Emotional instability (lower stability → higher deception risk)
        unstable = 1.0 - out.stability_score
        
        # Component 3: Micro-expression confidence (lower confidence → higher deception risk)
        micro_conf = 1.0  # Default: unknown/not detected
        if out.micro_emotion and out.micro_emotion not in ("UNKNOWN", "NEUTRAL"):
            micro_conf = out.micro_confidence
        micro_risk = 1.0 - np.clip(micro_conf, 0.0, 1.0)
        
        # Weighted fusion with 0.5/0.3/0.2 weights
        risk_score = 0.5 * forced + 0.3 * unstable + 0.2 * micro_risk
        
        # Sigmoid transformation: steepness=5, bias=-0.5
        # Maps [0,1] → ~[0.378, 0.622] (linear region) → full [0, 1] with asymmetry
        deception = float(expit(risk_score * 5.0 - 0.5))
        
        return np.clip(deception, 0.0, 1.0)

    def process_frame(self, frame: np.ndarray) -> Tuple[MMOutput, np.ndarray]:
        now = time.time()
        self.fps_buf.append(now - self._t_last)
        self._t_last = now

        evm_frame = self.magnifier.process(frame)
        H, W      = frame.shape[:2]
        is_cal    = self.au_calc._frame <= self.au_calc._wf

        # Update fatigue detector with actual FPS
        current_fps = 1.0 / (np.mean(self.fps_buf) + 1e-6)
        self.fatigue.fps = current_fps

        results = None
        if self.face_mesh:
            try:
                results = self.face_mesh.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.debug(f"FaceMesh error: {e}")

        face_ok = (
            results is not None
            and results.multi_face_landmarks
            and len(results.multi_face_landmarks) > 0
        )

        if not face_ok:
            self._lost += 1
            if self._last_out is not None and self._lost < self._LOST_RESET_THRESHOLD:
                out = replace(
                    self._last_out,
                    confidence=max(0.0, self._last_out.confidence * 0.9),
                    stability_score=max(0.0, self._last_out.stability_score * 0.95),
                    calibrating=is_cal,
                )
                if out.emotion_scores is None:
                    out.emotion_scores = {e: 0.0 for e in EmotionEnum}
            else:
                if self._lost >= self._LOST_RESET_THRESHOLD:
                    self.au_calc.reset_baseline()
                    self.head_pose.reset_filter()
                    self.apex_detector.reset()
                    self._pose_bad = False
                out = self._neutral_out(is_cal)
            # Update deception for this placeholder
            out.deception_prob = self._compute_deception_prob(out)
            return out, evm_frame

        self._lost   = 0
        lm_obj       = results.multi_face_landmarks[0]

        le     = lm_obj.landmark[LM.LEFT_EYE_OUTER]
        re     = lm_obj.landmark[LM.RIGHT_EYE_OUTER]
        iod_px = float(np.linalg.norm(
            np.array([le.x*W, le.y*H]) - np.array([re.x*W, re.y*H])))

        if iod_px < 1.0:
            out = self._neutral_out(is_cal)
            out.deception_prob = self._compute_deception_prob(out)
            return out, evm_frame

        lm    = np.array([[p.x*W, p.y*H, p.z*iod_px] for p in lm_obj.landmark])
        lm2d  = lm[:, :2].astype(np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1   = int(max(0, lm2d[:, 0].min() - 8))
        y1   = int(max(0, lm2d[:, 1].min() - 8))
        x2   = int(min(W, lm2d[:, 0].max() + 8))
        y2   = int(min(H, lm2d[:, 1].max() + 8))

        macro_flow = None
        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            roi     = gray[y1:y2, x1:x2]
            roi_lm  = [(int(x - x1), int(y - y1)) for x, y in lm2d.tolist()]
            macro_flow = self.flow_track_macro.compute(roi, roi_lm)

        evm_gray  = cv2.cvtColor(evm_frame, cv2.COLOR_BGR2GRAY)
        micro_flow_mean = self._compute_micro_flow(evm_gray, lm2d, H, W)

        micro_emotion    = None
        micro_confidence = 0.0
        try:
            if self.micro_model.is_trained_for_micro:
                face_crop  = _crop_face(evm_frame, lm, padding=0.05)
                apex_frame = self.apex_detector.feed(face_crop, micro_flow_mean)
                if apex_frame is not None:
                    micro_emotion, micro_confidence = self.micro_model.infer(apex_frame)
                    if micro_confidence > 0.3:
                        logger.info(
                            f"Micro apex: {micro_emotion} ({micro_confidence:.2f})"
                        )
        except Exception as exc:
            logger.error(f"Micro inference failed: {exc}")
            micro_emotion = "UNKNOWN"
            micro_confidence = 0.0

        aus    = self.au_calc.compute(lm)
        ear_l  = eye_aspect_ratio(lm, LM.LEFT_EYE)
        ear_r  = eye_aspect_ratio(lm, LM.RIGHT_EYE)
        mar    = mouth_aspect_ratio(lm)
        pitch, yaw, roll = self.head_pose.estimate(lm)
        gaze_h, gaze_v   = estimate_gaze_2d(lm)

        # Head pose gating
        PITCH_THRESHOLD = 25.0
        YAW_THRESHOLD   = 25.0
        ROLL_THRESHOLD  = 15.0

        if pitch is None or yaw is None or roll is None:
            unreliable = True
        else:
            entry = (abs(pitch) > PITCH_THRESHOLD 
                    or abs(yaw) > YAW_THRESHOLD 
                    or abs(roll) > ROLL_THRESHOLD)
            exit_ = (abs(pitch) < PITCH_THRESHOLD * 0.8
                    and abs(yaw) < YAW_THRESHOLD * 0.8
                    and abs(roll) < ROLL_THRESHOLD * 0.8)
            if self._pose_bad:
                unreliable = not exit_
            else:
                unreliable = entry
            self._pose_bad = unreliable

        if unreliable:
            prev_em    = self._last_out.emotion   if self._last_out else EmotionEnum.NEUTRAL
            prev_conf  = max(0.2, (self._last_out.confidence if self._last_out else 0.0) * 0.92)
            raw_scores = {e: 0.0 for e in EmotionEnum}
            self.pattern.update(prev_em, prev_conf, AUActivations(), macro_flow)
            emotion, confidence = prev_em, prev_conf
        else:
            emotion, confidence, _, raw_scores = self.clf.classify(aus)
            self.pattern.update(emotion, confidence, aus, macro_flow)

        raw_scores_macro = dict(raw_scores)
        emotion, confidence, raw_scores = self._integrate_micro(
            raw_scores, emotion, confidence, micro_emotion, micro_confidence
        )

        self.fatigue.update(aus.AU45, mar)
        self.chart.update(raw_scores)

        out = MMOutput(
            emotion=emotion, confidence=confidence, aus=aus,
            ear_l=ear_l, ear_r=ear_r, mar=mar,
            pitch=pitch, yaw=yaw, roll=roll,
            gaze_h=gaze_h, gaze_v=gaze_v,
            flow_mag=macro_flow['mean'] if macro_flow else None,
            is_fatigued=self.fatigue.is_fatigued,
            is_forced=self.pattern.is_forced,
            stability_score=self.pattern.stability_score,
            calibrating=is_cal,
            micro_emotion=micro_emotion,
            micro_confidence=micro_confidence,
            emotion_scores=raw_scores,
            emotion_scores_macro=raw_scores_macro,
        )
        out.deception_prob = self._compute_deception_prob(out)   # <-- BUG-MM1 computed

        self._last_out = replace(out)
        return out, evm_frame


if __name__ == "__main__":
    # Standalone test: open camera, feed frames to MMSystem.process_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit(1)
    
    mm_sys = MMSystem(frame_size=(800, 600), enable_evm=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 600))
        out, vis = mm_sys.process_frame(frame)
        
        # Use the built‑in HUDRenderer to draw on the frame
        mm_sys.hud.draw(vis, out, 30.0, mm_sys.chart)   # fps placeholder
        cv2.imshow("MM System Test", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()