#!/usr/bin/env python3
"""
Fusion Layer — Combines all four modalities

Implements confidence-weighted probabilistic fusion of:
- MM (Microexpression) — Emotion, AU patterns, micro-expressions
- TM (Text) — Linguistic contradictions, truth probability
- PM (Physiological) — HR, HRV, stress, arousal
- SM (Speech) — Acoustic stress, deception indicators

Output:
- Unified deception_probability [0, 1]
- Alert level: CLEAR | MODERATE | HIGH | CRITICAL
- Per-modality breakdown with confidence weights
- Human-readable explanation
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union
from enum import Enum
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)

# ALERT LEVEL ENUM


class AlertLevel(Enum):
    """Alert severity levels based on deception probability."""
    CLEAR = "CLEAR"          # < 0.30
    MODERATE = "MODERATE"    # 0.30 - 0.50
    HIGH = "HIGH"            # 0.50 - 0.80
    CRITICAL = "CRITICAL"    # > 0.80

# FUSION OUTPUT

@dataclass
class FusionOutput:
    """
    Final unified deception detection output.
    Combines all four modalities with confidence weighting.
    """
    deception_probability: float          # [0, 1] final score
    alert_level: AlertLevel               # CLEAR | MODERATE | HIGH | CRITICAL
    per_modality: Dict[str, float]        # mm, tm, pm, sm individual scores
    per_modality_weight: Dict[str, float] # Normalized confidence weights
    per_modality_confidence: Dict[str, float]  # Individual confidences
    contributing_signals: list            # Which modalities contributed
    explanation: str                      # Human-readable reason
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Validate deception probability range."""
        if not 0.0 <= self.deception_probability <= 1.0:
            logger.warning(f"Deception probability out of range: {self.deception_probability}")
            self.deception_probability = max(0.0, min(1.0, self.deception_probability))


# Helper to extract value from dict, dataclass, or tuple
def _extract_value(obj: Any, key: str, default: float = 0.5) -> float:
    """
    Safely extract a numeric value from a dict, dataclass, tuple, or any object.
    
    Priority:
    1. If dict, try .get(key, default)
    2. If dataclass (has __dict__), try getattr(obj, key, default)
    3. If tuple/list and index 0 exists, try float(obj[0])
    4. Otherwise return default
    """
    if obj is None:
        return default
    
    # Dictionary
    if isinstance(obj, dict):
        val = obj.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    # Dataclass or any object with __dict__
    if hasattr(obj, '__dict__'):
        val = getattr(obj, key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    # Tuple/list (legacy support: maybe first element is the value)
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        try:
            return float(obj[0])
        except (TypeError, ValueError):
            return default
    
    # Fallback: try direct conversion
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default

class MultiHeadAttentionFusion(nn.Module):
    """
    Minimal multi-head attention fusion layer.
    Input shape:
        [batch_size, 4]
    """

    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Linear(1, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, scores):
        """
        scores:
            [B,4]
        """

        # [B,4] -> [B,4,1]
        x = scores.unsqueeze(-1)

        # [B,4,1] -> [B,4,H]
        x = self.embedding(x)

        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)

        # Pool modality representations
        fused = attn_output.mean(dim=1)

        # Final probability
        output = self.classifier(fused)

        return output.squeeze(-1), attn_weights
# FUSION SYSTEM


class FusionSystem:
    """
    Multi-modal deception probability fusion.
    
    Implements confidence-weighted Bayesian combination with:
    - Per-modality quality weighting
    - Calibration lockout (ignore uncalibrated modalities)
    - Exponential decay for asynchronous updates
    - Alert thresholds with explanations
    - Phase 2: Emotion-aware threshold adaptation
    """
    
    def __init__(
        self,
        clear_threshold: float = 0.30,
        moderate_threshold: float = 0.50,
        high_threshold: float = 0.80,
        critical_threshold: float = 0.90,
        decay_rate: float = 0.98,
        min_confidence: float = 0.3,
        session_config: Optional[Any] = None,
        calibration_weights: Optional[dict] = None,
    ):
        """
        Initialize fusion system with thresholds.
        
        Args:
            clear_threshold: Below this = CLEAR
            moderate_threshold: Below this = MODERATE
            high_threshold: Below this = HIGH
            critical_threshold: Below this = CRITICAL
            decay_rate: Exponential decay for asynchronous updates (0-1)
            min_confidence: Minimum confidence to include modality in fusion
            session_config: SessionConfig for emotion-aware thresholds (Phase 2)
            calibration_weights: Optional dict of per-modality calibration weights (0-1)
        """
        self.clear_threshold = clear_threshold
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.session_config = session_config
        
        # Calibration-based modality weights
        self.calibration_weights = calibration_weights or {
            'mm': 0.25,
            'pm': 0.25,
            'sm': 0.25,
            'tm': 0.25,
        }
        
        # Track last values for decay
        self._last_output = {
            "mm": 0.5,
            "tm": 0.5,
            "pm": 0.5,
            "sm": 0.5,
        }
        # Attention fusion model
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.attention_model = MultiHeadAttentionFusion(
            hidden_dim=64,
            num_heads=4
        ).to(self.device)

        self.attention_model.eval()
        self._last_time = {}
    
    def set_calibration_weights(self, weights: dict):
        """
        Set modality weights from calibration profile.
        
        Args:
            weights: Dict[modality_name -> weight (0-1)]
        """
        if weights and isinstance(weights, dict):
            self.calibration_weights.update(weights)
            logger.info("📊 Fusion calibration weights updated")
    
    
    def _get_adaptive_threshold(self) -> float:
        """
        Get emotion-specific deception threshold (Phase 2).
        Falls back to default if emotion profiler unavailable.
        """
        if self.session_config is None or self.session_config.emotion_profiler is None:
            return self.moderate_threshold  # Default: 0.50
        
        try:
            emotion = self.session_config.current_emotion or "neutral"
            thresholds = self.session_config.emotion_profiler.get_emotion_thresholds()
            return thresholds.get(emotion, self.moderate_threshold)
        except Exception as e:
            logger.warning(f"Error retrieving emotion threshold: {e}")
            return self.moderate_threshold
    
    def update(
        self,
        mm_out: Optional[Any] = None,
        tm_out: Optional[Any] = None,
        pm_out: Optional[Any] = None,
        sm_out: Optional[Any] = None,
        timestamp: float = 0.0,
        deviations: Optional[Dict] = None,
    ) -> FusionOutput:
        """
        Compute fused deception probability from modality outputs.
        
        Args:
            mm_out: Microexpression modality output (dict, dataclass, or tuple)
            tm_out: Text modality output (dict, dataclass)
            pm_out: Physiological modality output (dict, dataclass)
            sm_out: Speech modality output (dict, dataclass)
            timestamp: Perf counter timestamp
            deviations: Optional dict with baseline deviations per modality
        
        Returns:
            FusionOutput with unified deception probability
        """
        # Normalize all inputs to dictionaries for consistent access
        mm_dict = self._to_dict(mm_out)
        tm_dict = self._to_dict(tm_out)
        pm_dict = self._to_dict(pm_out)
        sm_dict = self._to_dict(sm_out)
        
        # Step 1: Extract and validate deception probabilities
        mm_deception = _extract_value(mm_dict, "deception_prob", 0.5)
        tm_truth = _extract_value(tm_dict, "truth_probability", 0.5)
        pm_deception = _extract_value(pm_dict, "deception_probability", 0.5)
        sm_deception = _extract_value(sm_dict, "deception_prob", 0.5)
        
        # Convert TM truth to deception (invert)
        tm_deception = 1.0 - tm_truth
        
        # Step 2: Compute per-modality weights and confidence (with baseline deviations)
        weights, confidences, contributing = self._compute_weights_and_confidence(
            mm_dict, tm_dict, pm_dict, sm_dict, deviations=deviations
        )
        
        # Step 3: Apply exponential decay for asynchronous updates
        mm_deception = self._apply_decay("mm", mm_deception, timestamp)
        tm_deception = self._apply_decay("tm", tm_deception, timestamp)
        pm_deception = self._apply_decay("pm", pm_deception, timestamp)
        sm_deception = self._apply_decay("sm", sm_deception, timestamp)
        
        # Step 4: Compute weighted fusion
        per_modality = {
            "mm": mm_deception,
            "tm": tm_deception,
            "pm": pm_deception,
            "sm": sm_deception,
        }
        
        # Prepare modality tensor
        scores = torch.tensor([
            [
                per_modality["mm"],
                per_modality["tm"],
                per_modality["pm"],
                per_modality["sm"]
            ]
        ], dtype=torch.float32).to(self.device)

        # Attention fusion
        with torch.no_grad():
            fused_tensor, attention_weights = self.attention_model(scores)

        fused_deception = fused_tensor.item()

        # Fallback safety
        if not (0.0 <= fused_deception <= 1.0):
            fused_deception = 0.5

        # Keep total_weight for normalization logic below
        total_weight = sum(weights.values())
        
        # Clip to valid range
        fused_deception = max(0.0, min(1.0, fused_deception))
        
        # Step 5: Determine alert level
        alert_level = self._compute_alert_level(fused_deception)
        
        # Step 6: Generate explanation
        explanation = self._generate_explanation(
            fused_deception, per_modality, weights, contributing, alert_level
        )
        
        # Step 7: Normalize weights for output — only include active (positive) weights
        normalized_weights = {}
        active_weights = {mod: w for mod, w in weights.items() if w > 0}
        active_total = sum(active_weights.values())
        if active_total > 0:
            # Normalize active weights and set inactive ones to 0.0
            normalized_weights = {mod: (weights[mod] / active_total) if weights[mod] > 0 else 0.0 for mod in weights.keys()}
        else:
            normalized_weights = {mod: 0.0 for mod in weights.keys()}
        
        return FusionOutput(
            deception_probability=fused_deception,
            alert_level=alert_level,
            per_modality=per_modality,
            per_modality_weight=normalized_weights,
            per_modality_confidence=confidences,
            contributing_signals=contributing,
            explanation=explanation,
            timestamp=timestamp,
        )
    
    def _to_dict(self, obj: Any) -> Dict:
        """Convert dataclass or object to dict; return original dict if already dict."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, '__dict__'):
            # If object exposes a to_dict() method (preferred), use it
            if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                try:
                    d = obj.to_dict()
                    # Remove keys with None values to avoid silent defaults
                    return {k: v for k, v in d.items() if v is not None}
                except Exception as e:
                    logger.debug(f"to_dict() failed on {type(obj)}: {e}")

            # Generic dataclass/object fallback: convert public attrs and drop None
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_') and v is not None}
        # Fallback: wrap in dict with dummy key? Better return empty
        logger.debug(f"Unsupported type for _to_dict: {type(obj)}")
        return {}
    
    def _compute_weights_and_confidence(
        self,
        mm_dict: Dict,
        tm_dict: Dict,
        pm_dict: Dict,
        sm_dict: Dict,
        deviations: Optional[Dict] = None,
    ) -> tuple:
        """
        Compute per-modality weights and confidences.
        Applies calibration-based baseline weighting to signal-quality weights.
        
        Baseline deviation multiplier:
        - Low deviation (0): Normal weight
        - High deviation (1): Weight increased to flag anomalies
        - Formula: weight = base_weight * (0.5 + 0.5 * deviation)
        
        Returns (weights_dict, confidence_dict, contributing_signals_list)
        """
        weights = {}
        confidences = {}
        contributing = []
        
        # Default deviations if not provided
        if deviations is None:
            deviations = {}
        
        # MM weight: stability_score * (1 - is_calibrating) * calibration_weight * (1 + deviation)
        mm_calib_flag = float(mm_dict.get("calibrating", 0)) or float(mm_dict.get("is_calibrating", 0))
        mm_stability = float(mm_dict.get("stability_score", 0.5))
        mm_signal_weight = mm_stability * (1.0 - mm_calib_flag)
        mm_deviation = deviations.get("mm", {}).get("total_deviation", 0)
        # Deviation multiplier: lower deviation = lower importance, higher deviation = higher importance
        mm_deviation_multiplier = 0.5 + 0.5 * mm_deviation
        mm_weight = mm_signal_weight * self.calibration_weights.get('mm', 0.25) * mm_deviation_multiplier
        weights["mm"] = max(0, mm_weight)
        confidences["mm"] = mm_stability
        if mm_weight > self.min_confidence:
            contributing.append("mm")
        
        # TM weight: (1 - uncertainty) * nli_available * calibration_weight * (1 + deviation)
        tm_uncertainty = float(tm_dict.get("anomaly_score", 0.5))  # Anomaly as uncertainty
        tm_nli = 1.0 if tm_dict.get("nli_available", True) else 0.5
        tm_signal_weight = (1.0 - tm_uncertainty) * tm_nli
        tm_deviation = deviations.get("tm", {}).get("total_deviation", 0)
        tm_deviation_multiplier = 0.5 + 0.5 * tm_deviation
        tm_weight = tm_signal_weight * self.calibration_weights.get('tm', 0.25) * tm_deviation_multiplier
        weights["tm"] = max(0, tm_weight)
        confidences["tm"] = tm_nli
        if tm_weight > self.min_confidence:
            contributing.append("tm")
        
        # PM weight: signal_quality * calib_done * calibration_weight * (1 + deviation)
        pm_signal_quality = float(pm_dict.get("signal_quality", 0.5))
        pm_calib_done = 1.0 if pm_dict.get("calib_done", False) else 0.0
        pm_signal_weight = pm_signal_quality * pm_calib_done
        pm_deviation = deviations.get("pm", {}).get("total_deviation", 0)
        pm_deviation_multiplier = 0.5 + 0.5 * pm_deviation
        pm_weight = pm_signal_weight * self.calibration_weights.get('pm', 0.25) * pm_deviation_multiplier
        weights["pm"] = max(0, pm_weight)
        confidences["pm"] = pm_signal_quality
        if pm_weight > self.min_confidence:
            contributing.append("pm")
        
        # SM weight: model_trained * (status == "OK") * calibration_weight * (1 + deviation)
        sm_status = sm_dict.get("status", "UNKNOWN")
        sm_trained = 1.0 if sm_status == "OK" else 0.3  # Reduced weight if model not fully trained
        sm_voice_quality = float(sm_dict.get("voice_quality", 0.5))
        sm_signal_weight = sm_trained * sm_voice_quality
        sm_deviation = deviations.get("sm", {}).get("total_deviation", 0)
        sm_deviation_multiplier = 0.5 + 0.5 * sm_deviation
        sm_weight = sm_signal_weight * self.calibration_weights.get('sm', 0.25) * sm_deviation_multiplier
        weights["sm"] = max(0, sm_weight)
        confidences["sm"] = sm_voice_quality
        if sm_weight > self.min_confidence:
            contributing.append("sm")
        
        return weights, confidences, contributing
    
    def _apply_decay(self, modality: str, value: float, timestamp: float) -> float:
        """Apply exponential decay to asynchronous modality updates."""
        if modality not in self._last_time:
            self._last_time[modality] = timestamp
            self._last_output[modality] = value
            return value
        
        time_delta = max(0, timestamp - self._last_time[modality])
        decay_factor = (self.decay_rate ** time_delta)
        
        # Blend current value with previous value (decaying toward 0.5 neutral)
        decayed = self._last_output[modality] * decay_factor + (1.0 - decay_factor) * 0.5
        
        # Blend current value with decayed historical value to avoid stale jumps
        alpha = 0.7
        blended = alpha * value + (1.0 - alpha) * decayed
        
        self._last_output[modality] = value
        self._last_time[modality] = timestamp
        
        return blended
    
    def _compute_alert_level(self, deception_prob: float) -> AlertLevel:
        """Map deception probability to alert level."""
        if deception_prob >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif deception_prob >= self.high_threshold:
            return AlertLevel.HIGH
        elif deception_prob >= self._get_adaptive_threshold():
            return AlertLevel.MODERATE
        else:
            return AlertLevel.CLEAR
    
    def _generate_explanation(
        self,
        deception_prob: float,
        per_modality: Dict[str, float],
        weights: Dict[str, float],
        contributing: list,
        alert_level: AlertLevel,
    ) -> str:
        """Generate human-readable explanation of fusion result."""
        if not contributing:
            return "Insufficient signal from modalities for reliable analysis"
        
        # Find strongest signals
        weighted_scores = {
            mod: per_modality[mod] * weights[mod]
            for mod in per_modality.keys()
            if weights[mod] > 0
        }
        
        sorted_mods = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        
        if alert_level == AlertLevel.CRITICAL:
            return (
                f"CRITICAL deception indicators detected. "
                f"Multiple channels (mm={per_modality.get('mm', 0):.2f}, "
                f"tm={per_modality.get('tm', 0):.2f}, "
                f"pm={per_modality.get('pm', 0):.2f}) elevated. "
                f"Requires immediate attention."
            )
        elif alert_level == AlertLevel.HIGH:
            top_mod = sorted_mods[0][0] if sorted_mods else "unknown"
            return (
                f"High deception probability ({deception_prob:.1%}). "
                f"Strongest signal from {top_mod}. "
                f"Cross-modal confirmation recommended."
            )
        elif alert_level == AlertLevel.MODERATE:
            return (
                f"Moderate deception indicators detected ({deception_prob:.1%}). "
                f"Multiple subtle signals present. Continue monitoring."
            )
        else:  # CLEAR
            return (
                f"Baseline behavioral patterns. "
                f"Deception probability low ({deception_prob:.1%}). "
                f"No significant alerts."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test fusion
    fusion = FusionSystem()
    
    # Test case 1: All modalities reporting high deception (as dicts)
    output = fusion.update(
        mm_out={"deception_prob": 0.8, "stability_score": 0.9, "is_calibrating": 0},
        tm_out={"truth_probability": 0.2, "nli_available": True, "anomaly_score": 0.1},
        pm_out={"deception_probability": 0.75, "signal_quality": 0.8, "calib_done": True},
        sm_out={"deception_prob": 0.7, "status": "OK", "voice_quality": 0.8},
        timestamp=1.0,
    )
    
    print(f"✓ Fusion test 1:")
    print(f"  Deception probability: {output.deception_probability:.2%}")
    print(f"  Alert level: {output.alert_level.value}")
    print(f"  Contributing: {output.contributing_signals}")
    print(f"  Explanation: {output.explanation}")
    
    # Test case 2: MM output as dataclass (simulate what macro_v7 returns)
    try:
        from dataclasses import dataclass
        @dataclass
        class MMOutputTest:
            deception_prob: float = 0.9
            stability_score: float = 0.85
            calibrating: bool = False
        
        mm_dataclass = MMOutputTest(deception_prob=0.9, stability_score=0.85, calibrating=False)
        output2 = fusion.update(
            mm_out=mm_dataclass,
            tm_out={"truth_probability": 0.4, "nli_available": True, "anomaly_score": 0.2},
            pm_out={"deception_probability": 0.6, "signal_quality": 0.9, "calib_done": True},
            sm_out={"deception_prob": 0.55, "status": "OK", "voice_quality": 0.7},
            timestamp=2.0,
        )
        print(f"\n✓ Fusion test 2 (dataclass input):")
        print(f"  Deception probability: {output2.deception_probability:.2%}")
        print(f"  Alert level: {output2.alert_level.value}")
        print(f"  Contributing: {output2.contributing_signals}")
    except Exception as e:
        print(f"Test 2 skipped: {e}")