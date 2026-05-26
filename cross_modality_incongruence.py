#!/usr/bin/env python3
"""
Cross-Modality Incongruence Detector

Detects behavioral incongruences across modalities:
- Facial expressions vs verbal statements (deception indicators)
- Speech patterns vs physiological signals (stress indicators)
- Emotional consistency across all modalities
- Temporal patterns and contradiction markers

Produces incongruence scores that feed into fusion weighting.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class CrossModalityIncongruenceDetector:
    """
    Detects and scores incongruences between modality outputs.
    
    Incongruences are strong indicators of deception:
    - Subject claims innocence (TM/SM) but shows stress signs (PM, MM)
    - Facial expression contradicts spoken words
    - Physiological arousal but calm speech
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize incongruence detector.
        
        Args:
            window_size: Number of frames for temporal analysis
        """
        self.window_size = window_size
        
        # History buffers for temporal analysis
        self.mm_history = deque(maxlen=window_size)
        self.pm_history = deque(maxlen=window_size)
        self.sm_history = deque(maxlen=window_size)
        self.tm_history = deque(maxlen=window_size)
    
    def detect_facial_verbal_incongruence(
        self,
        mm_deviation: Dict,
        tm_deviation: Dict,
        sm_deviation: Dict
    ) -> Dict:
        """
        Detect incongruence between facial expressions and verbal statements.
        
        Strong indicators:
        - Subject shows fear/surprise (MM high) but claims innocence (TM low)
        - Subject shows stress (MM AU activation) but calm speech (SM low)
        - Facial micro-expressions contradict speech content
        
        Args:
            mm_deviation: Microexpression deviation scores
            tm_deviation: Text modality deviation scores
            sm_deviation: Speech modality deviation scores
        
        Returns:
            {
                "facial_verbal_match": float (0-1),  # 0=incongruent, 1=congruent
                "facial_verbal_incongruence": float (0-1),  # Opposite
                "confidence": float (0-1)
            }
        """
        
        # Facial deviation indicates heightened emotion/stress
        facial_dev = mm_deviation.get("total_deviation", 0)
        
        # Text deception indication
        text_deception = tm_deviation.get("total_deviation", 0)
        
        # Speech stress indication
        speech_stress = sm_deviation.get("stress_deviation", 0)
        
        # If subject shows facial stress but claims innocence (low text deception):
        # This is incongruent - potential deception
        facial_verbal_incongruence = abs(facial_dev - text_deception)
        
        # If subject shows facial stress but calm speech:
        # This is also incongruent
        facial_speech_incongruence = abs(facial_dev - speech_stress)
        
        # Combined incongruence
        combined_incongruence = max(facial_verbal_incongruence, facial_speech_incongruence)
        
        # Match score (inverse of incongruence)
        facial_verbal_match = 1.0 - combined_incongruence
        
        return {
            "facial_verbal_match": max(0, min(facial_verbal_match, 1.0)),
            "facial_verbal_incongruence": max(0, min(combined_incongruence, 1.0)),
            "confidence": 0.8  # High confidence in this metric
        }
    
    def detect_speech_physio_incongruence(
        self,
        sm_deviation: Dict,
        pm_deviation: Dict
    ) -> Dict:
        """
        Detect incongruence between speech patterns and physiological signals.
        
        Strong indicators:
        - Subject claims calmness (SM low stress) but HR elevated (PM high)
        - Subject's voice calm but breathing irregular, sweating detected
        - No vocal markers of stress but physiological signals present
        
        Args:
            sm_deviation: Speech modality deviation scores
            pm_deviation: Physiological modality deviation scores
        
        Returns:
            {
                "speech_physio_match": float (0-1),  # 0=incongruent, 1=congruent
                "speech_physio_incongruence": float (0-1),
                "stress_masking": float (0-1),  # Subject masking stress
                "confidence": float (0-1)
            }
        """
        
        # Speech stress indicators
        speech_stress_dev = sm_deviation.get("stress_deviation", 0)
        speech_pitch_dev = sm_deviation.get("pitch_deviation", 0)
        
        # Physiological stress indicators
        physio_stress_dev = pm_deviation.get("stress_deviation", 0)
        physio_hr_dev = pm_deviation.get("hr_deviation", 0)
        
        # Aggregate stress measures
        speech_stress_total = (speech_stress_dev + speech_pitch_dev) / 2.0
        physio_stress_total = (physio_stress_dev + physio_hr_dev) / 2.0
        
        # If physio shows stress but speech doesn't: stress masking
        stress_masking = max(0, physio_stress_total - speech_stress_total)
        
        # General incongruence: difference in stress signals
        speech_physio_incongruence = abs(physio_stress_total - speech_stress_total)
        
        # Match score (inverse)
        speech_physio_match = 1.0 - speech_physio_incongruence
        
        return {
            "speech_physio_match": max(0, min(speech_physio_match, 1.0)),
            "speech_physio_incongruence": max(0, min(speech_physio_incongruence, 1.0)),
            "stress_masking": max(0, min(stress_masking, 1.0)),
            "confidence": 0.7  # Medium confidence (noise in physio signals)
        }
    
    def detect_emotional_consistency(
        self,
        mm_deviation: Dict,
        pm_deviation: Dict,
        sm_deviation: Dict,
        tm_deviation: Dict
    ) -> Dict:
        """
        Detect if all modalities show consistent emotional/stress levels.
        
        Consistent patterns (all high or all low deviation):
        - Genuine emotional response
        - Consistent behavior across all modalities
        
        Inconsistent patterns:
        - Some modalities show stress, others calm
        - Potential controlled deception
        
        Args:
            mm_deviation: Microexpression deviation
            pm_deviation: Physiological deviation
            sm_deviation: Speech deviation
            tm_deviation: Text deviation
        
        Returns:
            {
                "emotional_consistency": float (0-1),
                "inconsistency_score": float (0-1),
                "variance_across_modalities": float,
                "modality_deviations": list of floats,
                "most_deviant_modality": str
            }
        """
        
        # Get total deviations for each modality
        mm_dev = mm_deviation.get("total_deviation", 0)
        pm_dev = pm_deviation.get("total_deviation", 0)
        sm_dev = sm_deviation.get("total_deviation", 0)
        tm_dev = tm_deviation.get("total_deviation", 0)
        
        deviations = [mm_dev, pm_dev, sm_dev, tm_dev]
        modalities = ["MM", "PM", "SM", "TM"]
        
        # Calculate statistics
        deviation_mean = np.mean(deviations)
        deviation_std = np.std(deviations)
        
        # Consistency: lower variance = more consistent (all similar)
        # Higher variance = inconsistent (some high, some low)
        if deviation_mean > 0:
            consistency_coefficient = 1.0 - (deviation_std / deviation_mean)
            consistency_coefficient = max(0, min(consistency_coefficient, 1.0))
        else:
            consistency_coefficient = 1.0
        
        # Variance (normalized)
        normalized_variance = deviation_std / (deviation_mean + 0.1)
        
        # Most deviant modality
        max_idx = np.argmax(deviations)
        most_deviant = modalities[max_idx]
        
        return {
            "emotional_consistency": consistency_coefficient,
            "inconsistency_score": 1.0 - consistency_coefficient,
            "variance_across_modalities": float(normalized_variance),
            "modality_deviations": deviations,
            "most_deviant_modality": most_deviant,
            "mean_deviation": float(deviation_mean),
            "std_deviation": float(deviation_std)
        }
    
    def detect_contradiction_patterns(
        self,
        tm_deviation: Dict,
        sm_deviation: Dict
    ) -> Dict:
        """
        Detect contradiction patterns in text and speech.
        
        Indicators:
        - Subject contradicts previous statements (TM)
        - Speech contains hesitations or filler words when making claims
        - Semantic drift (changing topic, backtracking)
        
        Args:
            tm_deviation: Text modality deviation scores
            sm_deviation: Speech modality deviation scores
        
        Returns:
            {
                "contradiction_level": float (0-1),
                "hesitation_level": float (0-1),
                "semantic_drift": float (0-1),
                "overall_contradiction": float (0-1)
            }
        """
        
        # Text contradictions
        text_contradictions = tm_deviation.get("contradiction_deviation", 0)
        
        # Speech hesitations
        speech_hesitations = sm_deviation.get("total_deviation", 0)
        
        # Semantic drift (if deception is high, likely drift)
        text_deception = tm_deviation.get("total_deviation", 0)
        semantic_drift = text_deception * 0.7  # Assumption: high deception correlates with drift
        
        # Overall contradiction score
        overall_contradiction = (text_contradictions * 0.4 + speech_hesitations * 0.3 + semantic_drift * 0.3)
        
        return {
            "contradiction_level": max(0, min(text_contradictions, 1.0)),
            "hesitation_level": max(0, min(speech_hesitations, 1.0)),
            "semantic_drift": max(0, min(semantic_drift, 1.0)),
            "overall_contradiction": max(0, min(overall_contradiction, 1.0))
        }
    
    def compute_all_incongruences(
        self,
        deviations: Dict
    ) -> Dict:
        """
        Compute all cross-modality incongruences.
        
        Args:
            deviations: Dict with deviation scores for all modalities
        
        Returns:
            Comprehensive incongruence report
        """
        
        mm_dev = deviations.get("mm", {})
        pm_dev = deviations.get("pm", {})
        sm_dev = deviations.get("sm", {})
        tm_dev = deviations.get("tm", {})
        
        # Detect all incongruence types
        facial_verbal = self.detect_facial_verbal_incongruence(mm_dev, tm_dev, sm_dev)
        speech_physio = self.detect_speech_physio_incongruence(sm_dev, pm_dev)
        emotional_consistency = self.detect_emotional_consistency(mm_dev, pm_dev, sm_dev, tm_dev)
        contradictions = self.detect_contradiction_patterns(tm_dev, sm_dev)
        
        # Aggregate incongruence score
        incongruence_weights = {
            "facial_verbal": 0.3,
            "speech_physio": 0.3,
            "emotional_consistency": 0.2,
            "contradictions": 0.2
        }
        
        overall_incongruence = (
            facial_verbal["facial_verbal_incongruence"] * incongruence_weights["facial_verbal"] +
            speech_physio["speech_physio_incongruence"] * incongruence_weights["speech_physio"] +
            (1.0 - emotional_consistency["emotional_consistency"]) * incongruence_weights["emotional_consistency"] +
            contradictions["overall_contradiction"] * incongruence_weights["contradictions"]
        )
        
        # Deception risk based on incongruences
        deception_risk = 0.5 * overall_incongruence + 0.5 * (
            tm_dev.get("total_deviation", 0) * 0.3 +
            pm_dev.get("stress_deviation", 0) * 0.3 +
            facial_verbal["facial_verbal_incongruence"] * 0.4
        )
        
        return {
            "facial_verbal": facial_verbal,
            "speech_physio": speech_physio,
            "emotional_consistency": emotional_consistency,
            "contradictions": contradictions,
            "overall_incongruence": max(0, min(overall_incongruence, 1.0)),
            "deception_risk_incongruence": max(0, min(deception_risk, 1.0)),
            "incongruence_weights": incongruence_weights
        }
    
    def update_history(self, deviations: Dict):
        """Update temporal history for pattern analysis."""
        self.mm_history.append(deviations.get("mm", {}))
        self.pm_history.append(deviations.get("pm", {}))
        self.sm_history.append(deviations.get("sm", {}))
        self.tm_history.append(deviations.get("tm", {}))
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns across history window.
        
        Returns:
            {
                "mm_trend": float,  # Increasing/decreasing stress
                "pm_trend": float,
                "sm_trend": float,
                "tm_trend": float,
                "escalation_pattern": bool,  # Stress escalating over time
                "stabilization_pattern": bool  # Stress stabilizing
            }
        """
        
        if len(self.mm_history) < 5:
            return {}
        
        # Compute trends for each modality
        mm_devs = [h.get("total_deviation", 0) for h in list(self.mm_history)[-10:]]
        pm_devs = [h.get("total_deviation", 0) for h in list(self.pm_history)[-10:]]
        sm_devs = [h.get("total_deviation", 0) for h in list(self.sm_history)[-10:]]
        tm_devs = [h.get("total_deviation", 0) for h in list(self.tm_history)[-10:]]
        
        # Trend: slope of linear fit
        x = np.arange(len(mm_devs))
        mm_trend = np.polyfit(x, mm_devs, 1)[0] if len(mm_devs) > 1 else 0
        pm_trend = np.polyfit(x, pm_devs, 1)[0] if len(pm_devs) > 1 else 0
        sm_trend = np.polyfit(x, sm_devs, 1)[0] if len(sm_devs) > 1 else 0
        tm_trend = np.polyfit(x, tm_devs, 1)[0] if len(tm_devs) > 1 else 0
        
        # Pattern detection
        all_trends = [mm_trend, pm_trend, sm_trend, tm_trend]
        escalation = sum(1 for t in all_trends if t > 0.01) >= 3
        stabilization = max(abs(t) for t in all_trends) < 0.005
        
        return {
            "mm_trend": float(mm_trend),
            "pm_trend": float(pm_trend),
            "sm_trend": float(sm_trend),
            "tm_trend": float(tm_trend),
            "escalation_pattern": escalation,
            "stabilization_pattern": stabilization
        }
