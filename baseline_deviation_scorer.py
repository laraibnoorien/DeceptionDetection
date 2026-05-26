"""
Baseline Deviation Scorer

Computes per-modality deviations from subject baseline:
- MM: Microexpression AU activation deviation
- PM: Physiological HR/HRV deviation
- SM: Speech acoustic feature deviation
- TM: Text linguistic pattern deviation

Deviations feed into fusion weighting:
  Higher deviation = Higher weight for that modality (flag anomalies)
"""

import logging
import numpy as np
from typing import Dict, Optional
from subject_behavior_profile import SubjectBehaviorProfile

logger = logging.getLogger(__name__)


class BaselineDeviationScorer:
    """
    Computes baseline deviation scores for all modalities.
    
    Formula: deviation = (live_output - baseline_mean) / (baseline_std + epsilon)
    Result: Float in range [0, 1] where:
      - 0 = No deviation from baseline
      - 1 = Extreme deviation (> 3 standard deviations)
    """
    
    def __init__(self, behavior_profile: Optional[SubjectBehaviorProfile] = None):
        """
        Initialize deviation scorer with subject baseline profile.
        
        Args:
            behavior_profile: SubjectBehaviorProfile with baseline data
        """
        self.behavior_profile = behavior_profile
        self.epsilon = 1e-6  # Prevent division by zero
        
    def score_mm_deviation(self, mm_out: Dict) -> Dict:
        """
        Microexpression deviation from baseline.
        
        Compare:
        - Emotion label confidence
        - AU activation patterns
        - Facial asymmetry
        
        Args:
            mm_out: {"emotion": str, "confidence": float, "au_activations": [...]}
        
        Returns:
            {
                "emotion": str,
                "confidence": float,
                "baseline_confidence_mean": float,
                "confidence_deviation": float (0-1),
                "au_deviation": float (0-1),
                "total_deviation": float (0-1)
            }
        """
        if not mm_out or not self.behavior_profile:
            return {
                "emotion": mm_out.get("emotion", "neutral") if mm_out else "neutral",
                "confidence": mm_out.get("confidence", 0) if mm_out else 0,
                "baseline_confidence_mean": 0.5,
                "confidence_deviation": 0,
                "au_deviation": 0,
                "total_deviation": 0
            }
        
        emotion = mm_out.get("emotion", "neutral")
        confidence = float(mm_out.get("confidence", 0))
        
        # Get baseline for this emotion
        mm_baseline = self.behavior_profile.mm_baseline
        emotion_baseline = mm_baseline.emotion_baselines.get(emotion)
        
        if not emotion_baseline:
            # Use average baseline if emotion-specific not found
            baseline_conf_mean = 0.5
            baseline_conf_std = 0.15
        else:
            # Use apex intensity or average AU activation as confidence proxy
            baseline_conf_mean = emotion_baseline.apex_intensity_mean or 0.5
            baseline_conf_std = emotion_baseline.apex_intensity_std or 0.15
        
        # Compute confidence deviation: (live - mean) / std
        confidence_deviation = abs(confidence - baseline_conf_mean) / (baseline_conf_std + self.epsilon)
        confidence_deviation = min(confidence_deviation / 3.0, 1.0)  # Normalize to 0-1
        
        # AU activation deviation (mock - would need real AU data from macro_v7)
        au_deviation = 0  # TODO: Extract real AU activations from mm_out
        
        # Total deviation: average of components
        total_deviation = (confidence_deviation + au_deviation) / 2.0
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "baseline_confidence_mean": baseline_conf_mean,
            "confidence_deviation": confidence_deviation,
            "au_deviation": au_deviation,
            "total_deviation": total_deviation
        }
    
    def score_pm_deviation(self, pm_out: Dict) -> Dict:
        """
        Physiological deviation from baseline.
        
        Compare:
        - Heart rate elevation
        - HRV change
        - Stress level
        - SpO2 deviation
        
        Args:
            pm_out: {"heart_rate": float, "hrv": float, "stress_probability": float, ...}
        
        Returns:
            {
                "heart_rate": float,
                "baseline_hr_mean": float,
                "hr_deviation": float (0-1),
                "stress_deviation": float (0-1),
                "total_deviation": float (0-1)
            }
        """
        if not pm_out or not self.behavior_profile:
            return {
                "heart_rate": pm_out.get("heart_rate", 0) if pm_out else 0,
                "baseline_hr_mean": 70,
                "hr_deviation": 0,
                "stress_deviation": 0,
                "total_deviation": 0
            }
        
        heart_rate = float(pm_out.get("heart_rate", 0))
        stress_prob = float(pm_out.get("stress_probability", 0))
        
        # Get baseline physiological metrics
        pm_baseline = self.behavior_profile.pm_baseline
        baseline_hr_mean = pm_baseline.hr_baseline_mean or 70
        baseline_hr_std = pm_baseline.hr_baseline_std or 8
        baseline_stress = pm_baseline.stress_baseline or 0.3
        
        # HR deviation: (live_hr - baseline) / std
        hr_deviation = abs(heart_rate - baseline_hr_mean) / (baseline_hr_std + self.epsilon)
        hr_deviation = min(hr_deviation / 3.0, 1.0)  # Normalize to 0-1
        
        # Stress deviation: how much above baseline stress
        stress_deviation = (stress_prob - baseline_stress) / (0.3 + self.epsilon)
        stress_deviation = max(stress_deviation, 0)  # Only positive deviations
        stress_deviation = min(stress_deviation, 1.0)  # Cap at 1.0
        
        # Total deviation: weighted average (stress more important for deception)
        total_deviation = (hr_deviation * 0.4 + stress_deviation * 0.6)
        
        return {
            "heart_rate": heart_rate,
            "baseline_hr_mean": baseline_hr_mean,
            "hr_deviation": hr_deviation,
            "stress_deviation": stress_deviation,
            "total_deviation": total_deviation
        }
    
    def score_sm_deviation(self, sm_out: Dict) -> Dict:
        """
        Speech acoustic deviation from baseline.
        
        Compare:
        - Pitch variance
        - Speech rate changes
        - Filler word frequency
        - Stress in voice
        
        Args:
            sm_out: {"pitch": float, "speech_rate": float, "stress_prob": float, ...}
        
        Returns:
            {
                "pitch": float,
                "baseline_pitch_mean": float,
                "pitch_deviation": float (0-1),
                "stress_deviation": float (0-1),
                "total_deviation": float (0-1)
            }
        """
        if not sm_out or not self.behavior_profile:
            return {
                "pitch": sm_out.get("pitch", 0) if sm_out else 0,
                "baseline_pitch_mean": 150,
                "pitch_deviation": 0,
                "stress_deviation": 0,
                "total_deviation": 0
            }
        
        pitch = float(sm_out.get("pitch", 0))
        speech_stress = float(sm_out.get("stress_prob", 0))
        
        # Get baseline speech metrics
        sm_baseline = self.behavior_profile.sm_baseline
        baseline_pitch = sm_baseline.pitch_mean or 150
        baseline_pitch_std = sm_baseline.pitch_std or 25
        baseline_speech_stress = 0.3  # Default value (no specific field)
        
        # Pitch deviation: (live - baseline) / std
        pitch_deviation = abs(pitch - baseline_pitch) / (baseline_pitch_std + self.epsilon)
        pitch_deviation = min(pitch_deviation / 3.0, 1.0)  # Normalize to 0-1
        
        # Speech stress deviation
        stress_deviation = (speech_stress - baseline_speech_stress) / (0.3 + self.epsilon)
        stress_deviation = max(stress_deviation, 0)
        stress_deviation = min(stress_deviation, 1.0)
        
        # Total deviation: weighted average
        total_deviation = (pitch_deviation * 0.4 + stress_deviation * 0.6)
        
        return {
            "pitch": pitch,
            "baseline_pitch_mean": baseline_pitch,
            "pitch_deviation": pitch_deviation,
            "stress_deviation": stress_deviation,
            "total_deviation": total_deviation
        }
    
    def score_tm_deviation(self, tm_out: Dict) -> Dict:
        """
        Text linguistic deviation from baseline.
        
        Compare:
        - Contradiction markers
        - Hesitation patterns
        - Semantic drift
        - Sentiment changes
        
        Args:
            tm_out: {"deception_probability": float, "contradictions": int, ...}
        
        Returns:
            {
                "deception_probability": float,
                "baseline_deception": float,
                "contradiction_deviation": float (0-1),
                "hesitation_deviation": float (0-1),
                "total_deviation": float (0-1)
            }
        """
        if not tm_out or not self.behavior_profile:
            return {
                "deception_probability": tm_out.get("deception_probability", 0) if tm_out else 0,
                "baseline_deception": 0.2,
                "contradiction_deviation": 0,
                "hesitation_deviation": 0,
                "total_deviation": 0
            }
        
        deception_prob = float(tm_out.get("deception_probability", 0))
        contradictions = int(tm_out.get("contradictions", 0))
        hesitations = int(tm_out.get("hesitations", 0))
        
        # Get baseline text metrics
        tm_baseline = self.behavior_profile.tm_baseline
        baseline_deception = 0.2  # Default value (no specific deception_prob field)
        baseline_contradictions = tm_baseline.contradiction_tendency or 0.1
        baseline_hesitations = tm_baseline.hesitation_pattern or 0.1
        
        # Deception probability deviation
        deception_deviation = abs(deception_prob - baseline_deception) / (0.2 + self.epsilon)
        deception_deviation = min(deception_deviation, 1.0)
        
        # Contradiction deviation (per utterance)
        contradiction_deviation = contradictions / (baseline_contradictions + 0.01)
        contradiction_deviation = min(contradiction_deviation / 3.0, 1.0)
        
        # Hesitation deviation
        hesitation_deviation = hesitations / (baseline_hesitations + 0.01)
        hesitation_deviation = min(hesitation_deviation / 3.0, 1.0)
        
        # Total deviation: weighted by importance
        total_deviation = (
            deception_deviation * 0.5 +
            contradiction_deviation * 0.3 +
            hesitation_deviation * 0.2
        )
        
        return {
            "deception_probability": deception_prob,
            "baseline_deception": baseline_deception,
            "contradiction_deviation": contradiction_deviation,
            "hesitation_deviation": hesitation_deviation,
            "total_deviation": total_deviation
        }
    
    def compute_all_deviations(
        self,
        mm_out: Dict,
        pm_out: Dict,
        sm_out: Dict,
        tm_out: Dict
    ) -> Dict:
        """
        Compute deviations for all modalities.
        
        Returns:
            {
                "mm": {scores},
                "pm": {scores},
                "sm": {scores},
                "tm": {scores},
                "cross_modality_incongruence": {incongruence_scores}
            }
        """
        deviations = {
            "mm": self.score_mm_deviation(mm_out),
            "pm": self.score_pm_deviation(pm_out),
            "sm": self.score_sm_deviation(sm_out),
            "tm": self.score_tm_deviation(tm_out)
        }
        
        # Cross-modality incongruence
        incongruence = self._compute_incongruence(
            deviations["mm"],
            deviations["pm"],
            deviations["sm"],
            deviations["tm"]
        )
        deviations["cross_modality_incongruence"] = incongruence
        
        return deviations
    
    def _compute_incongruence(
        self,
        mm_dev: Dict,
        pm_dev: Dict,
        sm_dev: Dict,
        tm_dev: Dict
    ) -> Dict:
        """
        Detect cross-modality incongruence:
        - Facial vs verbal mismatch
        - Speech vs physiological mismatch
        - Emotional consistency
        - Contradiction patterns
        
        Returns:
            {
                "facial_verbal_incongruence": float (0-1),
                "speech_physio_incongruence": float (0-1),
                "emotional_consistency": float (0-1),
                "overall_incongruence": float (0-1)
            }
        """
        
        # Facial vs verbal: if MM shows emotion but TM shows deception
        mm_emotion_deviation = mm_dev.get("total_deviation", 0)
        tm_deception_deviation = tm_dev.get("total_deviation", 0)
        facial_verbal_incongruence = abs(mm_emotion_deviation - tm_deception_deviation)
        
        # Speech vs physiological: if SM is calm but PM shows stress
        sm_stress_dev = sm_dev.get("stress_deviation", 0)
        pm_stress_dev = pm_dev.get("stress_deviation", 0)
        speech_physio_incongruence = abs(sm_stress_dev - pm_stress_dev)
        
        # Emotional consistency: all deviate similarly or differently
        all_deviations = [
            mm_dev.get("total_deviation", 0),
            pm_dev.get("total_deviation", 0),
            sm_dev.get("total_deviation", 0),
            tm_dev.get("total_deviation", 0)
        ]
        deviation_mean = np.mean(all_deviations)
        deviation_std = np.std(all_deviations)
        emotional_consistency = 1.0 - (deviation_std / (deviation_mean + 0.1)) if deviation_mean > 0 else 1.0
        emotional_consistency = max(min(emotional_consistency, 1.0), 0.0)
        
        # Overall incongruence: average of all incongruence measures
        overall_incongruence = (
            facial_verbal_incongruence * 0.35 +
            speech_physio_incongruence * 0.35 +
            (1.0 - emotional_consistency) * 0.3
        )
        
        return {
            "facial_verbal_incongruence": float(min(facial_verbal_incongruence, 1.0)),
            "speech_physio_incongruence": float(min(speech_physio_incongruence, 1.0)),
            "emotional_consistency": float(emotional_consistency),
            "overall_incongruence": float(overall_incongruence)
        }
