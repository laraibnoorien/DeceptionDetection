#!/usr/bin/env python3
"""
Text Modality System v4 - REAL-TIME SPEECH INTEGRATION
=======================================================

Optimized for:
1. speech_recognition (Google STT) as primary backend
2. Continuous streaming with real-time updates in solo mode
3. Complete deception detection pipeline:
   - Audio streaming → Speech recognition (Google API)
   - NLI contradiction detection
   - Behavioral profiler (per-speaker baseline)
   - Temporal dynamics tracking
   - Entity memory updates (spaCy NER)
   - Truth probability fusion (sigmoid-mapped risk score)
   - Deception probability with real-time output

4. Dual-mode operation: standalone (continuous stream) vs integrated (main.py)
5. Per-speaker session management with baseline learning

Architecture:
┌──────────────────────────────────────┐
│ Live Microphone Audio Stream          │
│ (Continuous, real-time)              │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Silence Detection & Utterance Segmentation
│ Split continuous stream into utterances
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Speech Recognition (Google API)      │
│ Real-time transcription              │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Linguistic Features → Behavioral      │
│ Profiler → Temporal Dynamics →        │
│ NLI Contradiction → Entity Memory →   │
│ Risk Fusion → Truth Probability      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Real-Time Output                      │
│ Display results immediately           │
│ Continue streaming...                 │
└──────────────────────────────────────┘
"""

import os
import sys
import math
import time
import logging
import asyncio
import json
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Callable, Any, Generator
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Cache configuration
_project_root = Path(__file__).resolve().parents[1]
_cache_dir = os.environ.get('TRANSFORMERS_CACHE') or str(_project_root / '.cache' / 'transformers')
_hf_home = os.environ.get('HF_HOME') or str(_project_root / '.cache' / 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = _cache_dir
os.environ['HF_HOME'] = _hf_home

if os.environ.get('HF_TOKEN'):
    os.environ['HUGGINGFACE_HUB_TOKEN'] = os.environ.get('HF_TOKEN')

try:
    Path(_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(_hf_home).mkdir(parents=True, exist_ok=True)
except Exception:
    logger.warning(f"Unable to create transformers cache dirs: {_cache_dir} {_hf_home}")

# Optional dependencies
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed – NLI disabled")

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        NLP = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("Downloading spacy model...")
        os.system("python -m spacy download en_core_web_sm")
        NLP = spacy.load("en_core_web_sm")
except Exception as e:
    SPACY_AVAILABLE = False
    NLP = None
    logger.warning(f"spaCy not available – entity extraction disabled: {e}")

# Speech recognition - PRIMARY: speech_recognition (Google STT)
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
    RECOGNIZER = sr.Recognizer()
except ImportError:
    SPEECH_AVAILABLE = False
    RECOGNIZER = None
    logger.warning("speech_recognition not installed – mic input disabled")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("pyaudio not installed – may affect audio capture")



# LINGUISTIC FEATURE EXTRACTOR (from v3, unchanged)

class LinguisticFeatureExtractor:
    """Extracts linguistic features for baseline and anomaly detection."""
    
    MULTI_FILLERS = ['you know', 'i mean', 'i think', 'i guess', 'could be', 'kind of', 'sort of']
    SINGLE_FILLERS = {'um', 'uh', 'like', 'basically', 'literally', 'actually', 'hmm', 'uh-huh'}
    CERTAINTY_MARKERS = {'definitely', 'absolutely', 'certainly', 'always', 'never', 'obviously', 
                         'clearly', 'sure', 'certain'}
    UNCERTAINTY_MARKERS = {'maybe', 'perhaps', 'could', 'might', 'seem', 'appear', 'possibly',
                           'probably', 'uncertain', 'doubt', 'hesitate'}
    POSITIVE_WORDS = {'happy', 'good', 'great', 'excellent', 'wonderful', 'nice', 'perfect',
                      'amazing', 'fantastic', 'love', 'beautiful', 'awesome'}
    NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'disgusted',
                      'hate', 'ugly', 'worst', 'disgusting', 'painful'}
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text."""
        if not text or len(text.strip()) == 0:
            return {
                "filler_rate": 0.0,
                "certainty_score": 0.0,
                "sentiment_polarity": 0.0,
            }
        
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Filler rate
        filler_count = 0
        for phrase in self.MULTI_FILLERS:
            filler_count += text_lower.count(phrase)
        for word in words:
            clean_word = word.strip('.,?!;:\'"')
            if clean_word in self.SINGLE_FILLERS:
                filler_count += 1
        filler_rate = filler_count / max(word_count, 1)
        
        # Certainty score
        certainty_count = 0
        uncertainty_count = 0
        for word in words:
            clean_word = word.strip('.,?!;:\'"')
            if clean_word in self.CERTAINTY_MARKERS:
                certainty_count += 1
            elif clean_word in self.UNCERTAINTY_MARKERS:
                uncertainty_count += 1
        
        net_certainty = certainty_count - uncertainty_count
        certainty_score = net_certainty / max(word_count, 1)
        certainty_score = np.clip(certainty_score, -1.0, 1.0)
        
        # Sentiment polarity
        pos_count = 0
        neg_count = 0
        for word in words:
            clean_word = word.strip('.,?!;:\'"')
            if clean_word in self.POSITIVE_WORDS:
                pos_count += 1
            elif clean_word in self.NEGATIVE_WORDS:
                neg_count += 1
        
        net_sentiment = pos_count - neg_count
        sentiment_polarity = net_sentiment / max(word_count, 1)
        sentiment_polarity = np.clip(sentiment_polarity, -1.0, 1.0)
        
        return {
            "filler_rate": float(filler_rate),
            "certainty_score": float(certainty_score),
            "sentiment_polarity": float(sentiment_polarity),
        }



# BEHAVIORAL PROFILER (from v3, with session awareness)

class BehaviouralProfiler:
    """Tracks behavioral baseline per speaker with session management."""
    
    def __init__(self):
        self.baseline = {
            "filler_rate": 0.0,
            "certainty_score": 0.0,
            "sentiment_polarity": 0.0,
        }
        self.feature_history = []
        self.updated = False
        self.session_id = None
        self.session_start = time.time()
    
    def update(self, features: Dict[str, float]) -> None:
        """Update baseline with new features (moving average)."""
        self.feature_history.append(features)
        if len(self.feature_history) > 20:
            self.feature_history = self.feature_history[-20:]
        
        for key in self.baseline:
            values = [f[key] for f in self.feature_history]
            self.baseline[key] = float(np.mean(values))
        self.updated = True
    
    def anomaly_score(self, features: Dict[str, float]) -> float:
        """Compute anomaly score based on distance from baseline."""
        if not self.updated or len(self.feature_history) == 0:
            return 0.5
        
        distances = []
        for key in self.baseline:
            dist = abs(features.get(key, 0.0) - self.baseline[key])
            distances.append(dist)
        
        mean_dist = np.mean(distances)
        score = np.clip(mean_dist / 1.5, 0.0, 1.0)
        return float(score)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session metadata."""
        return {
            "session_id": self.session_id,
            "session_duration": time.time() - self.session_start,
            "utterance_count": len(self.feature_history),
            "baseline_established": self.updated
        }



# ENTITY MEMORY (with session tracking)

class EntityMemory:
    """Extracts named entities and updates session baseline."""
    
    def __init__(self):
        self.entities = []  # List per utterance
        self.entity_stats = {
            "PERSON": {},
            "ORG": {},
            "GPE": {},
            "DATE": {},
            "TIME": {}
        }
        self.session_entities = []
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not SPACY_AVAILABLE or NLP is None:
            return {}
        
        try:
            doc = NLP(text)
            out = {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "TIME": []}
            
            for ent in doc.ents:
                if ent.label_ in out:
                    out[ent.label_].append(ent.text)
                    # Track entity frequency
                    if ent.text not in self.entity_stats[ent.label_]:
                        self.entity_stats[ent.label_][ent.text] = 0
                    self.entity_stats[ent.label_][ent.text] += 1
            
            self.entities.append(out)
            self.session_entities.append({
                "timestamp": time.time(),
                "entities": out
            })
            return out
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of entities mentioned in session."""
        return {
            "total_mentions": len(self.session_entities),
            "entity_frequency": self.entity_stats,
            "unique_persons": len(self.entity_stats["PERSON"]),
            "unique_locations": len(self.entity_stats["GPE"]),
            "unique_organizations": len(self.entity_stats["ORG"])
        }
    
    def update_baseline_from_entities(self, profiler: BehaviouralProfiler) -> None:
        """Update behavioral baseline using entity-based signals."""
        if len(self.session_entities) < 5:
            return
        
        # If many different entities mentioned consistently = baseline consistency signal
        consistency_bonus = min(0.1, len(self.entity_stats["PERSON"]) * 0.01)
        logger.info(f"Entity consistency bonus: {consistency_bonus}")



# TEMPORAL DYNAMICS (from v3, unchanged)

class TemporalDynamics:
    """Tracks inter-utterance feature deltas."""
    
    def __init__(self):
        self.prev_features = None
        self.delta_history = {
            "delta_filler_rate": deque(maxlen=50),
            "delta_certainty": deque(maxlen=50),
            "delta_sentiment": deque(maxlen=50),
        }
    
    def compute(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute temporal deltas with z-score normalization."""
        if self.prev_features is None:
            self.prev_features = features.copy()
            return {
                "delta_filler_rate": 0.0,
                "delta_certainty": 0.0,
                "delta_sentiment": 0.0
            }
        
        delta = {
            "delta_filler_rate": features["filler_rate"] - self.prev_features["filler_rate"],
            "delta_certainty": features["certainty_score"] - self.prev_features["certainty_score"],
            "delta_sentiment": features["sentiment_polarity"] - self.prev_features["sentiment_polarity"],
        }
        
        for key, val in delta.items():
            hist = self.delta_history[key]
            if len(hist) > 1:
                std = float(np.std(list(hist)))
                if std > 1e-6:
                    delta[key] = val / std
            hist.append(val)
        
        self.prev_features = features.copy()
        return delta



# CONTRADICTION DETECTOR (from v3)

class ImprovedContradictionDetector:
    """Detects contradictions using NLI."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.nli = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.nli = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=-1
                )
                logger.info(f"NLI model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}")
                self.nli = None
    
    def detect(self, current_text: str, previous_texts: List[str]) -> Tuple[float, List[str]]:
        """Detect contradictions with previous utterances."""
        if not previous_texts or self.nli is None:
            return 0.0, []
        
        contradiction_scores = []
        conflicting = []
        
        for prev in previous_texts[-10:]:
            if not prev or len(prev.strip()) == 0:
                continue
            
            try:
                result = self.nli(
                    current_text,
                    candidate_labels=["entailment", "neutral", "contradiction"],
                    hypothesis_template=prev,
                )
                scores = result["scores"]
                labels = result["labels"]
                
                idx = labels.index("contradiction") if "contradiction" in labels else -1
                contradiction_score = scores[idx] if idx >= 0 else 0.0
                
                if contradiction_score > 0.5:
                    contradiction_scores.append(contradiction_score)
                    if contradiction_score > 0.65:
                        conflicting.append(prev)
            except Exception as e:
                logger.warning(f"Contradiction detection failed: {e}")
                continue
        
        final_score = max(contradiction_scores) if contradiction_scores else 0.0
        return float(final_score), conflicting



# TRUTH ESTIMATOR (from v3)

class ImprovedTruthEstimator:
    """Fuses multiple deception indicators into truth probability."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "contradiction": 0.45,
            "anomaly": 0.35,
            "temporal": 0.20,
        }
        
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {total}. Normalizing...")
            for key in self.weights:
                self.weights[key] /= total
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Logistic sigmoid function."""
        return 1.0 / (1.0 + math.exp(-np.clip(x, -500, 500)))
    
    def compute(self, contradiction: float, anomaly: float, temporal: float) -> float:
        """Compute truth probability from component scores."""
        risk = (self.weights["contradiction"] * contradiction +
                self.weights["anomaly"] * anomaly +
                self.weights["temporal"] * temporal)
        
        truth = 1.0 - self.sigmoid((risk - 0.5) * 5.0)
        return float(np.clip(truth, 0.0, 1.0))



# OUTPUT DATACLASS

@dataclass
class TMOutput:
    """Output from text modality processing."""
    text: str
    timestamp: float
    truth_probability: float  # [0, 1] - 1 = truthful
    deception_probability: float  # [0, 1] - 1 = deceptive (1 - truth)
    contradiction_score: float
    anomaly_score: float
    temporal_score: float
    inconsistency_flag: bool
    conflicting_previous: Optional[List[str]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    linguistic_features: Dict[str, float] = field(default_factory=dict)
    nli_available: bool = True
    speech_confidence: Optional[float] = None  # From STT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)



# CONTINUOUS SPEECH RECOGNITION (Google STT - speech_recognition)

class ContinuousSpeechRecognizer:
    """
    Optimized for continuous streaming with real-time utterance detection.
    
    Uses speech_recognition (Google API) with:
    - Continuous microphone stream
    - Automatic silence-based utterance segmentation
    - Real-time transcription callbacks
    - Confidence scoring
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 silence_threshold: float = 0.03,
                 silence_duration: float = 0.8,
                 phrase_time_limit: int = 15):
        """
        Initialize continuous recognizer.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            silence_threshold: Volume threshold for silence detection
            silence_duration: Duration of silence before utterance ends (seconds)
            phrase_time_limit: Max utterance length (seconds)
        """
        self.recognizer = RECOGNIZER if SPEECH_AVAILABLE else None
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.phrase_time_limit = phrase_time_limit
        
        if self.recognizer:
            logger.info("Google Speech Recognition initialized")
        else:
            logger.error("speech_recognition not available")
    
    def recognize_audio(self, audio_data) -> Tuple[str, float]:
        """
        Synchronous recognition (for single utterances).
        
        Args:
            audio_data: Audio from speech_recognition.listen()
            
        Returns:
            (text, confidence)
        """
        if not self.recognizer:
            return "", 0.0
        
        try:
            result = self.recognizer.recognize_google(audio_data)
            # Google doesn't provide confidence, but successful recognition = high confidence
            return result, 0.85
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Google API error: {e}")
            return "", 0.0
    
    def stream_from_mic(self, utterance_callback: Optional[Callable] = None) -> Generator[Tuple[str, float], None, None]:
        """
        Continuous streaming with real-time utterance detection.
        
        Yields utterances as they're recognized, with silence-based segmentation.
        
        Args:
            utterance_callback: Optional callback called when utterance recognized
                               callback(text: str, confidence: float)
            
        Yields:
            (text, confidence) tuples for each recognized utterance
        """
        if not self.recognizer or not SPEECH_AVAILABLE:
            logger.error("Microphone not available")
            return
        
        try:
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                logger.info("Listening from microphone (continuous stream)...")
                logger.info("Speak naturally. Press Ctrl+C to stop.")
                
                # Adjust for ambient noise
                logger.info("Calibrating for ambient noise (5 seconds)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                utterance_count = 0
                
                while True:
                    try:
                        logger.info("Listening...")
                        
                        # Listen with silence detection
                        # phrase_time_limit sets max utterance length
                        audio = self.recognizer.listen(
                            source,
                            timeout=None,  # Wait indefinitely for speech
                            phrase_time_limit=self.phrase_time_limit
                        )
                        
                        logger.info("Processing audio...")
                        
                        # Recognize
                        text, confidence = self.recognize_audio(audio)
                        
                        if text:
                            utterance_count += 1
                            logger.info(f"[{utterance_count}] Recognized: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                            logger.info(f"    Confidence: {confidence:.1%}")
                            
                            if utterance_callback:
                                utterance_callback(text, confidence)
                            
                            yield text, confidence
                        else:
                            logger.warning("No speech detected in audio")
                    
                    except sr.UnknownValueError:
                        logger.warning("Could not understand audio, continuing...")
                        continue
                    
                    except sr.RequestError as e:
                        logger.error(f"API error: {e}")
                        logger.info("Retrying...")
                        time.sleep(1)
                        continue
                    
                    except KeyboardInterrupt:
                        logger.info("\nStopping...")
                        break
                    
                    except Exception as e:
                        logger.error(f"Unexpected error: {e}")
                        continue
        
        except KeyboardInterrupt:
            logger.info("Stream ended by user")
        except Exception as e:
            logger.error(f"Microphone error: {e}")
    
    async def async_stream_from_mic(self, utterance_callback: Optional[Callable] = None):
        """
        Async generator for continuous streaming.
        
        Yields utterances in an async context.
        
        Args:
            utterance_callback: Async callback called when utterance recognized
            
        Yields:
            (text, confidence) tuples
        """
        # Wrap the synchronous generator in async
        loop = asyncio.get_event_loop()
        
        for text, confidence in self.stream_from_mic(utterance_callback):
            # Allow async operations between utterances
            await asyncio.sleep(0.01)
            yield text, confidence



# MAIN TEXT MODALITY PROCESSOR WITH CONTINUOUS STREAMING

class TextModalityProcessor:
    """
    Unified text modality with real-time speech integration.
    
    Dual-mode operation:
    1. Standalone: Listen from mic, process, output results
    2. Integrated: Accept text from main.py, process through pipeline
    """
    
    def __init__(self,
                 baseline_update_threshold: float = 0.7,
                 enable_speech: bool = True,
                 stT_backend: str = 'speech_recognition',
                 output_callback: Optional[Callable] = None):
        """
        Initialize processor.
        
        Args:
            baseline_update_threshold: Truth threshold for baseline update
            enable_speech: Enable speech recognition
            stT_backend: STT backend ('speech_recognition' or 'whisper')
            output_callback: Callback function for results (useful for main.py integration)
        """
        self._histories: Dict[str, deque] = {}
        self._temporal: Dict[str, TemporalDynamics] = {}
        self._profilers: Dict[str, BehaviouralProfiler] = {}
        
        self.feature_extractor = LinguisticFeatureExtractor()
        self.entity_memory = EntityMemory()
        self.contradiction_detector = ImprovedContradictionDetector()
        self.truth_estimator = ImprovedTruthEstimator()
        self.baseline_update_threshold = baseline_update_threshold
        
        # Initialize speech recognizer only if enabled and available
        self.speech_recognizer = None
        self.mic_capture = None
        
        if enable_speech and SPEECH_AVAILABLE:
            try:
                self.speech_recognizer = ContinuousSpeechRecognizer()
            except Exception as e:
                logger.warning(f"Speech recognizer init failed: {e}")
        
        self.nli_available = self.contradiction_detector.nli is not None
        self.output_callback = output_callback
        self._last_output = None
        
        logger.info(f"TextModalityProcessor initialized (NLI: {self.nli_available}, Speech: {self.speech_recognizer is not None})")
    
    def _get_history(self, speaker_id: str) -> deque:
        """Get or create history for speaker."""
        if speaker_id not in self._histories:
            self._histories[speaker_id] = deque(maxlen=20)
        return self._histories[speaker_id]
    
    def _get_temporal(self, speaker_id: str) -> TemporalDynamics:
        """Get or create temporal tracker for speaker."""
        if speaker_id not in self._temporal:
            self._temporal[speaker_id] = TemporalDynamics()
        return self._temporal[speaker_id]
    
    def _get_profiler(self, speaker_id: str) -> BehaviouralProfiler:
        """Get or create profiler for speaker."""
        if speaker_id not in self._profilers:
            self._profilers[speaker_id] = BehaviouralProfiler()
            self._profilers[speaker_id].session_id = f"{speaker_id}_{int(time.time())}"
        return self._profilers[speaker_id]
    
    async def process(self,
                     text: str,
                     speaker_id: str = "speaker_1",
                     speech_confidence: Optional[float] = None) -> TMOutput:
        """
        Process utterance through full pipeline.
        
        Args:
            text: Input text/utterance
            speaker_id: Speaker identifier
            speech_confidence: Confidence from STT (if from speech)
            
        Returns:
            TMOutput with all scores
        """
        timestamp = time.time()
        
        # Extract linguistic features
        features = self.feature_extractor.extract(text)
        
        # Compute temporal deltas
        temporal_engine = self._get_temporal(speaker_id)
        temporal_deltas = temporal_engine.compute(features)
        
        temporal_score = min(1.0, (abs(temporal_deltas["delta_filler_rate"]) +
                                   abs(temporal_deltas["delta_certainty"]) +
                                   abs(temporal_deltas["delta_sentiment"])) / 3.0)
        
        # Get behavioral anomaly
        profiler = self._get_profiler(speaker_id)
        anomaly = profiler.anomaly_score(features)
        
        # Check contradictions
        history = self._get_history(speaker_id)
        contradiction, conflicting = self.contradiction_detector.detect(text, list(history))
        
        # Estimate truth
        truth_prob = self.truth_estimator.compute(contradiction, anomaly, temporal_score)
        
        # Update baseline for truthful statements
        if truth_prob > self.baseline_update_threshold:
            profiler.update(features)
        
        # Extract entities (updates baseline indirectly)
        entities = self.entity_memory.extract(text)
        
        # Add to history
        history.append(text)
        
        # Create output
        output = TMOutput(
            text=text,
            timestamp=timestamp,
            truth_probability=round(truth_prob, 3),
            deception_probability=round(1.0 - truth_prob, 3),
            contradiction_score=round(contradiction, 3),
            anomaly_score=round(anomaly, 3),
            temporal_score=round(temporal_score, 3),
            inconsistency_flag=truth_prob < 0.45,
            conflicting_previous=conflicting,
            entities=entities,
            linguistic_features=features,
            nli_available=self.nli_available,
            speech_confidence=speech_confidence
        )
        
        # Call output callback if provided (for main.py integration)
        if self.output_callback:
            await self.output_callback(output)
        
        # Cache output for retrieval by get_last_output
        self._last_output = output
        
        return output
    
    def process_sync(self,
                    text: str,
                    speaker_id: str = "speaker_1",
                    speech_confidence: Optional[float] = None) -> TMOutput:
        """Synchronous wrapper using asyncio.run()."""
        output = asyncio.run(self.process(text, speaker_id, speech_confidence))
        self._last_output = output
        return output
    
    def get_last_output(self) -> Optional[Any]:
        """Get the last cached output from process_sync."""
        return self._last_output
    
    def stream_from_mic_continuous(self, speaker_id: str = "speaker_1") -> Generator[TMOutput, None, None]:
        """
        Continuous streaming from microphone with real-time output.
        
        Generator yields TMOutput for each recognized utterance.
        Automatically handles silence detection and utterance segmentation.
        
        Args:
            speaker_id: Speaker identifier
            
        Yields:
            TMOutput for each recognized utterance
        """
        if not self.speech_recognizer:
            logger.error("Speech recognizer not available")
            return
        
        utterance_count = 0
        
        # Use sync generator for continuous stream
        for text, confidence in self.speech_recognizer.stream_from_mic():
            utterance_count += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing utterance {utterance_count}...")
            logger.info(f"{'='*70}")
            
            try:
                # Process through modality
                output = self.process_sync(text, speaker_id, speech_confidence=confidence)
                
                logger.info(f"Truth: {output.truth_probability:.1%} | "
                          f"Deception: {output.deception_probability:.1%}")
                logger.info(f"Contradiction: {output.contradiction_score:.3f} | "
                          f"Anomaly: {output.anomaly_score:.3f} | "
                          f"Temporal: {output.temporal_score:.3f}")
                
                if output.inconsistency_flag:
                    logger.warning("⚠️  INCONSISTENCY DETECTED")
                
                # Yield for caller to process
                yield output
                
            except Exception as e:
                logger.error(f"Error processing utterance: {e}", exc_info=True)
                continue
    
    async def stream_from_mic_async(self, speaker_id: str = "speaker_1"):
        """
        Async version of continuous streaming.
        
        Args:
            speaker_id: Speaker identifier
            
        Yields:
            TMOutput for each recognized utterance
        """
        if not self.speech_recognizer:
            logger.error("Speech recognizer not available")
            return
        
        utterance_count = 0
        
        async for text, confidence in self.speech_recognizer.async_stream_from_mic():
            utterance_count += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing utterance {utterance_count}...")
            logger.info(f"{'='*70}")
            
            try:
                output = await self.process(text, speaker_id, speech_confidence=confidence)
                
                logger.info(f"Truth: {output.truth_probability:.1%} | "
                          f"Deception: {output.deception_probability:.1%}")
                
                yield output
                
            except Exception as e:
                logger.error(f"Error processing utterance: {e}", exc_info=True)
                continue
    
    def reset_speaker(self, speaker_id: str) -> Dict[str, Any]:
        """
        Reset speaker state and return session summary.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            Session summary dict
        """
        summary = {}
        
        if speaker_id in self._profilers:
            summary["profiler_info"] = self._profilers[speaker_id].get_session_info()
            del self._profilers[speaker_id]
        
        summary["entity_summary"] = self.entity_memory.get_session_summary()
        
        if speaker_id in self._histories:
            del self._histories[speaker_id]
        if speaker_id in self._temporal:
            del self._temporal[speaker_id]
        
        logger.info(f"Reset speaker state: {speaker_id}")
        return summary



# HELPER: Format output for display

def format_output(output: TMOutput) -> str:
    """Pretty-print TMOutput."""
    s = f"""
╔════════════════════════════════════════════════════════════════╗
║                     DECEPTION ANALYSIS                         ║
╚════════════════════════════════════════════════════════════════╝

Statement: "{output.text[:80]}{'...' if len(output.text) > 80 else ''}"

┌─ TRUTH & DECEPTION ─────────────────────────────────────────┐
│  Truth Probability:       {output.truth_probability:.1%}
│  Deception Probability:   {output.deception_probability:.1%}
│  Inconsistency Flag:      {'🚩 YES' if output.inconsistency_flag else '✓ NO'}
└─────────────────────────────────────────────────────────────┘

┌─ COMPONENT SCORES ──────────────────────────────────────────┐
│  Contradiction Score:     {output.contradiction_score:.3f}
│  Anomaly Score:           {output.anomaly_score:.3f}
│  Temporal Dynamics Score: {output.temporal_score:.3f}
└─────────────────────────────────────────────────────────────┘

┌─ LINGUISTIC FEATURES ───────────────────────────────────────┐
│  Filler Rate:             {output.linguistic_features.get('filler_rate', 0):.3f}
│  Certainty Score:         {output.linguistic_features.get('certainty_score', 0):.3f}
│  Sentiment Polarity:      {output.linguistic_features.get('sentiment_polarity', 0):.3f}
└─────────────────────────────────────────────────────────────┘

┌─ ENTITIES MENTIONED ────────────────────────────────────────┐
"""
    for entity_type, entities in output.entities.items():
        if entities:
            s += f"│  {entity_type:12} {', '.join(entities)}\n"
    
    if output.conflicting_previous:
        s += f"""└─────────────────────────────────────────────────────────────┘

⚠️  CONTRADICTIONS DETECTED:
"""
        for i, conflicting in enumerate(output.conflicting_previous[:3], 1):
            s += f"  {i}. \"{conflicting[:70]}{'...' if len(conflicting) > 70 else ''}\"\n"
    else:
        s += "└─────────────────────────────────────────────────────────────┘\n"
    
    if output.speech_confidence:
        s += f"\n🎤 Speech Confidence: {output.speech_confidence:.1%}\n"
    
    return s



# STANDALONE MODE: Listen from microphone

async def standalone_mode():
    """Run in standalone mode: continuous listening from microphone."""
    logger.info("="*70)
    logger.info("TEXT MODALITY - STANDALONE MODE (CONTINUOUS STREAMING)")
    logger.info("Listening from microphone with real-time utterance processing...")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*70)
    
    processor = TextModalityProcessor(enable_speech=True)
    
    try:
        # Use continuous streaming
        for output in processor.stream_from_mic_continuous():
            print(format_output(output))
            
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
        summary = processor.reset_speaker("speaker_1")
        logger.info(f"Session summary:\n{json.dumps(summary, indent=2, default=str)}")



# INTEGRATION: Accept input from main.py

class MainPyIntegration:
    """
    Integration class for main.py to send utterances to text modality.
    
    Usage in main.py:
        from text_modality import MainPyIntegration
        
        tm_integration = MainPyIntegration(output_handler=my_callback)
        
        # When you have utterance text:
        output = tm_integration.process_utterance(text, speaker_id)
    """
    
    def __init__(self, output_handler: Optional[Callable] = None):
        """
        Initialize integration.
        
        Args:
            output_handler: Callback function called with TMOutput
        """
        self.processor = TextModalityProcessor(
            enable_speech=False,  # Don't initialize mic
            output_callback=output_handler
        )
        self._loop = None
    
    def process_utterance(self, text: str, speaker_id: str = "speaker_1",
                         speech_confidence: Optional[float] = None) -> TMOutput:
        """
        Process utterance from main.py.
        
        Args:
            text: Utterance text
            speaker_id: Speaker identifier
            speech_confidence: Optional confidence score
            
        Returns:
            TMOutput with all scores
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            output = loop.run_until_complete(
                self.processor.process(text, speaker_id, speech_confidence)
            )
            return output
        finally:
            loop.close()
    
    def get_speaker_info(self, speaker_id: str) -> Dict[str, Any]:
        """Get speaker session info."""
        profiler = self.processor._get_profiler(speaker_id)
        return {
            "baseline": profiler.baseline,
            "session_info": profiler.get_session_info(),
            "entity_summary": self.processor.entity_memory.get_session_summary()
        }
    
    def reset_session(self, speaker_id: str) -> Dict[str, Any]:
        """Reset speaker and return summary."""
        return self.processor.reset_speaker(speaker_id)



# MAIN ENTRY POINT

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if running as standalone or integrated
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode with sample utterances
        print("\n" + "="*80)
        print("TEST MODE: Processing sample utterances")
        print("="*80)
        
        processor = TextModalityProcessor(enable_speech=False)
        
        test_utterances = [
            ("I was definitely at home all evening.", "speaker_1"),
            ("Uh, like, I was maybe at the office?", "speaker_2"),
            ("I was there, you know, I mean I was definitely there.", "speaker_1"),
            ("I'm certain I saw him at the café yesterday.", "speaker_3"),
            ("Maybe I was confused, I'm not sure if I remember correctly.", "speaker_3"),
        ]
        
        for text, speaker_id in test_utterances:
            output = processor.process_sync(text, speaker_id)
            print(format_output(output))
    else:
        # Standalone mode: continuous listening from mic
        standalone_mode()