#!/usr/bin/env python3
"""
Deception Detection Pipeline

Stable real-time orchestration pipeline for:
- MM (Microexpression)
- PM (Physiological)
- SM (Speech)
- TM (Text)

Includes:
- live camera streaming
- microphone transcription
- wearable support
- HUD rendering
- fusion engine
- session logging
- graceful shutdown
"""

import asyncio
import cv2
import logging
import threading
import time

from pathlib import Path
from typing import Optional

from session import SessionConfig, SubjectProfile
from run_logger import UnifiedRunLogger
from fusion import FusionSystem
from hud import HUDRenderer
from baseline_deviation_scorer import BaselineDeviationScorer
from cross_modality_incongruence import CrossModalityIncongruenceDetector

logger = logging.getLogger(__name__)

# Graceful import of speech_recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    logger.warning("speech_recognition not available - using mock for audio transcription")
    SR_AVAILABLE = False
    # Create a mock module to prevent errors
    class MockRecognizer:
        def __init__(self): pass
        def adjust_for_ambient_noise(self, *args, **kwargs): pass
        def listen(self, *args, **kwargs): raise TimeoutError()
        def recognize_google(self, *args, **kwargs): raise Exception("Not available")
    
    class MockMicrophone:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class sr:
        Recognizer = MockRecognizer
        Microphone = MockMicrophone
        UnknownValueError = Exception
        WaitTimeoutError = TimeoutError

# ============================================================
# Optional imports
# ============================================================

try:
    from camera_input import CameraStreamer
    CAMERA_AVAILABLE = True
except Exception as e:
    logger.warning(f"Camera input unavailable: {e}")
    CAMERA_AVAILABLE = False

try:
    from gpu_acceleration import (
        get_device,
        initialize_device
    )
    GPU_AVAILABLE = True
except Exception as e:
    logger.warning(f"GPU unavailable: {e}")
    GPU_AVAILABLE = False

try:
    from O3_SpeechModality.microphone_input import (
        MicrophoneStreamer
    )
    MICROPHONE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Microphone unavailable: {e}")
    MICROPHONE_AVAILABLE = False

try:
    from O2_PhysiologicalModality.wearable_connector import (
        BLEWearableConnector
    )
    WEARABLE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Wearable unavailable: {e}")
    WEARABLE_AVAILABLE = False

try:
    from O1_TextModality.text_v4 import (
        TextModalityProcessor
    )
    TM_AVAILABLE = True
except Exception as e:
    logger.warning(f"TM unavailable: {e}")
    TM_AVAILABLE = False

try:
    from O2_PhysiologicalModality.physio_v3 import (
        PMSystem,
        PMConfig
    )
    PM_AVAILABLE = True
except Exception as e:
    logger.warning(f"PM unavailable: {e}")
    PM_AVAILABLE = False
    PMConfig = None
    PM_AVAILABLE = False

try:
    from O3_SpeechModality.speech_v3 import (
        StreamingSMProcessor
    )
    SM_AVAILABLE = True
except Exception as e:
    logger.warning(f"SM unavailable: {e}")
    SM_AVAILABLE = False

try:
    from O4_MicroexpressionModality.macro_v7 import (
        MicroExpressionModel
    )

    MM_AVAILABLE = True

except Exception as e:
    logger.warning(f"MM unavailable: {e}")
    MM_AVAILABLE = False


# ============================================================
# Realtime Audio Transcriber
# ============================================================

class RealtimeAudioTranscriber:

    def __init__(
        self,
        callback,
        energy_threshold=500,
        pause_threshold=1.0
    ):

        self.callback = callback

        self.recognizer = sr.Recognizer()

        self.recognizer.energy_threshold = (
            energy_threshold
        )

        self.recognizer.pause_threshold = (
            pause_threshold
        )

        try:
            self.microphone = sr.Microphone()

        except Exception as e:

            logger.warning(
                f"Microphone unavailable: {e}"
            )

            self.microphone = None

        self.stop_event = threading.Event()

        self.thread = None

        self._running = False

    def _adjust_noise(self):

        if self.microphone is None:
            return

        try:

            with self.microphone as source:

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=2
                )

        except Exception as e:

            logger.warning(
                f"Noise adjustment failed: {e}"
            )

    def _listen_loop(self):

        if self.microphone is None:
            return

        self._adjust_noise()

        with self.microphone as source:

            while not self.stop_event.is_set():

                try:

                    audio = self.recognizer.listen(
                        source,
                        timeout=0.5,
                        phrase_time_limit=5
                    )

                except sr.WaitTimeoutError:
                    continue

                except Exception as e:

                    logger.debug(
                        f"Audio listen error: {e}"
                    )

                    continue

                try:

                    text = (
                        self.recognizer.recognize_google(
                            audio
                        )
                    )

                    if text and text.strip():

                        logger.info(
                            f"🎤 {text}"
                        )

                        self.callback(text)

                except sr.UnknownValueError:
                    continue

                except Exception as e:

                    logger.debug(
                        f"Speech recognition failed: {e}"
                    )

    def start(self):

        if self._running:
            return

        self._running = True

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )

        self.thread.start()

        logger.info(
            "🎙️ Audio transcription started"
        )

    def stop(self):

        self._running = False

        self.stop_event.set()

        if (
            self.thread and
            self.thread.is_alive()
        ):
            self.thread.join(timeout=1.0)

        logger.info(
            "🎙️ Audio transcription stopped"
        )


# ============================================================
# Pipeline
# ============================================================

class DeceptionDetectionPipeline:

    def __init__(
        self,
        config: SessionConfig,
        subject_profile: Optional[
            SubjectProfile
        ] = None,
        behavior_profile = None,  # SubjectBehaviorProfile
    ):

        self.config = config

        self.subject_profile = subject_profile
        
        self.behavior_profile = behavior_profile

        self.clock = config.session_clock

        self.running = True
        
        # Modality weights (initialized from calibration if available)
        self.modality_weights = self._init_modality_weights()

        # ====================================================
        # GPU
        # ====================================================

        self.device = None

        if GPU_AVAILABLE:

            try:

                initialize_device(
                    prefer_gpu=True
                )

                self.device = get_device()

                logger.info(
                    f"GPU enabled: {self.device}"
                )

            except Exception as e:

                logger.warning(
                    f"GPU init failed: {e}"
                )

        # ====================================================
        # Logger
        # ====================================================

        self.run_logger = UnifiedRunLogger(
            config.session_id
        )

        # ====================================================
        # Fusion + HUD
        # ====================================================

        self.fusion = FusionSystem(calibration_weights=self.modality_weights)
        
        # Update fusion with calibration weights if available
        if self.behavior_profile:
            self.fusion.set_calibration_weights(self.modality_weights)
        
        # ====================================================
        # Baseline Deviation Scorer
        # ====================================================
        
        self.deviation_scorer = BaselineDeviationScorer(
            behavior_profile=self.behavior_profile
        )
        
        # ====================================================
        # Cross-Modality Incongruence Detector
        # ====================================================
        
        self.incongruence_detector = CrossModalityIncongruenceDetector(
            window_size=30
        )

        self.hud = HUDRenderer(
            frame_width=800,
            frame_height=600
        )
        
        # ====================================================
        # HUD Deviation Display
        # ====================================================
        
        try:
            from hud_deviation_display import DeviationDisplay
            self.deviation_display = DeviationDisplay(
                frame_width=800,
                frame_height=600
            )
        except ImportError:
            logger.warning("HUD deviation display not available")
            self.deviation_display = None

        # ====================================================
        # Camera
        # ====================================================

        self.camera = None

        if CAMERA_AVAILABLE:

            try:

                self.camera = CameraStreamer(
                    camera_id=0,
                    frame_width=1280,
                    frame_height=720,
                    fps=30,
                    validate_frames=False  # Disable validation for stability
                )

                logger.info(
                    "Camera initialized"
                )

            except Exception as e:

                logger.warning(
                    f"Camera init failed: {e}"
                )

        # ====================================================
        # Microphone Stream
        # ====================================================

        self.microphone = None

        if MICROPHONE_AVAILABLE:

            try:

                self.microphone = (
                    MicrophoneStreamer(
                        sample_rate=16000,
                        chunk_size=512,
                        use_vad=True
                    )
                )

                logger.info(
                    "Microphone initialized"
                )

            except Exception as e:

                logger.warning(
                    f"Microphone init failed: {e}"
                )

        # ====================================================
        # Wearable
        # ====================================================

        self.wearable = None

        if WEARABLE_AVAILABLE:

            try:

                self.wearable = (
                    BLEWearableConnector()
                )

                logger.info(
                    "Wearable initialized"
                )

            except Exception as e:

                logger.warning(
                    f"Wearable init failed: {e}"
                )

        # ====================================================
        # Modalities
        # ====================================================

        self.tm = (
            TextModalityProcessor()
            if TM_AVAILABLE
            else None
        )

        self.pm = None
        if PM_AVAILABLE:
            try:
                import os
                # Try to initialize PMSystem with proper configuration
                # Priority: timex_mac from env/config > gadgetbridge_dir > None (will use rPPG if camera available)
                timex_mac = os.environ.get('TIMEX_MAC') or self.config.config_dict.get('timex_mac')
                gadgetbridge_dir = os.environ.get('GADGETBRIDGE_DIR') or self.config.config_dict.get('gadgetbridge_dir')
                
                if timex_mac or gadgetbridge_dir:
                    pm_cfg = PMConfig(
                        timex_mac=timex_mac,
                        gadgetbridge_dir=gadgetbridge_dir,
                        camera_id=self.config.config_dict.get('camera_id', 0)
                    )
                    self.pm = PMSystem(cfg=pm_cfg)
                    logger.info(f"PMSystem initialized with device: {self.pm._device_name if hasattr(self.pm, '_device_name') else 'unknown'}")
                else:
                    # Without wearable device, PM will be disabled
                    logger.info("PM disabled: No TIMEX_MAC or GADGETBRIDGE_DIR configured. To enable, set TIMEX_MAC env var or configure timex_mac in config.yaml")
                    self.pm = None
            except ValueError as e:
                logger.warning(f"PM initialization error: {e}")
                self.pm = None
            except Exception as e:
                logger.warning(f"PM failed: {e}")
                self.pm = None

        self.sm = None
        if SM_AVAILABLE:
            try:
                self.sm = StreamingSMProcessor(
                    input_source="external",  # Fed externally from mic
                    sr=16000,
                    chunk_ms=20,
                    vad_energy_threshold=3e-4,
                    vad_zcr_threshold=0.15,
                    min_speech_ms=300
                )
                logger.info("SM initialized with external audio source")
            except Exception as e:
                logger.warning(f"SM init failed: {e}")
                self.sm = None

        self.mm = None

        if MM_AVAILABLE:

            try:

                weights = str(
                    Path(__file__).resolve().parent
                    / 'micro_expression_models'
                    / 'phase1'
                    / 'best_78.06.pth'
                )

                self.mm = (
                    MicroExpressionModel(
                        weights_path=weights
                    )
                )

                logger.info(
                    "MM model loaded"
                )

            except Exception as e:

                logger.warning(
                    f"MM init failed: {e}"
                )

        # ====================================================
        # Audio Transcriber
        # ====================================================

        self.audio_transcriber = None

        try:

            self.audio_transcriber = (
                RealtimeAudioTranscriber(
                    callback=self.process_utterance
                )
            )

        except Exception as e:

            logger.warning(
                f"Audio transcriber failed: {e}"
            )

        # ====================================================
        # FPS
        # ====================================================

        self.frame_idx = 0

        self.fps = 0.0

        self._fps_counter = 0

        self._last_fps_time = time.time()

    # ========================================================
    # Modality Weights Initialization (from calibration)
    # ========================================================
    
    def _init_modality_weights(self) -> dict:
        """
        Initialize modality fusion weights based on calibration confidence.
        
        If behavior_profile is available, weights are set based on per-modality
        calibration quality. Otherwise, default equal weights are used.
        
        Returns:
            Dict[modality_name -> weight (0-1)]
        """
        default_weights = {
            'mm': 0.25,
            'pm': 0.25,
            'sm': 0.25,
            'tm': 0.25,
        }
        
        if self.behavior_profile is None:
            logger.warning("⚠️  No calibration profile - using default equal modality weights")
            return default_weights
        
        # Compute weights from calibration confidence scores
        confidences = {
            'mm': self.behavior_profile.mm_baseline.calibration_quality,
            'pm': self.behavior_profile.pm_baseline.calibration_quality,
            'sm': self.behavior_profile.sm_baseline.calibration_quality,
            'tm': self.behavior_profile.tm_baseline.calibration_quality,
        }
        
        # Normalize to sum to 1.0
        total_conf = sum(confidences.values())
        if total_conf > 0:
            weights = {k: v / total_conf for k, v in confidences.items()}
        else:
            weights = default_weights
        
        logger.info(f"📊 Fusion weights from calibration:")
        for mod, weight in sorted(weights.items()):
            conf = confidences.get(mod, 0)
            logger.info(f"   {mod.upper()}: {weight:.1%} (confidence: {conf:.1%})")
        
        return weights

    # ========================================================
    # Initialize modalities
    # ========================================================

    def initialize_modalities(self):

        logger.info(
            "Initializing modalities..."
        )

        if self.pm:

            try:

                self.pm.connect()

                self.pm.start()

                logger.info(
                    "PM started"
                )

            except Exception as e:

                logger.warning(
                    f"PM failed: {e}"
                )

        if self.sm:

            try:

                self.sm.get_last_output()

                logger.info(
                    "SM ready"
                )

            except Exception as e:

                logger.warning(
                    f"SM failed: {e}"
                )

    # ========================================================
    # Process utterance
    # ========================================================

    def process_utterance(
        self,
        text: str,
        speaker_id=None
    ):

        if not self.tm:
            return

        try:

            if speaker_id is None:
                speaker_id = self.subject_profile.speaker_id if self.subject_profile else "speaker_1"

            # Use process_sync from text_v4
            tm_out = (
                self.tm.process_sync(
                    text,
                    speaker_id=speaker_id
                )
            )

            if tm_out:

                self.run_logger.log_contradiction(
                    self.clock.now(),
                    text,
                    getattr(
                        tm_out,
                        "deception_probability",
                        0.0
                    ),
                    getattr(
                        tm_out,
                        "conflicting_previous",
                        []
                    )
                )
                
                logger.info(
                    f"📝 TM: {text[:50]}... | "
                    f"Deception: {getattr(tm_out, 'deception_probability', 0.0):.2f}"
                )

        except Exception as e:

            logger.debug(
                f"TM processing failed: {e}"
            )

    # ========================================================
    # Process frame
    # ========================================================

    def process_frame(self, frame):

        ts = self.clock.now()

        mm_out = {}
        tm_out = {}
        pm_out = {}
        sm_out = {}

        # MM (Microexpression) - Real-time from frame
        if self.mm:

            try:

                label, conf = (
                    self.mm.infer(frame)
                )

                mm_out = {
                    "emotion": str(label),
                    "confidence": float(conf)
                }
                
                logger.debug(
                    f"MM: {label} ({conf:.2f})"
                )

            except Exception as e:

                logger.debug(
                    f"MM failed: {e}"
                )

        # PM (Physiological) - Real-time from wearable
        if self.pm:

            try:

                out = self.pm.get_output()

                if out:

                    if hasattr(
                        out,
                        "__dict__"
                    ):
                        pm_out = out.__dict__

                    elif isinstance(
                        out,
                        dict
                    ):
                        pm_out = out
                
                if pm_out:
                    logger.debug(
                        f"PM: HR={pm_out.get('heart_rate', 0)}, "
                        f"Stress={pm_out.get('stress_probability', 0):.2f}"
                    )

            except Exception as e:

                logger.debug(
                    f"PM failed: {e}"
                )

        # SM (Speech) - Real-time from microphone
        if self.sm:

            try:

                sm_out = (
                    self.sm.get_last_output()
                    or {}
                )
                
                if sm_out:
                    logger.debug(
                        f"SM: Stress={sm_out.get('stress_prob', 0):.2f}, "
                        f"Deception={sm_out.get('deception_prob', 0):.2f}"
                    )

            except Exception as e:

                logger.debug(
                    f"SM failed: {e}"
                )

        # TM (Text) - From previous utterance, non-blocking
        if self.tm:
            try:
                # Try to get cached output without blocking
                if hasattr(self.tm, '_last_output'):
                    tm_out = self.tm._last_output or {}
                elif hasattr(self.tm, 'get_last_output'):
                    tm_out = self.tm.get_last_output() or {}
            except Exception as e:
                logger.debug(f"TM output fetch failed: {e}")

        # Baseline Deviation Scoring
        deviations = {}
        if self.behavior_profile:
            try:
                deviations = self.deviation_scorer.compute_all_deviations(
                    mm_out=mm_out,
                    pm_out=pm_out,
                    sm_out=sm_out,
                    tm_out=tm_out
                )
                logger.debug(
                    f"Deviations - MM: {deviations['mm'].get('total_deviation', 0):.2f}, "
                    f"PM: {deviations['pm'].get('total_deviation', 0):.2f}, "
                    f"SM: {deviations['sm'].get('total_deviation', 0):.2f}, "
                    f"TM: {deviations['tm'].get('total_deviation', 0):.2f}"
                )
            except Exception as e:
                logger.debug(f"Deviation scoring failed: {e}")
        
        # Cross-Modality Incongruence Detection
        incongruences = {}
        if self.incongruence_detector and deviations:
            try:
                incongruences = self.incongruence_detector.compute_all_incongruences(deviations)
                self.incongruence_detector.update_history(deviations)
                
                logger.debug(
                    f"Incongruences - F-V: {incongruences.get('facial_verbal', {}).get('facial_verbal_incongruence', 0):.2f}, "
                    f"S-P: {incongruences.get('speech_physio', {}).get('speech_physio_incongruence', 0):.2f}, "
                    f"Overall: {incongruences.get('overall_incongruence', 0):.2f}"
                )
            except Exception as e:
                logger.debug(f"Incongruence detection failed: {e}")

        # Fusion - Combine all modalities
        fusion_out = self.fusion.update(
            mm_out=mm_out,
            tm_out=tm_out,
            pm_out=pm_out,
            sm_out=sm_out,
            timestamp=ts,
            deviations=deviations  # Pass deviations to fusion
        )

        # HUD - Render all outputs
        try:

            hud_frame = (
                self.hud.composite_hud(
                    frame,
                    mm_out,
                    tm_out,
                    pm_out,
                    sm_out,
                    fusion_out.__dict__,
                    fps=self.fps
                )
            )
            
            # Render baseline deviations if available
            if self.deviation_display and deviations:
                try:
                    # Extract weights from fusion output
                    weights = getattr(fusion_out, 'modality_weights', {})
                    
                    # Render deviation display on HUD frame
                    hud_frame = self.deviation_display.render_all_deviations(
                        hud_frame,
                        deviations,
                        weights
                    )
                except Exception as e:
                    logger.debug(f"Deviation display rendering failed: {e}")
            
            # Print to console for real-time feedback
            if self.frame_idx % 30 == 0:
                logger.info(
                    f"✓ Frame {self.frame_idx} | "
                    f"MM: {mm_out.get('emotion', 'N/A')} | "
                    f"TM: {'📝' if tm_out else '—'} | "
                    f"PM: {'♥' if pm_out else '—'} | "
                    f"SM: {'🎤' if sm_out else '—'} | "
                    f"Fusion: {fusion_out.deception_probability:.2f}"
                )

        except Exception as e:

            logger.debug(
                f"HUD failed: {e}"
            )

            hud_frame = frame.copy()

        # Logger
        try:

            self.run_logger.log_frame(
                self.frame_idx,
                ts,
                mm_out=mm_out,
                tm_out=tm_out,
                pm_out=pm_out,
                sm_out=sm_out,
                fusion_out=fusion_out.__dict__,
                fps=self.fps
            )

        except Exception as e:

            logger.debug(
                f"Logger failed: {e}"
            )

        # FPS
        self.frame_idx += 1

        self._fps_counter += 1

        now = time.time()

        if now - self._last_fps_time >= 1.0:

            self.fps = (
                self._fps_counter /
                (now - self._last_fps_time)
            )

            self._fps_counter = 0

            self._last_fps_time = now

        return hud_frame

    # ========================================================
    # Main loop
    # ========================================================

    def run_camera_loop(
        self,
        camera_id=0,
        headless=False
    ):

        if self.camera is None:

            logger.error(
                "Camera unavailable"
            )

            return

        try:

            self.camera.start()

            logger.info(
                "Camera started"
            )
            
            # Wait for camera to actually be ready
            if hasattr(self.camera, '_camera_ready_event'):
                ready = self.camera._camera_ready_event.wait(timeout=5.0)
                if not ready:
                    logger.warning("Camera initialization timeout")

        except Exception as e:

            logger.error(
                f"Camera failed: {e}"
            )

            return

        # Microphone
        if self.microphone:

            try:

                self.microphone.start()

            except Exception as e:

                logger.warning(
                    f"Mic failed: {e}"
                )

        # Wearable
        if self.wearable:

            try:

                if hasattr(self.wearable, "start"):

                    wearable_thread = threading.Thread(
                        target=self.wearable.start,
                        daemon=True
                    )

                    wearable_thread.start()

                    logger.info(
                        "Wearable started"
                    )

                elif hasattr(self.wearable, "run"):

                    wearable_thread = threading.Thread(
                        target=self.wearable.run,
                        daemon=True
                    )

                    wearable_thread.start()

                    logger.info(
                        "Wearable running"
                    )

                else:

                    logger.warning(
                        f"Wearable API unknown: "
                        f"{dir(self.wearable)}"
                    )

            except Exception as e:

                logger.warning(
                    f"Wearable failed: {e}"
                )

        # Audio transcription
       # Audio transcription disabled to prevent
        # duplicate microphone capture conflicts
        # with MicrophoneStreamer on macOS

        logger.info(
            "RealtimeAudioTranscriber disabled "
            "(MicrophoneStreamer active)"
        )

        consecutive_failures = 0

        MAX_FAILURES = 100

        try:

            attempt_count = 0
            while self.running:
                attempt_count += 1
                frame = self.camera.get_frame(timeout=0.05)  # 50ms timeout
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 50 == 0:
                        logger.debug(f"Frame fetch attempt {attempt_count}: None (failures={consecutive_failures})")
                    if (
                        consecutive_failures >=
                        MAX_FAILURES
                    ):
                        logger.error(
                            "Camera permanently unavailable"
                        )
                        break
                    time.sleep(0.01)  # 10ms before retry
                    continue
                consecutive_failures = 0
                
                if attempt_count == 1 or attempt_count % 200 == 0:
                    logger.info(f"✓ Got frame {attempt_count}, frame_idx={self.frame_idx}")
                
                try:
                    frame = cv2.resize(
                        frame,
                        (
                            self.hud.frame_width,
                            self.hud.frame_height
                        )
                    )

                except Exception as e:
                    logger.warning(
                        f"Resize failed: {e}"
                    )
                    continue

                try:
                    # Reduce heavy inference load
                    # Process every 2nd frame
                    if (self._fps_counter % 2) == 0:
                        hud_frame = (self.process_frame(frame))
                    else:
                        hud_frame = frame

                except Exception as e:

                    logger.exception(
                        f"Frame processing failed: {e}"
                    )
                    continue

                if self.frame_idx % 300 == 0:
                    logger.info(
                        f"Pipeline healthy | "
                        f"FPS={self.fps:.2f} | "
                        f"Frames={self.frame_idx}"
                    )
                    try:
                        stats = (
                            self.camera.get_stats()
                        )
                        logger.info(
                            f"Camera stats: {stats}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Stats failed: {e}"
                        )
                if not headless:
                    cv2.imshow(
                        "Deception Detection",
                        hud_frame
                    )

                key = (
                    cv2.waitKey(2) & 0xFF
                )

                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info(
                "User shutdown"
            )

        except Exception as e:
            logger.exception(
                f"Fatal pipeline error: {e}"
            )

        finally:
            self.shutdown()

    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self):

        logger.info(
            "Shutting down pipeline"
        )

        self.running = False

        try:

            if (
                self.audio_transcriber and
                self.audio_transcriber._running
            ):
                self.audio_transcriber.stop()

        except Exception as e:

            logger.debug(
                f"Audio cleanup failed: {e}"
            )

        try:

            if self.microphone:
                self.microphone.stop()

        except Exception as e:

            logger.debug(
                f"Mic cleanup failed: {e}"
            )

        try:
            if self.camera:
                self.camera.stop()
        except Exception as e:
            logger.debug(f"Camera cleanup failed: {e}")

        try:
            if (self.wearable and hasattr(self.wearable, "disconnect")):
                self.wearable.disconnect()

        except Exception as e:

            logger.debug(
                f"Wearable cleanup failed: {e}"
            )

        try:

            if self.pm:
                self.pm.stop()

        except Exception as e:

            logger.debug(
                f"PM cleanup failed: {e}"
            )

        try:

            cv2.destroyAllWindows()

        except Exception as e:

            logger.debug(
                f"OpenCV cleanup failed: {e}"
            )

        try:

            self.run_logger.log_run_end(
                self.frame_idx,
                self.run_logger.apex_count,
                self.clock.elapsed(),
                self.fps
            )

            self.run_logger.close()

        except Exception as e:

            logger.debug(
                f"Logger cleanup failed: {e}"
            )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )
    cfg = SessionConfig.from_yaml(
        'config.yaml',
        speaker_id='demo_subject'
    )
    profile = SubjectProfile.get_or_create(
        cfg.speaker_id
    )
    pipeline = (
        DeceptionDetectionPipeline(
            cfg,
            profile
        )
    )
    pipeline.initialize_modalities()
    pipeline.run_camera_loop(
        camera_id=0,
        headless=False
    )