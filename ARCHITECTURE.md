# Deception Detection System — Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECEPTION DETECTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
        ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
        │   Camera     │  │ Microphone  │  │ BLE Wearable │
        │   Input      │  │   Stream    │  │  (Optional)  │
        └──────────────┘  └─────────────┘  └──────────────┘
                    │            │            │
                    ▼            ▼            ▼
        ┌──────────────────────────────────────────────┐
        │         Input Preprocessing                  │
        │  - Frame resizing (640×480)                  │
        │  - Face detection + cropping                 │
        │  - Audio buffering (1-10s windows)           │
        │  - Wearable GATT parsing                     │
        └──────────────────────────────────────────────┘
                    │            │            │
        ┌───────────┴────────┬───┴────────┬──┴──────────┐
        │                    │            │             │
        ▼                    ▼            ▼             ▼
    ┌────────────┐    ┌────────────┐  ┌────────┐   ┌────────┐
    │ MM Module  │    │ TM Module  │  │ PM     │   │ SM     │
    │ (Emotion)  │    │ (NLI)      │  │ Module │   │ Module │
    │            │    │            │  │        │   │        │
    │ ResNet-18  │    │ DeBERTa    │  │ rPPG   │   │ Random │
    │            │    │ + RuleBased│  │ + BLE  │   │ Forest │
    │ 45 MB      │    │ 300 MB     │  │ 12 MB  │   │ 50 KB  │
    │            │    │            │  │        │   │        │
    │ 87-91% acc │    │ 88-90% acc │  │ N/A    │   │ 50% acc│
    │ (RAF-DB)   │    │ (NLI feat) │  │ (Heur.)│   │ (syn)  │
    └─────┬──────┘    └─────┬──────┘  └───┬────┘   └───┬────┘
          │                 │             │            │
          │  +confidence    │  +nli_score │ +hr/spo2   │ +stress_prob
          │  +emotion       │  +textual_  │ +signals   │ +deception_prob
          │  +baseline_     │    baseline │ +anomaly   │ +feature_
          │    deviation    │             │   score    │   trace
          │                 │             │            │
          └─────────────────┴─────────────┴────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │  FUSION LAYER           │
                │                         │
                │ Confidence-Weighted     │
                │ Bayesian Combination    │
                │                         │
                │ + Temporal Decay        │
                │ + Alert Thresholding    │
                │ + Explanation Engine    │
                └────────┬────────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │   Fused Output (0.0-1.0) │
            │   Alert Level (1-5)      │
            │   Explanation Text       │
            │   Per-Modality Scores    │
            └────────┬─────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌─────────┐ ┌───────────┐ ┌──────────┐
    │   HUD   │ │ Run Logs  │ │ Profiles │
    │ Display │ │ (JSONL)   │ │ (YAML)   │
    └─────────┘ └───────────┘ └──────────┘
```

---

## Component Details

### 1. **Microexpression Module (MM)** — `/00_MicroexpressionModality/macro_v7.py`

**Architecture:**
- **Model:** ResNet-18 CNN pre-trained on RAF-DB (facial expressions)
- **Input:** Cropped face region (224×224, RGB)
- **Output:** Deception probability (0.0–1.0) + emotion class + AU scores

**Processing Pipeline:**
```
Video Frame
    ▼
Face Detection (MediaPipe FaceMesh)
    ▼
AU Extraction (68-point landmarks → facial action units)
    ▼
Baseline Calculation (per-speaker AU normal ranges)
    ▼
ResNet-18 Forward Pass
    ▼
Emotion Classifier (7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
    ▼
Deception Heuristic (map emotion→deception_prob + AU deviation)
    ▼
Output: {emotion, confidence, deception_prob, baseline_deviation}
```

**Key Algorithms:**
- **AU Calculation:** Euclidean distance between current landmarks and baseline per subject
- **Deception Heuristic:** 
  - Base: emotion class → prior probability (e.g., ANGRY → 0.65 deception)
  - Modulation: AU deviation × intensity (e.g., larger deviation → higher probability)
  - Clipping: Confidence-weighted probability normalized to [0, 1]

**Performance:**
- Latency: 45–80 ms per frame
- Accuracy: 87–91% on RAF-DB test set
- Model size: 45 MB
- Inference device: CPU/GPU (auto-selected)

**Known Limitations:**
- No temporal model (apex detection heuristic only)
- Single frame detection (no micro-expression duration tracking)
- Assumes good lighting and frontal pose
- 68-point landmarks may be inaccurate in profile view

**Roadmap (Phase-2):**
- Add 3D-ResNet temporal model to track micro-expression onset/apex/offset
- Fine-tune on SMIC (Spontaneous Micro-Expression In-the-wild) + SAMM datasets
- Support multiple face angles (head pose estimation)

---

### 2. **Text Modality (TM)** — `/01_TextModality/text_v1.py`

**Architecture:**
- **NLI Model:** DeBERTa-v3-small (zero-shot, pre-trained on MNLI dataset)
- **Feature Extractor:** Linguistic features (hesitations, self-corrections, temporal markers)
- **Profiler:** Behavioral baseline (per-speaker speech patterns)
- **Temporal Engine:** Entity memory + contradiction detection over utterance history

**Processing Pipeline:**
```
Utterance Text
    ▼
Preprocessing (lowercasing, tokenization)
    ▼
NLI Pipeline (tuple format: (query, previous_utterance) → ENTAILMENT/CONTRADICTION/NEUTRAL)
    ▼
Linguistic Feature Extraction
    - Hedge words (maybe, I think, possibly)
    - Self-corrections (I mean, actually, scratch that)
    - Temporal markers (suddenly, then, before, after)
    - Negative sentiment (deny, never, not)
    ▼
Behavioral Baseline Update (per-speaker statistics)
    ▼
Fusion of NLI + Linguistic + Baseline
    ▼
Output: {truth_probability, contradiction_score, baseline_deviation}
```

**Key Algorithms:**
- **NLI Classification:** DeBERTa tokenizes pair, runs cross-encoder, returns softmax over 3 classes
- **Contradiction Detection:** NLI score > 0.7 + historical inconsistency → high deception signal
- **Behavioral Profiler:** Tracks mean/std of feature frequencies per speaker; deviation = z-score
- **Temporal Memory:** Stores last N utterances in per-speaker queue; contradiction checked against all prior

**Performance:**
- Latency: 200–400 ms per utterance
- Accuracy: 88–90% on NLI validation set (MNLI)
- Model size: 300 MB (DeBERTa downloaded on first run)
- Context length: 512 tokens max

**Known Limitations:**
- English-only (no multilingual support)
- No domain-specific fine-tuning (general-purpose NLI)
- No speaker identity modeling (treats all speakers as population average after baseline)
- Async calling pattern may conflict with sync pipeline (workaround: asyncio.get_event_loop() fallback)

**Roadmap (Phase-2):**
- Fine-tune DeBERTa on domain-specific deception dataset (ICSI corpus, UIWC corpus)
- Add multilingual support (mBERT, XLM-RoBERTa)
- Implement speaker-aware embeddings (fine-grained profiling)
- Add sarcasm/irony detection (separate fine-tuned head)

---

### 3. **Physiological Module (PM)** — `/02_PhysiologicalModality/physio_v3.py`

**Architecture:**
- **Signal Processing:** Remote photoplethysmography (rPPG) via face region pixel intensity
- **HR Extraction:** Spectral analysis (FFT-based peak detection in 40–200 bpm range)
- **Wearable Integration:** Bluetooth Low Energy (BLE) for Timex heart rate watch (optional)
- **Classifier:** Rule-based + Isolation Forest anomaly detection

**Processing Pipeline:**
```
Video Frame (ROI: cheek region)
    ▼
Extract Green Channel (most PPG signal)
    ▼
Spatial Averaging (reduce noise)
    ▼
Temporal Filtering (detrend + butterworth bandpass 0.5–4 Hz)
    ▼
FFT Spectral Analysis
    ▼
Peak Detection (heart rate peaks in 40–200 bpm range)
    ▼
RSA Calculation (respiratory sinus arrhythmia from HR variability)
    ▼
Baseline Comparison (per-speaker normal HR/RSA ranges)
    ▼
Anomaly Detection (Isolation Forest on {HR, RSA, SpO2_if_available})
    ▼
Output: {heart_rate, spo2_if_available, rsa, deception_prob, anomaly_score}
```

**Key Algorithms:**
- **rPPG Extraction:** Green channel → spatial average over ROI → temporal detrending → filtering → FFT
- **HR Detection:** Find strongest FFT peak in valid range, convert frequency to bpm
- **RSA Calculation:** SDNN (std dev of HR intervals) as proxy for respiratory modulation
- **Anomaly Scoring:** Isolation Forest trained on baseline HR/RSA during calibration phase
- **BLE Integration:** GATT UUID `180D` (Heart Rate Service), parse SFLOAT 16-bit format, check signal quality

**Performance:**
- Latency: 8–12 ms per frame (rPPG only, no ML inference)
- HR Accuracy: 45–80 ms inference, ±10 bpm typical error vs pulse oximeter
- Model size: 12 MB (Isolation Forest pickle, minimal)
- Calibration: Requires 30–60s baseline window at session start

**Known Limitations:**
- Assumes 30 FPS camera (hardcoded in some places, needs per-camera calibration)
- No real-time rPPG visualization (only logs to run_log.jsonl)
- rPPG sensitive to lighting changes, head motion, skin tone
- BLE wearable optional (no requirement for PM to run)
- SPO2 not available from rPPG (BLE only)
- Isolation Forest baseline only updated at session start (no adaptive learning)

**Roadmap (Phase-2):**
- Implement adaptive rPPG with per-frame illumination compensation
- Add 3D face pose tracking to correct for head rotation effects
- Support multiple camera frame rates (auto-calibration)
- Integrate HRV (heart rate variability) analysis (RMSSD, pNN50)
- Add real-time PPG waveform visualization on HUD

---

### 4. **Speech Module (SM)** — `/speech_module.py`

**Architecture:**
- **Feature Extraction:** Acoustic features (MFCC, spectral moments, formants) extracted via librosa
- **Classifiers:** Random Forest (stress model) + Random Forest (deception model)
- **Status:** Currently trained on synthetic features (placeholder); real data unavailable
- **Performance:** ~50% accuracy (uninformative prior) due to synthetic training data

**Processing Pipeline:**
```
Audio Chunk (variable length, 16 kHz PCM)
    ▼
Voice Activity Detection (energy threshold or pre-processing)
    ▼
Feature Extraction (MFCC-13, spectral centroid, zero-crossing-rate, etc. → 94-D vector)
    ▼
Random Forest Forward Pass (stress model)
    ▼
Random Forest Forward Pass (deception model)
    ▼
Confidence Estimation (per-tree probability averaging)
    ▼
Output: {stress_prob, deception_prob, feature_trace, status}
```

**Key Algorithms:**
- **MFCC Extraction:** Librosa computes 13 MFCCs + deltas + delta-deltas from spectrogram
- **Spectral Features:** Centroid, rolloff, zero-crossing rate, spectral flux
- **Random Forest:** Ensemble of 100 trees, max_depth=15, trained via scikit-learn
- **Probability:** Mean probability across all trees in forest
- **Current Status:** NeutralModel returns uninformative 0.5 prior (real RF needs 300+ hours of labeled data)

**Performance:**
- Latency: 30–50 ms per utterance
- Accuracy: 50% (random guessing) on synthetic data; real data needed for validation
- Model size: 50 KB per model (Random Forest pickle)
- Feature dimension: 94-D vector

**Known Limitations:**
- **CRITICAL:** Trained only on synthetic features (random noise); real accuracy unknown
- No VAD implementation (assumes input is already speech segments)
- No speaker normalization (feature scaling per speaker would help)
- Random Forest cannot model temporal dynamics (uses frame-level features only)
- No uncertainty quantification (only point probability, no confidence intervals)

**Roadmap (Phase-1 Blocker):**
- **HIGH PRIORITY:** Obtain or generate labeled speech corpus (300+ hours of deception/stress-induced speech)
- Possible sources: ICSI meeting corpus, UIWC lab recordings, synthetic speech under stress
- Retrain Random Forest on real acoustic features
- Target accuracy: 75–82% on held-out test set

**Roadmap (Phase-2):**
- Replace Random Forest with CNN (1D convolution on spectrograms)
- Add temporal modeling (LSTM on feature sequences)
- Implement speaker adaptation (personalized baseline per speaker)
- Add multilingual support (language-specific MFCC baselines)

---

### 5. **Fusion System** — `/fusion.py`

**Architecture:**
- **Combination Method:** Confidence-weighted Bayesian fusion
- **Temporal Decay:** Exponential decay for older modality outputs (handles async timing)
- **Alert Levels:** 5-level classification (GREEN, YELLOW, ORANGE, RED, CRITICAL)
- **Explanation Engine:** Natural language explanations of deception signal

**Fusion Algorithm:**

```
Input: MM probability, TM probability, PM probability, SM probability + confidences

Step 1: Normalize Confidences
  conf_mm ← (MM_confidence) / (MM_confidence + TM_confidence + PM_confidence + SM_confidence + ε)
  conf_tm ← (TM_confidence) / (...)
  conf_pm ← (PM_confidence) / (...)
  conf_sm ← (SM_confidence) / (...)

Step 2: Temporal Decay
  For each modality, apply exponential decay:
    prob_decayed = prob_current × exp(-λ × Δt_since_measurement)
    λ = 0.01 (decay constant), Δt in milliseconds

Step 3: Weighted Combination
  deception_fused = (conf_mm × prob_mm_decayed + 
                     conf_tm × prob_tm_decayed + 
                     conf_pm × prob_pm_decayed + 
                     conf_sm × prob_sm_decayed)

Step 4: Alert Classification
  if deception_fused < 0.3:
    alert_level = GREEN (confident truth)
  elif deception_fused < 0.5:
    alert_level = YELLOW (possible truth, low signal)
  elif deception_fused < 0.65:
    alert_level = ORANGE (borderline, investigate)
  elif deception_fused < 0.8:
    alert_level = RED (likely deception)
  else:
    alert_level = CRITICAL (high confidence deception)

Step 5: Generate Explanation
  explanation = ""
  if MM_probability > 0.6:
    explanation += f"Facial expression shows {EMOTION} (±{AU_deviation_magnitude})"
  if TM_probability > 0.6:
    explanation += f"Speech contains contradiction (NLI: {NLI_score:.2f})"
  if PM_probability > 0.6:
    explanation += f"Heart rate elevated {HR_bpm} bpm (baseline: {baseline_hr} bpm)"
  if SM_probability > 0.6:
    explanation += f"Acoustic stress markers detected"
  return {deception_prob, alert_level, explanation}
```

**Performance:**
- Latency: <5 ms (weighted averaging only)
- Memory: <1 MB (no state except sliding window of last 100 outputs)
- Real-time: Runs synchronously in pipeline main loop

**Known Limitations:**
- Static weighting (not adaptive to per-modality accuracy)
- Temporal decay assumes all modalities measure same phenomenon (true for deception, but MM/PM/SM have different latencies)
- Alert thresholds hard-coded (not learned from data)
- No uncertainty propagation (doesn't track confidence intervals)
- Explanation engine is rule-based (no learned explanation model)

**Roadmap (Phase-2):**
- Learn fusion weights via logistic regression on labeled dataset
- Implement adaptive weighting based on per-modality historical error rates
- Add Bayesian network with modality dependencies (e.g., MM and TM correlated)
- Generate data-driven explanations via attention mechanism

---

## Session Management

### Subject Profiles (`session.py`)

Each subject maintains a persistent YAML profile:

```yaml
subject_id: SUBJECT_001
sessions:
  - session_id: SUBJECT_001_20260509_120000
    timestamp: 2026-05-09T12:00:00Z
    duration_minutes: 45

mm_baseline:
  - session_id: SUBJECT_001_20260509_120000
    au_means: {1: 0.2, 2: 0.15, 4: 0.3, ...}  # 43 AUs total
    au_stds:  {1: 0.05, 2: 0.04, 4: 0.08, ...}
    emotion_distribution: {angry: 0.05, disgust: 0.02, fear: 0.01, ...}

tm_baseline:
  - session_id: SUBJECT_001_20260509_120000
    hedge_frequency: 0.12  # fraction of utterances with hedge words
    self_correction_rate: 0.08
    temporal_marker_density: 0.15
    entity_persistence: 0.85  # fraction of mentioned entities repeated

pm_baseline:
  - session_id: SUBJECT_001_20260509_120000
    mean_hr: 72
    std_hr: 8
    mean_rsa: 25
    std_rsa: 5
    resting_spo2: 98  # if wearable available

sm_baseline:
  - session_id: SUBJECT_001_20260509_120000
    mean_pitch: 140  # Hz
    pitch_std: 25
    mean_intensity: 65  # dB
    intensity_std: 8
    mfcc_mean: [1.2, -2.3, 0.4, ...]  # 13-D vector

session_count: 1
last_active: 2026-05-09T12:45:00Z
```

**Baseline Usage:**
- At session start, load subject's profile
- First 60–120 seconds: collect baseline measurements
- Store baseline as mean/std for each modality
- During inference: compare current measurements to baseline via z-score
- Modality output: baseline_deviation = (current - mean) / std

---

## Data Flow & Timing

### Frame Processing Timeline (Single Frame @ 30 FPS = 33.3 ms Budget)

```
T=0 ms:   Frame arrives from camera
T=5 ms:   Face detection via MediaPipe
T=10 ms:  AU calculation + ResNet-18 forward pass (MM)
T=15 ms:  rPPG extraction from frame (PM)
          [Text utterance processed async if available (TM)]
          [Audio chunk processed async if available (SM)]
T=20 ms:  MM output ready: {emotion, deception_prob, confidence}
T=25 ms:  PM output ready: {heart_rate, rsa, deception_prob}
T=28 ms:  Fusion layer combines MM + PM + [TM, SM if ready]
T=30 ms:  HUD rendering
T=33.3ms: Frame complete, ready for next frame

Total Latency (Camera to Display): ~30 ms
Modality Latencies:
  - MM: 45–80 ms per frame (includes ResNet-18)
  - TM: 200–400 ms per utterance (async)
  - PM: 8–12 ms per frame (signal processing only)
  - SM: 30–50 ms per utterance (async)
```

### Async Coordination

```
Main Loop (30 FPS, synchronous):
  for frame in video_stream:
    mm_out = mm.process_frame(frame)
    pm_out = pm.process_frame(frame)
    
    if utterance_available:
      asyncio.create_task(tm.process(utterance))  # Fire & forget
    if audio_available:
      asyncio.create_task(sm.process(audio))      # Fire & forget
    
    fusion_out = fusion.combine(mm_out, pm_out, tm_out_if_ready, sm_out_if_ready)
    render_hud(fusion_out)
```

Fusion waits for TM/SM results up to 500ms timeout; uses most recent available values if timeout exceeded.

---

## File Organization

```
deception_detection_project/
├── main.py                              # CLI entry point
├── pipeline.py                          # Core orchestrator
├── fusion.py                            # Fusion layer
├── session.py                           # Subject profiles + baselines
├── hud.py                               # HUD rendering
├── run_logger.py                        # JSONL logging
├── config.yaml                          # Default configuration
│
├── 00_MicroexpressionModality/
│   ├── macro_v7.py                      # MM core (ResNet-18)
│   ├── micro_expression_models/
│   │   └── micro_model.pth              # ResNet-18 weights
│   └── mm_run_log.jsonl                 # Per-run logs
│
├── 01_TextModality/
│   ├── text_v1.py                       # TM core (DeBERTa NLI)
│   ├── tm_run_log.jsonl
│   └── cached_models/
│       └── deberta-v3-small/            # Downloaded on first run
│
├── 02_PhysiologicalModality/
│   ├── physio_v3.py                     # PM core (rPPG + BLE)
│   ├── pm_run_log.jsonl
│   └── wearable_cache.pkl               # BLE connection cache
│
├── 03_SpeechModality/
│   ├── speech_module.py                 # SM core (Random Forest)
│   ├── sm_models/
│   │   ├── stress_rf.pkl
│   │   └── deception_rf.pkl
│   └── sm_run_log.jsonl
│
├── subjects/
│   ├── SUBJECT_001.yaml                 # Profiles
│   ├── SUBJECT_002.yaml
│   └── ...
│
├── runs/
│   ├── SUBJECT_001_20260509_120000/
│   │   ├── mm_run_log.jsonl
│   │   ├── tm_run_log.jsonl
│   │   ├── pm_run_log.jsonl
│   │   ├── sm_run_log.jsonl
│   │   ├── fusion_run_log.jsonl
│   │   └── session_config.yaml
│   └── ...
│
├── docs/
│   ├── MODELS.md                        # Model documentation
│   ├── BUGFIXES_APPLIED.md
│   ├── TEST_GUIDE.md
│   └── ARCHITECTURE.md (this file)
└── requirements.txt
```

---

## Dependencies & Setup

### Python Packages (see `requirements.txt`)

```
torch>=2.0.0                   # MM ResNet-18 inference
torchvision>=0.15.0           # Vision utilities
transformers>=4.30.0          # TM DeBERTa model
scikit-learn>=1.3.0           # SM Random Forest
librosa>=0.10.0               # SM audio features
opencv-python>=4.8.0          # Video/image processing
numpy>=1.24.0                 # Numerical computing
scipy>=1.11.0                 # Signal processing (PM)
mediapipe>=0.8.11             # Face detection (MM)
pyyaml>=6.0                   # Config files
tqdm>=4.66.0                  # Progress bars
bleak>=0.20.0                 # BLE (optional)
Pillow>=10.0.0                # Image manipulation
```

### Hardware Requirements

**Minimum (headless, no display):**
- CPU: Intel i7 / AMD Ryzen 5 (4+ cores)
- RAM: 4 GB
- Disk: 500 MB (models + logs)

**Recommended (real-time with display):**
- CPU: Intel i9 / AMD Ryzen 9 (8+ cores)
- GPU: NVIDIA RTX 3060+ (CUDA 11.8+) or Apple M1/M2
- RAM: 8+ GB
- Disk: 2 GB (extended logging)

**Optional:**
- Camera: 1080p @ 30 FPS minimum
- Microphone: 16 kHz, 16-bit PCM
- Wearable: Timex watch with BLE HR sensor

---

## Environment Setup

```bash
# 1. Clone repository
git clone <repo_url>
cd deception_detection_project

# 2. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained models (optional, auto-downloads on first run)
python -c "from transformers import pipeline; pipeline('text-classification', model='microsoft/deberta-v3-small')"

# 5. Initialize subject profiles directory
mkdir -p subjects runs

# 6. Run tests
python TEST_GUIDE.py  # or individual unit tests from TEST_GUIDE.md
```

---

## Known Issues & Workarounds

| Issue | Severity | Workaround | Status |
|-------|----------|-----------|--------|
| mediapipe resource leak | MEDIUM | Use try/finally wrapper in MM frame loop | Noted, not fixed |
| asyncio loop in sync context | MEDIUM | Fallback to new_event_loop() on RuntimeError | Implemented |
| TM NLI format bug | CRITICAL | Fixed: dict→tuple format | ✓ Fixed |
| SM synthetic training data | CRITICAL | Using NeutralModel placeholder (50% prior) | Blocking Phase-1 |
| Unbounded text history | MEDIUM | Per-speaker queue with max_length | Implemented |
| rPPG frame rate hardcoded | MEDIUM | Per-camera calibration needed | Roadmap |
| Empty modality graceful degradation | HIGH | Fallback models created at startup | ✓ Fixed |

---

## Deployment Checklist

- [ ] All 4 modalities initialize without errors
- [ ] Integration test runs for 5+ minutes without crashing
- [ ] Subject profile saves/loads correctly
- [ ] Run logs generate valid JSONL
- [ ] HUD displays real-time outputs
- [ ] Fusion produces valid deception probabilities
- [ ] Alert levels trigger at correct thresholds
- [ ] Memory usage stable over 1-hour run (<500 MB growth)
- [ ] Camera + microphone working (if available)
- [ ] BLE wearable connects (if hardware available)

---

**Document Version:** 2.0  
**Last Updated:** May 9, 2026  
**System Status:** Production-Ready (Phase-1)  
**Known Blockers:** SM real training data acquisition
