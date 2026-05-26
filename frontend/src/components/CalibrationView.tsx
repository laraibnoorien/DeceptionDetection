import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useStore } from '../store/useStore';
import { motion, AnimatePresence } from 'motion/react';
import {
  Camera,
  Activity,
  Mic,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Info,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { cn } from '../lib/utils';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type CalibrationStep = 'EMOTIONAL' | 'PHYSIOLOGICAL' | 'SPEECH' | 'COMPLETED';

interface CalibrationProgress {
  overall_progress: number;   // 0-1
  current_item: string;
  stage: string;
  estimated_time_remaining: number;
  frames_processed: number;
  samples_collected: number;
  feature_extraction_complete: boolean;
}

interface CalibrationConfig {
  sessionId: string;
  subjectId: string;
  stimulusVideos: string[];   // list of video filenames
  questions: string[];
}

// ---------------------------------------------------------------------------
// API Helpers (adjust base URL as needed)
// Defaults to backend at localhost:8000 for development; override with VITE_API_BASE/VITE_WS_BASE
// ---------------------------------------------------------------------------
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000';

// Capture duration constants (ms)
const EMOTION_CAPTURE_MS = Number(import.meta.env.VITE_EMOTION_CAPTURE_MS) || 3000;
const PHYSIO_CAPTURE_MS = Number(import.meta.env.VITE_PHYSIO_CAPTURE_MS) || 6000;
// Backend feature-ack timeout (ms)
const FEATURE_ACK_TIMEOUT_MS = Number(import.meta.env.VITE_FEATURE_ACK_TIMEOUT_MS) || 15000;

async function apiCall<T>(path: string, options?: RequestInit): Promise<T | null> {
  const res = await fetch(`${API_BASE}${path}`, {
    // Only set Content-Type when sending a body
    ...(options && options.body ? { headers: { 'Content-Type': 'application/json' } } : {}),
    ...options,
  });
  if (res.status === 204) return null;
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`API error: ${res.status} ${txt}`);
  }
  // Try to parse JSON safely; return null on empty body
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    return null;
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export const CalibrationView: React.FC = () => {
  const { setStage, updateCalibration, calibration, telemetry } = useStore();

  const [step, setStep] = useState<CalibrationStep>('EMOTIONAL');
  const [config, setConfig] = useState<CalibrationConfig | null>(null);
  const [progress, setProgress] = useState<CalibrationProgress>({
    overall_progress: 0,
    current_item: '',
    stage: 'EMOTIONAL',
    estimated_time_remaining: 0,
    frames_processed: 0,
    samples_collected: 0,
    feature_extraction_complete: false,
  });
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sessionActive, setSessionActive] = useState(false);
  const [stepCompletion, setStepCompletion] = useState<Record<CalibrationStep, boolean>>({
    EMOTIONAL: false,
    PHYSIOLOGICAL: false,
    SPEECH: false,
    COMPLETED: false,
  });

  // Demo / manual start controls
  const [autoStartAllowed, setAutoStartAllowed] = useState(false);
  const [capturedEmotions, setCapturedEmotions] = useState<string[]>([]);
  const [startingEmotion, setStartingEmotion] = useState<string | null>(null);

  // Capture lifecycle flags
  const [isEmotionCapturing, setIsEmotionCapturing] = useState(false);
  const [isPhysioCapturing, setIsPhysioCapturing] = useState(false);
  const [liveFeedActive, setLiveFeedActive] = useState(false);

  // Fallback stimulus/emotion flow when no server stimulus videos are available
  const [stimulusIndex, setStimulusIndex] = useState<number>(0);
  const emotionsFallback = ['Neutral', 'Happy', 'Sad', 'Surprised'];
  const [showStimulusText, setShowStimulusText] = useState<string | null>(null);
  const [speechRecorded, setSpeechRecorded] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timersRef = useRef<number[]>([]);
  // Guard against React StrictMode double-mounting in development
  const initCalledRef = useRef<boolean>(false);
  // Track whether camera stream is active (prevent duplicate opens)
  const cameraActiveRef = useRef<boolean>(false);
  // Guard to allow a single WS reconnection attempt
  const reconnectAttemptedRef = useRef<boolean>(false);

  // Allow runtime-adjustable capture durations (seconds in UI -> ms internally)
  const [emotionCaptureMs, setEmotionCaptureMs] = useState<number>(EMOTION_CAPTURE_MS);
  const [physioCaptureMs, setPhysioCaptureMs] = useState<number>(PHYSIO_CAPTURE_MS);

  // Lightweight sleep helper
  const sleep = (ms: number) => new Promise<void>(res => {
    const t = window.setTimeout(res, ms);
    timersRef.current.push(t);
  });

  // Play simple text-based stimuli when no stimulus videos are provided by server
  const playStimuliFallback = useCallback(async () => {
    try {
      for (let i = 0; i < emotionsFallback.length; i++) {
        const emo = emotionsFallback[i];
        setStimulusIndex(i);
        setShowStimulusText(emo);
        // allow a few seconds for expression capture (frames are already being sent)
        await sleep(3000);
      }
      setShowStimulusText(null);
      setStepCompletion(prev => ({ ...prev, EMOTIONAL: true }));
    } catch (e) {
      console.warn('Stimuli fallback interrupted', e);
    }
  }, [emotionsFallback]);

  // Record answers sequentially for speech calibration and send audio blobs to backend over WebSocket
  const recordAnswersSequentially = useCallback(async (questions: string[] = []) => {
    if (!questions || !questions.length) return;
    try {
      for (let i = 0; i < questions.length; i++) {
        const q = questions[i];
        // get audio stream for short duration
        let stream: MediaStream | null = null;
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        } catch (err) {
          console.warn('Audio permission denied', err);
          break;
        }

        const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        const chunks: Blob[] = [];
        mr.ondataavailable = (ev: BlobEvent) => {
          if (ev.data && ev.data.size) chunks.push(ev.data);
        };

        try {
          mr.start();
        } catch (e) {
          console.warn('MediaRecorder start failed', e);
          setCameraError('Microphone recording failed. Please check permissions.');
          try { cancelCurrentCapture(); } catch(_) {}
          stream.getTracks().forEach(t => t.stop());
          break;
        }
        // Record for fixed window (4 seconds) per question
        await sleep(4000);
        try { mr.stop(); } catch (e) { /* ignore */ }

        // wait for stop and data to be available
        await new Promise<void>((res) => {
          mr.onstop = () => res();
        });

        const blob = new Blob(chunks, { type: 'audio/webm' });
        // send audio blob over calibration websocket as binary payload
        try {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(blob);
          }
        } catch (e) {
          console.warn('Failed to send audio blob over websocket', e);
          // as fallback, POST to /api/metrics/push with base64 (best-effort)
          try {
            const buf = await blob.arrayBuffer();
            const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
            await fetch(`${API_BASE}/api/metrics/push`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ type: 'audio_blob', question: q, audio_b64: b64 }),
            });
          } catch (e2) {
            console.warn('Fallback audio upload failed', e2);
          }
        }

        // stop and release tracks
        try {
          stream.getTracks().forEach(t => t.stop());
        } catch (_) {}
      }
      setSpeechRecorded(true);
      setStepCompletion(prev => ({ ...prev, SPEECH: true }));
    } catch (e) {
      console.warn('Speech recording failed', e);
    }
  }, []);

  // -----------------------------------------------------------------------
  // Initialize calibration session & open camera
  // -----------------------------------------------------------------------
  useEffect(() => {
    // Prevent double initialization (React StrictMode mounts twice in dev)
    if (initCalledRef.current) return;
    initCalledRef.current = true;

    let isMounted = true; // Guard against post-async updates
    
    // Prevent auto-start; session must be initiated by the user via the UI
    if (!autoStartAllowed) {
      setIsLoading(false);
      return;
    }

    const init = async () => {
      try {
        setIsLoading(true);
        // Start calibration session on backend (subjectId from store or context)
        const subjectId = telemetry?.speaker_id || telemetry?.subjectId || telemetry?.subject || 'web-subject';
        const raw = await apiCall<any>('/api/calibration/start', {
          method: 'POST',
          body: JSON.stringify({ subject_id: subjectId }),
        });
        
        // Check if component is still mounted
        if (!isMounted) return;

        // Normalize various possible response shapes
        let sessionId: string | undefined;
        let stimulusVideos: string[] = [];
        let questions: string[] = [];
        if (raw) {
          if (raw.session_id || raw.sessionId) {
            sessionId = raw.session_id || raw.sessionId;
            stimulusVideos = raw.stimulus_videos || raw.stimulusVideos || raw.stimulus_videos || [];
            questions = raw.questions || raw.questions || [];
          } else if (raw.success && raw.data) {
            const d = raw.data;
            sessionId = d.session_id || d.sessionId || d.id;
            stimulusVideos = d.stimulus_videos || d.stimulusVideos || [];
            questions = d.questions || [];
          }
        }

        if (!sessionId) throw new Error('Could not obtain calibration session id from backend');

        setConfig({
          sessionId,
          subjectId,
          stimulusVideos,
          questions,
        });
        setSessionActive(true);
        // Open camera
        await openCamera();
        // Connect WebSocket with subject id as query param
        connectWebSocket(sessionId, subjectId);
      } catch (err: any) {
        setCameraError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    init();

    return () => {
      isMounted = false; // Mark as unmounted
      // Cleanup
      try { if (wsRef.current) wsRef.current.close(); } catch(_) {}
      try {
        // Prefer stopping tracks from the video element's srcObject if present
        const s = (videoRef.current && (videoRef.current.srcObject as MediaStream | null)) || cameraStream;
        if (s && s.getTracks) s.getTracks().forEach(t => t.stop());
      } catch (_) {}
    };
  }, [cameraStream]);

  // If server provided no stimulus videos, run a local fallback stimuli playback
  useEffect(() => {
    if (!config || !sessionActive) return;
    const hasServerVideos = Array.isArray(config.stimulusVideos) && config.stimulusVideos.length > 0;
    if (!hasServerVideos) {
      // play simple text stimuli locally to capture facial baseline
      playStimuliFallback();
    }
  }, [config, sessionActive, playStimuliFallback]);

  // Trigger speech recording flow when entering SPEECH step (once)
  useEffect(() => {
    if (step === 'SPEECH') {
      // Ensure camera is stopped before speech stage (mic-only)
      try { stopCamera(); } catch (_) {}
      setIsEmotionCapturing(false);
      setIsPhysioCapturing(false);
      // Start mic-only speech recording
      if (config?.questions && !speechRecorded) {
        recordAnswersSequentially(config.questions);
      }
    }
  }, [step, config, speechRecorded, recordAnswersSequentially]);

  // Auto-start physio capture when entering PHYSIOLOGICAL
  useEffect(() => {
    if (step === 'PHYSIOLOGICAL' && sessionActive && !stepCompletion.PHYSIOLOGICAL) {
      startPhysioCapture(physioCaptureMs);
    }
  }, [step, sessionActive, stepCompletion, physioCaptureMs]);

  // After calibration completed, start persistent live optical feed (separate lifecycle)
  useEffect(() => {
    if (step === 'COMPLETED') {
      // Ensure calibration camera is stopped first
      try { stopCamera(); } catch (_) {}
      startLiveOpticalFeed();
      setStepCompletion(prev => ({ ...prev, COMPLETED: true }));
    }
  }, [step]);

  // Global unmount cleanup: stop cameras, live feed, close websocket, clear timers
  useEffect(() => {
    return () => {
      try { stopCamera(); } catch(_) {}
      try { stopLiveOpticalFeed(); } catch(_) {}
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.close();
        }
      } catch(_) {}
      // clear pending timers
      try {
        timersRef.current.forEach(t => clearTimeout(t));
      } catch(_) {}
      timersRef.current = [];
    };
  }, []);

  const openCamera = async () => {
    try {
      if (cameraActiveRef.current) return; // prevent duplicate opens
      if (cameraStream) return; // already open
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      console.log('[CAMERA_START]');
      cameraActiveRef.current = true;
      setCameraStream(stream);
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err: any) {
      setCameraError('Camera access denied. Please allow camera permissions.');
      // recover capture state but keep websocket alive
      try { cancelCurrentCapture(); } catch(_) {}
      throw err;
    }
  };

  const stopCamera = () => {
    try {
      console.log('[CAMERA_STOP]');
      // Stop video element's srcObject tracks first
      const s = (videoRef.current && (videoRef.current.srcObject as MediaStream | null)) || cameraStream;
      if (s && s.getTracks) {
        s.getTracks().forEach(t => t.stop());
      }
    } catch (e) {
      console.warn('stopCamera error', e);
    } finally {
      try { if (videoRef.current) videoRef.current.srcObject = null; } catch(_) {}
      setCameraStream(null);
      cameraActiveRef.current = false;
    }
  };

  const cancelCurrentCapture = () => {
    // Stop timers, stop camera, reset capture flags but keep websocket alive
    try {
      console.log('[CAPTURE_END]');
      timersRef.current.forEach(t => clearTimeout(t));
      timersRef.current = [];
    } catch (_) {}
    try { stopCamera(); } catch(_) {}
    setIsEmotionCapturing(false);
    setIsPhysioCapturing(false);
    setStartingEmotion(null);
    setShowStimulusText(null);
  };

  const tryOpenCameraRetry = async () => {
    // Clear previous camera error and attempt a single retry
    setCameraError(null);
    try {
      await openCamera();
    } catch (e) {
      setCameraError('Camera retry failed. Check permissions and device.');
    }
  };

  const startLiveOpticalFeed = async () => {
    if (liveFeedActive) return;
    await openCamera();
    console.log('[CAMERA_START] [LIVE_FEED]');
    setLiveFeedActive(true);
  };

  const stopLiveOpticalFeed = () => {
    console.log('[CAMERA_STOP] [LIVE_FEED]');
    setLiveFeedActive(false);
    stopCamera();
  };

  const startPhysioCapture = async (durationMs = PHYSIO_CAPTURE_MS) => {
    if (isPhysioCapturing) return;
    try {
      console.log('[CAPTURE_START] [PHYSIO]');
      setIsPhysioCapturing(true);
      await openCamera();
      // let capture run for durationMs, then stop
      const t = window.setTimeout(() => {
        stopPhysioCapture();
      }, durationMs);
      timersRef.current.push(t);
    } catch (e) {
      console.warn('startPhysioCapture failed', e);
      setIsPhysioCapturing(false);
      try { cancelCurrentCapture(); } catch(_) {}
    }
  };

  const stopPhysioCapture = () => {
    setIsPhysioCapturing(false);
    stopCamera();
    setStepCompletion(prev => ({ ...prev, PHYSIOLOGICAL: true }));
  };

  const connectWebSocket = (sessionId: string, subjectId?: string) => {
    // Prevent duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const qs = subjectId ? `?subject_id=${encodeURIComponent(subjectId)}` : '';
    const wsUrl = `${WS_BASE}/ws/calibration/${sessionId}${qs}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsError(null);
      reconnectAttemptedRef.current = false;
      console.log('[WS_CONNECTED]');
      // Inform backend we're ready (optional)
      try { ws.send(JSON.stringify({ type: 'ready', sessionId })); } catch (e) {}
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
          setProgress(data.payload as CalibrationProgress);
        } else if (data.type === 'stage_complete') {
          // Mark current step as complete
          const completedStage = data.payload.stage as CalibrationStep;
          setStepCompletion(prev => ({ ...prev, [completedStage]: true }));
          // Store baseline profile data if provided
          if (data.payload.baseline) {
            updateCalibration(data.payload.baseline);
          }
        } else if (data.type === 'error') {
          setWsError(data.payload.message);
        }
      } catch (err) {
        console.error('WebSocket message parse error', err);
      }
    };

    ws.onerror = (e) => setWsError('WebSocket connection error');
    ws.onclose = (e) => {
      console.log('[WS_CLOSED]');
      // Attempt a single reconnect if calibration is active and we have session info
      if (sessionActive && config?.sessionId && !reconnectAttemptedRef.current) {
        reconnectAttemptedRef.current = true;
        console.log('[WS_CLOSED] attempting one-time reconnect');
        setTimeout(() => {
          try { connectWebSocket(config.sessionId, config.subjectId); } catch (_) {}
        }, 1000);
        return;
      }
      setWsError('WebSocket connection closed');
    };
  };

  // Start a real calibration session (manual start)
  const startCalibration = async () => {
    try {
      setIsLoading(true);
      const subjectId = telemetry?.speaker_id || telemetry?.subjectId || telemetry?.subject || 'web-subject';
      const raw = await apiCall<any>('/api/calibration/start', {
        method: 'POST',
        body: JSON.stringify({ subject_id: subjectId }),
      });
      if (!raw || !raw.data) throw new Error('Calibration start failed');
      const d = raw.data;
      const sessionId = d.id || d.sessionId || d.id;
      setConfig({ sessionId, subjectId, stimulusVideos: d.stimulus_videos || [], questions: d.questions || [] });
      setSessionActive(true);
      setAutoStartAllowed(true);
      await openCamera();
      connectWebSocket(sessionId, subjectId);
    } catch (e: any) {
      setWsError(e.message || 'Calibration start failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Demo mode starter (calls backend /api/demo/start to create a synthetic session)
  const startDemo = async () => {
    try {
      setIsLoading(true);
      const subjectId = telemetry?.speaker_id || telemetry?.subjectId || telemetry?.subject || 'demo-subject';
      const raw = await apiCall<any>('/api/demo/start', {
        method: 'POST',
        body: JSON.stringify({ subject_id: subjectId }),
      });
      if (!raw || !raw.data) throw new Error('Demo start failed');
      const d = raw.data;
      const sessionId = d.id || d.sessionId || d.id;
      setConfig({ sessionId, subjectId, stimulusVideos: d.stimulus_videos || [], questions: d.questions || [] });
      setSessionActive(true);
      setAutoStartAllowed(true);
      await openCamera();
      connectWebSocket(sessionId, subjectId);
    } catch (e: any) {
      setWsError(e.message || 'Demo start failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to capture a single emotion stimulus (manual trigger)
  const startEmotionCapture = async (emotion: string) => {
    if (!sessionActive) {
      setWsError('Please start a calibration session first');
      return;
    }
    if (capturedEmotions.includes(emotion)) return;
    try {
      console.log('[CAPTURE_START] [EMOTION]', emotion);
      setStartingEmotion(emotion);
      setShowStimulusText(emotion);
      setIsEmotionCapturing(true);
      // ensure camera is open for capture
      try { await openCamera(); } catch (e) { console.warn('openCamera failed', e); }
      // Notify backend of stimulus start (best-effort)
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'start_stimulus', stimulus: emotion }));
        }
      } catch (e) {}

      // capture window (frames will be sent only during isEmotionCapturing)
      await sleep(emotionCaptureMs);

      // update captured list synchronously
      const newCaptured = Array.from(new Set([...capturedEmotions, emotion]));
      setCapturedEmotions(newCaptured);
      setShowStimulusText(null);
      setStartingEmotion(null);
      setIsEmotionCapturing(false);
      // stop camera after per-emotion capture (per requirements)
      stopCamera();

      // If all emotions captured, wait for backend to acknowledge feature extraction
      const all = (config?.stimulusVideos && config.stimulusVideos.length>0) ? config.stimulusVideos : emotionsFallback;
      const allCaptured = all.every((e:any)=> newCaptured.includes(e));
      if (allCaptured) {
        // poll for backend acknowledgement (feature_extraction_complete or stepCompletion.EMOTIONAL)
        const start = Date.now();
        const timeout = 10000; // 10s
        let acknowledged = false;
        while (Date.now() - start < timeout) {
          if (progress.feature_extraction_complete || stepCompletion.EMOTIONAL) { acknowledged = true; break; }
          await sleep(500);
        }
        if (acknowledged) {
          setStepCompletion(prev=>({...prev, EMOTIONAL: true}));
        } else {
          setWsError('Backend did not acknowledge emotional feature extraction. Please retry capture.');
        }
      }

      console.log('[CAPTURE_END] [EMOTION]', emotion);
    } catch (e: any) {
      console.warn('startEmotionCapture failed', e);
      setCameraError(e?.message || 'Capture failed');
      try { cancelCurrentCapture(); } catch(_) {}
      setIsEmotionCapturing(false);
      setStartingEmotion(null);
      setShowStimulusText(null);
    }
  };

  // -----------------------------------------------------------------------
  // Handle step transitions (triggered by user or completion)
  // -----------------------------------------------------------------------
  const nextStep = useCallback(() => {
    setStep(prev => {
      if (prev === 'EMOTIONAL') return 'PHYSIOLOGICAL';
      if (prev === 'PHYSIOLOGICAL') return 'SPEECH';
      if (prev === 'SPEECH') return 'COMPLETED';
      return prev;
    });
  }, []);

  // If a step is completed by backend, automatically advance after a short delay
  useEffect(() => {
    if (stepCompletion[step] && step !== 'COMPLETED') {
      const timer = setTimeout(nextStep, 2000);
      return () => clearTimeout(timer);
    }
  }, [stepCompletion, step, nextStep]);

  // -----------------------------------------------------------------------
  // Send camera frames to backend for analysis (simplified – you may want
  // to throttle the frame rate and send via WebSocket binary messages)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!cameraStream || !canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    const captureFrame = () => {
      // Only send frames when actively capturing or live feed is on
      if (!(isEmotionCapturing || isPhysioCapturing || liveFeedActive)) return;
      if (!context || !video.videoWidth) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          try { wsRef.current.send(blob); } catch (e) { /* swallow */ }
        }
      }, 'image/jpeg', 0.6);
    };

    const interval = window.setInterval(captureFrame, 100); // ~10 fps (throttled)
    timersRef.current.push(interval);
    return () => {
      clearInterval(interval);
      // remove interval id from timersRef
      timersRef.current = timersRef.current.filter(t => t !== interval);
    };
  }, [cameraStream, isEmotionCapturing, isPhysioCapturing, liveFeedActive]);

  // -----------------------------------------------------------------------
  // Render helpers
  // -----------------------------------------------------------------------
  const renderCameraFeed = () => (
    <div className="relative w-full max-w-lg mx-auto rounded-2xl overflow-hidden border border-white/10 shadow-2xl">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-auto bg-black"
      />
      <canvas ref={canvasRef} className="hidden" />
      {cameraError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-red-400 p-6 text-center text-sm">
          <AlertTriangle className="w-6 h-6 mr-2" />
          {cameraError}
        </div>
      )}
      {liveFeedActive && (
      <div className="absolute top-3 left-3 bg-red-600 text-white text-xs px-3 py-1 rounded-md opacity-90">
        LIVE
      </div>
      )}
    </div>
  );

  const renderProgressBar = () => {
    const pct = Math.round(progress.overall_progress * 100);
    return (
      <div className="space-y-3">
        <div className="flex justify-between text-xs uppercase tracking-wider text-white/30">
          <span>{progress.current_item || 'Initialising'}</span>
          <span>{pct}%</span>
        </div>
        <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gold-muted"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ ease: 'linear' }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-white/20">
          <span>Frames: {progress.frames_processed}</span>
          <span>Samples: {progress.samples_collected}</span>
        </div>
      </div>
    );
  };

  // -----------------------------------------------------------------------
  // Step-specific content
  // -----------------------------------------------------------------------
  const renderEmotionalStep = () => (
    <div className="flex-1 w-full flex flex-col items-center justify-center p-8 space-y-10">
      <div className="luxury-glass p-8 rounded-3xl w-full max-w-3xl text-center">
        <h3 className="text-4xl font-serif italic text-white mb-4">
          Emotional Baseline Calibration
        </h3>
        <p className="text-white/40 font-light max-w-lg mx-auto">
          Watch each stimulus video naturally. The system will record your micro‑expressions to establish a personal emotional baseline.
        </p>
        {config?.stimulusVideos && (
          <p className="text-sm text-gold-muted/70 mt-4">
            {config.stimulusVideos.length} videos queued
          </p>
        )}
      </div>
      <div className="relative w-full max-w-lg mx-auto">
        {renderCameraFeed()}
        {showStimulusText && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-black/60 text-white rounded-xl px-6 py-4 text-2xl font-bold">{showStimulusText}</div>
          </div>
        )}
      </div>
      {renderProgressBar()}
      {wsError && (
        <div className="bg-red-900/30 border border-red-500/40 rounded-xl p-4 text-red-300 text-sm flex items-center gap-3 max-w-lg">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{wsError}</span>
        </div>
      )}
    </div>
  );

  const renderPhysiologicalStep = () => (
    <div className="flex-1 w-full flex flex-col items-center justify-center p-8 space-y-10">
      <div className="text-center space-y-4 max-w-lg">
        <Activity className="w-16 h-16 text-gold-muted/60 mx-auto animate-pulse" />
        <h3 className="text-4xl font-serif italic text-white">
          Physiological Baseline
        </h3>
        <p className="text-white/40 font-light">
          Analysing heart rate, HRV, and respiratory patterns from facial blood flow.
        </p>
      </div>
      {renderCameraFeed()}
      {renderProgressBar()}
      {/* Live metric display (real values from progress or store) */}
      <div className="grid grid-cols-2 gap-6 w-full max-w-sm">
        <MetricBox label="Resting BPM" value={`${telemetry?.heart_rate ?? '--'}`} />
        <MetricBox label="HR Variability" value={`${telemetry?.hrv ?? '--'} ms`} />
      </div>
    </div>
  );

  const renderSpeechStep = () => {
    const currentQuestion = progress.current_item || (config?.questions?.[0] ?? '');
    return (
      <div className="flex-1 w-full flex flex-col items-center justify-center p-8 space-y-10">
        <div className="luxury-glass p-12 rounded-3xl w-full max-w-3xl text-center space-y-8">
          <div className="flex items-center gap-3 text-gold-muted justify-center">
            <Mic className="w-5 h-5" />
            <span className="text-xs uppercase tracking-[0.3em] font-black">Vocal Baseline</span>
          </div>
          <p className="text-4xl font-serif text-white italic leading-snug">
            "{currentQuestion}"
          </p>
          <p className="text-white/30 text-sm">
            Please answer clearly. Your voice patterns will be analysed.
          </p>
        </div>
        {renderCameraFeed()}
        {renderProgressBar()}
        {/* Audio level indicator – could be added with Web Audio API */}
      </div>
    );
  };

  const renderCompletedStep = () => (
    <div className="flex-1 flex flex-col items-center justify-center text-center p-12 max-w-2xl mx-auto">
      <div className="w-28 h-28 rounded-full border border-gold-muted/30 flex items-center justify-center mb-8 relative">
        <div className="absolute inset-0 bg-gold-muted/5 blur-3xl rounded-full" />
        <CheckCircle2 className="w-10 h-10 text-gold-muted" />
      </div>
      <h3 className="text-5xl font-serif italic text-editorial-gradient mb-4">
        Calibration Complete
      </h3>
      <p className="text-lg text-white/30 mb-12 max-w-md">
        Your personal baseline has been stored. The system is now tuned to detect deviations specific to you.
      </p>
      <button
        onClick={() => setStage('live')}
        className="px-14 py-5 bg-white text-black font-black text-xs uppercase tracking-[0.4em] rounded-2xl hover:bg-gold-bright transition-all shadow-2xl"
      >
        Enter Analytical Space
      </button>
    </div>
  );

  // -----------------------------------------------------------------------
  // Main render
  // -----------------------------------------------------------------------
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white/50">
        <RefreshCw className="animate-spin mr-3" /> Initialising calibration...
      </div>
    );
  }

  return (
    <div className="max-w-[1400px] mx-auto pt-16 pb-24 px-12">
      {/* Floating controls: Abort + capture duration (seconds) */}
      <div className="absolute top-6 right-8 z-50 flex items-center gap-3">
        {(isEmotionCapturing || isPhysioCapturing) && (
          <button onClick={cancelCurrentCapture} className="px-3 py-2 bg-red-600 text-white text-xs rounded">Abort</button>
        )}
        <div className="flex items-center gap-2 bg-black/40 p-2 rounded">
          <label className="text-white/60 text-xs">E(s)</label>
          <input type="number" min="0.5" step="0.5" value={emotionCaptureMs/1000} onChange={e => setEmotionCaptureMs(Math.max(500, Number(e.target.value)*1000))} className="w-16 text-xs p-1 rounded bg-white/5 text-white" />
          <label className="text-white/60 text-xs">P(s)</label>
          <input type="number" min="1" step="0.5" value={physioCaptureMs/1000} onChange={e => setPhysioCaptureMs(Math.max(1000, Number(e.target.value)*1000))} className="w-16 text-xs p-1 rounded bg-white/5 text-white" />
        </div>
      </div>
      {/* Header with step indicators */}
      <div className="flex items-center justify-between mb-16 px-4">
        <div className="flex gap-12">
          {(['EMOTIONAL', 'PHYSIOLOGICAL', 'SPEECH'] as CalibrationStep[]).map((s, idx) => (
            <StepIndicator
              key={s}
              active={step === s}
              completed={stepCompletion[s]}
              index={idx + 1}
              label={s === 'EMOTIONAL' ? 'Visual Plane' : s === 'PHYSIOLOGICAL' ? 'Physiometrics' : 'Acoustics'}
            />
          ))}
        </div>
        <div className="text-right flex items-center gap-4">
          <div>
            <span className="text-[10px] font-black text-white/20 uppercase tracking-[0.3em] block mb-1">Status</span>
            <span className="text-xs font-mono text-gold-muted uppercase tracking-widest">{step === 'COMPLETED' ? 'Baseline Acquired' : 'In Progress'}</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={startDemo} className="px-3 py-2 bg-white/5 hover:bg-white/10 text-white text-xs rounded">Demo Mode</button>
            <button onClick={startCalibration} className="px-3 py-2 bg-white/5 hover:bg-white/10 text-white text-xs rounded">Interrogate</button>
          </div>
        </div>
      </div>

      {/* Main calibration container */}
      <div className="luxury-glass rounded-[3rem] overflow-hidden min-h-[600px] flex flex-col items-center">
        <AnimatePresence mode="wait">
          {step === 'EMOTIONAL' && (
            <motion.div
              key="emotional"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full flex-1 flex flex-col"
            >
              {renderEmotionalStep()}
              <CalibrationFooter
                progress={progress.overall_progress * 100}
                label="Facial Micro‑Expression Capture"
                onNext={nextStep}
                canNext={stepCompletion.EMOTIONAL}
              />
            </motion.div>
          )}

          {step === 'PHYSIOLOGICAL' && (
            <motion.div
              key="physiological"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full flex-1 flex flex-col"
            >
              {renderPhysiologicalStep()}
              <CalibrationFooter
                progress={progress.overall_progress * 100}
                label="Neuro‑Cardio Mapping"
                onNext={nextStep}
                canNext={stepCompletion.PHYSIOLOGICAL}
              />
            </motion.div>
          )}

          {step === 'SPEECH' && (
            <motion.div
              key="speech"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full flex-1 flex flex-col"
            >
              {renderSpeechStep()}
              <CalibrationFooter
                progress={progress.overall_progress * 100}
                label="Vocal Fingerprint Mapping"
                onNext={nextStep}
                canNext={stepCompletion.SPEECH}
              />
            </motion.div>
          )}

          {step === 'COMPLETED' && (
            <motion.div
              key="completed"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex-1 w-full flex items-center justify-center p-12"
            >
              {renderCompletedStep()}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer note */}
      <div className="mt-12 flex items-center gap-6 px-4">
        <Info className="w-5 h-5 text-white/20" />
        <p className="text-[10px] text-white/20 tracking-widest uppercase font-bold">
          Encrypted End‑to‑End Analysis Sequence — Case Ref: #DL-8829-X
        </p>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Sub-components (modified for real data)
// ---------------------------------------------------------------------------

const StepIndicator: React.FC<{
  active: boolean;
  completed: boolean;
  index: number;
  label: string;
}> = ({ active, completed, index, label }) => (
  <div className="flex items-center gap-6 relative">
    <div
      className={cn(
        'w-10 h-10 rounded-full flex items-center justify-center text-[10px] font-black border transition-all duration-700',
        active
          ? 'bg-gold-muted border-gold-muted text-black shadow-[0_0_15px_rgba(197,160,89,0.4)]'
          : completed
          ? 'bg-white/5 border-white/20 text-white'
          : 'bg-transparent border-white/5 text-white/20'
      )}
    >
      {completed ? <CheckCircle2 className="w-4 h-4 stroke-[3px]" /> : index}
    </div>
    <div className="flex flex-col">
      <span
        className={cn(
          'text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-700',
          active ? 'text-gold-muted' : completed ? 'text-white/60' : 'text-white/10'
        )}
      >
        {label}
      </span>
      {active && (
        <motion.div
          layoutId="step-label-active"
          className="text-[9px] text-white/40 uppercase tracking-widest mt-1"
        >
          Active Processing
        </motion.div>
      )}
    </div>
  </div>
);

const CalibrationFooter: React.FC<{
  progress: number;
  label: string;
  onNext: () => void;
  canNext: boolean;
}> = ({ progress, label, onNext, canNext }) => (
  <div className="flex flex-col gap-10 w-full max-w-2xl mx-auto px-12 pb-12">
    <div className="space-y-4">
      <div className="flex justify-between items-end">
        <span className="text-[10px] font-black text-white/30 uppercase tracking-[0.4em]">{label}</span>
        <span className="text-xs font-mono text-gold-muted tracking-widest">
          {Math.round(progress)}%
        </span>
      </div>
      <div className="h-[2px] w-full bg-white/5 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gold-muted shadow-[0_0_15px_rgba(197,160,89,0.6)]"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ ease: 'linear' }}
        />
      </div>
    </div>
    <div className="flex justify-center">
      <button
        disabled={!canNext}
        onClick={onNext}
        className={cn(
          'px-12 py-5 rounded-2xl text-[10px] font-black uppercase tracking-[0.4em] transition-all border',
          canNext
            ? 'bg-white text-black border-white hover:bg-gold-bright hover:border-gold-bright'
            : 'bg-transparent border-white/10 text-white/20 cursor-not-allowed'
        )}
      >
        {canNext ? 'Continue Protocol' : 'Processing…'}
      </button>
    </div>
  </div>
);

const MetricBox: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="p-8 rounded-[2rem] border border-white/5 bg-white/[0.02] text-center group hover:border-white/10 transition-colors">
    <span className="text-[10px] text-white/20 font-black uppercase tracking-[0.3em] block mb-3">
      {label}
    </span>
    <span className="text-4xl font-serif italic text-white tracking-widest">{value}</span>
  </div>
);