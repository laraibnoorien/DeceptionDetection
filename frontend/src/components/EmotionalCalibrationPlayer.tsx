import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Camera, Activity, Zap, Shield, Play, Pause, RefreshCcw } from 'lucide-react';
import { cn } from '../lib/utils';
import { useStore } from '../store/useStore';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000';

interface VideoSegment {
  id: string;
  label: string;
  message: string;
  color: string;
  duration: number; // in seconds
}

const CALIBRATION_SEGMENTS: VideoSegment[] = [
  {
    id: 'NEUTRAL',
    label: 'Neutral / Normal',
    message: 'Baseline neutral state: establish subject baseline under relaxed conditions.',
    color: 'from-slate-400 to-slate-600',
    duration: 8
  },
  { 
    id: 'DISGUST', 
    label: 'Disgust Baseline', 
    message: 'Analyzing Ocular and Nasal Micro-signatures: Monitoring corrugator supercilii and levator labii superioris.',
    color: 'from-emerald-600 to-teal-800',
    duration: 8
  },
  { 
    id: 'HAPPINESS', 
    label: 'Happiness Baseline', 
    message: 'Zygomatic Major Activation: Tracking involuntary orbicularis oculi contraction (Duchenne markers).',
    color: 'from-amber-400 to-orange-500',
    duration: 8
  },
  { 
    id: 'SURPRISE', 
    label: 'Surprise Baseline', 
    message: 'Rapid Frontalis Response: Establishing baseline for sudden eyebrow and upper lid elevation.',
    color: 'from-blue-400 to-cyan-600',
    duration: 8
  },
  { 
    id: 'SADNESS', 
    label: 'Sadness Baseline', 
    message: 'Inner Brow Contraction: Calibrating for subtle triangular patterns in the forehead and mouth-corner depressions.',
    color: 'from-indigo-500 to-violet-700',
    duration: 8
  },
  { 
    id: 'FEAR', 
    label: 'Fear Response', 
    message: 'Lateral Eyebrow Tension: Monitoring upper eyelid retraction and horizontal forehead furrowing.',
    color: 'from-purple-500 to-fuchsia-700',
    duration: 8
  },
  { 
    id: 'REPRESSION', 
    label: 'Repression Sync', 
    message: 'Involuntary Micro-leakage: Monitoring attempts to neutralize or mask genuine emotional artifacts.',
    color: 'from-slate-600 to-slate-800',
    duration: 8
  }
];

interface EmotionalCalibrationPlayerProps {
  onComplete: () => void;
}

export const EmotionalCalibrationPlayer: React.FC<EmotionalCalibrationPlayerProps> = ({ onComplete }) => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [completed, setCompleted] = useState<string[]>([]);
  const [calibrationId, setCalibrationId] = useState<string | null>(null);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const wsRef = React.useRef<WebSocket | null>(null);
  const captureTimerRef = React.useRef<number | null>(null);

  const segment = CALIBRATION_SEGMENTS[currentIdx];
  const { updateCalibration, updateHardware, pushTelemetry, telemetry } = useStore();

  // Initialize WebSocket connection lazily when user starts capture
  const ensureWebSocket = async (sessionId?: string, subjectId?: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return wsRef.current;
    try {
      if (!sessionId) return null;
      const qs = subjectId ? `?subject_id=${encodeURIComponent(subjectId)}` : '';
      const wsUrl = `${WS_BASE}/ws/calibration/${sessionId}${qs}`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => console.debug('Calibration WS connected');
      ws.onclose = () => console.debug('Calibration WS closed');
      ws.onerror = (e) => console.warn('WS error', e);
      ws.onmessage = (ev) => {
        try {
          const msg = typeof ev.data === 'string' ? JSON.parse(ev.data) : null;
          if (!msg) return;
          // Handle calibration progress and metrics
          if (msg.type === 'progress' && msg.payload) {
            const p = msg.payload;
            const overall = (typeof p.overall_progress === 'number') ? p.overall_progress : (p.stage_progress || 0);
            updateCalibration({
              emotional: Math.round((p.modality_confidence?.mm || 0) * 100),
              physiological: Math.round((p.modality_confidence?.pm || 0) * 100),
              speech: Math.round((p.modality_confidence?.sm || 0) * 100),
              currentStep: p.current_item || p.stage || 'Calibrating',
            });
            setProgress(Math.round(Math.max(0, Math.min(1, overall)) * 100));
          }
          if (msg.type === 'stage_complete') {
            const completedStage = msg.payload.stage;
            setCompleted(prev => {
              if (!prev.includes(completedStage)) return [...prev, completedStage];
              return prev;
            });
            if (msg.payload.baseline) {
              // Map baseline to calibration store fields
              const b = msg.payload.baseline;
              updateCalibration({
                emotional: Math.round((b.mm_confidence || b.mm_confidence?.value || 0) * 100),
                physiological: Math.round((b.pm_confidence || b.pm_confidence?.value || 0) * 100),
                speech: Math.round((b.sm_confidence || b.sm_confidence?.value || 0) * 100),
                currentStep: completedStage,
              });
            }
          }
          if (msg.type === 'error') {
            console.warn('Calibration error', msg.payload?.message || msg.payload);
          }
        } catch (e) {
          // ignore
        }
      };
      wsRef.current = ws;
      return ws;
    } catch (e) {
      console.warn('Could not open websocket', e);
      return null;
    }
  };

  const requestCalibrationStart = async (subjectId?: string): Promise<string | null> => {
    try {
      const sid = subjectId || telemetry?.speaker_id || telemetry?.subjectId || 'web-subject';
      const res = await fetch(`${API_BASE}/api/calibration/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ subject_id: sid }) });
      const j = await res.json();
      // backend returns { success: true, data: { id, ... } } or { session_id, ... }
      if (j) {
        if (j.session_id) return j.session_id;
        if (j.sessionId) return j.sessionId;
        if (j.data && (j.data.id || j.data.sessionId || j.data.session_id)) return j.data.id || j.data.sessionId || j.data.session_id;
      }
      return null;
    } catch (e) {
      console.warn('Failed to start calibration', e);
      return null;
    }
  };

  const startStreamAndSendFrames = async (calId: string) => {
    try {
      await ensureWebSocket();
      setCalibrationId(calId);

      // get camera
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // ensure websocket for this session
      await ensureWebSocket(calId, telemetry?.speaker_id || telemetry?.subjectId || 'web-subject');

      // create canvas
      if (!canvasRef.current) {
        const c = document.createElement('canvas');
        c.width = 320;
        c.height = 240;
        canvasRef.current = c;
      }

      // capture loop ~5 fps
      const sendFrame = () => {
        try {
          const v = videoRef.current;
          const c = canvasRef.current;
          if (!v || !c) return;
          const ctx = c.getContext('2d');
          if (!ctx) return;
          ctx.drawImage(v, 0, 0, c.width, c.height);
          c.toBlob((blob) => {
            if (!blob) return;
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              try { wsRef.current.send(blob); } catch (e) { /* ignore send errors */ }
            }
          }, 'image/jpeg', 0.6);
        } catch (e) {
          console.debug('frame send failed', e);
        }
      };

      captureTimerRef.current = window.setInterval(sendFrame, 200);
    } catch (e) {
      console.warn('Camera access failed', e);
    }
  };

  const stopStream = () => {
    try {
      if (captureTimerRef.current) {
        clearInterval(captureTimerRef.current);
        captureTimerRef.current = null;
      }
      if (videoRef.current) {
        const s = videoRef.current.srcObject as MediaStream | null;
        if (s) {
          s.getTracks().forEach(t => t.stop());
        }
        videoRef.current.srcObject = null;
      }
    } catch (e) {
      console.warn('Error stopping stream', e);
    }
  };


  useEffect(() => {
    if (progress >= 100) {
      setIsPlaying(false);
      setProgress(100);
      if (!completed.includes(segment.id)) {
        setCompleted(prev => [...prev, segment.id]);
      }
    }
  }, [progress, segment.id, completed]);

  // Cleanup on unmount: stop capture and close websocket
  useEffect(() => {
    return () => {
      try {
        stopStream();
        if (wsRef.current) {
          try { wsRef.current.close(); } catch (e) {}
          wsRef.current = null;
        }
      } catch (e) {}
    };
  }, []);

  const handleStart = async () => {
    // Start calibration session and begin streaming frames to backend
    if (!calibrationId) {
      const calId = await requestCalibrationStart();
      if (calId) {
        await startStreamAndSendFrames(calId);
        setIsPlaying(true);
      } else {
        console.warn('Could not obtain calibration ID from backend');
      }
    } else {
      // already have calibration id
      await startStreamAndSendFrames(calibrationId);
      setIsPlaying(true);
    }
  };
  const handleStop = () => {
    setIsPlaying(false);
    stopStream();
  };
  const handleReset = () => {
    setIsPlaying(false);
    setProgress(0);
    stopStream();
  };

  const handleFinish = () => {
    if (completed.length >= CALIBRATION_SEGMENTS.length) {
      onComplete();
    }
  };

  return (
    <div className="relative w-full h-full bg-slate-950 flex flex-col overflow-hidden">
      {/* Main Video Area */}
      <div className="flex-1 relative flex overflow-hidden">
        {/* Left Side: Emotions List */}
        <div className="w-80 border-r border-white/5 bg-black/40 backdrop-blur-xl p-8 flex flex-col gap-6 overflow-y-auto no-scrollbar">
           <div className="space-y-1">
             <h3 className="text-[10px] font-black text-white/30 uppercase tracking-[0.4em]">Calibration Nodes</h3>
             <p className="text-[10px] text-white/10 uppercase font-mono">CASME II Protocol v3.0</p>
           </div>

           <div className="space-y-3">
              {CALIBRATION_SEGMENTS.map((s, i) => (
                <button
                  key={s.id}
                  onClick={() => {
                    if (isPlaying) return;
                    setCurrentIdx(i);
                    setProgress(completed.includes(s.id) ? 100 : 0);
                  }}
                  className={cn(
                    "w-full p-4 rounded-2xl border transition-all duration-500 text-left group relative overflow-hidden",
                    currentIdx === i 
                      ? "bg-white/5 border-gold-muted/40" 
                      : "bg-transparent border-white/5 hover:border-white/20"
                  )}
                >
                   {currentIdx === i && (
                     <div className={cn("absolute inset-y-0 left-0 w-1 bg-gradient-to-b", s.color)} />
                   )}
                   <div className="flex items-center justify-between mb-1">
                     <span className={cn(
                       "text-[10px] font-black uppercase tracking-widest",
                       currentIdx === i ? "text-white" : "text-white/40"
                     )}>
                       {s.id}
                     </span>
                     {completed.includes(s.id) && (
                       <Zap className="w-3 h-3 text-gold-muted fill-gold-muted/20" />
                     )}
                   </div>
                   <span className="text-xs text-white/60 block truncate">{s.label}</span>
                </button>
              ))}
           </div>

           <div className="mt-auto pt-8">
              <button
                disabled={completed.length < CALIBRATION_SEGMENTS.length}
                onClick={handleFinish}
                className={cn(
                  "w-full py-4 rounded-full border text-[10px] font-black uppercase tracking-[0.4em] transition-all",
                  completed.length < CALIBRATION_SEGMENTS.length
                    ? "bg-transparent border-white/5 text-white/10 cursor-not-allowed"
                    : "bg-gold-muted text-black border-gold-muted hover:shadow-[0_0_20px_rgba(197,160,89,0.4)]"
                )}
              >
                Assemble Engine
              </button>
           </div>
        </div>

        {/* Right Side: Active Workspace */}
        <div className="flex-1 relative flex flex-col">
          <AnimatePresence mode="wait">
            <motion.div 
              key={segment.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={cn("absolute inset-0 bg-gradient-to-br opacity-10 blur-3xl", segment.color)}
            />
          </AnimatePresence>

          <div className="flex-1 flex flex-col items-center justify-center px-12 relative z-10 text-center">
             <motion.div
               key={segment.id}
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="space-y-8 max-w-2xl"
             >
                <div className="space-y-4">
                  <h2 className={cn(
                    "text-6xl font-black uppercase tracking-tighter bg-clip-text text-transparent bg-gradient-to-r",
                    segment.color
                  )}>
                    {segment.label}
                  </h2>
                  {segment.id === 'NEUTRAL' && (
                    <div className="inline-block px-3 py-1 rounded-full bg-white/5 text-xs font-bold text-white/60 uppercase tracking-widest">Baseline State</div>
                  )}
                  <p className="text-xl font-serif italic text-white/60 leading-relaxed">
                    "{segment.message}"
                  </p>
                </div>

                <div className="flex items-center justify-center gap-6">
                   {!isPlaying ? (
                     <button
                      onClick={handleStart}
                      className="px-8 py-4 bg-white text-black rounded-full text-[10px] font-black uppercase tracking-[0.4em] flex items-center gap-3 hover:scale-105 active:scale-95 transition-all shadow-2xl"
                     >
                       <Play className="w-4 h-4 fill-current" />
                       Initiate Capture
                     </button>
                   ) : (
                     <button
                      onClick={handleStop}
                      className="px-8 py-4 bg-white/5 border border-white/10 text-white rounded-full text-[10px] font-black uppercase tracking-[0.4em] flex items-center gap-3 hover:bg-white/10 transition-all"
                     >
                       <Pause className="w-4 h-4 fill-current" />
                       Halt Stream
                     </button>
                   )}
                   
                   <button
                    onClick={handleReset}
                    className="w-14 h-14 rounded-full border border-white/5 flex items-center justify-center hover:bg-white/5 text-white/40 hover:text-white transition-all"
                   >
                     <RefreshCcw className="w-5 h-5" />
                   </button>
                </div>
             </motion.div>

             {/* Progress HUD */}
             <div className="absolute top-12 right-12 text-right">
                <span className="text-[10px] font-black text-white/20 uppercase tracking-[0.4em] block mb-2">Quantization Progress</span>
                <span className="text-4xl font-mono text-white tracking-tighter">
                  {Math.round(progress)}%
                </span>
             </div>
          </div>

          {/* Camera HUD Overlays */}
          <div className="h-64 border-t border-white/5 bg-black/20 p-8 flex gap-8">
             <div className="w-80 h-full bg-slate-900 rounded-3xl border border-white/5 overflow-hidden relative">
                <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
                <div className="absolute inset-0 border-[30px] border-transparent">
                  <div className="w-full h-full border border-gold-muted/30 rounded-xl" />
                </div>
                <div className="absolute bottom-4 left-4 flex items-center gap-2 text-[8px] font-bold text-white/40 uppercase tracking-widest">
                  <div className={cn("w-1.5 h-1.5 rounded-full animate-pulse", isPlaying ? "bg-rose-500" : "bg-white/20")} />
                  Primary Optic Feed
                </div>
             </div>

             <div className="flex-1 grid grid-cols-2 gap-4">
                <HUDitem icon={<Camera className="w-4 h-4 text-gold-muted" />} label="Visual Plane" value={telemetry?.mm?.emotion || 'Neutral'} />
                <HUDitem icon={<Shield className="w-4 h-4 text-gold-muted" />} label="Integrity" value={telemetry?.fusion?.confidence ? `${Math.round(telemetry.fusion.confidence * 100)}%` : '—'} />
                <HUDitem icon={<Zap className="w-4 h-4 text-gold-muted" />} label="Engine Latency" value={telemetry?.engineLatencyMs ? `${telemetry.engineLatencyMs}ms` : '—'} />
                <HUDitem icon={<Activity className="w-4 h-4 text-gold-muted" />} label="Biometric Sync" value={typeof telemetry?.pm?.signalQuality === 'number' ? (telemetry.pm.signalQuality > 0.75 ? 'Locked' : 'Degraded') : '—'} />
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const HUDitem: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
  <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-slate-900/40 border border-white/5 backdrop-blur-md">
    <div className="p-1.5 bg-slate-950 rounded-lg border border-white/5">
      {icon}
    </div>
    <div>
      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest block leading-none mb-1">{label}</span>
      <span className="text-xs font-mono text-slate-300 leading-none">{value}</span>
    </div>
  </div>
);
