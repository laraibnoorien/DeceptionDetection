import React, { useState, useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000';
import { FusionGauge } from './FusionGauge';
import { ModalityCard, MetricRow } from './ModalityCard';
import { TelemetryChart } from './TelemetryChart';
import { 
  Camera, 
  Activity, 
  Mic, 
  History,
  BrainCircuit
} from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '../lib/utils';
import { TelemetryEvent, AlertLevel } from '../types';

export const DashboardView: React.FC = () => {
  const { history, pushTelemetry } = useStore();
  
  // Realtime simulation or Live interrogation
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const primaryStreamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const latestModalitiesRef = useRef<any>({});
  const reconnectAttemptsRef = useRef<number>(0);

  const [demoActive, setDemoActive] = useState<boolean>(true);
  const [interrogateActive, setInterrogateActive] = useState<boolean>(false);
  const [primaryFeedError, setPrimaryFeedError] = useState<string | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);

  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraDeviceId, setCameraDeviceId] = useState<string | null>(null);
  const [autoReconnect, setAutoReconnect] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    const enumerate = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (!mounted) return;
        const cams = devices.filter(d => d.kind === 'videoinput');
        setAvailableCameras(cams);
        if (cams.length && !cameraDeviceId) setCameraDeviceId(cams[0].deviceId);
      } catch (e) {
        console.warn('enumerateDevices failed', e);
      }
    };
    enumerate();

    const startPrimaryFeed = async (attempts = 3) => {
      for (let i = 0; i < attempts; i++) {
        try {
          const constraints: any = { video: { width: 1280, height: 720, facingMode: 'user' }, audio: false };
          if (cameraDeviceId) constraints.video = { deviceId: { exact: cameraDeviceId }, width: 1280, height: 720 };
          const s = await navigator.mediaDevices.getUserMedia(constraints);
          if (!mounted) { s.getTracks().forEach(t=>t.stop()); return; }
          primaryStreamRef.current = s;
          if (videoRef.current) videoRef.current.srcObject = s;
          setPrimaryFeedError(null);
          console.log('[CAMERA_START] [PRIMARY_FEED]');
          return;
        } catch (e) {
          console.warn('Primary optic feed attempt failed', e);
          setPrimaryFeedError('Primary optic feed unavailable. Retry later.');
          // exponential backoff
          await new Promise(r => setTimeout(r, 500 * Math.pow(2, i)));
        }
      }
    };
    // Attempt to start primary optic feed once on mount (with retries)
    startPrimaryFeed();

    let interval: any = null;
    if (demoActive) {
      interval = setInterval(() => {
        const prob = 0.1 + Math.random() * 0.4;
        const drift = Math.sin(Date.now() / 5000) * 0.2;
        const finalProb = Math.max(0, Math.min(1, prob + drift));
        
        let level: AlertLevel = 'GREEN';
        if (finalProb > 0.7) level = 'RED';
        else if (finalProb > 0.5) level = 'ORANGE';
        else if (finalProb > 0.3) level = 'YELLOW';

        const event: TelemetryEvent = {
          timestamp: new Date().toISOString(),
          mm: {
            emotion: finalProb > 0.6 ? 'Anxiety' : 'Neutral',
            auIntensities: { 'AU1': 0.1 + drift, 'AU4': 0.2 + drift },
            gazeStability: 0.95 - (drift * 0.5),
            blinkRate: 15 + Math.round(drift * 10),
            score: finalProb * 0.8,
            anomalies: finalProb > 0.7 ? ['Micro-stutter detected'] : []
          },
          pm: {
            bpm: 72 + Math.round(drift * 20),
            hrv: 45 - Math.round(drift * 10),
            spo2: 98,
            stressIndex: finalProb,
            anomalyScore: drift > 0.15 ? 0.8 : 0.1,
            signalQuality: 1.0
          },
          sm: {
            pitch: 110 + (drift * 10),
            jitter: 0.01 + (drift * 0.05),
            shimmer: 0.02 + (drift * 0.05),
            vocalStress: finalProb * 0.9,
            score: finalProb * 0.7,
            transcript: 'Subject describing location of events.'
          },
          tm: {
            transcript: 'I was not at the location during that hour.',
            contradictionScore: finalProb > 0.6 ? 0.75 : 0.1,
            certaintyScore: 1 - finalProb,
            inconsistencies: finalProb > 0.6 ? ['Temporal conflict', 'Negative phrasing'] : []
          },
          fusion: {
            deceptionProbability: finalProb,
            confidence: 0.85 + (Math.random() * 0.1),
            dominantModality: drift > 0.1 ? 'PM' : 'MM',
            alertLevel: level
          },
          explainability: {
            drivers: level === 'RED' 
              ? ['Acute physiological spike', 'Cognitive load detected in speech'] 
              : ['Consistent baseline patterns'],
            summary: level === 'RED'
              ? 'High probability of deception. Autonomic arousal coupled with linguistic deflection.'
              : 'Subject behavior is consistent with calibrated baseline.'
          }
        };
        pushTelemetry(event);
      }, 1000);
    }

    return () => {
      mounted = false;
      try { if (interval) clearInterval(interval); } catch(_) {}
    };
  }, [pushTelemetry, demoActive]);
  
  const defaultLatest: TelemetryEvent = {
    timestamp: new Date().toISOString(),
    mm: { emotion: 'Neutral', auIntensities: {}, gazeStability: 1.0, blinkRate: 0, score: 0, anomalies: [] },
    pm: { bpm: 0, hrv: 0, spo2: 0, stressIndex: 0, anomalyScore: 0, signalQuality: 0 },
    sm: { pitch: 0, jitter: 0, shimmer: 0, vocalStress: 0, score: 0, transcript: '' },
    tm: { transcript: '', contradictionScore: 0, certaintyScore: 1.0, inconsistencies: [] },
    fusion: { deceptionProbability: 0, confidence: 0, dominantModality: 'MM', alertLevel: 'GREEN' },
    explainability: { drivers: [], summary: '' }
  };
  const latest = history[0] || defaultLatest;
  
  useEffect(() => {
    return () => {
      try { if (primaryStreamRef.current) primaryStreamRef.current.getTracks().forEach(t=>t.stop()); } catch(_) {}
      try { if (wsRef.current) wsRef.current.close(); } catch(_) {}
    };
  }, []);

  const connectLiveWebSocket = (sessionId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    reconnectAttemptsRef.current = 0;
    const openWs = () => {
      const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS_CONNECTED] [LIVE]');
        setWsError(null);
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'metrics' || data.type === 'progress' || data.type === 'fusion') {
            const metrics = data.payload || data;
            const mm = latestModalitiesRef.current.mm || {};
            const pm = latestModalitiesRef.current.pm || {};
            const sm = latestModalitiesRef.current.sm || {};
            const tm = latestModalitiesRef.current.tm || {};
            const fusion = metrics.fusion || metrics || {};
            const event: TelemetryEvent = {
              timestamp: new Date().toISOString(),
              mm: { emotion: mm.emotion || mm.emotionLabel || 'Neutral', auIntensities: mm.auIntensities || {}, gazeStability: mm.gazeStability || 0.9, blinkRate: mm.blinkRate || 12, score: mm.microexpressionScore || (fusion?.modalities?.mm || fusion?.modalities?.mm || metrics?.modalities?.mm || 0), anomalies: mm.anomalies || [] },
              pm: { bpm: pm.bpm || pm.heart_rate || 70, hrv: pm.hrv || 40, spo2: pm.spo2 || 98, stressIndex: pm.stressIndex || (fusion?.modalities?.pm || metrics?.modalities?.pm || 0), anomalyScore: pm.anomalyScore || 0, signalQuality: pm.signalQuality || 1.0 },
              sm: { pitch: sm.pitch || sm.meanPitch || 120, jitter: sm.jitter || 0.01, shimmer: sm.shimmer || 0.02, vocalStress: sm.vocalStress || (fusion?.modalities?.sm || metrics?.modalities?.sm || 0), score: sm.score || 0.0, transcript: sm.transcript || '' },
              tm: { transcript: tm.transcript || '', contradictionScore: tm.contradiction || tm.contradictionScore || (fusion?.modalities?.tm || metrics?.modalities?.tm || 0), certaintyScore: tm.certainty || 1.0, inconsistencies: tm.inconsistencies || [] },
              fusion: { deceptionProbability: fusion.deceptionProbability || metrics.deceptionProbability || metrics.probability || 0, confidence: fusion.confidence || metrics.confidence || 0.9, dominantModality: fusion.dominantModality || metrics.dominantModality || 'MM', alertLevel: fusion.alertLevel || (metrics.riskLevel || 'GREEN') },
              explainability: { drivers: data.explainability?.drivers || data.payload?.explainability?.drivers || [], summary: data.explainability?.summary || data.payload?.explainability?.summary || '' }
            };
            pushTelemetry(event);
          } else if (data.type && data.type.endsWith('_update')) {
            const mod = data.type.split('_')[0];
            latestModalitiesRef.current[mod] = data.payload;
          }
        } catch (e) {
          console.warn('Live WS message parse error', e);
        }
      };

        ws.onclose = () => {
        console.log('[WS_CLOSED] [LIVE]');
        if (interrogateActive && autoReconnect && reconnectAttemptsRef.current < 10) {
          const backoff = 500 * Math.pow(2, reconnectAttemptsRef.current);
          reconnectAttemptsRef.current += 1;
          console.log('[WS_CLOSED] attempting reconnect in', backoff, 'ms');
          setTimeout(() => openWs(), backoff);
          return;
        }
        setInterrogateActive(false);
        setDemoActive(true);
        setWsError('Live connection closed');
      };

      ws.onerror = (e) => {
        console.warn('Live WS error', e);
        setWsError('WebSocket error');
      };
    };
    openWs();
  };

  const startInterrogate = async () => {
    if (interrogateActive) return;
    setDemoActive(false);
    setWsError(null);
    try {
      const res = await fetch(`${API_BASE}/api/sessions/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ subjectId: 'live-subject' }) });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Failed to start live session: ${res.status} ${txt}`);
      }
      const data = await res.json();
      // robust session id parsing
      const sessionId = data?.data?.id || data?.data?.sessionId || data?.sessionId || data?.id || data?.session_id || data?.session || data?.sid;
      if (!sessionId) {
        throw new Error('No session id returned from backend');
      }
      sessionIdRef.current = sessionId;
      setInterrogateActive(true);
      connectLiveWebSocket(sessionId);
    } catch (e: any) {
      console.warn('Interrogate start failed', e);
      setWsError(e?.message || 'Interrogate start failed');
      setDemoActive(true);
    }
  };

  const stopInterrogate = async () => {
    try {
      if (wsRef.current) wsRef.current.close();
      if (sessionIdRef.current) {
        await fetch(`${API_BASE}/api/sessions/${sessionIdRef.current}/end`, { method: 'POST' });
      }
    } catch (_) {}
    setInterrogateActive(false);
    setDemoActive(true);
  };

  return (
    <div className="h-[calc(100vh-80px)] overflow-hidden flex flex-col px-12 pb-8">
      <div className="flex-1 flex overflow-hidden gap-12 pt-8">
        {/* Left Panel: Sensory Feed */}
        <div className="w-[480px] flex flex-col gap-10 overflow-y-auto no-scrollbar pb-12">
          <div className="space-y-6">
             <div className="flex items-center justify-between px-2">
                <div className="flex items-center gap-3">
                  <Camera className="w-4 h-4 text-gold-muted" />
                  <h4 className="text-[10px] font-black text-white uppercase tracking-[0.4em]">Primary Optical Feed</h4>
                </div>
                <span className="text-[10px] font-mono text-white/20 uppercase">RAW-4K / 60 FPS</span>
             </div>
             
             <div className="aspect-[3/4] luxury-glass rounded-[3rem] relative overflow-hidden group">
                {/* Primary video element for live optic feed */}
                <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover rounded-[3rem]" />
                {/* Primary feed error / retry control */}
                {primaryFeedError && (
                  <div className="absolute top-4 right-4 z-30 flex items-center gap-2">
                    <div className="text-xs text-red-400 bg-black/60 px-3 py-1 rounded">{primaryFeedError}</div>
                    <button onClick={() => { setPrimaryFeedError(null); (async ()=>{try{const s=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,facingMode:'user'},audio:false}); primaryStreamRef.current = s; if (videoRef.current) videoRef.current.srcObject = s; setPrimaryFeedError(null); console.log('[CAMERA_START] [PRIMARY_FEED RETRY]');}catch(e){setPrimaryFeedError('Retry failed');}})(); }} className="px-2 py-1 bg-white/5 text-white text-xs rounded">Retry</button>
                  </div>
                )}
                {/* Fallback Facial Overlay Simulation when no video available */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-56 h-80 border border-gold-muted/20 rounded-[4rem] relative opacity-60">
                    <div className="absolute top-1/4 left-1/4 w-6 h-6 border-t border-l border-gold-muted/40" />
                    <div className="absolute top-1/4 right-1/4 w-6 h-6 border-t border-r border-gold-muted/40" />
                    <div className="absolute bottom-1/4 left-1/4 w-6 h-6 border-b border-l border-gold-muted/40" />
                    <div className="absolute bottom-1/4 right-1/4 w-6 h-6 border-b border-r border-gold-muted/40" />
                  </div>
                </div>
                 
                <div className="absolute top-6 left-6 flex gap-3">
                   <div className="px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-full border border-white/5 text-[9px] text-gold-muted font-black uppercase tracking-[0.2em] shadow-xl">Subject Locked</div>
                   <div className="px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-full border border-white/5 text-[9px] text-white/60 font-black uppercase tracking-[0.2em] shadow-xl">AU Link Active</div>
                </div>

                <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/80 via-transparent to-transparent" />
                <div className="absolute bottom-6 left-6 right-6 flex items-center justify-between">
                    <div className="space-y-2">
                      <span className="text-[10px] text-white/40 font-black uppercase tracking-[0.3em] block">Saccadic Drift</span>
                      <div className="w-40 h-[2px] bg-white/10 rounded-full overflow-hidden">
                        <motion.div 
                          animate={{ width: "75%" }}
                          transition={{ duration: 10, repeat: Infinity, repeatType: 'reverse' }}
                          className="h-full bg-gold-muted" 
                        />
                      </div>
                    </div>
                    <div className="text-right">
                       <span className="text-[10px] text-gold-muted font-black uppercase block tracking-[0.2em]">Signal 10/10</span>
                    </div>
                </div>
             </div>
          </div>

          <div className="space-y-6">
            <h4 className="text-[10px] font-black text-white/40 uppercase tracking-[0.4em] border-b border-white/5 pb-4 px-2">Cognitive Inference</h4>
            <div className="p-8 luxury-glass rounded-[2.5rem] relative overflow-hidden group">
               <div className="absolute top-0 right-0 p-4">
                  <BrainCircuit className="w-5 h-5 text-gold-muted/20" />
               </div>
               <p className="text-xl font-serif italic text-white leading-relaxed">
                 "{latest.explainability.summary}"
               </p>
               <div className="flex flex-wrap gap-2 mt-8">
                 {latest.explainability.drivers.map((d, i) => (
                   <span key={i} className="text-[9px] font-black px-3 py-1.5 bg-white/5 text-white/40 border border-white/5 rounded-full uppercase tracking-widest group-hover:border-gold-muted/20 transition-colors">
                     {d}
                   </span>
                 ))}
               </div>
            </div>
          </div>

          <div className="flex-1 luxury-glass rounded-[2.5rem] p-8 flex flex-col relative overflow-hidden min-h-[300px]">
             <div className="flex items-center gap-3 mb-8">
               <History className="w-4 h-4 text-white/20" />
               <h4 className="text-[10px] font-black text-white/30 uppercase tracking-[0.4em]">Artifact Timeline</h4>
             </div>
             <div className="flex-1 space-y-4 overflow-y-auto no-scrollbar">
                <TimelineEvent time="11:42:04" label="Baseline match confirmed" type="OK" />
                <TimelineEvent time="11:42:15" label="Physiological drift detected" type="WARN" />
                {latest.fusion.alertLevel === 'RED' && <TimelineEvent time="11:43:01" label="High-Risk microexpression cluster" type="ALERT" />}
             </div>
          </div>
        </div>

        {/* Center Panel: Fusion Matrix */}
        <div className="flex-1 flex flex-col gap-12 overflow-y-auto no-scrollbar pb-12">
          <div className="flex flex-col items-center pt-8">
            <div className="flex items-center gap-4 mb-6">
              <button onClick={() => { setDemoActive(true); setInterrogateActive(false); }} className="px-3 py-2 bg-white/5 text-white text-xs rounded">Demo</button>
              {!interrogateActive ? (
                <button onClick={startInterrogate} className="px-3 py-2 bg-white/5 text-white text-xs rounded">Interrogate</button>
              ) : (
                <button onClick={stopInterrogate} className="px-3 py-2 bg-red-600 text-white text-xs rounded">Stop Interrogate</button>
              )}
              <div className="text-[10px] text-white/40 ml-4">Mode: {interrogateActive ? 'Interrogate' : demoActive ? 'Demo' : 'Idle'}</div>
            </div>
            <div className="flex items-center gap-3 mb-4">
              <label className="text-[10px] text-white/40">Camera:</label>
              <select value={cameraDeviceId || ''} onChange={e => setCameraDeviceId(e.target.value)} className="text-xs p-1 rounded bg-white/5 text-white">
                <option value="">Default</option>
                {availableCameras.map((c) => (
                  <option key={c.deviceId} value={c.deviceId}>{c.label || c.deviceId}</option>
                ))}
              </select>
              <label className="text-[10px] text-white/40 ml-4">Auto-reconnect:</label>
              <input type="checkbox" checked={autoReconnect} onChange={e => setAutoReconnect(e.target.checked)} />
            </div>
            {wsError && (
              <div className="text-red-400 text-sm mb-4 flex items-center gap-3">
                <div>{wsError}</div>
                <button onClick={() => { setWsError(null); if (sessionIdRef.current) connectLiveWebSocket(sessionIdRef.current); else startInterrogate(); }} className="px-2 py-1 bg-white/5 text-white text-xs rounded">Retry</button>
              </div>
            )}
            <FusionGauge 
               probability={latest.fusion.deceptionProbability} 
               confidence={latest.fusion.confidence} 
               level={latest.fusion.alertLevel}
               className="w-full max-w-xl"
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            <ModalityCard 
              id="V-SYNC" 
              title="Micro-Expressions" 
              subtitle="Facial Action Unit Analysis"
              score={latest.mm.score}
              icon={<Camera className="w-4 h-4" />}
            >
              <MetricRow label="Dominant State" value={latest.mm.emotion} alert={latest.mm.emotion !== 'Neutral'} />
              <MetricRow label="Eye Blink Delta" value={`${latest.mm.blinkRate} bpm`} />
              <MetricRow label="Saccade Entropy" value={`${(latest.mm.gazeStability*100).toFixed(1)}%`} />
              <div className="pt-6 h-20">
                 <TelemetryChart data={history.slice().reverse()} dataKey="mm.score" color="#c5a059" domain={[0, 1]} />
              </div>
            </ModalityCard>

            <ModalityCard 
              id="P-LINK" 
              title="Physio-Signature" 
              subtitle="Autonomic Nervous Response"
              score={latest.pm.anomalyScore}
              icon={<Activity className="w-4 h-4" />}
            >
              <MetricRow label="Pulse Resonance" value={`${latest.pm.bpm} BPM`} />
              <MetricRow label="HR Variability" value={`${latest.pm.hrv} ms`} />
              <MetricRow label="Stress Index" value={`${(latest.pm.stressIndex*100).toFixed(0)}%`} alert={latest.pm.stressIndex > 0.6} />
              <div className="pt-6 h-20">
                 <TelemetryChart data={history.slice().reverse()} dataKey="pm.bpm" color="#c5a059" domain={[60, 120]} />
              </div>
            </ModalityCard>

            <ModalityCard 
              id="A-FEED" 
              title="Acoustic Jitter" 
              subtitle="Prosodic Stress Delta"
              score={latest.sm.score}
              icon={<Mic className="w-4 h-4" />}
            >
              <MetricRow label="Vocal Tension" value={`${(latest.sm.vocalStress*100).toFixed(0)}%`} />
              <MetricRow label="Jitter Anomaly" value={`${latest.sm.jitter.toFixed(3)}`} />
              <MetricRow label="Pitch Drift" value={`${latest.sm.pitch.toFixed(1)} Hz`} />
              <div className="pt-6 h-20">
                 <TelemetryChart data={history.slice().reverse()} dataKey="sm.vocalStress" color="#c5a059" domain={[0, 1]} />
              </div>
            </ModalityCard>

            <ModalityCard 
              id="T-CORE" 
              title="Cognitive Load" 
              subtitle="Verbal Logic Coherence"
              score={latest.tm.contradictionScore}
              icon={<BrainCircuit className="w-4 h-4" />}
            >
              <MetricRow label="Logic Entropy" value={`${(latest.tm.certaintyScore*100).toFixed(0)}%`} alert={latest.tm.certaintyScore < 0.4} />
              <MetricRow label="Conflict Nodes" value={latest.tm.inconsistencies.length} />
              <div className="mt-4 text-[10px] bg-black/40 p-4 rounded-2xl border border-white/5 text-white/40 font-light italic leading-relaxed">
                "{latest.tm.transcript}"
              </div>
            </ModalityCard>
          </div>
        </div>
      </div>

      {/* Luxury Timeline Bar */}
      <div className="h-24 luxury-glass rounded-[2rem] mt-8 flex items-center px-12 gap-8 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-[1px] bg-white/10" />
        <div className="flex flex-col shrink-0">
          <span className="text-[10px] text-white/20 uppercase font-black tracking-[0.4em] mb-1">Temporal Sequence</span>
          <span className="text-[10px] font-mono text-gold-muted uppercase tracking-widest leading-none">LIVE-LINK</span>
        </div>
        
        <div className="flex-1 flex items-center justify-center gap-1 overflow-x-auto no-scrollbar h-12 px-8">
           {history.slice(0, 120).map((h, i) => (
             <div 
              key={i} 
              className={cn(
                "w-[2px] shrink-0 transition-all cursor-crosshair rounded-full",
                h.fusion.alertLevel === 'RED' ? "h-10 bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.3)]" : 
                h.fusion.alertLevel === 'ORANGE' ? "h-6 bg-gold-muted shadow-[0_0_8px_rgba(197,160,89,0.3)]" :
                h.fusion.alertLevel === 'YELLOW' ? "h-4 bg-gold-muted/40" : "h-2 bg-white/5"
              )} 
             />
           ))}
        </div>

        <div className="flex gap-12 shrink-0 border-l border-white/10 pl-12">
           <div className="flex flex-col">
             <span className="text-[10px] text-white/20 uppercase font-black tracking-[0.3em] mb-1">Confidence</span>
             <span className="text-sm font-serif italic text-white uppercase leading-none">{(latest.fusion.confidence * 100).toFixed(0)}%</span>
           </div>
           <div className="flex flex-col">
             <span className="text-[10px] text-white/20 uppercase font-black tracking-[0.3em] mb-1">Inference Engine</span>
             <span className="text-sm font-serif italic text-white uppercase leading-none">v4.2-STB</span>
           </div>
        </div>
      </div>
    </div>
  );
};

const TimelineEvent: React.FC<{ time: string; label: string; type: 'OK' | 'WARN' | 'ALERT' }> = ({ time, label, type }) => (
  <div className="flex items-start gap-4 py-2 group">
    <span className="text-[10px] font-mono text-white/20 mt-0.5 group-hover:text-gold-muted transition-colors">{time}</span>
    <div className="flex flex-col">
      <span className={cn(
        "text-[10px] font-black uppercase tracking-widest",
        type === 'ALERT' ? "text-rose-500" : type === 'WARN' ? "text-gold-muted" : "text-white/40"
      )}>
        {label}
      </span>
    </div>
  </div>
);
