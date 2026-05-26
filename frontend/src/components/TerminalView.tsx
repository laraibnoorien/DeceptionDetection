import React from 'react';
import { useStore } from '../store/useStore';
import { StatusIndicator } from './StatusIndicator';
import { Camera, Mic, Watch, Server, Cpu, Zap, Play, Bluetooth, ChevronRight, CheckCircle2, Info } from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '../lib/utils';

export const TerminalView: React.FC = () => {
  const { hardware, setStage, updateHardware } = useStore();
  const [standbyActive, setStandbyActive] = React.useState(false);
  const standbyStreamRef = React.useRef<MediaStream | null>(null);
  const [availableCameras, setAvailableCameras] = React.useState<MediaDeviceInfo[]>([]);
  const [cameraDeviceId, setCameraDeviceId] = React.useState<string | null>(null);
  
  const isReady = hardware.camera === 'connected' && hardware.microphone === 'connected';

  React.useEffect(() => {
    const enumDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === 'videoinput');
        setAvailableCameras(cams);
        if (cams.length && !cameraDeviceId) setCameraDeviceId(cams[0].deviceId);
      } catch (e) { console.warn('enum devices failed', e); }
    };
    enumDevices();
  }, []);

  const startOpticsStandby = async () => {
    if (standbyActive) return;
    try {
      const constraints: any = { video: { width: 640, height: 480, facingMode: 'user' }, audio: false };
      if (cameraDeviceId) constraints.video = { deviceId: { exact: cameraDeviceId }, width: 640, height: 480 };
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      standbyStreamRef.current = s;
      updateHardware({ camera: 'connected' });
      setStandbyActive(true);
      console.log('[CAMERA_START] [STANDBY]');
    } catch (e) {
      console.warn('Optics standby failed', e);
      updateHardware({ camera: 'disconnected' });
    }
  };

  const stopOpticsStandby = () => {
    try {
      console.log('[CAMERA_STOP] [STANDBY]');
      const s = standbyStreamRef.current;
      if (s && s.getTracks) s.getTracks().forEach(t => t.stop());
    } catch (_) {}
    standbyStreamRef.current = null;
    setStandbyActive(false);
    updateHardware({ camera: 'disconnected' });
  };

  React.useEffect(() => {
    // cleanup on unmount
    return () => stopOpticsStandby();
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto pt-24 pb-32 px-12">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-24 items-start">
        <div className="lg:col-span-7 space-y-16">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h2 className="text-8xl font-serif italic text-editorial-gradient leading-[0.9] mb-8">
              The science of <br />
              <span className="not-italic text-gold-muted">Catching a Lie.</span>
            </h2>
            <p className="text-xl text-white/40 max-w-xl font-light leading-relaxed">
              Our neural-fusion engine processes multimodal micro-signatures across four behavioral planes to detect deception in realtime.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
            <StatusItem icon={<Camera />} label="Visual Plane" status={hardware.camera} sub="Facial AU Mapping" />
            <StatusItem icon={<Mic />} label="Acoustic Plane" status={hardware.microphone} sub="Vocal Stress Telemetry" />
            <StatusItem icon={<Watch />} label="Physiometric" status={hardware.wearable} sub="Autonomic Link" />
            <StatusItem icon={<Server />} label="Cognitive Link" status={hardware.server} sub="Fusion v4.2 Engine" />
          </div>

          <motion.div 
            whileHover={{ y: -5 }}
            className="p-10 luxury-glass rounded-[2rem] flex items-center justify-between group cursor-pointer"
          >
            <div className="flex items-center gap-8">
              <div className="w-20 h-20 rounded-full border border-white/10 flex items-center justify-center bg-black/40 group-hover:border-gold-muted/50 transition-colors">
                <Bluetooth className="w-8 h-8 text-gold-muted/60 group-hover:text-gold-muted" />
              </div>
              <div>
                <h4 className="text-white text-lg font-bold uppercase tracking-widest">Biometric Wearable</h4>
                <p className="text-white/40 text-sm mt-1">Initiate BLE handshake for physiological sync.</p>
              </div>
            </div>
            <div className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center group-hover:bg-gold-muted group-hover:text-black transition-all">
              <ChevronRight className="w-5 h-5" />
            </div>
          </motion.div>
        </div>

        <div className="lg:col-span-5 pt-12">
          <div className="luxury-glass rounded-[3rem] p-10 relative overflow-hidden backdrop-blur-2xl">
            <div className="absolute top-0 left-0 w-full h-1 bg-gold-muted/10 blur-md" />
            
            <div className="aspect-[4/5] bg-black/60 rounded-[2rem] border border-white/5 mb-10 flex flex-col items-center justify-center gap-4 relative group overflow-hidden">
               <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/80" />
               <Camera className="w-12 h-12 text-white/5 group-hover:text-gold-muted/20 transition-colors relative z-10" />
               <span className="text-[10px] text-white/20 uppercase tracking-[0.5em] font-bold relative z-10">Optics Standby</span>
               <div className="absolute bottom-4 left-4 right-4 flex justify-between z-20">
                 <div className="flex gap-3">
                   <select value={cameraDeviceId || ''} onChange={e=>setCameraDeviceId(e.target.value)} className="text-xs p-1 rounded bg-white/5 text-white">
                     {availableCameras.map(c => <option key={c.deviceId} value={c.deviceId}>{c.label || c.deviceId}</option>)}
                   </select>
                   {!standbyActive ? (
                     <button onClick={startOpticsStandby} className="px-3 py-2 bg-white/5 text-white text-xs rounded">Start Standby</button>
                   ) : (
                     <button onClick={stopOpticsStandby} className="px-3 py-2 bg-red-600 text-white text-xs rounded">Stop Standby</button>
                   )}
                 </div>
                 <div className="text-[10px] text-white/30">{standbyActive ? 'Standby Active' : 'Idle'}</div>
               </div>
            </div>

            <div className="space-y-6">
               <h4 className="text-[10px] font-bold text-gold-muted uppercase tracking-[0.3em] opacity-80">Pre-Session Protocol</h4>
               <CheckItem label="Focal Point Acquired" checked={hardware.camera === 'connected'} />
               <CheckItem label="Signal Encryption Set" checked={true} />
               <CheckItem label="Baseline Nodes Ready" checked={hardware.aiModels === 'connected'} />
            </div>

            <button 
              disabled={!isReady}
              onClick={() => setStage('calibration')}
              className={cn(
                "w-full mt-12 py-5 rounded-2xl font-black text-[10px] uppercase tracking-[0.4em] flex items-center justify-center gap-4 transition-all border",
                isReady 
                  ? "bg-white text-black border-white hover:bg-gold-bright hover:border-gold-bright shadow-[0_20px_40px_rgba(0,0,0,0.4)]" 
                  : "bg-transparent text-white/20 border-white/10 cursor-not-allowed"
              )}
            >
              <Zap className="w-4 h-4 fill-current" />
              Begin Induction
            </button>
          </div>
          
          <div className="mt-8 px-10 flex gap-4 text-white/30 italic">
             <Info className="w-5 h-5 shrink-0 opacity-50" />
             <p className="text-[10px] leading-loose tracking-wide">
               LIAR LIAR, PANTS ON FIRE!
             </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatusItem: React.FC<{ icon: React.ReactNode; label: string; sub: string; status: any }> = ({ icon, label, sub, status }) => (
  <div className="p-8 rounded-[2rem] luxury-glass flex gap-6 items-start group hover:border-white/20 transition-all">
    <div className={cn(
      "w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 border border-white/5 bg-black/40",
      status === 'connected' ? "text-gold-muted" : "text-white/10"
    )}>
      {React.cloneElement(icon as React.ReactElement, { className: 'w-5 h-5' })}
    </div>
    <div className="flex-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-black text-white uppercase tracking-widest">{label}</span>
        <div className={cn(
            "w-1.5 h-1.5 rounded-full",
            status === 'connected' ? "bg-gold-muted shadow-[0_0_10px_rgba(197,160,89,0.8)]" : "bg-white/10"
        )} />
      </div>
      <p className="text-[10px] text-white/30 uppercase tracking-widest mt-1.5">{sub}</p>
    </div>
  </div>
);

const CheckItem: React.FC<{ label: string; checked: boolean }> = ({ label, checked }) => (
  <div className="flex items-center justify-between py-2 border-b border-white/[0.03]">
    <span className="text-[10px] text-white/40 font-bold uppercase tracking-widest">{label}</span>
    <div className={cn(
      "w-5 h-5 rounded-full border flex items-center justify-center transition-all",
      checked ? "bg-gold-muted border-gold-muted text-black" : "bg-transparent border-white/10 text-transparent"
    )}>
      {checked && <CheckCircle2 className="w-3.5 h-3.5 stroke-[3px]" />}
    </div>
  </div>
);
