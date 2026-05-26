import React from 'react';
import { useStore } from '../store/useStore';
import { 
  FileText, 
  Download, 
  Share2, 
  TrendingUp, 
  BarChart, 
  Clock, 
  Hash,
  ArrowRight,
  ShieldCheck
} from 'lucide-react';
import { TelemetryChart } from './TelemetryChart';
import { cn } from '../lib/utils';

export const ReportView: React.FC = () => {
  const { history, resetSession } = useStore();
  
  if (history.length === 0) return (
     <div className="flex flex-col items-center justify-center p-20 text-center space-y-4">
        <FileText className="w-16 h-16 text-slate-800" />
        <h3 className="text-xl font-bold text-slate-500 uppercase tracking-widest">No Case Data Found</h3>
        <p className="text-slate-600 text-sm">Please finalize a live monitoring session to generate a report.</p>
     </div>
  );

  const averageDeception = history.reduce((acc, h) => acc + h.fusion.deceptionProbability, 0) / history.length;

  return (
    <div className="max-w-6xl mx-auto py-12 px-8 space-y-8 pb-32">
      <div className="flex items-start justify-between border-b border-slate-800 pb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2 py-0.5 bg-red-600 text-[10px] font-bold text-white rounded uppercase tracking-widest">Confidential</span>
            <span className="text-[10px] font-mono text-slate-500 uppercase">Case Reference: #DL-2026-0519-A</span>
          </div>
          <h2 className="text-4xl font-black text-white tracking-tighter uppercase mb-2">Post-Action Analysis Report</h2>
          <p className="text-slate-500 max-w-xl">Comprehensive behavioral and physiological breakdown of session biometrics for investigative review.</p>
        </div>
        <div className="flex gap-4">
          <ReportAction icon={<Download />} label="Download PDF" />
          <ReportAction icon={<Share2 />} label="Export JSON" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <StatCard label="Session Duration" value="12m 42s" icon={<Clock />} />
        <StatCard label="Mean Deception Prob" value={`${(averageDeception * 100).toFixed(1)}%`} icon={<TrendingUp />} />
        <StatCard label="Model Confidence" value="94.2%" icon={<ShieldCheck />} />
        <StatCard label="Anomaly Count" value="14" icon={<Hash />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
           <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8">
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-lg font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-red-500" />
                  Temporal Deception Trend
                </h3>
                <span className="text-xs font-mono text-slate-500 italic">Full Session Playback</span>
              </div>
              <div className="h-[300px] w-full">
                 <TelemetryChart data={history.slice().reverse()} dataKey="fusion.deceptionProbability" color="#ef4444" height={300} />
              </div>
           </div>

           <div className="grid grid-cols-2 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                 <h4 className="text-xs font-bold text-slate-400 uppercase mb-4">Modality Contributions</h4>
                 <div className="space-y-4">
                    <ContributionRow label="Physiological (PM)" value={45} color="bg-emerald-500" />
                    <ContributionRow label="Microexpressions (MM)" value={30} color="bg-blue-500" />
                    <ContributionRow label="Speech (SM)" value={15} color="bg-amber-500" />
                    <ContributionRow label="Linguistic (TM)" value={10} color="bg-purple-500" />
                 </div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
                 <h4 className="text-xs font-bold text-slate-400 uppercase mb-4">Final Verdict</h4>
                 <div className="flex-1 flex flex-col items-center justify-center text-center py-4">
                   <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Inference Engine Result</span>
                   <span className={cn(
                     "text-4xl font-black uppercase tracking-tighter",
                     averageDeception > 0.6 ? "text-red-500" : "text-emerald-500"
                   )}>
                     {averageDeception > 0.6 ? "High Risk" : "Normal"}
                   </span>
                 </div>
                 <button className="w-full py-3 bg-slate-800 text-white text-[10px] font-bold uppercase tracking-widest rounded-lg flex items-center justify-center gap-2">
                   View Full Timeline
                   <ArrowRight className="w-3 h-3" />
                 </button>
              </div>
           </div>
        </div>

        <div className="space-y-6">
           <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 overflow-hidden relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-red-600/5 blur-3xl rounded-full" />
              <h4 className="text-xs font-bold text-white uppercase tracking-widest mb-6">Investigative Summary</h4>
              <div className="space-y-6">
                 <SummaryItem 
                   title="Autonomic Arousal" 
                   desc="Significant BPM shift and HRV compression noted during questioned phases."
                 />
                 <SummaryItem 
                   title="Vocal Inconsistency" 
                   desc="Pitch jitter increased by 22% when discussing temporal location."
                 />
                 <SummaryItem 
                   title="Cognitive Load" 
                   desc="Response latency increased during session midpoint. Sentence structure became significantly more complex."
                 />
              </div>
           </div>

           <button 
             onClick={resetSession}
             className="w-full py-4 border border-slate-800 rounded-xl text-slate-500 hover:text-slate-300 hover:bg-slate-900 transition-all font-bold uppercase text-[10px] tracking-[0.3em]"
           >
             Close Case File & Wipe Data
           </button>
        </div>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ label: string; value: string; icon: React.ReactNode }> = ({ label, value, icon }) => (
  <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg relative group transition-all hover:border-slate-700">
     <div className="text-slate-500 mb-4 group-hover:text-slate-400 transition-colors">{icon}</div>
     <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">{label}</span>
     <span className="text-2xl font-black text-white tracking-tight">{value}</span>
  </div>
);

const ReportAction: React.FC<{ icon: React.ReactNode; label: string }> = ({ icon, label }) => (
  <button className="flex items-center gap-2 px-4 py-2.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-all rounded-xl text-xs font-bold font-mono">
    {React.cloneElement(icon as React.ReactElement, { className: 'w-4 h-4' })}
    {label}
  </button>
);

const ContributionRow: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div>
    <div className="flex justify-between text-[10px] font-bold uppercase mb-1.5 tracking-tight">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-200">{value}%</span>
    </div>
    <div className="h-1 w-full bg-slate-950 rounded-full overflow-hidden">
      <div className={cn("h-full", color)} style={{ width: `${value}%` }} />
    </div>
  </div>
);

const SummaryItem: React.FC<{ title: string; desc: string }> = ({ title, desc }) => (
  <div className="space-y-1">
    <h5 className="text-[11px] font-bold text-red-500 uppercase tracking-tight">• {title}</h5>
    <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
  </div>
);
