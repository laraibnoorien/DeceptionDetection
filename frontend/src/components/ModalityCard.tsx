import React from 'react';
import { cn } from '../lib/utils';
import { motion } from 'motion/react';

interface ModalityCardProps {
  id: string;
  title: string;
  subtitle: string;
  score: number; // 0 to 1
  children: React.ReactNode;
  className?: string;
}

export const ModalityCard: React.FC<ModalityCardProps> = ({ id, title, subtitle, score, children, className }) => {
  const isHighAlert = score > 0.7;
  
  return (
    <div className={cn(
      "flex flex-col p-5 bg-white/5 backdrop-blur-md border transition-all duration-500 rounded-xl",
      isHighAlert ? "border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.1)]" : "border-white/10",
      className
    )}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-900 text-slate-300 ring-1 ring-white/10">{id}</span>
            <h4 className="text-sm font-bold text-slate-100 uppercase tracking-wide">{title}</h4>
          </div>
          <p className="text-[10px] text-slate-500 mt-0.5">{subtitle}</p>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-slate-500 uppercase block leading-none mb-1">Risk Score</span>
          <span className={cn(
            "text-lg font-mono font-bold leading-none",
            score > 0.7 ? "text-rose-500" : score > 0.4 ? "text-amber-500" : "text-emerald-500"
          )}>
            {(score).toFixed(2)}
          </span>
        </div>
      </div>
      
      <div className="flex-1 space-y-4">
        {children}
      </div>

      <div className="mt-4 pt-4 border-t border-white/5">
        <div className="flex items-center justify-between text-[10px] text-slate-600 uppercase tracking-tighter">
          <span>Signal Quality</span>
          <span className="text-emerald-500 font-mono italic">Optimal</span>
        </div>
      </div>
    </div>
  );
};

export const MetricRow: React.FC<{ label: string; value: string | number; sub?: string; alert?: boolean }> = ({ label, value, sub, alert }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0">
    <span className="text-xs text-slate-400 font-medium">{label}</span>
    <div className="text-right">
      <span className={cn("text-xs font-mono font-bold", alert ? "text-red-400" : "text-slate-200")}>{value}</span>
      {sub && <span className="text-[10px] text-slate-500 ml-1.5 font-normal">{sub}</span>}
    </div>
  </div>
);
