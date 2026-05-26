import React from 'react';
import { motion } from 'motion/react';
import { AlertLevel } from '../types';
import { cn } from '../lib/utils';

interface FusionGaugeProps {
  probability: number; // 0 to 1
  confidence: number; // 0 to 1
  level: AlertLevel;
  className?: string;
}

const levelConfigs: Record<AlertLevel, { color: string; bg: string; shadow: string }> = {
  GREEN: { color: 'text-emerald-500', bg: 'bg-emerald-500', shadow: 'shadow-emerald-500/20' },
  YELLOW: { color: 'text-amber-400', bg: 'bg-amber-400', shadow: 'shadow-amber-400/20' },
  ORANGE: { color: 'text-orange-500', bg: 'bg-orange-500', shadow: 'shadow-orange-500/20' },
  RED: { color: 'text-rose-500', bg: 'bg-rose-500', shadow: 'shadow-rose-500/20' },
};

export const FusionGauge: React.FC<FusionGaugeProps> = ({ probability, confidence, level, className }) => {
  const config = levelConfigs[level];
  const percentage = Math.round(probability * 100);
  
  return (
    <div className={cn("relative flex flex-col items-center justify-center p-8 bg-white/5 border border-white/10 rounded-2xl overflow-hidden backdrop-blur-md", className)}>
      {/* Background radial glow */}
      <div className={cn("absolute inset-0 opacity-5 blur-3xl transition-colors duration-500", config.bg)} />
      
      <div className="relative z-10 text-center">
        <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-1">Live Deception Probability</h3>
        <div className="flex flex-col items-center">
          <motion.span 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn("text-7xl font-black tracking-tighter transition-colors duration-500", config.color)}
          >
            {percentage}%
          </motion.span>
          
          <div className="flex items-center gap-4 mt-4">
            <div className="flex flex-col items-center">
              <span className="text-[10px] text-slate-500 uppercase mb-1">Confidence</span>
              <span className="text-sm font-mono text-slate-300">{Math.round(confidence * 100)}%</span>
            </div>
            <div className="w-[1px] h-8 bg-slate-800" />
            <div className="flex flex-col items-center">
              <span className="text-[10px] text-slate-500 uppercase mb-1">Alert Level</span>
              <span className={cn("text-sm font-bold uppercase", config.color)}>{level}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar under-glow */}
      <div className="absolute bottom-0 left-0 w-full h-1 bg-slate-900 overflow-hidden">
        <motion.div 
          className={cn("h-full transition-all duration-1000", config.bg)}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
