import React from 'react';
import { useStore } from '../store/useStore';
import { Shield, BrainCircuit, Activity, RotateCcw } from 'lucide-react';
import { cn } from '../lib/utils';
import { motion } from 'motion/react';

export const Header: React.FC = () => {
  const { stage, setStage, resetSession } = useStore();
  
  return (
    <header className="h-20 border-b border-white/5 bg-transparent flex items-center justify-between px-12 sticky top-0 z-50">
      <div className="flex items-center gap-12">
        <div className="flex items-center gap-3 group cursor-pointer" onClick={() => setStage('idle')}>
          <div className="relative">
            <Shield className="w-5 h-5 text-gold-muted stroke-[1.5px]" />
            <div className="absolute inset-0 bg-gold-muted/20 blur-lg rounded-full" />
          </div>
          <div>
            <h1 className="text-sm font-black tracking-[0.3em] text-white leading-none uppercase font-sans">DON’T LIE</h1>
            <span className="text-[10px] font-medium text-gold-muted/60 tracking-[0.2em] uppercase mt-1 block">Behavioral Intelligence</span>
          </div>
        </div>
        
        <nav className="flex items-center gap-10">
          <NavItem active={stage === 'idle'} label="Terminal" onClick={() => setStage('idle')} />
          <NavItem active={stage === 'calibration'} label="Calibration" onClick={() => setStage('calibration')} />
          <NavItem active={stage === 'live'} label="Live Link" disabled={stage === 'idle'} onClick={() => setStage('live')} />
          <NavItem active={stage === 'report'} label="Archives" onClick={() => setStage('report')} />
        </nav>
      </div>

      <div className="flex items-center gap-6">
        {stage !== 'idle' && (
          <button 
            onClick={resetSession}
            className="flex items-center gap-2 px-3 py-1.5 text-[10px] uppercase tracking-widest font-bold text-slate-500 hover:text-white transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Clear Session
          </button>
        )}
        
        <div className="flex items-center gap-4 px-5 py-2 rounded-full border border-white/5 bg-white/[0.02]">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-gold-muted shadow-[0_0_8px_rgba(197,160,89,0.5)]" />
            <span className="text-[10px] font-mono text-gold-muted uppercase tracking-widest">Auth Valid</span>
          </div>
        </div>
      </div>
    </header>
  );
};

const NavItem: React.FC<{ active: boolean; label: string; disabled?: boolean; onClick: () => void }> = ({ active, label, disabled, onClick }) => (
  <button 
    onClick={onClick}
    disabled={disabled}
    className={cn(
      "text-[10px] font-black uppercase tracking-[0.25em] transition-all relative py-2",
      active ? "text-gold-muted" : "text-white/40 hover:text-white/80",
      disabled && "opacity-20 cursor-not-allowed"
    )}
  >
    {label}
    {active && (
      <motion.span 
        layoutId="nav-glow"
        className="absolute bottom-0 left-0 w-full h-[1px] bg-gold-muted shadow-[0_0_15px_rgba(197,160,89,0.8)]" 
      />
    )}
  </button>
);
