import React from 'react';
import { cn } from '../lib/utils';
import { ModalityStatus } from '../types';

interface StatusIndicatorProps {
  label: string;
  status: ModalityStatus;
  className?: string;
}

const statusConfig: Record<ModalityStatus, { color: string; label: string }> = {
  connected: { color: 'bg-emerald-500', label: 'Connected' },
  connecting: { color: 'bg-amber-500 animate-pulse', label: 'Connecting' },
  disconnected: { color: 'bg-red-500', label: 'Disconnected' },
  unavailable: { color: 'bg-slate-600', label: 'Unavailable' },
  degraded: { color: 'bg-orange-500', label: 'Degraded' },
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ label, status, className }) => {
  const config = statusConfig[status];
  
  return (
    <div className={cn("flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800", className)}>
      <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono text-slate-500 uppercase">{config.label}</span>
        <div className={cn("w-2 h-2 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.5)]", config.color)} />
      </div>
    </div>
  );
};
