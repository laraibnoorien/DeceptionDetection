/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Header } from './components/Header';
import { TerminalView } from './components/TerminalView';
import { CalibrationView } from './components/CalibrationView';
import { DashboardView } from './components/DashboardView';
import { ReportView } from './components/ReportView';
import { useStore } from './store/useStore';
import { motion, AnimatePresence } from 'motion/react';

export default function App() {
  const { stage } = useStore();

  return (
    <div className="min-h-screen bg-[#070707] text-slate-300 font-sans selection:bg-gold-muted/30 relative">
      {/* Cinematic Background Layer */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_20%_20%,_rgba(197,160,89,0.05)_0%,_transparent_50%)]" />
        <div className="absolute bottom-0 right-0 w-full h-full bg-[radial-gradient(circle_at_80%_80%,_rgba(139,92,246,0.03)_0%,_transparent_50%)]" />
      </div>

      <div className="relative z-10">
        <Header />
        
        <main className="relative">
          <AnimatePresence mode="wait">
          {stage === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="w-full"
            >
              <TerminalView />
            </motion.div>
          )}

          {stage === 'calibration' && (
            <motion.div
              key="calibration"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 1.02 }}
              className="w-full"
            >
              <CalibrationView />
            </motion.div>
          )}

          {stage === 'live' && (
            <motion.div
              key="live"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full"
            >
              <DashboardView />
            </motion.div>
          )}

          {stage === 'report' && (
            <motion.div
              key="report"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="w-full"
            >
              <ReportView />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      </div>

      {/* Global Background Elements */}
      <div className="fixed inset-0 pointer-events-none z-[-1] overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-red-600/5 blur-[120px] rounded-full" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-slate-800/10 blur-[150px] rounded-full" />
      </div>
    </div>
  );
}

