import { create } from 'zustand';
import { 
  HardwareStatus, 
  TelemetryEvent, 
  SessionStage, 
  CalibrationProgress,
  AlertLevel 
} from '../types';

interface AppState {
  stage: SessionStage;
  hardware: HardwareStatus;
  calibration: CalibrationProgress;
  telemetry: TelemetryEvent | null;
  history: TelemetryEvent[];
  
  // Actions
  setStage: (stage: SessionStage) => void;
  updateHardware: (update: Partial<HardwareStatus>) => void;
  updateCalibration: (update: Partial<CalibrationProgress>) => void;
  pushTelemetry: (event: TelemetryEvent) => void;
  resetSession: () => void;
}


export const useStore = create<AppState>((set) => ({
  stage: 'idle',
  hardware: {
    camera: 'unknown',
    microphone: 'unknown',
    wearable: 'unavailable',
    server: 'unknown',
    aiModels: 'unknown',
    websocket: 'unknown',
  },
  calibration: {
    emotional: 0,
    physiological: 0,
    speech: 0,
    currentStep: 'Initializing...',
  },
  telemetry: null,
  history: [],

  setStage: (stage) => set({ stage }),
  updateHardware: (update) => set((state) => ({ 
    hardware: { ...state.hardware, ...update } 
  })),
  updateCalibration: (update) => set((state) => ({ 
    calibration: { ...state.calibration, ...update } 
  })),
  pushTelemetry: (event) => set((state) => ({
    telemetry: event,
    history: [event, ...state.history].slice(0, 100) // Keep last 100 events
  })),
  resetSession: () => set({
    stage: 'idle',
    telemetry: null,
    history: [],
    calibration: {
      emotional: 0,
      physiological: 0,
      speech: 0,
      currentStep: 'Initializing...',
    }
  })
}));
