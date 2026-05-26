/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ModalityStatus = 'connected' | 'connecting' | 'disconnected' | 'unavailable' | 'degraded';

export interface HardwareStatus {
  camera: ModalityStatus;
  microphone: ModalityStatus;
  wearable: ModalityStatus;
  server: ModalityStatus;
  aiModels: ModalityStatus;
  websocket: ModalityStatus;
}

export type AlertLevel = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED';

export interface MicroexpressionModality {
  emotion: string;
  auIntensities: Record<string, number>;
  gazeStability: number;
  blinkRate: number;
  score: number;
  anomalies: string[];
}

export interface PhysiologicalModality {
  bpm: number;
  hrv: number;
  spo2: number;
  stressIndex: number;
  anomalyScore: number;
  signalQuality: number;
}

export interface SpeechModality {
  pitch: number;
  jitter: number;
  shimmer: number;
  vocalStress: number;
  score: number;
  transcript: string;
}

export interface LinguisticModality {
  transcript: string;
  contradictionScore: number;
  certaintyScore: number;
  inconsistencies: string[];
}

export interface MultimodalFusion {
  deceptionProbability: number;
  confidence: number;
  dominantModality: 'MM' | 'PM' | 'SM' | 'TM';
  alertLevel: AlertLevel;
}

export interface TelemetryEvent {
  timestamp: string;
  mm: MicroexpressionModality;
  pm: PhysiologicalModality;
  sm: SpeechModality;
  tm: LinguisticModality;
  fusion: MultimodalFusion;
  explainability: {
    drivers: string[];
    summary: string;
  };
}

export type SessionStage = 'idle' | 'calibration' | 'live' | 'report';

export interface CalibrationProgress {
  emotional: number;
  physiological: number;
  speech: number;
  currentStep: string;
}
