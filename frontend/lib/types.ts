export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectionResult {
  detected: boolean;
  gesture: string | null;
  confidence: number;
  landmarks: HandLandmark[][];
  num_hands: number;
}

export interface GestureHistoryItem {
  id: string;
  gesture: string;
  confidence: number;
  timestamp: Date;
}
