export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectionResult {
  detected:    boolean;
  gesture:     string | null;   // combined "English / Swahili" — backward compat
  english:     string | null;
  swahili:     string | null;
  confidence:  number;
  landmarks:   HandLandmark[][];
  num_hands:   number;
  gesture_id?: number;
}

export interface GestureHistoryItem {
  id:         string;
  gesture:    string;
  english:    string | null;
  swahili:    string | null;
  confidence: number;
  timestamp:  Date;
}
