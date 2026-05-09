import axios from 'axios';
import type { DetectionResult } from './types';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE  = BASE_URL.replace(/^http/, 'ws');

const api = axios.create({ baseURL: BASE_URL, timeout: 5000 });

/** Legacy HTTP polling — uses /predict (AI-based) */
export async function detectGesture(imageBase64: string): Promise<DetectionResult> {
  const { data } = await api.post<DetectionResult>('/predict', { image: imageBase64 });
  return data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    await api.get('/health');
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// WebSocket streaming client
// ---------------------------------------------------------------------------
export type WsMessageHandler = (result: DetectionResult) => void;
export type WsErrorHandler   = (err: Event) => void;

export class WSDetectionClient {
  private ws: WebSocket | null = null;
  private onMessage: WsMessageHandler;
  private onError: WsErrorHandler | undefined;
  private _open = false;

  constructor(onMessage: WsMessageHandler, onError?: WsErrorHandler) {
    this.onMessage = onMessage;
    this.onError   = onError;
  }

  connect(): void {
    if (this.ws) return;
    this.ws = new WebSocket(`${WS_BASE}/ws/detect`);

    this.ws.onopen  = () => { this._open = true; };
    this.ws.onclose = () => { this._open = false; this.ws = null; };
    this.ws.onerror = (e) => { this.onError?.(e); };
    this.ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as DetectionResult;
        this.onMessage(data);
      } catch {
        // ignore malformed frames
      }
    };
  }

  sendFrame(imageBase64: string): void {
    if (this._open && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ image: imageBase64 }));
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws    = null;
    this._open = false;
  }

  get isOpen(): boolean {
    return this._open;
  }
}
