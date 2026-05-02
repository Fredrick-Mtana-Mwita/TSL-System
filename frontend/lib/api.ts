import axios from 'axios';
import type { DetectionResult } from './types';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 5000,
});

export async function detectGesture(imageBase64: string): Promise<DetectionResult> {
  const { data } = await api.post<DetectionResult>('/detect', { image: imageBase64 });
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
