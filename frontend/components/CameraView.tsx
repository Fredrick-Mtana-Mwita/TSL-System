'use client';

import { useRef, useCallback, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera, CameraOff, FlipHorizontal, Loader2 } from 'lucide-react';
import type { DetectionResult, HandLandmark } from '@/lib/types';
import { WSDetectionClient } from '@/lib/api';
import styles from './CameraView.module.css';

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],  [1, 2],   [2, 3],   [3, 4],
  [0, 5],  [5, 6],   [6, 7],   [7, 8],
  [0, 9],  [9, 10],  [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9],  [9, 13],  [13, 17],
];
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

export interface CameraViewProps {
  isActive:        boolean;
  onToggle:        () => void;
  onDetection:     (result: DetectionResult) => void;
  detectionResult: DetectionResult | null;
}

export function CameraView({ isActive, onToggle, onDetection, detectionResult }: CameraViewProps) {
  const webcamRef  = useRef<Webcam>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const wsRef      = useRef<WSDetectionClient | null>(null);
  const frameTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [camReady,   setCamReady]   = useState(false);
  const [camError,   setCamError]   = useState(false);

  // Draw hand skeleton whenever detection result changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!detectionResult?.detected) return;

    const W = canvas.width;
    const H = canvas.height;

    detectionResult.landmarks.forEach((hand: HandLandmark[]) => {
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo((1 - hand[a].x) * W, hand[a].y * H);
        ctx.lineTo((1 - hand[b].x) * W, hand[b].y * H);
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.75)';
        ctx.lineWidth   = 2;
        ctx.lineCap     = 'round';
        ctx.stroke();
      });
      hand.forEach((lm, idx) => {
        const x     = (1 - lm.x) * W;
        const y     = lm.y * H;
        const isTip = FINGERTIPS.has(idx);
        const isWrist = idx === 0;
        ctx.beginPath();
        ctx.arc(x, y, isWrist ? 7 : isTip ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle   = isWrist ? '#ec4899' : isTip ? '#a78bfa' : '#7c3aed';
        ctx.shadowColor = isWrist ? '#ec4899' : '#7c3aed';
        ctx.shadowBlur  = 12;
        ctx.fill();
        ctx.shadowBlur  = 0;
      });
    });
  }, [detectionResult]);

  // Send one frame via WebSocket
  const sendFrame = useCallback(() => {
    if (!webcamRef.current || !wsRef.current) return;
    const img = webcamRef.current.getScreenshot();
    if (img) wsRef.current.sendFrame(img);
  }, []);

  // Start / stop WebSocket + frame timer
  useEffect(() => {
    if (isActive) {
      const client = new WSDetectionClient(
        (result) => onDetection(result),
        () => {/* WebSocket error — keep going, backend may be starting */},
      );
      client.connect();
      wsRef.current  = client;
      frameTimer.current = setInterval(sendFrame, 200);   // ~5 fps to server
    } else {
      frameTimer.current && clearInterval(frameTimer.current);
      wsRef.current?.disconnect();
      wsRef.current = null;
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx && canvasRef.current) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    return () => {
      frameTimer.current && clearInterval(frameTimer.current);
      wsRef.current?.disconnect();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  const videoConstraints = { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode };

  return (
    <div className="flex flex-col gap-3 h-full">
      <div
        className={`relative w-full rounded-2xl overflow-hidden bg-[#090914] border border-purple-500/15 shadow-2xl shadow-purple-950/30 ${styles.videoContainer}`}
      >
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          screenshotQuality={0.8}
          videoConstraints={videoConstraints}
          onUserMedia={() => { setCamReady(true); setCamError(false); }}
          onUserMediaError={() => { setCamError(true); setCamReady(false); }}
          className={`absolute inset-0 w-full h-full object-cover ${styles.webcamVideo}`}
          mirrored={false}
        />

        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {!camReady && !camError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#090914] gap-3">
            <Loader2 size={32} className="text-purple-500 animate-spin" />
            <p className="text-slate-400 text-sm">Starting camera…</p>
          </div>
        )}

        {camError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#090914] gap-3 p-6 text-center">
            <div className="w-14 h-14 rounded-full bg-red-500/15 flex items-center justify-center">
              <CameraOff size={24} className="text-red-400" />
            </div>
            <p className="text-slate-300 font-medium">Camera access denied</p>
            <p className="text-slate-500 text-sm max-w-xs">
              Please allow camera permission in your browser settings and reload the page.
            </p>
          </div>
        )}

        {isActive && camReady && (
          <>
            <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-purple-500 rounded-tl-md opacity-70" />
            <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-purple-500 rounded-tr-md opacity-70" />
            <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-purple-500 rounded-bl-md opacity-70" />
            <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-purple-500 rounded-br-md opacity-70" />
          </>
        )}

        {detectionResult?.detected && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 text-[11px] font-medium px-2.5 py-1 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
            {detectionResult.num_hands} hand{detectionResult.num_hands !== 1 ? 's' : ''} detected
          </div>
        )}

        <button
          onClick={() => setFacingMode(m => m === 'user' ? 'environment' : 'user')}
          className="absolute bottom-3 right-3 w-9 h-9 rounded-full glass flex items-center justify-center text-slate-300 hover:text-white transition-colors border border-purple-500/20 active:scale-95"
          title="Flip camera"
        >
          <FlipHorizontal size={16} />
        </button>
      </div>

      <button
        onClick={onToggle}
        disabled={camError}
        className={`w-full py-3.5 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed ${
          isActive
            ? 'bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/30'
            : 'bg-gradient-to-r from-purple-700 to-purple-600 hover:from-purple-600 hover:to-purple-500 text-white shadow-lg shadow-purple-900/40'
        }`}
      >
        {isActive ? (
          <>
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Stop Detection
          </>
        ) : (
          <>
            <Camera size={16} />
            Start Detection
          </>
        )}
      </button>
    </div>
  );
}
