'use client';

import { useState, useCallback, useEffect } from 'react';
import { Header }           from '@/components/Header';
import { CameraWrapper }    from '@/components/CameraWrapper';
import { TranslationPanel } from '@/components/TranslationPanel';
import { checkHealth }      from '@/lib/api';
import type { DetectionResult, GestureHistoryItem } from '@/lib/types';
import styles from './page.module.css';

export default function Home() {
  const [isDetecting,      setIsDetecting]      = useState(false);
  const [detectionResult,  setDetectionResult]  = useState<DetectionResult | null>(null);
  const [history,          setHistory]          = useState<GestureHistoryItem[]>([]);
  const [isBackendOnline,  setIsBackendOnline]  = useState(false);

  // Poll backend health
  useEffect(() => {
    const probe = async () => { setIsBackendOnline(await checkHealth()); };
    probe();
    const id = setInterval(probe, 5000);
    return () => clearInterval(id);
  }, []);

  const handleToggleDetection = useCallback(() => {
    setIsDetecting(prev => {
      if (prev) setDetectionResult(null);
      return !prev;
    });
  }, []);

  const handleDetection = useCallback((result: DetectionResult) => {
    setIsBackendOnline(true);
    setDetectionResult(result);

    if (!result.detected || !result.gesture) return;

    setHistory(prev => {
      if (prev[0]?.gesture === result.gesture) return prev;
      const item: GestureHistoryItem = {
        id:         `${Date.now()}-${Math.random()}`,
        gesture:    result.gesture!,
        english:    result.english ?? null,
        swahili:    result.swahili ?? null,
        confidence: result.confidence,
        timestamp:  new Date(),
      };
      return [item, ...prev.slice(0, 49)];
    });

  }, []);

  const handleClearHistory = useCallback(() => setHistory([]), []);

  return (
    <div className={`min-h-screen flex flex-col ${styles.container}`}>
      <Header isDetecting={isDetecting} isBackendOnline={isBackendOnline} />

      {/* Backend offline banner */}
      {!isBackendOnline && (
        <div className="mx-4 mt-3 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs text-center">
          Backend offline — start the server with{' '}
          <code className="font-mono bg-amber-500/15 px-1.5 py-0.5 rounded">
            uvicorn main:app --reload
          </code>{' '}
          in{' '}
          <code className="font-mono bg-amber-500/15 px-1.5 py-0.5 rounded">backend/</code>
        </div>
      )}

      <main className="flex-1 flex flex-col lg:flex-row gap-4 p-4 lg:p-5 lg:items-start">
        <section className="w-full lg:flex-[2] lg:min-w-0">
          <CameraWrapper
            isActive={isDetecting}
            onToggle={handleToggleDetection}
            onDetection={handleDetection}
            detectionResult={detectionResult}
          />
        </section>

        <section className="w-full lg:w-[340px] xl:w-[380px] lg:flex-shrink-0 lg:sticky lg:top-[68px]">
          <TranslationPanel
            result={detectionResult}
            history={history}
            isDetecting={isDetecting}
            onClearHistory={handleClearHistory}
          />
        </section>
      </main>

      <footer className="px-4 pb-4 text-center">
        <p className="text-[11px] text-slate-700">
          TSL System — Tanzania Sign Language · Transformer AI · Bilingual EN/SW
        </p>
      </footer>
    </div>
  );
}
