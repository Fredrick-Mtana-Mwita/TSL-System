'use client';

import { useEffect, useState } from 'react';
import type { DetectionResult } from '@/lib/types';
import { CameraView } from './CameraView';
import styles from './CameraWrapper.module.css';

interface CameraWrapperProps {
  isActive: boolean;
  onToggle: () => void;
  onDetection: (result: DetectionResult) => void;
  detectionResult: DetectionResult | null;
}

export function CameraWrapper({
  isActive,
  onToggle,
  onDetection,
  detectionResult,
}: CameraWrapperProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className={`w-full rounded-2xl bg-[#090914] border border-purple-500/15 flex items-center justify-center ${styles.loadingContainer}`}>
        <p className="text-slate-500 text-sm">Loading camera...</p>
      </div>
    );
  }

  return (
    <CameraView
      isActive={isActive}
      onToggle={onToggle}
      onDetection={onDetection}
      detectionResult={detectionResult}
    />
  );
}
