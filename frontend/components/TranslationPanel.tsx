'use client';

import { Clock, Copy, Trash2, HandMetal, TrendingUp, Volume2 } from 'lucide-react';
import { useState } from 'react';
import type { DetectionResult, GestureHistoryItem } from '@/lib/types';
import { speak, isTtsSupported, type TtsLang } from '@/lib/tts';
import styles from './TranslationPanel.module.css';

interface TranslationPanelProps {
  result:         DetectionResult | null;
  history:        GestureHistoryItem[];
  isDetecting:    boolean;
  onClearHistory: () => void;
}

function ConfidenceBar({ value }: { value: number }) {
  const pct   = Math.round(value * 100);
  const color = pct >= 75 ? 'bg-emerald-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500';
  const pctClass = Math.round(pct / 5) * 5; // Round to nearest 5%
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color} ${styles.confidenceBar} ${styles[`pct${pctClass}`]}`}
        />
      </div>
      <span className="text-xs text-slate-400 w-8 text-right font-mono">{pct}%</span>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function SpeakButton({ text, lang, label }: { text: string; lang: TtsLang; label: string }) {
  return (
    <button
      type="button"
      onClick={() => speak(text, lang)}
      title={`Speak in ${label}`}
      className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg border border-purple-500/20
                 text-slate-400 hover:text-purple-300 hover:bg-purple-500/10 transition-colors"
    >
      <Volume2 size={11} />
      {label}
    </button>
  );
}

export function TranslationPanel({
  result,
  history,
  isDetecting,
  onClearHistory,
}: TranslationPanelProps) {
  const [copied, setCopied] = useState(false);
  const ttsOk = isTtsSupported();

  const handleCopy = () => {
    const text = result?.english && result?.swahili
      ? `${result.english} / ${result.swahili}`
      : result?.gesture ?? '';
    if (text) {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      });
    }
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Current Detection */}
      <div className="glass rounded-2xl p-4 border border-purple-500/18 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-medium uppercase tracking-widest">
            <HandMetal size={13} />
            Current Gesture
          </div>
          {result?.gesture && (
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-purple-400 transition-colors px-2 py-1 rounded-lg hover:bg-purple-500/10"
            >
              <Copy size={11} />
              {copied ? 'Copied!' : 'Copy'}
            </button>
          )}
        </div>

        <div className="min-h-[96px] flex items-center justify-center mb-3">
          {!isDetecting ? (
            <p className="text-slate-600 text-sm text-center">
              Press <span className="text-purple-400">Start Detection</span> to begin
            </p>
          ) : !result?.detected ? (
            <div className="flex flex-col items-center gap-2">
              <div className="w-8 h-8 rounded-full border-2 border-purple-500/40 border-t-purple-500 animate-spin" />
              <p className="text-slate-500 text-sm">Scanning for hands…</p>
            </div>
          ) : (
            <div className="w-full animate-fade-up" key={result.gesture}>
              {/* English label */}
              <p className="text-xl font-bold text-white text-center leading-snug">
                {result.english ?? result.gesture}
              </p>
              {/* Swahili label */}
              {result.swahili && (
                <p className="text-base font-semibold text-purple-300 text-center mt-0.5">
                  {result.swahili}
                </p>
              )}
              <p className="text-xs text-slate-500 text-center mt-1 mb-3">
                {result.num_hands} hand{result.num_hands !== 1 ? 's' : ''} · TSL
              </p>
              <ConfidenceBar value={result.confidence} />

              {ttsOk && (
                <div className="flex gap-2 justify-center mt-3">
                  {result.english && (
                    <SpeakButton text={result.english} lang="english" label="EN" />
                  )}
                  {result.swahili && (
                    <SpeakButton text={result.swahili} lang="swahili" label="SW" />
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {result?.detected && (
          <div className="grid grid-cols-2 gap-2 mt-2">
            <div className="bg-white/3 rounded-xl p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Hands</p>
              <p className="text-lg font-bold text-purple-400">{result.num_hands}</p>
            </div>
            <div className="bg-white/3 rounded-xl p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Confidence</p>
              <p className="text-lg font-bold text-emerald-400">
                {Math.round(result.confidence * 100)}%
              </p>
            </div>
          </div>
        )}
      </div>

      {/* History */}
      <div className="glass rounded-2xl p-4 border border-purple-500/18 flex flex-col min-h-0 flex-1">
        <div className="flex items-center justify-between mb-3 flex-shrink-0">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-medium uppercase tracking-widest">
            <TrendingUp size={13} />
            History
            {history.length > 0 && (
              <span className="bg-purple-500/20 text-purple-400 text-[10px] px-1.5 py-0.5 rounded-full font-mono">
                {history.length}
              </span>
            )}
          </div>
          {history.length > 0 && (
            <button
              type="button"
              onClick={onClearHistory}
              className="flex items-center gap-1 text-xs text-slate-600 hover:text-red-400 transition-colors px-2 py-1 rounded-lg hover:bg-red-500/10"
            >
              <Trash2 size={11} />
              Clear
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-slate-600 text-sm text-center">Detected gestures will appear here</p>
          </div>
        ) : (
          <ul className="flex flex-col gap-1.5 overflow-y-auto flex-1 pr-1">
            {history.map((item, i) => (
              <li
                key={item.id}
                className={`flex items-center justify-between gap-2 bg-white/3 hover:bg-white/5 rounded-xl px-3 py-2 transition-colors ${styles.historyItem} ${styles[`delay${Math.min(i, 49)}`]}`}
              >
                <div className="flex flex-col min-w-0">
                  <span className="text-sm text-slate-200 truncate">
                    {item.english ?? item.gesture}
                  </span>
                  {item.swahili && (
                    <span className="text-xs text-purple-400 truncate">{item.swahili}</span>
                  )}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className="text-[11px] font-mono text-slate-500">
                    {Math.round(item.confidence * 100)}%
                  </span>
                  <span className="flex items-center gap-1 text-[10px] text-slate-600">
                    <Clock size={9} />
                    {formatTime(item.timestamp)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
