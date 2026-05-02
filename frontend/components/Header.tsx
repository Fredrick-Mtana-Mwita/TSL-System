'use client';

import { Hand, Wifi, WifiOff, Activity } from 'lucide-react';

interface HeaderProps {
  isDetecting: boolean;
  isBackendOnline: boolean;
}

export function Header({ isDetecting, isBackendOnline }: HeaderProps) {
  return (
    <header className="glass sticky top-0 z-50 px-4 py-3 flex items-center justify-between border-b border-purple-500/20">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-purple-600 to-purple-900 flex items-center justify-center shadow-lg shadow-purple-900/40">
          <Hand size={18} className="text-white" />
          {isDetecting && (
            <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-red-500 border-2 border-[#0d0d1a] animate-pulse-dot" />
          )}
        </div>
        <div className="leading-none">
          <p className="text-base font-bold text-white tracking-tight">TSL System</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Tanzania Sign Language</p>
        </div>
      </div>

      {/* Status badges */}
      <div className="flex items-center gap-2">
        {/* Backend connection */}
        <span
          className={`hidden sm:flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1.5 rounded-full transition-colors ${
            isBackendOnline
              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
              : 'bg-red-500/15 text-red-400 border border-red-500/20'
          }`}
        >
          {isBackendOnline ? <Wifi size={11} /> : <WifiOff size={11} />}
          {isBackendOnline ? 'Connected' : 'Offline'}
        </span>

        {/* Active detection indicator */}
        {isDetecting && (
          <span className="flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1.5 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/20 animate-fade-up">
            <Activity size={11} className="animate-pulse" />
            Detecting
          </span>
        )}
      </div>
    </header>
  );
}
