'use client';

export type TtsLang = 'english' | 'swahili';

const LANG_CODES: Record<TtsLang, string> = {
  english: 'en-US',
  swahili: 'sw',
};

export function speak(text: string, lang: TtsLang = 'english'): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang  = LANG_CODES[lang];
  utt.rate  = 0.9;
  utt.pitch = 1.0;
  window.speechSynthesis.speak(utt);
}

export function stopSpeaking(): void {
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}

export function isTtsSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}
