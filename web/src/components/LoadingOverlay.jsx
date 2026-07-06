import { useState, useEffect } from 'react';

const STAGES = [
  { key: 'upload',    label: 'Uploading CAD file...',               icon: '📤' },
  { key: 'parse',     label: 'Parsing STEP geometry...',            icon: '🔍' },
  { key: 'mesh',      label: 'Generating triangulated mesh...',     icon: '🔺' },
  { key: 'inference', label: 'Running UV-Net inference on GPU...',  icon: '🧠' },
  { key: 'segment',   label: 'Segmenting face classifications...',  icon: '🎨' },
  { key: 'done',      label: 'Rendering results...',                icon: '✨' },
];

/**
 * Full-screen loading overlay with staged progress messages.
 */
export default function LoadingOverlay({ progress, stage }) {
  const [dots, setDots] = useState('');
  const [currentStage, setCurrentStage] = useState(0);

  // Animate dots
  useEffect(() => {
    const timer = setInterval(() => {
      setDots(prev => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);
    return () => clearInterval(timer);
  }, []);

  // Auto-advance stages for mock mode (when no explicit stage is provided)
  useEffect(() => {
    if (stage !== undefined) return;

    const timers = STAGES.map((_, idx) =>
      setTimeout(() => setCurrentStage(idx), idx * 1200)
    );
    return () => timers.forEach(clearTimeout);
  }, [stage]);

  const activeStageIdx = stage !== undefined
    ? STAGES.findIndex(s => s.key === stage)
    : currentStage;

  const activeStage = STAGES[Math.max(0, activeStageIdx)] || STAGES[0];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-dark-900/90 backdrop-blur-md">
      <div className="flex flex-col items-center gap-8 animate-fade-in-up">
        {/* Spinner ring */}
        <div className="relative w-24 h-24">
          {/* Outer ring */}
          <div className="absolute inset-0 rounded-none border-2 border-dark-500/30" />
          {/* Spinning arc */}
          <div className="absolute inset-0 rounded-none border-2 border-transparent border-t-accent-primary border-r-accent-bright animate-spin-slow" />
          {/* Inner glow */}
          <div className="absolute inset-3 rounded-none bg-gradient-to-br from-accent-primary/10 to-accent-bright/5 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-accent-primary drop-shadow-[0_0_8px_rgba(85,239,196,0.5)]"
              viewBox="0 0 24 24"
              fill="currentColor"
              fillOpacity="0.2"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="12 3 22 21 2 21" />
            </svg>
          </div>
        </div>

        {/* Stage text */}
        <div className="text-center">
          <p className="text-lg font-medium text-white mb-2">
            {activeStage.label}{dots}
          </p>
          {progress !== undefined && progress > 0 && (
            <p className="text-sm font-mono text-dark-200">
              {progress}% complete
            </p>
          )}
        </div>

        {/* Progress bar */}
        <div className="w-72 h-1.5 rounded-none bg-dark-600 overflow-hidden">
          <div
            className="h-full rounded-none bg-gradient-to-r from-accent-primary to-accent-bright transition-all duration-500 ease-out"
            style={{
              width: progress
                ? `${progress}%`
                : `${Math.min(((activeStageIdx + 1) / STAGES.length) * 100, 95)}%`
            }}
          />
        </div>

        {/* Stage list */}
        <div className="flex flex-col gap-1.5 w-72">
          {STAGES.slice(0, -1).map((s, idx) => (
            <div
              key={s.key}
              className={`flex items-center gap-2.5 text-xs transition-all duration-300 ${
                idx < activeStageIdx
                  ? 'text-emerald-400'
                  : idx === activeStageIdx
                  ? 'text-white font-medium'
                  : 'text-dark-400'
              }`}
            >
              <span className="w-4 text-center">
                {idx < activeStageIdx ? '✓' : idx === activeStageIdx ? '●' : '○'}
              </span>
              <span>{s.label.replace('...', '')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
