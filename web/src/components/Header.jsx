import { useState, useEffect } from 'react';

/**
 * Application header with title, status indicator, and ambient gradient.
 */
export default function Header({ status, fileName }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const statusConfig = {
    idle: { color: 'bg-dark-400', text: 'Ready', dot: 'bg-dark-300' },
    uploading: { color: 'bg-amber-500/20', text: 'Uploading', dot: 'bg-amber-400' },
    processing: { color: 'bg-blue-500/20', text: 'Processing', dot: 'bg-blue-400' },
    complete: { color: 'bg-emerald-500/20', text: 'Complete', dot: 'bg-emerald-400' },
    error: { color: 'bg-red-500/20', text: 'Error', dot: 'bg-red-400' },
    mock: { color: 'bg-purple-500/20', text: 'Mock Mode', dot: 'bg-purple-400' },
  };

  const currentStatus = statusConfig[status] || statusConfig.idle;

  return (
    <header className="glass sticky top-0 z-50 px-6 py-3" style={{ paddingTop: "10px", paddingBottom: "10px" }}>
      {/* Ambient top-edge glow */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-accent-primary/50 to-transparent" />

      <div className="flex items-center justify-between max-w-[1800px] mx-auto">
        {/* Left: Branding */}
        <div className="flex items-center gap-3">
          {/* Logo icon */}
          <div className="relative w-9 h-9 rounded-none bg-gradient-to-br from-accent-primary to-accent-bright flex items-center justify-center shadow-lg shadow-accent-glow">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-semibold text-white tracking-tight leading-tight">
              CAD Feature Recognizer
            </h1>
            <p className="text-[11px] text-dark-200 font-mono leading-tight">
              UV-Net Powered Analysis
            </p>
          </div>
        </div>

        {/* Center: File info */}
        {fileName && (
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-none bg-dark-700/60 border border-dark-500/40">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-dark-200">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
              <path d="M14 2v6h6" />
            </svg>
            <span className="text-xs text-dark-100 font-mono truncate max-w-[200px]">
              {fileName}
            </span>
          </div>
        )}

        {/* Right: Status + Time */}
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-none text-xs font-medium ${currentStatus.color}`}>
            <span className={`w-2 h-2 rounded-none ${currentStatus.dot} ${status === 'processing' || status === 'uploading' ? 'animate-pulse' : ''}`} />
            <span className="text-dark-100">{currentStatus.text}</span>
          </div>

          <div className="hidden sm:block text-[11px] text-dark-300 font-mono tabular-nums">
            {time.toLocaleTimeString('en-US', { hour12: false })}
          </div>
        </div>
      </div>
    </header>
  );
}
