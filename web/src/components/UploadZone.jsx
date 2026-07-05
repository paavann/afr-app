import { useCallback, useState } from 'react';

/**
 * Drag-and-drop file upload zone with click-to-browse fallback.
 * Accepts only .step / .stp files.
 */
export default function UploadZone({ onFileSelect, disabled }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    const stepFile = files.find(f =>
      f.name.toLowerCase().endsWith('.step') || f.name.toLowerCase().endsWith('.stp')
    );

    if (stepFile) {
      onFileSelect(stepFile);
    }
  }, [disabled, onFileSelect]);

  const handleClick = () => {
    if (disabled) return;
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.step,.stp';
    input.onchange = (e) => {
      const file = e.target.files?.[0];
      if (file) onFileSelect(file);
    };
    input.click();
  };

  const handleLoadMock = (e) => {
    e.stopPropagation();
    onFileSelect(null); // null signals "use mock data"
  };

  return (
    <div className="w-full max-w-2xl mx-auto animate-fade-in-up">
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative group cursor-pointer
          rounded-2xl border-2 border-dashed
          transition-all duration-300 ease-out
          px-8 py-14
          ${isDragging
            ? 'drag-active'
            : 'border-dark-400/60 hover:border-accent-primary/50 hover:bg-dark-700/30'
          }
          ${disabled ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        {/* Shimmer overlay on hover */}
        <div className="absolute inset-0 rounded-2xl overflow-hidden opacity-0 group-hover:opacity-100 transition-opacity duration-500">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.02] to-transparent animate-shimmer" />
        </div>

        <div className="relative flex flex-col items-center gap-5 text-center">
          {/* Upload icon */}
          <div className={`
            w-16 h-16 rounded-2xl flex items-center justify-center
            bg-gradient-to-br from-accent-primary/20 to-accent-bright/10
            border border-accent-primary/20
            transition-transform duration-300
            group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-accent-glow
            ${isDragging ? 'scale-110 shadow-lg shadow-accent-glow' : ''}
          `}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-accent-primary">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>

          {/* Text */}
          <div>
            <p className="text-base font-medium text-white mb-1">
              {isDragging ? 'Drop your STEP file here' : 'Upload CAD File'}
            </p>
            <p className="text-sm text-dark-200">
              Drag & drop a <span className="text-accent-secondary font-mono text-xs">.step</span> or{' '}
              <span className="text-accent-secondary font-mono text-xs">.stp</span> file, or{' '}
              <span className="text-accent-primary underline underline-offset-2">browse</span>
            </p>
          </div>

          {/* File type badges */}
          <div className="flex gap-2">
            {['.STEP', '.STP'].map(ext => (
              <span
                key={ext}
                className="px-2.5 py-1 rounded-md bg-dark-600/80 text-[11px] font-mono text-dark-200 border border-dark-500/50"
              >
                {ext}
              </span>
            ))}
          </div>

          {/* Mock data button */}
          <button
            type="button"
            onClick={handleLoadMock}
            className="
              mt-2 px-4 py-2 rounded-lg
              text-xs font-medium
              bg-dark-600/60 text-dark-200
              border border-dark-500/40
              hover:bg-accent-primary/15 hover:text-accent-secondary hover:border-accent-primary/30
              transition-all duration-200
              cursor-pointer
            "
          >
            ⚡ Load Demo Data (No Backend)
          </button>
        </div>
      </div>
    </div>
  );
}
