import { getDetectedFeatures } from '../utils/colorMap';

/**
 * Side-panel legend showing detected features, their colors, and face counts.
 */
export default function FeatureLegend({ predictions, fileInfo, onReset }) {
  const features = getDetectedFeatures(predictions);
  const totalFaces = predictions.length;

  return (
    <div className="glass rounded-none p-6 w-full max-w-[300px] animate-fade-in-up flex flex-col gap-4 max-h-[calc(100vh-140px)] overflow-y-auto">
      {/* Panel header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-none bg-gradient-to-br from-accent-primary/30 to-accent-bright/15 flex items-center justify-center">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-secondary">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
            </svg>
          </div>
          <h2 className="text-sm font-semibold text-white">Feature Legend</h2>
        </div>
        {onReset && (
          <button
            onClick={onReset}
            className="text-[11px] text-dark-300 hover:text-error transition-colors px-2 py-1 rounded-none hover:bg-red-500/10 cursor-pointer"
          >
            Reset
          </button>
        )}
      </div>

      {/* Feature list */}
      <div className="flex flex-col gap-1.5">
        {features.map((feature) => {
          const percentage = ((feature.count / totalFaces) * 100).toFixed(1);
          return (
            <div
              key={feature.label}
              className="group flex items-center gap-3 p-3 rounded-none bg-dark-700/40 hover:bg-dark-600/60 transition-all duration-200 border border-transparent hover:border-dark-500/30"
            >
              {/* Color swatch */}
              <div
                className="w-3.5 h-3.5 rounded-none flex-shrink-0 shadow-sm"
                style={{ backgroundColor: feature.hex }}
              />

              {/* Label & count */}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-dark-100 truncate">
                  {feature.label}
                </p>
                <p className="text-[10px] text-dark-300 font-mono">
                  {feature.count} face{feature.count !== 1 ? 's' : ''} · {percentage}%
                </p>
              </div>

              {/* Mini bar */}
              <div className="w-12 h-1 rounded-none bg-dark-600 overflow-hidden flex-shrink-0">
                <div
                  className="h-full rounded-none transition-all duration-500"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: feature.hex,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Separator */}
      <div className="h-px bg-dark-500/40" />

      {/* File info */}
      {fileInfo && (
        <div className="flex flex-col gap-2">
          <h3 className="text-[11px] font-semibold text-dark-200 uppercase tracking-wider">
            File Details
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Faces',    value: fileInfo.num_faces },
              { label: 'Vertices', value: fileInfo.num_vertices?.toLocaleString() },
              { label: 'Triangles', value: fileInfo.num_triangles?.toLocaleString() },
              { label: 'Inference', value: `${fileInfo.processing_time_ms}ms` },
              { label: 'Model',    value: fileInfo.model_version },
              { label: 'Confidence', value: `${(fileInfo.confidence * 100).toFixed(0)}%` },
            ].map(item => (
              <div key={item.label} className="flex flex-col p-3 rounded-none bg-dark-700/30">
                <span className="text-[10px] text-dark-300 uppercase tracking-wider">{item.label}</span>
                <span className="text-xs text-dark-100 font-mono font-medium">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Controls hint */}
      <div className="flex flex-col gap-1 pt-1">
        <p className="text-[10px] text-dark-400 uppercase tracking-wider font-semibold">Controls</p>
        <div className="flex flex-col gap-0.5 text-[10px] text-dark-300">
          <span>🖱 Left drag — Orbit</span>
          <span>🖱 Right drag — Pan</span>
          <span>🖱 Scroll — Zoom</span>
        </div>
      </div>
    </div>
  );
}
