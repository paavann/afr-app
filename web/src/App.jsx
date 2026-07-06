import { useState, useCallback } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import LoadingOverlay from './components/LoadingOverlay';
import Viewport3D from './components/Viewport3D';
import FeatureLegend from './components/FeatureLegend';
import { predictCAD, fetchSTL } from './utils/api';
import { getMockResponse } from './utils/mockData';

/**
 * Main application component — single-page CAD Feature Recognizer.
 *
 * State machine:
 *   idle → uploading → processing → complete
 *                                 → error → idle
 */
export default function App() {
  // App state
  const [status, setStatus] = useState('idle'); // idle | uploading | processing | complete | error | mock
  const [fileName, setFileName] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loadingStage, setLoadingStage] = useState(undefined);

  // Result data
  const [predictions, setPredictions] = useState(null);
  const [stlData, setStlData] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  /**
   * Handle file selection (or null for mock data).
   */
  const handleFileSelect = useCallback(async (file) => {
    // Reset error state
    setErrorMsg(null);

    // ── Mock mode ──
    if (file === null) {
      setStatus('processing');
      setFileName('demo_bracket.step');
      setLoadingStage(undefined);

      // Simulate processing delay
      await new Promise(r => setTimeout(r, 4500));

      const mock = getMockResponse();
      setPredictions(mock.data.predictions);
      setFileInfo(mock.data.file_info);
      setStlData(null); // Will use procedural geometry
      setStatus('mock');
      return;
    }

    // ── Real upload ──
    try {
      setFileName(file.name);
      setStatus('uploading');
      setProgress(0);
      setLoadingStage('upload');

      const response = await predictCAD(file, (pct) => {
        setProgress(pct);
        if (pct >= 100) {
          setLoadingStage('parse');
          setStatus('processing');
        }
      });

      // Process response
      setLoadingStage('inference');
      setPredictions(response.data?.predictions || response.predictions);
      setFileInfo(response.data?.file_info || response.file_info);

      // Fetch STL if URL is provided
      const stlUrl = response.data?.mesh_url || response.mesh_url;
      if (stlUrl) {
        setLoadingStage('segment');
        const stlBuffer = await fetchSTL(stlUrl);
        setStlData(stlBuffer);
      } else {
        setStlData(null);
      }

      setStatus('complete');
    } catch (err) {
      console.error('Upload/prediction failed:', err);
      setErrorMsg(
        err.response?.data?.detail ||
        err.message ||
        'Failed to process CAD file. Is the backend running?'
      );
      setStatus('error');

      // Auto-recover after 5s
      setTimeout(() => {
        if (status === 'error') setStatus('idle');
      }, 5000);
    }
  }, []);

  /**
   * Reset to initial state.
   */
  const handleReset = useCallback(() => {
    setStatus('idle');
    setFileName(null);
    setPredictions(null);
    setStlData(null);
    setFileInfo(null);
    setErrorMsg(null);
    setProgress(0);
    setLoadingStage(undefined);
  }, []);

  const showViewport = predictions && (status === 'complete' || status === 'mock');
  const showUpload = status === 'idle' || status === 'error';
  const showLoading = status === 'uploading' || status === 'processing';

  return (
    <div className="flex flex-col min-h-screen">
      <Header status={status} fileName={fileName} />

      {/* ── Loading overlay ── */}
      {showLoading && (
        <LoadingOverlay progress={progress} stage={loadingStage} />
      )}

      {/* ── Main content ── */}
      <main className="flex-1 flex flex-col">
        {/* Upload state */}
        {showUpload && (
          <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 gap-6">
            {/* Hero text */}
            <div className="text-center animate-fade-in-up max-w-xl">
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3 tracking-tight">
                Analyze Your{' '}
                <span className="bg-gradient-to-r from-accent-primary to-accent-bright bg-clip-text text-transparent animate-gradient">
                  CAD Models
                </span>
              </h2>
              <p className="text-dark-200 text-base sm:text-lg leading-relaxed">
                Upload a STEP file and let UV-Net identify fillets, chamfers,
                cylinders, and other geometric features with AI-powered inference.
              </p>
            </div>

            {/* Upload zone */}
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={showLoading}
            />

            {/* Error message */}
            {errorMsg && (
              <div className="animate-fade-in-up max-w-lg w-full px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-300 text-center">
                <span className="font-medium">⚠ Error:</span> {errorMsg}
              </div>
            )}

            {/* Feature chips */}
            <div className="flex flex-wrap gap-2 justify-center mt-4 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
              {['Fillet Detection', 'Chamfer Analysis', 'Surface Classification', 'GPU Inference', 'B-Rep Segmentation'].map(tag => (
                <span
                  key={tag}
                  className="px-3 py-1.5 rounded-full text-[11px] font-medium bg-dark-700/50 text-dark-200 border border-dark-500/30"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Viewport state */}
        {showViewport && (
          <div className="flex-1 flex relative">
            {/* 3D Canvas — fills remaining space */}
            <div className="flex-1 min-h-[500px] p-3">
              <Viewport3D stlData={stlData} predictions={predictions} />
            </div>

            {/* Legend panel — overlaid on the right */}
            <div className="absolute top-4 right-4 z-10">
              <FeatureLegend
                predictions={predictions}
                fileInfo={fileInfo}
                onReset={handleReset}
              />
            </div>

            {/* Mode badge */}
            {status === 'mock' && (
              <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10">
                <div className="glass-light px-4 py-2 rounded-full text-xs font-medium text-accent-secondary flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                  Demo Mode — Using procedural geometry
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer — subtle */}
      <footer className="px-6 py-3 border-t border-dark-700/30 text-center">
        <p className="text-[11px] text-dark-400">
          CAD Feature Recognizer · Powered by UV-Net · Built with React & Three.js
        </p>
      </footer>
    </div>
  );
}
