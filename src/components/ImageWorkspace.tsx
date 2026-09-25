import { useCallback, useEffect, useRef, useState } from 'react';
import { MoveIcon, MaximizeIcon, RefreshIcon, ZoomInIcon, ZoomOutIcon } from './icons';

export type ComparisonMode = 'side-by-side' | 'before-after' | 'split';

interface ImageWorkspaceProps {
  original: ImageData | null;
  converted: ImageData | null;
  isProcessing: boolean;
  presetLabel: string;
}

const TABS: { id: ComparisonMode; label: string }[] = [
  { id: 'side-by-side', label: 'Side by Side' },
  { id: 'before-after', label: 'Before / After' },
  { id: 'split', label: 'Split View' },
];

export default function ImageWorkspace({ original, converted, isProcessing, presetLabel }: ImageWorkspaceProps) {
  const [mode, setMode] = useState<ComparisonMode>('side-by-side');
  const containerRef = useRef<HTMLDivElement>(null);
  const stageWrapRef = useRef<HTMLDivElement>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const convertedCanvasRef = useRef<HTMLCanvasElement>(null);
  const originalCanvasRef2 = useRef<HTMLCanvasElement>(null); // second copy for side-by-side left pane

  const [sliderPct, setSliderPct] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const draggingSlider = useRef(false);
  const panState = useRef<{ dragging: boolean; startX: number; startY: number; origin: { x: number; y: number } }>({
    dragging: false,
    startX: 0,
    startY: 0,
    origin: { x: 0, y: 0 },
  });

  const drawInto = useCallback((canvas: HTMLCanvasElement | null, data: ImageData | null) => {
    if (!canvas || !data) return;
    canvas.width = data.width;
    canvas.height = data.height;
    canvas.getContext('2d')?.putImageData(data, 0, 0);
  }, []);

  useEffect(() => {
    drawInto(originalCanvasRef.current, original);
    drawInto(originalCanvasRef2.current, original);
  }, [original, drawInto, mode]);

  useEffect(() => {
    drawInto(convertedCanvasRef.current, converted);
  }, [converted, drawInto, mode]);

  const updateSliderFromClientX = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSliderPct(Math.min(100, Math.max(0, pct)));
  }, []);

  const onSliderPointerDown = (e: React.PointerEvent) => {
    draggingSlider.current = true;
    (e.target as Element).setPointerCapture(e.pointerId);
    updateSliderFromClientX(e.clientX);
  };
  const onSliderPointerMove = (e: React.PointerEvent) => {
    if (!draggingSlider.current) return;
    updateSliderFromClientX(e.clientX);
  };
  const onSliderPointerUp = (e: React.PointerEvent) => {
    draggingSlider.current = false;
    (e.target as Element).releasePointerCapture(e.pointerId);
  };
  const onSliderKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setSliderPct((p) => Math.max(0, p - 2));
    if (e.key === 'ArrowRight') setSliderPct((p) => Math.min(100, p + 2));
    if (e.key === 'Home') setSliderPct(0);
    if (e.key === 'End') setSliderPct(100);
  };

  const onPanPointerDown = (e: React.PointerEvent) => {
    if (zoom <= 1) return;
    panState.current = { dragging: true, startX: e.clientX, startY: e.clientY, origin: pan };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };
  const onPanPointerMove = (e: React.PointerEvent) => {
    if (!panState.current.dragging) return;
    const dx = e.clientX - panState.current.startX;
    const dy = e.clientY - panState.current.startY;
    setPan({ x: panState.current.origin.x + dx, y: panState.current.origin.y + dy });
  };
  const onPanPointerUp = (e: React.PointerEvent) => {
    panState.current.dragging = false;
    (e.currentTarget as Element).releasePointerCapture(e.pointerId);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    const el = stageWrapRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => undefined);
    } else {
      el.requestFullscreen?.().catch(() => undefined);
    }
  };

  if (!original) return null;

  const stageStyle: React.CSSProperties = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    cursor: zoom > 1 ? 'grab' : 'default',
  };

  return (
    <section className="panel workspace" aria-labelledby="workspace-heading">
      <h2 id="workspace-heading" className="visually-hidden">
        Comparison workspace
      </h2>
      <div className="workspace-tabs" role="tablist" aria-label="Comparison mode">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={mode === tab.id}
            className={`workspace-tab ${mode === tab.id ? 'workspace-tab--active' : ''}`}
            onClick={() => setMode(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="workspace__stage-wrap" ref={stageWrapRef}>
        <div
          className="workspace__stage-inner"
          ref={containerRef}
          onPointerDown={onPanPointerDown}
          onPointerMove={onPanPointerMove}
          onPointerUp={onPanPointerUp}
        >
          {mode === 'side-by-side' && (
            <div className="workspace__side-by-side" style={stageStyle}>
              <div className="workspace__side-pane">
                <canvas ref={originalCanvasRef2} className="workspace__canvas" aria-label="Original infrared image" />
                <span className="workspace__pane-label workspace__pane-label--left">Original IR Image</span>
              </div>
              <div className="workspace__side-pane">
                {converted && (
                  <canvas ref={convertedCanvasRef} className="workspace__canvas" aria-label="Converted visible-light approximation" />
                )}
                <span className="workspace__pane-label workspace__pane-label--right">
                  Converted Visible ({presetLabel})
                </span>
              </div>
            </div>
          )}

          {(mode === 'before-after' || mode === 'split') && (
            <>
              <div className="workspace__stage" style={stageStyle}>
                <canvas ref={originalCanvasRef} className="workspace__canvas" aria-label="Original infrared image" />
                {converted && (
                  <canvas
                    ref={convertedCanvasRef}
                    className="workspace__canvas workspace__canvas--overlay"
                    style={{ clipPath: `inset(0 ${100 - (mode === 'split' ? 50 : sliderPct)}% 0 0)` }}
                    aria-label="Converted visible-light approximation"
                  />
                )}
              </div>

              {converted && mode === 'before-after' && (
                <div
                  className="workspace__slider-handle"
                  style={{ left: `${sliderPct}%` }}
                  role="slider"
                  tabIndex={0}
                  aria-label="Before/after comparison position"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(sliderPct)}
                  onPointerDown={onSliderPointerDown}
                  onPointerMove={onSliderPointerMove}
                  onPointerUp={onSliderPointerUp}
                  onKeyDown={onSliderKeyDown}
                >
                  <div className="workspace__slider-line" />
                  <div className="workspace__slider-grip">&#8596;</div>
                </div>
              )}
              {converted && mode === 'split' && (
                <div className="workspace__slider-handle workspace__slider-handle--static" style={{ left: '50%' }}>
                  <div className="workspace__slider-line" />
                </div>
              )}

              <div className="workspace__labels" aria-hidden="true">
                <span>Original IR Image</span>
                {converted && <span>Converted Visible ({presetLabel})</span>}
              </div>
            </>
          )}

          {isProcessing && (
            <div className="workspace__processing-badge" role="status">
              Processing&hellip;
            </div>
          )}
        </div>
      </div>

      <div className="workspace__toolbar">
        <div className="workspace__toolbar-group">
          <button
            type="button"
            className="tool-btn"
            aria-pressed={zoom > 1}
            title={zoom > 1 ? 'Center pan position' : 'Zoom in to enable panning'}
            onClick={() => (zoom > 1 ? setPan({ x: 0, y: 0 }) : setZoom(1.5))}
          >
            <MoveIcon size={16} />
          </button>
          <button type="button" className="tool-btn" onClick={() => setZoom((z) => Math.max(0.25, z / 1.25))} aria-label="Zoom out">
            <ZoomOutIcon size={16} />
          </button>
          <button type="button" className="tool-btn tool-btn--readout" onClick={() => setZoom(1)}>
            {Math.round(zoom * 100)}%
          </button>
          <button type="button" className="tool-btn" onClick={() => setZoom((z) => Math.min(6, z * 1.25))} aria-label="Zoom in">
            <ZoomInIcon size={16} />
          </button>
          <button type="button" className="tool-btn tool-btn--pill" onClick={resetView}>
            Fit
          </button>
          <button type="button" className="tool-btn" onClick={resetView} title="Reset view">
            <RefreshIcon size={15} />
          </button>
        </div>
        <div className="workspace__toolbar-group">
          <span className="workspace__dims">
            {original.width} &times; {original.height}
          </span>
          <button type="button" className="tool-btn" onClick={toggleFullscreen} aria-label="Toggle fullscreen">
            <MaximizeIcon size={16} />
          </button>
        </div>
      </div>
    </section>
  );
}
