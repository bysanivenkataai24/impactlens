import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

import { AboutView } from './components/AboutView';
import { CaseDetails } from './components/CaseDetails';
import { ConnectionView } from './components/ConnectionView';
import { ErrorNotice } from './components/ErrorNotice';
import { ExportPanel } from './components/ExportPanel';
import { FindingsPanel } from './components/FindingsPanel';
import { PhotoStrip } from './components/PhotoStrip';
import { PhotoViewer } from './components/PhotoViewer';
import { SavedCasesView } from './components/SavedCasesView';
import { fetchHealth, predict, type ClientConfig, type UiError } from './lib/api';
import { createDemoCase } from './lib/demo';
import { findingNumbers, findingsFromResponse, isEditedPrediction, manualFinding, modelRunFrom } from './lib/findings';
import { regionFromPercentRect, regionFromPolygon } from './lib/geometry';
import { acceptFiles, sha256Hex, type Rejection } from './lib/photos';
import { loadCase, recallServiceUrl } from './lib/storage';
import { DEFAULT_LIMITS, type HealthResponse, type Point } from './lib/types';
import { caseReducer, emptyCase } from './state/caseReducer';

type View = 'inspect' | 'connection' | 'saved' | 'about';
type Inference = { kind: 'idle' } | { kind: 'running'; startedAt: number } | { kind: 'error'; error: UiError };

const VIEWS: Array<[View, string]> = [
  ['inspect', 'Inspect'],
  ['connection', 'Model connection'],
  ['saved', 'Saved cases'],
  ['about', 'About & limits'],
];

export function App() {
  const [view, setView] = useState<View>('inspect');
  const [inspection, dispatch] = useReducer(caseReducer, undefined, emptyCase);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [config, setConfig] = useState<ClientConfig>(() => ({ baseUrl: recallServiceUrl(), apiKey: '' }));
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [authOk, setAuthOk] = useState<boolean | null>(null);
  const [inference, setInference] = useState<Record<string, Inference>>({});
  const [threshold, setThreshold] = useState<number | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showRejected, setShowRejected] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<Point[]>([]);
  const [rejected, setRejected] = useState<Rejection[]>([]);
  const [announcement, setAnnouncement] = useState('');
  const files = useRef(new Map<string, File>());
  const controllers = useRef(new Map<string, AbortController>());

  const limits = health?.limits ?? DEFAULT_LIMITS;
  const scoreThreshold = threshold ?? limits.score_threshold.default;
  const isDemo = inspection.mode === 'illustrative_demo';
  const current = inspection.photos.find((p) => p.id === currentId) ?? inspection.photos[0] ?? null;
  const numbers = useMemo(() => findingNumbers(inspection.findings), [inspection.findings]);
  const announce = useCallback((message: string) => setAnnouncement(`${message} (${new Date().toLocaleTimeString()})`), []);

  // Read-only health probe of the same-origin (or remembered) service. No key is sent.
  useEffect(() => {
    const controller = new AbortController();
    void fetchHealth(config, 15_000, controller.signal).then((result) => {
      if (result.ok) setHealth(result.value);
    });
    return () => controller.abort();
    // Only on first load; the connection page re-checks explicitly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function releaseFiles() {
    for (const photo of inspection.photos) if (photo.object_url) URL.revokeObjectURL(photo.object_url);
    for (const controller of controllers.current.values()) controller.abort();
    files.current.clear();
    controllers.current.clear();
  }

  function replaceCase(next: typeof inspection, question: string | null) {
    if (question && inspection.photos.length > 0 && inspection.mode === 'live' && !window.confirm(question)) return;
    releaseFiles();
    dispatch({ type: 'replace_case', inspection: next });
    setCurrentId(next.photos[0]?.id ?? null);
    setSelectedId(null);
    setInference({});
    setDrawing(false);
    setDraft([]);
    setRejected([]);
    setView('inspect');
  }

  async function addFiles(list: File[]) {
    if (isDemo || list.length === 0) return;
    const { accepted, rejected: refused } = await acceptFiles(list, { existing: inspection.photos, limits });
    for (const { photo, file } of accepted) files.current.set(photo.id, file);
    if (accepted.length) {
      dispatch({ type: 'add_photos', photos: accepted.map((a) => a.photo) });
      setCurrentId(accepted[0].photo.id);
      setSelectedId(null);
    }
    setRejected(refused);
    announce(`${accepted.length} photo(s) added${refused.length ? `, ${refused.length} not added` : ''}.`);
  }

  function removePhoto(photoId: string) {
    const photo = inspection.photos.find((p) => p.id === photoId);
    const count = inspection.findings.filter((f) => f.photo_id === photoId).length;
    if (count > 0 && !window.confirm(`Remove this photo and its ${count} finding(s)?`)) return;
    controllers.current.get(photoId)?.abort();
    if (photo?.object_url) URL.revokeObjectURL(photo.object_url);
    files.current.delete(photoId);
    dispatch({ type: 'remove_photo', photoId });
    if (currentId === photoId) setCurrentId(inspection.photos.find((p) => p.id !== photoId)?.id ?? null);
    setSelectedId(null);
  }

  async function reattach(photoId: string, file: File) {
    const photo = inspection.photos.find((p) => p.id === photoId);
    if (!photo) return;
    const hash = await sha256Hex(await file.arrayBuffer());
    if (hash !== photo.sha256) {
      setRejected([{ fileName: file.name, reason: 'This is not the same file: its SHA-256 differs from the saved record.' }]);
      return;
    }
    files.current.set(photoId, file);
    dispatch({ type: 'attach_photo', photoId, objectUrl: URL.createObjectURL(file) });
    setRejected([]);
    announce('Photo re-attached.');
  }

  async function runModel() {
    if (!current || isDemo) return;
    const file = files.current.get(current.id);
    if (!file) return;
    const previous = inspection.findings.filter((f) => f.photo_id === current.id && f.origin === 'model_output');
    const reviewed = previous.filter((f) => f.review_status !== 'pending_review' || isEditedPrediction(f) || f.notes || f.part || f.quote_inr !== null);
    if (
      reviewed.length > 0 &&
      !window.confirm(`Running the model again replaces this photo's ${previous.length} model finding(s), including ${reviewed.length} you have reviewed. Continue?`)
    ) {
      return;
    }
    const photoId = current.id;
    const controller = new AbortController();
    controllers.current.set(photoId, controller);
    setInference((s) => ({ ...s, [photoId]: { kind: 'running', startedAt: Date.now() } }));
    announce('Sending the photo to the model.');
    const result = await predict(config, {
      file,
      fileName: current.file_name,
      scoreThreshold,
      timeoutMs: (limits.inference_timeout_s + 45) * 1000,
      maxPolygonPoints: limits.max_polygon_points,
      signal: controller.signal,
    });
    controllers.current.delete(photoId);
    if (!result.ok) {
      setInference((s) => ({ ...s, [photoId]: { kind: 'error', error: result.error } }));
      announce(result.error.title);
      return;
    }
    const response = result.value;
    // Overlays are only trustworthy if the service saw the same bytes, oriented the same way.
    if (response.image.sha256 !== current.sha256 || response.image.width !== current.width || response.image.height !== current.height) {
      const error: UiError = {
        kind: 'invalid_response',
        title: 'Response does not match this photo',
        detail: `The service reports ${response.image.width}×${response.image.height} px (hash ${response.image.sha256.slice(0, 12)}…) but this photo is ${current.width}×${current.height} px (hash ${current.sha256.slice(0, 12)}…). Nothing from this response is shown.`,
        requestId: response.request_id,
      };
      setInference((s) => ({ ...s, [photoId]: { kind: 'error', error } }));
      return;
    }
    dispatch({ type: 'model_result', photoId, run: modelRunFrom(response), findings: findingsFromResponse(photoId, response) });
    setInference((s) => ({ ...s, [photoId]: { kind: 'idle' } }));
    setSelectedId(null);
    announce(response.message);
  }

  function finishDrawing() {
    if (!current) return;
    const region = regionFromPolygon(draft);
    if (!region) {
      announce('A region needs at least three points that enclose an area.');
      return;
    }
    const finding = manualFinding(current.id, region);
    dispatch({ type: 'add_finding', finding });
    setDrawing(false);
    setDraft([]);
    setSelectedId(finding.id);
    announce('Region added as a human annotation. Choose its damage type.');
  }

  const state = current ? (inference[current.id] ?? { kind: 'idle' }) : { kind: 'idle' as const };
  // A live photo has an object URL exactly when its bytes are attached in this tab.
  const canRun = Boolean(current && !isDemo && current.object_url && config.apiKey && state.kind !== 'running');
  const runBlocker = !current
    ? 'Add a photo first.'
    : isDemo
      ? 'Demo photos are illustrations and are never sent to a model.'
      : !current.object_url
        ? 'Re-attach this photo to run the model.'
        : !config.apiKey
          ? 'Enter the service key on the Model connection page.'
          : health && !health.ready
            ? 'The service reports that no usable model is loaded.'
            : null;

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p className="brand-name">ImpactLens</p>
            <p className="brand-tag">Vehicle damage review · decision support</p>
          </div>
        </div>
        <nav aria-label="Main">
          <ul>
            {VIEWS.map(([key, label]) => (
              <li key={key}>
                <button type="button" className="nav-btn" aria-current={view === key ? 'page' : undefined} onClick={() => setView(key)}>
                  {label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
        <p className={`conn-pill ${health?.ready ? 'is-ready' : 'is-not-ready'}`} data-testid="conn-pill">
          {health ? (health.ready ? `Model ready: ${health.model_id}` : 'Service up, no model') : 'Service not connected'}
          {config.apiKey ? ' · key set' : ''}
        </p>
      </header>

      {isDemo && (
        <p className="demo-banner" role="note" data-testid="demo-banner">
          Illustrative demo: the photo is a drawing and every finding is a hand-made sample, not model output.{' '}
          <button type="button" className="link-btn" onClick={() => replaceCase(emptyCase(), null)}>
            Leave demo
          </button>
        </p>
      )}

      <main id="main" tabIndex={-1}>
        {view === 'inspect' && (
          <div className="inspect">
            <div className="case-actions">
              <button type="button" className="btn btn-secondary" onClick={() => replaceCase(emptyCase(), 'Start a new case? Unsaved findings in this case will be lost.')}>
                New case
              </button>
              {!isDemo && (
                <button type="button" className="btn btn-secondary" onClick={() => replaceCase(createDemoCase(), 'Open the illustrative demo? Unsaved findings in this case will be lost.')}>
                  Open illustrative demo
                </button>
              )}
            </div>

            <div className="inspect-grid">
              <PhotoStrip
                photos={inspection.photos}
                findings={inspection.findings}
                currentId={current?.id ?? null}
                canAdd={!isDemo}
                rejected={rejected}
                onSelect={(id) => {
                  setCurrentId(id);
                  setSelectedId(null);
                  setDrawing(false);
                  setDraft([]);
                }}
                onRemove={removePhoto}
                onFiles={(list) => void addFiles(list)}
                onDismissRejected={() => setRejected([])}
              />

              <section className="panel viewer" aria-labelledby="viewer-heading">
                <h2 id="viewer-heading">{current ? `Photo ${inspection.photos.indexOf(current) + 1}: ${current.file_name}` : 'Photo viewer'}</h2>
                {!current ? (
                  <div className="stage-empty">
                    <p>Add vehicle photos to begin, or open the illustrative demo to explore the workflow without a model.</p>
                  </div>
                ) : (
                  <>
                    <div className="toolbar" role="group" aria-label="Viewer controls">
                      <label className="check">
                        <input type="checkbox" checked={showOverlays} onChange={(e) => setShowOverlays(e.target.checked)} /> Overlays
                      </label>
                      <label className="check">
                        <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} /> Show rejected
                      </label>
                      {!drawing ? (
                        <button type="button" className="btn btn-small" onClick={() => { setDrawing(true); setDraft([]); setSelectedId(null); }} disabled={!(current.object_url || current.sample_url)}>
                          Draw region
                        </button>
                      ) : (
                        <>
                          <button type="button" className="btn btn-small btn-primary" onClick={finishDrawing} disabled={draft.length < 3}>
                            Finish region ({draft.length} points)
                          </button>
                          <button type="button" className="btn btn-small" onClick={() => setDraft((d) => d.slice(0, -1))} disabled={draft.length === 0}>
                            Undo point
                          </button>
                          <button type="button" className="btn btn-small" onClick={() => { setDrawing(false); setDraft([]); }}>
                            Cancel
                          </button>
                        </>
                      )}
                    </div>
                    {drawing && <p className="help" role="status">Click or tap the photo to place points around the damage, then choose “Finish region”.</p>}

                    <PhotoViewer
                      photo={current}
                      position={inspection.photos.indexOf(current) + 1}
                      findings={inspection.findings.filter((f) => f.photo_id === current.id)}
                      numbers={numbers}
                      selectedId={selectedId}
                      onSelect={setSelectedId}
                      showOverlays={showOverlays}
                      showRejected={showRejected}
                      drawing={drawing}
                      draft={draft}
                      onAddPoint={(point) => setDraft((d) => [...d, point])}
                      onCancelDrawing={() => { setDrawing(false); setDraft([]); }}
                    />

                    {!current.object_url && !current.sample_url && (
                      <div className="reattach">
                        <label htmlFor="reattach-input">Re-attach {current.file_name}</label>
                        <input id="reattach-input" type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => { const f = e.target.files?.[0]; if (f) void reattach(current.id, f); e.target.value = ''; }} />
                      </div>
                    )}

                    <RectangleForm onAdd={(left, top, width, height) => {
                      const region = regionFromPercentRect(left, top, width, height);
                      if (!region) return false;
                      const finding = manualFinding(current.id, region);
                      dispatch({ type: 'add_finding', finding });
                      setSelectedId(finding.id);
                      announce('Rectangular region added as a human annotation.');
                      return true;
                    }} />

                    <div className="run-box">
                      <div className="threshold">
                        <label htmlFor="threshold">Minimum model score</label>
                        <input
                          id="threshold"
                          type="range"
                          min={limits.score_threshold.min}
                          max={limits.score_threshold.max}
                          step={0.01}
                          value={scoreThreshold}
                          onChange={(e) => setThreshold(Number(e.target.value))}
                          aria-describedby="threshold-help"
                        />
                        <output htmlFor="threshold">{scoreThreshold.toFixed(2)}</output>
                        <p className="help" id="threshold-help">
                          Uncalibrated score; regions below it are not returned. Default {limits.score_threshold.default} (chosen by the service operator).
                        </p>
                      </div>
                      <div className="run-actions">
                        <button type="button" className="btn btn-primary" onClick={() => void runModel()} disabled={!canRun} aria-describedby="run-help" data-testid="run-model">
                          {state.kind === 'running' ? 'Running…' : 'Run model on this photo'}
                        </button>
                        {state.kind === 'running' && (
                          <button type="button" className="btn" onClick={() => controllers.current.get(current.id)?.abort()}>
                            Cancel
                          </button>
                        )}
                        <p className="help" id="run-help">
                          {runBlocker ?? 'Sends only this photo, only when you press the button.'}
                        </p>
                      </div>
                    </div>
                    <div aria-live="polite" data-testid="inference-status">
                      {state.kind === 'running' && <Running startedAt={state.startedAt} />}
                      {state.kind === 'error' && <ErrorNotice error={state.error} onRetry={canRun ? () => void runModel() : undefined} />}
                    </div>
                  </>
                )}
              </section>

              {current && (
                <FindingsPanel
                  inspection={inspection}
                  photoId={current.id}
                  numbers={numbers}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  onUpdate={(findingId, patch) => dispatch({ type: 'update_finding', findingId, patch })}
                  onDelete={(findingId) => {
                    dispatch({ type: 'delete_finding', findingId });
                    setSelectedId(null);
                  }}
                />
              )}
            </div>

            <div className="lower-grid">
              <CaseDetails
                inspection={inspection}
                onDetails={(patch) => dispatch({ type: 'set_details', patch })}
                onVehicle={(patch) => dispatch({ type: 'set_vehicle', patch })}
              />
              <ExportPanel inspection={inspection} currentPhoto={current} announce={announce} />
            </div>
          </div>
        )}
        {view === 'connection' && (
          <ConnectionView
            config={config}
            health={health}
            authOk={authOk}
            onConfig={setConfig}
            onHealth={(h, auth) => {
              setHealth(h);
              setAuthOk(auth);
              setThreshold(null);
            }}
          />
        )}
        {view === 'saved' && (
          <SavedCasesView
            announce={announce}
            onLoad={(id) => {
              const saved = loadCase(id);
              if (saved) replaceCase(saved, 'Load the saved case? Unsaved findings in the current case will be lost.');
            }}
          />
        )}
        {view === 'about' && <AboutView sourceUrl={health?.source_url ?? null} />}
      </main>

      <footer className="footer">
        <p>
          Decision support only: findings need human review and are not a claim decision or repair estimate. Photos are processed in memory and
          not stored by this page.{' '}
          {health?.source_url ? (
            <a href={health.source_url} rel="noopener noreferrer" target="_blank">
              Source code (AGPL-3.0)
            </a>
          ) : (
            'Licensed AGPL-3.0.'
          )}
        </p>
      </footer>
      <p className="sr-only" aria-live="polite" data-testid="announcer">
        {announcement}
      </p>
    </div>
  );
}

function Running({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const seconds = Math.max(0, Math.round((now - startedAt) / 1000));
  return (
    <div className="notice notice-info" role="status">
      <span className="spinner" aria-hidden="true" /> Waiting for the model… {seconds} s
      {seconds >= 15 && <p className="small">A service that was asleep can take a minute or more to start on its first request.</p>}
    </div>
  );
}

function RectangleForm({ onAdd }: { onAdd: (left: number, top: number, width: number, height: number) => boolean }) {
  const [values, setValues] = useState({ left: '', top: '', width: '', height: '' });
  const [error, setError] = useState<string | null>(null);
  const fields: Array<[keyof typeof values, string]> = [
    ['left', 'Left edge (%)'],
    ['top', 'Top edge (%)'],
    ['width', 'Width (%)'],
    ['height', 'Height (%)'],
  ];
  return (
    <details className="rect-form">
      <summary>Add a rectangular region by numbers (keyboard alternative to drawing)</summary>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const ok = onAdd(Number(values.left), Number(values.top), Number(values.width), Number(values.height));
          setError(ok ? null : 'Enter numbers from 0 to 100 with a positive width and height.');
          if (ok) setValues({ left: '', top: '', width: '', height: '' });
        }}
      >
        <div className="rect-fields">
          {fields.map(([key, label]) => (
            <label key={key}>
              {label}
              <input type="number" min={0} max={100} step="0.1" required value={values[key]} onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))} />
            </label>
          ))}
        </div>
        <button type="submit" className="btn btn-small">
          Add region
        </button>
        {error && (
          <p className="notice notice-error" role="alert">
            {error}
          </p>
        )}
      </form>
    </details>
  );
}
