import { useState } from 'react';

import { checkAuth, fetchHealth, normalizeBaseUrl, REASON_TEXT, type ClientConfig, type UiError } from '../lib/api';
import { formatBytes } from '../lib/photos';
import { rememberServiceUrl } from '../lib/storage';
import type { HealthResponse } from '../lib/types';
import { ErrorNotice } from './ErrorNotice';

interface Props {
  config: ClientConfig;
  health: HealthResponse | null;
  authOk: boolean | null;
  onConfig: (config: ClientConfig) => void;
  onHealth: (health: HealthResponse | null, authOk: boolean | null) => void;
}

export function ConnectionView({ config, health, authOk, onConfig, onHealth }: Props) {
  const [url, setUrl] = useState(config.baseUrl);
  const [key, setKey] = useState(config.apiKey);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [error, setError] = useState<UiError | null>(null);
  const [checking, setChecking] = useState(false);

  async function connect() {
    const normalized = normalizeBaseUrl(url);
    if ('error' in normalized) {
      setUrlError(normalized.error);
      return;
    }
    setUrlError(null);
    const next = { baseUrl: normalized.url, apiKey: key.trim() };
    onConfig(next);
    rememberServiceUrl(next.baseUrl);
    setChecking(true);
    setError(null);
    const healthResult = await fetchHealth(next);
    if (!healthResult.ok) {
      setChecking(false);
      setError(healthResult.error);
      onHealth(null, null);
      return;
    }
    let auth: boolean | null = null;
    if (next.apiKey) {
      const authResult = await checkAuth(next);
      auth = authResult.ok;
      if (!authResult.ok) setError(authResult.error);
    }
    setChecking(false);
    onHealth(healthResult.value, auth);
  }

  function forgetKey() {
    setKey('');
    onConfig({ ...config, apiKey: '' });
    onHealth(health, null);
  }

  return (
    <div className="view connection">
      <section className="panel" aria-labelledby="connection-heading">
        <h2 id="connection-heading">Model connection</h2>
        <p>
          Predictions come only from an ImpactLens inference service. Leave the URL empty when this page is served by the service itself
          (for example on its Hugging Face Space).
        </p>
        <form
          className="field-grid"
          onSubmit={(event) => {
            event.preventDefault();
            void connect();
          }}
        >
          <label htmlFor="service-url">Service URL</label>
          <div>
            <input
              id="service-url"
              type="url"
              inputMode="url"
              placeholder="Same origin as this page"
              value={url}
              aria-invalid={urlError ? true : undefined}
              aria-describedby="service-url-help"
              onChange={(e) => setUrl(e.target.value)}
            />
            <p className="help" id="service-url-help">
              {urlError ?? 'https:// only (http is accepted for localhost). Another origin must list this page in its ALLOWED_ORIGINS.'}
            </p>
          </div>
          <label htmlFor="service-key">Service key</label>
          <div>
            <input
              id="service-key"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={key}
              aria-describedby="service-key-help"
              onChange={(e) => setKey(e.target.value)}
            />
            <p className="help" id="service-key-help">
              Developer-pilot access. The key stays in this tab's memory only: it is not saved, not put in the URL and is sent only to the
              service URL above. Closing or reloading the tab forgets it. A public product needs per-user sign-in instead of a shared key.
            </p>
          </div>
          <div className="button-row span-2">
            <button type="submit" className="btn btn-primary" disabled={checking}>
              {checking ? 'Checking…' : 'Check connection'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={forgetKey} disabled={!config.apiKey && !key}>
              Forget key
            </button>
          </div>
        </form>
        <div aria-live="polite">
          {error && <ErrorNotice error={error} onRetry={() => void connect()} />}
        </div>
      </section>

      {health && (
        <section className="panel" aria-labelledby="health-heading" data-testid="health">
          <h2 id="health-heading">Service status</h2>
          <p className={`notice ${health.ready ? 'notice-ok' : 'notice-error'}`} data-testid="readiness">
            {health.ready ? 'Ready: the service has a verified model loaded.' : 'Not ready: the service cannot make predictions.'}
            {authOk === true && ' Service key accepted.'}
            {authOk === false && ' Service key rejected.'}
          </p>
          {health.reasons.length > 0 && (
            <ul>
              {health.reasons.map((reason) => (
                <li key={reason}>
                  <code>{reason}</code>: {REASON_TEXT[reason] ?? 'See the service log.'}
                </li>
              ))}
            </ul>
          )}
          <dl className="kv">
            <dt>Model id</dt>
            <dd>{health.model_id ?? '—'}</dd>
            <dt>Model revision</dt>
            <dd>{health.model_revision ?? '—'}</dd>
            <dt>Checkpoint SHA-256</dt>
            <dd className="mono">{health.checkpoint_sha256 ?? '—'}</dd>
            <dt>Task / device</dt>
            <dd>
              {health.task ?? '—'} / {health.device ?? '—'}
            </dd>
            <dt>Classes</dt>
            <dd>{health.classes.length ? health.classes.join(', ') : '—'}</dd>
            <dt>Upload limit</dt>
            <dd>
              {formatBytes(health.limits.max_upload_bytes)}, {health.limits.max_image_pixels.toLocaleString('en-IN')} pixels
            </dd>
            <dt>Score threshold</dt>
            <dd>
              default {health.limits.score_threshold.default}, range {health.limits.score_threshold.min}–{health.limits.score_threshold.max}
            </dd>
            <dt>Service version</dt>
            <dd>{health.service_version}</dd>
            {health.source_url && (
              <>
                <dt>Source code</dt>
                <dd>
                  <a href={health.source_url} rel="noopener noreferrer" target="_blank">
                    {health.source_url}
                  </a>
                </dd>
              </>
            )}
          </dl>
        </section>
      )}
    </div>
  );
}
