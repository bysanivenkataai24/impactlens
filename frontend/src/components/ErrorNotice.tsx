import { REASON_TEXT, type UiError } from '../lib/api';

export function ErrorNotice({ error, onRetry }: { error: UiError; onRetry?: () => void }) {
  const tone = error.kind === 'cancelled' ? 'notice-info' : error.kind === 'cold_start' || error.kind === 'busy' ? 'notice-warn' : 'notice-error';
  return (
    <div className={`notice ${tone}`} role="alert" data-error-kind={error.kind}>
      <p className="notice-title">{error.title}</p>
      <p>{error.detail}</p>
      {error.reasons && error.reasons.length > 0 && (
        <ul>
          {error.reasons.map((reason) => (
            <li key={reason}>
              <code>{reason}</code>: {REASON_TEXT[reason] ?? 'See the service log.'}
            </li>
          ))}
        </ul>
      )}
      {error.retryAfterS !== undefined && <p>Retry after about {error.retryAfterS} s.</p>}
      {error.requestId && <p className="muted small">Request id: {error.requestId}</p>}
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
