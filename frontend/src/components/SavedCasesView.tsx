import { useState } from 'react';

import { deleteAllCases, deleteCase, listCases, storageAvailable, type StoredCaseSummary } from '../lib/storage';

interface Props {
  onLoad: (id: string) => void;
  announce: (message: string) => void;
}

export function SavedCasesView({ onLoad, announce }: Props) {
  const [cases, setCases] = useState<StoredCaseSummary[]>(() => listCases());
  const available = storageAvailable();

  const refresh = () => setCases(listCases());

  return (
    <div className="view">
      <section className="panel" aria-labelledby="saved-heading">
        <h2 id="saved-heading">Saved in this browser</h2>
        <p className="notice notice-info">
          Developer demo storage. Records live only in this browser's local storage: other people, devices and browsers cannot see them,
          and clearing site data deletes them. Photo bytes are never saved; after loading a case, re-attach each photo (its SHA-256 is
          checked). The service key is never saved.
        </p>
        {!available && <p className="notice notice-warn">Local storage is unavailable in this browser session, so nothing can be saved.</p>}
        {cases.length === 0 ? (
          <p className="empty">No saved cases.</p>
        ) : (
          <ul className="saved-list">
            {cases.map((c) => (
              <li key={c.id}>
                <div>
                  <strong>{c.title || '(untitled case)'}</strong>
                  {c.mode === 'illustrative_demo' && <span className="badge badge-sample">Illustrative demo</span>}
                  <p className="small muted">
                    {c.photos} photo(s), {c.findings} finding(s), updated {new Date(c.updated_at).toLocaleString('en-IN')}
                  </p>
                </div>
                <div className="button-row">
                  <button type="button" className="btn btn-small" onClick={() => onLoad(c.id)}>
                    Load
                  </button>
                  <button
                    type="button"
                    className="btn btn-small btn-danger"
                    onClick={() => {
                      deleteCase(c.id);
                      refresh();
                      announce('Saved case deleted.');
                    }}
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
        <button
          type="button"
          className="btn btn-danger"
          disabled={cases.length === 0}
          onClick={() => {
            if (window.confirm('Delete every ImpactLens case saved in this browser?')) {
              deleteAllCases();
              refresh();
              announce('All saved cases deleted.');
            }
          }}
        >
          Delete all saved cases
        </button>
      </section>
    </div>
  );
}
