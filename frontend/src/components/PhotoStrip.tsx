import { useRef, useState, type DragEvent } from 'react';

import { photoSource } from '../lib/annotate';
import type { Rejection } from '../lib/photos';
import { MAX_PHOTOS, type Finding, type Photo } from '../lib/types';

interface Props {
  photos: Photo[];
  findings: Finding[];
  currentId: string | null;
  canAdd: boolean;
  rejected: Rejection[];
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onFiles: (files: File[]) => void;
  onDismissRejected: () => void;
}

export function PhotoStrip({ photos, findings, currentId, canAdd, rejected, onSelect, onRemove, onFiles, onDismissRejected }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const full = photos.length >= MAX_PHOTOS;

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (canAdd) onFiles(Array.from(event.dataTransfer.files));
  }

  return (
    <section className="panel photos" aria-labelledby="photos-heading">
      <h2 id="photos-heading">
        Photos <span className="muted">({photos.length} of {MAX_PHOTOS})</span>
      </h2>
      {canAdd && (
        <div
          className={`dropzone${dragging ? ' is-dragging' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          data-testid="dropzone"
        >
          <input
            ref={input}
            id="photo-input"
            className="sr-only"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            disabled={full}
            onChange={(e) => {
              onFiles(Array.from(e.target.files ?? []));
              e.target.value = '';
            }}
          />
          <label htmlFor="photo-input" className={`btn btn-primary${full ? ' is-disabled' : ''}`}>
            Add photos
          </label>
          <p className="help">
            {full ? 'Photo limit reached.' : 'Or drop JPG, PNG or WebP files here. Up to six per case. Nothing is uploaded until you run the model.'}
          </p>
        </div>
      )}
      {rejected.length > 0 && (
        <div className="notice notice-error" role="alert" data-testid="rejected-files">
          <p className="notice-title">Not added</p>
          <ul>
            {rejected.map((r, i) => (
              <li key={`${r.fileName}-${i}`}>
                <strong>{r.fileName}</strong>: {r.reason}
              </li>
            ))}
          </ul>
          <button type="button" className="btn btn-small" onClick={onDismissRejected}>
            Dismiss
          </button>
        </div>
      )}
      {photos.length === 0 ? (
        <p className="empty">No photos yet. Add close-ups and wider context views of the vehicle.</p>
      ) : (
        <ul className="thumbs">
          {photos.map((photo, index) => {
            const count = findings.filter((f) => f.photo_id === photo.id).length;
            const src = photoSource(photo);
            const current = photo.id === currentId;
            return (
              <li key={photo.id} className={current ? 'is-current' : undefined}>
                <button
                  type="button"
                  className="thumb"
                  aria-pressed={current}
                  aria-label={`Photo ${index + 1}: ${photo.file_name}, ${photo.width} by ${photo.height}, ${count} finding(s)${photo.model_run ? ', model run' : ''}`}
                  onClick={() => onSelect(photo.id)}
                  data-testid={`thumb-${index + 1}`}
                >
                  {src ? <img src={src} alt="" /> : <span className="thumb-missing">Not attached</span>}
                  <span className="thumb-meta">
                    {index + 1}. {count} finding{count === 1 ? '' : 's'}
                  </span>
                </button>
                {canAdd && (
                  <button type="button" className="thumb-remove" aria-label={`Remove photo ${index + 1}`} onClick={() => onRemove(photo.id)}>
                    ×
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
